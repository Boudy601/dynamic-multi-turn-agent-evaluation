#!/usr/bin/env python3
"""Serve the local-network manual chat review UI."""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_SCRIPT_DIR = PROJECT_ROOT / "src" / "dataset_generation"
sys.path.insert(0, str(ASSIGNMENT_SCRIPT_DIR))

from assign import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    DEFAULT_ALIASES,
    DEFAULT_GUIDELINES,
    DEFAULT_SOURCE,
    AssignmentError,
    Chat,
    InputDataError,
    SopResolver,
    build_sop_resolver,
    load_reviewers,
    load_source_chats,
    read_assignment_folders,
    resolve_chat_guideline,
    run_top_up,
    validate_reviewer_name,
)


ASSIGNMENTS_DIR = PROJECT_ROOT / "assignments"
REVIEWERS_FILE = PROJECT_ROOT / "config" / "reviewers.json"
REVIEWS_FILE = ASSIGNMENTS_DIR / "reviews.json"
PAGE_FILE = PROJECT_ROOT / "tools" / "chat_review.html"
VALID_STATUSES = {"error", "no_error"}
MAX_REQUEST_BYTES = 64 * 1024


class ApiError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class AppContext:
    reviewers: list[str]
    resolver: SopResolver
    canonical_names: dict[str, str]
    sop_ids: list[str]
    source_chats: list[Chat]
    assignments_dir: Path = ASSIGNMENTS_DIR
    reviews_file: Path = REVIEWS_FILE
    lock: Any = field(default_factory=threading.RLock, repr=False)


def load_reviews(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ApiError(f"Invalid JSON in {path.name}: {exc}", 500) from exc
    except OSError as exc:
        raise ApiError(f"Unable to read {path.name}: {exc}", 500) from exc
    if not isinstance(data, Mapping):
        raise ApiError(f"{path.name} must contain a JSON object", 500)

    reviews: dict[str, dict[str, str]] = {}
    for reviewer, entries in data.items():
        if not isinstance(reviewer, str) or not isinstance(entries, Mapping):
            raise ApiError(f"{path.name} contains an invalid reviewer entry", 500)
        reviews[reviewer] = {}
        for chat_id, status in entries.items():
            if (
                not isinstance(chat_id, str)
                or not isinstance(status, str)
                or status not in VALID_STATUSES
            ):
                raise ApiError(f"{path.name} contains an invalid review status", 500)
            reviews[reviewer][chat_id] = status
    return reviews


def save_reviews(path: Path, reviews: Mapping[str, Mapping[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=".reviews-",
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(reviews, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
    except OSError as exc:
        raise ApiError(f"Unable to save {path.name}: {exc}", 500) from exc
    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def message_list(record: Mapping[str, Any]) -> list[dict[str, str]]:
    messages = record.get("original")
    if messages is None:
        messages = record.get("messages", record.get("turns", record.get("delexed")))
    if not isinstance(messages, list):
        raise AssignmentError("Chat has no valid message list")

    result: list[dict[str, str]] = []
    for position, message in enumerate(messages, start=1):
        if isinstance(message, Mapping):
            speaker = message.get("speaker", "unknown")
            content = message.get("text", message.get("content"))
        elif isinstance(message, (list, tuple)) and len(message) >= 2:
            speaker, content = message[0], message[1]
        else:
            raise AssignmentError(f"Invalid message at position {position}")
        if content is None:
            raise AssignmentError(f"Message at position {position} has no text")
        result.append({"speaker": str(speaker), "text": str(content)})
    return result


def create_app_context() -> AppContext:
    reviewers = load_reviewers(REVIEWERS_FILE)
    resolver, canonical_names, sop_ids = build_sop_resolver(
        DEFAULT_GUIDELINES, DEFAULT_ALIASES
    )
    source_chats = load_source_chats(DEFAULT_SOURCE, resolver)
    return AppContext(
        reviewers=reviewers,
        resolver=resolver,
        canonical_names=canonical_names,
        sop_ids=sop_ids,
        source_chats=source_chats,
    )


class ReviewRequestHandler(BaseHTTPRequestHandler):
    app: AppContext

    def _send_json(self, status_code: int, payload: Mapping[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_error(self, error: ApiError) -> None:
        self._send_json(error.status_code, {"error": str(error)})

    def _validate_reviewer(self, value: Any) -> str:
        if not isinstance(value, str):
            raise ApiError("A reviewer name is required")
        try:
            reviewer = validate_reviewer_name(value)
        except AssignmentError as exc:
            raise ApiError(str(exc)) from exc
        if reviewer not in self.app.reviewers:
            raise ApiError("Unknown reviewer", 404)
        return reviewer

    def _read_json_body(self) -> dict[str, Any]:
        content_type = self.headers.get("Content-Type", "").split(";", 1)[0].strip()
        if content_type != "application/json":
            raise ApiError("Content-Type must be application/json", 415)
        try:
            content_length = int(self.headers.get("Content-Length", ""))
        except ValueError as exc:
            raise ApiError("A valid Content-Length header is required") from exc
        if content_length < 1 or content_length > MAX_REQUEST_BYTES:
            raise ApiError("Request body size is invalid")
        try:
            data = json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ApiError("Request body must contain valid JSON") from exc
        if not isinstance(data, dict):
            raise ApiError("Request body must be a JSON object")
        return data

    def _reviewer_chats(self, reviewer: str) -> list[dict[str, Any]]:
        _, reviewer_chats = read_assignment_folders(
            self.app.assignments_dir, self.app.resolver
        )
        reviews = load_reviews(self.app.reviews_file)
        statuses = reviews.get(reviewer, {})
        chats = [
            {
                "chat_id": chat.chat_id,
                "sop_id": chat.sop_id,
                "sop_name": self.app.canonical_names[chat.sop_id],
                "message_count": chat.message_count,
                "status": statuses.get(chat.chat_id, "pending"),
                "scenario": chat.record.get("scenario", {}),
                "sop_guideline": resolve_chat_guideline(
                    chat.record, 0, self.app.resolver
                ),
                "messages": message_list(chat.record),
            }
            for chat in reviewer_chats.get(reviewer, [])
        ]
        return sorted(
            chats,
            key=lambda chat: (
                -chat["message_count"],
                chat["sop_name"].casefold(),
                chat["chat_id"],
            ),
        )

    def _require_assigned_chat(
        self, reviewer: str, chat_id: Any
    ) -> tuple[str, list[dict[str, Any]]]:
        if not isinstance(chat_id, str) or not chat_id:
            raise ApiError("A chat_id is required")
        chats = self._reviewer_chats(reviewer)
        if not any(chat["chat_id"] == chat_id for chat in chats):
            raise ApiError("Chat is not assigned to this reviewer", 404)
        return chat_id, chats

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/":
            self._serve_page()
            return
        prefix = "/api/chats/"
        if path.startswith(prefix):
            reviewer_name = unquote(path[len(prefix) :])
            if not reviewer_name or "/" in reviewer_name:
                self._send_error(ApiError("Invalid reviewer path"))
                return
            try:
                reviewer = self._validate_reviewer(reviewer_name)
                with self.app.lock:
                    chats = self._reviewer_chats(reviewer)
                self._send_json(200, {"reviewer": reviewer, "chats": chats})
            except (ApiError, AssignmentError, InputDataError, OSError) as exc:
                error = exc if isinstance(exc, ApiError) else ApiError(str(exc), 500)
                self._send_error(error)
            return
        self._send_error(ApiError("Not found", 404))

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        try:
            data = self._read_json_body()
            if path == "/api/add-chats":
                self._add_chats(data)
            elif path == "/api/review":
                self._save_review(data)
            elif path == "/api/rollback":
                self._rollback_review(data)
            else:
                raise ApiError("Not found", 404)
        except (ApiError, AssignmentError, InputDataError, OSError) as exc:
            error = exc if isinstance(exc, ApiError) else ApiError(str(exc), 500)
            self._send_error(error)

    def _add_chats(self, data: Mapping[str, Any]) -> None:
        reviewer = self._validate_reviewer(data.get("reviewer"))
        count = data.get("count")
        if type(count) is not int or count < 1:
            raise ApiError("count must be a positive integer")
        with self.app.lock:
            added, _, _ = run_top_up(
                self.app.source_chats,
                reviewer,
                count,
                self.app.assignments_dir,
                self.app.resolver,
            )
            chats = self._reviewer_chats(reviewer)
        warning = (
            f"Only {len(added)} of {count} requested chats were available."
            if len(added) < count
            else None
        )
        self._send_json(
            200,
            {
                "reviewer": reviewer,
                "added": len(added),
                "added_chat_ids": [chat.chat_id for chat in added],
                "warning": warning,
                "chats": chats,
            },
        )

    def _save_review(self, data: Mapping[str, Any]) -> None:
        reviewer = self._validate_reviewer(data.get("reviewer"))
        chat_id, _ = self._require_assigned_chat(reviewer, data.get("chat_id"))
        status = data.get("status")
        if status not in VALID_STATUSES:
            raise ApiError("status must be 'error' or 'no_error'")
        with self.app.lock:
            reviews = load_reviews(self.app.reviews_file)
            reviews.setdefault(reviewer, {})[chat_id] = status
            save_reviews(self.app.reviews_file, reviews)
            chats = self._reviewer_chats(reviewer)
        self._send_json(
            200,
            {"reviewer": reviewer, "chat_id": chat_id, "status": status, "chats": chats},
        )

    def _rollback_review(self, data: Mapping[str, Any]) -> None:
        reviewer = self._validate_reviewer(data.get("reviewer"))
        chat_id, _ = self._require_assigned_chat(reviewer, data.get("chat_id"))
        with self.app.lock:
            reviews = load_reviews(self.app.reviews_file)
            reviewer_reviews = reviews.get(reviewer, {})
            reviewer_reviews.pop(chat_id, None)
            if not reviewer_reviews:
                reviews.pop(reviewer, None)
            save_reviews(self.app.reviews_file, reviews)
            chats = self._reviewer_chats(reviewer)
        self._send_json(
            200,
            {
                "reviewer": reviewer,
                "chat_id": chat_id,
                "status": "pending",
                "chats": chats,
            },
        )

    def _serve_page(self) -> None:
        try:
            page = PAGE_FILE.read_text(encoding="utf-8")
        except OSError as exc:
            self._send_error(ApiError(f"Unable to load review page: {exc}", 500))
            return
        reviewer_json = json.dumps(self.app.reviewers, ensure_ascii=False)
        page = page.replace("__REVIEWERS_JSON__", reviewer_json)
        body = page.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:
        sys.stdout.write(f"{self.address_string()} - {format % args}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def print_urls(host: str, port: int) -> None:
    print(f"Local: http://127.0.0.1:{port}")
    if host not in {"0.0.0.0", "::"}:
        print(f"Network: http://{host}:{port}")
        return
    try:
        addresses = {
            info[4][0]
            for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)
            if not info[4][0].startswith("127.")
        }
    except OSError:
        addresses = set()
    for address in sorted(addresses):
        print(f"Network: http://{address}:{port}")
    if not addresses:
        print("Network: use this computer's LAN IP with the port shown above.")


def main() -> int:
    args = parse_args()
    if not 1 <= args.port <= 65535:
        print("Error: --port must be between 1 and 65535", file=sys.stderr)
        return 2
    try:
        app = create_app_context()
    except (AssignmentError, InputDataError, OSError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    ReviewRequestHandler.app = app
    try:
        with ThreadingHTTPServer((args.host, args.port), ReviewRequestHandler) as server:
            print_urls(args.host, args.port)
            print("Press Ctrl+C to stop the server.")
            server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping review server.")
    except OSError as exc:
        print(f"Error: unable to start server: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
