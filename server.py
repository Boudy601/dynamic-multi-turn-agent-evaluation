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
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import unquote, urlsplit


PROJECT_ROOT = Path(__file__).resolve().parent
ASSIGNMENT_SCRIPT_DIR = PROJECT_ROOT / "src" / "dataset_generation"
sys.path.insert(0, str(ASSIGNMENT_SCRIPT_DIR))

from assign import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    DEFAULT_ALIASES,
    DEFAULT_REVIEWER_SOPS,
    DEFAULT_SOURCE,
    AssignmentError,
    Chat,
    SopResolver,
    build_sop_resolver,
    load_reviewer_sops,
    load_reviewers,
    load_source_chats,
    read_assignment_folders,
    resolve_chat_guideline,
    run_top_up,
    validate_reviewer_name,
)
from prepare_prompts import (  # noqa: E402  # pyright: ignore[reportMissingImports]
    InputDataError,
    load_json,
    normalize_name,
)


ASSIGNMENTS_DIR = PROJECT_ROOT / "assignments"
REVIEWERS_FILE = PROJECT_ROOT / "config" / "reviewers.json"
REVIEWER_SOPS_FILE = DEFAULT_REVIEWER_SOPS
GUIDELINES_FILE = (
    PROJECT_ROOT / "data" / "source" / "abcd" / "data" / "guidelines_full_context.json"
)
ACTION_SCHEMA_FILE = PROJECT_ROOT / "config" / "subflow_actions_schema.json"
REVIEWS_FILE = ASSIGNMENTS_DIR / "reviews.json"
PAGE_FILE = PROJECT_ROOT / "tools" / "chat_review.html"
VALID_STATUSES = {"error", "no_error"}
VALID_REVIEWED_BY = {"ai", "human", "ai_human"}
VALID_SPEAKERS = {"agent", "customer", "action"}
MAX_REVIEW_REASON_LENGTH = 300
MAX_REQUEST_BYTES = 256 * 1024


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def parse_edited_timestamp(value: Any) -> float:
    if not isinstance(value, str) or not value.strip():
        return 0.0
    try:
        parsed = datetime.fromisoformat(value.strip())
    except ValueError:
        return 0.0
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


class ApiError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class AppContext:
    reviewers: list[str]
    reviewer_sops: dict[str, list[str]]
    resolver: SopResolver
    canonical_names: dict[str, str]
    sop_ids: list[str]
    source_chats: list[Chat]
    action_schema: dict[str, dict[str, dict[str, list[str]]]]
    action_display_names: dict[str, str] = field(default_factory=dict)
    assignments_dir: Path = ASSIGNMENTS_DIR
    reviews_file: Path = REVIEWS_FILE
    lock: Any = field(default_factory=threading.RLock, repr=False)


def load_reviews(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
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

    reviews: dict[str, dict[str, dict[str, Any]]] = {}
    for reviewer, entries in data.items():
        if not isinstance(reviewer, str) or not isinstance(entries, Mapping):
            raise ApiError(f"{path.name} contains an invalid reviewer entry", 500)
        reviews[reviewer] = {}
        for chat_id, review in entries.items():
            if (
                not isinstance(chat_id, str)
                or not isinstance(review, (str, Mapping))
            ):
                raise ApiError(f"{path.name} contains an invalid review status", 500)
            if isinstance(review, str):
                status = review
                reason = ""
                edited = False
                edited_at = ""
                reviewed_by = ""
                reviewed_at = ""
            else:
                status = review.get("status")
                reason = review.get("reason", "")
                edited = bool(review.get("edited", False))
                raw_edited_at = review.get("edited_at", "")
                edited_at = raw_edited_at if isinstance(raw_edited_at, str) else ""
                raw_reviewed_by = review.get("reviewed_by", "")
                reviewed_by = (
                    raw_reviewed_by
                    if isinstance(raw_reviewed_by, str) and raw_reviewed_by in VALID_REVIEWED_BY
                    else ""
                )
                raw_reviewed_at = review.get("reviewed_at", "")
                reviewed_at = raw_reviewed_at if isinstance(raw_reviewed_at, str) else ""
            if (
                not isinstance(status, str)
                or status not in VALID_STATUSES
                or not isinstance(reason, str)
                or len(reason) > MAX_REVIEW_REASON_LENGTH
            ):
                raise ApiError(f"{path.name} contains an invalid review status", 500)
            entry: dict[str, Any] = {"status": status, "reason": reason}
            if edited:
                entry["edited"] = True
                if edited_at:
                    entry["edited_at"] = edited_at
            if reviewed_by:
                entry["reviewed_by"] = reviewed_by
            if reviewed_at:
                entry["reviewed_at"] = reviewed_at
            reviews[reviewer][chat_id] = entry
    return reviews


def save_json_atomic(path: Path, data: Any, prefix: str = ".tmp-") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=prefix,
            suffix=".tmp",
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            json.dump(data, file, ensure_ascii=False, indent=2)
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


def save_reviews(
    path: Path,
    reviews: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> None:
    save_json_atomic(path, reviews, prefix=".reviews-")


def validate_edited_messages(raw_messages: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ApiError("messages must be a non-empty list")

    validated: list[dict[str, Any]] = []
    for index, item in enumerate(raw_messages, start=1):
        if not isinstance(item, Mapping):
            raise ApiError(f"Message #{index} must be an object")
        speaker_raw = item.get("speaker")
        if not isinstance(speaker_raw, str):
            raise ApiError(f"Message #{index} is missing a valid speaker")
        speaker = speaker_raw.strip().casefold()
        if speaker not in VALID_SPEAKERS:
            raise ApiError(
                f"Message #{index} speaker must be 'agent', 'customer', or 'action'"
            )
        text = item.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ApiError(f"Message #{index} must have non-empty text")

        entry: dict[str, Any] = {"speaker": speaker, "text": text}
        if speaker == "action":
            function_name = item.get("function_name", "")
            if function_name is None:
                function_name = ""
            if not isinstance(function_name, str):
                raise ApiError(f"Action message #{index} function_name must be a string")
            parameters = item.get("parameters", [])
            if parameters is None:
                parameters = []
            if not isinstance(parameters, list) or any(
                not isinstance(param, str) for param in parameters
            ):
                raise ApiError(
                    f"Action message #{index} parameters must be a list of strings"
                )
            entry["function_name"] = function_name.strip()
            entry["parameters"] = list(parameters)
        validated.append(entry)
    return validated


def apply_edited_messages(
    record: Mapping[str, Any],
    messages: Sequence[Mapping[str, Any]],
    edited_at: str | None = None,
) -> dict[str, Any]:
    updated = dict(record)
    scenario = updated.get("scenario")
    subflow = (
        scenario.get("subflow", "") if isinstance(scenario, Mapping) else ""
    )
    updated["original"] = [[msg["speaker"], msg["text"]] for msg in messages]
    delexed: list[dict[str, Any]] = []
    for turn_count, msg in enumerate(messages, start=1):
        if msg["speaker"] == "action":
            function_name = msg.get("function_name") or None
            parameters = list(msg.get("parameters", []))
            targets: list[Any] = [subflow, None, function_name, parameters, -1]
        else:
            targets = [subflow, None, None, [], -1]
        delexed.append({
            "speaker": msg["speaker"],
            "text": msg["text"].lower(),
            "turn_count": turn_count,
            "targets": targets,
        })
    updated["delexed"] = delexed
    updated["edited_at"] = edited_at or utc_now_iso()
    return updated


def validate_review_entry(status: Any, reason: Any = "") -> dict[str, Any]:
    if not isinstance(status, str) or status not in VALID_STATUSES:
        raise ApiError("status must be 'error' or 'no_error'")
    if reason is None:
        reason = ""
    if not isinstance(reason, str):
        raise ApiError("reason must be a string")
    reason = reason.strip()
    if status == "error" and not reason:
        raise ApiError(
            "A short reason is required when marking a chat as having an error"
        )
    if len(reason) > MAX_REVIEW_REASON_LENGTH:
        raise ApiError(
            f"Reason must be {MAX_REVIEW_REASON_LENGTH} characters or fewer"
        )
    return {"status": status, "reason": reason}


def message_list(record: Mapping[str, Any]) -> list[dict[str, Any]]:
    original_messages = record.get("original")
    delexed_messages = record.get("delexed")
    messages = original_messages
    if messages is None:
        messages = record.get("messages", record.get("turns", delexed_messages))
    if not isinstance(messages, list):
        raise AssignmentError("Chat has no valid message list")
    if delexed_messages is not None:
        if (
            not isinstance(delexed_messages, list)
            or len(delexed_messages) != len(messages)
        ):
            raise AssignmentError("Original and delexed message lists are not aligned")

    result: list[dict[str, Any]] = []
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
        message_data: dict[str, Any] = {
            "speaker": str(speaker),
            "text": str(content),
        }
        metadata_message = (
            delexed_messages[position - 1]
            if isinstance(delexed_messages, list)
            else None
        )
        if metadata_message is not None:
            if isinstance(metadata_message, Mapping):
                metadata_speaker = metadata_message.get("speaker", "unknown")
            elif isinstance(metadata_message, (list, tuple)) and metadata_message:
                metadata_speaker = metadata_message[0]
            else:
                raise AssignmentError(
                    f"Invalid delexed message at position {position}"
                )
            if str(metadata_speaker).casefold() != str(speaker).casefold():
                raise AssignmentError(
                    f"Original and delexed speakers differ at message position {position}"
                )
        if (
            str(speaker).casefold() == "action"
            and isinstance(metadata_message, Mapping)
        ):
            targets = metadata_message.get("targets")
            if isinstance(targets, list) and len(targets) > 3:
                function_name, parameters = targets[2], targets[3]
                if isinstance(function_name, str):
                    message_data["function_name"] = function_name
                if isinstance(parameters, list):
                    message_data["parameters"] = parameters
        result.append(message_data)
    return result


def load_action_schema(
    path: Path,
) -> dict[str, dict[str, dict[str, list[str]]]]:
    data = load_json(path)
    if not isinstance(data, Mapping):
        raise InputDataError(f"{path} must contain a JSON object")

    schema: dict[str, dict[str, dict[str, list[str]]]] = {}
    for flow, flow_data in data.items():
        if flow == "_description":
            continue
        if not isinstance(flow_data, Mapping):
            raise InputDataError(f"Action schema flow {flow!r} must be an object")
        subflows = flow_data.get("subflows")
        if not isinstance(subflows, Mapping):
            raise InputDataError(
                f"Action schema flow {flow!r} must contain a subflows object"
            )

        normalized_subflows: dict[str, dict[str, list[str]]] = {}
        for subflow, subflow_data in subflows.items():
            if not isinstance(subflow_data, Mapping):
                raise InputDataError(
                    f"Action schema subflow {flow!r}/{subflow!r} must be an object"
                )
            actions = subflow_data.get("actions")
            if not isinstance(actions, list):
                raise InputDataError(
                    f"Action schema subflow {flow!r}/{subflow!r} must contain an actions list"
                )

            normalized_actions: dict[str, list[str]] = {}
            for action in actions:
                if not isinstance(action, Mapping) or len(action) != 1:
                    raise InputDataError(
                        f"Actions in {flow!r}/{subflow!r} must be single-entry objects"
                    )
                action_name, parameters = next(iter(action.items()))
                if (
                    not isinstance(action_name, str)
                    or not isinstance(parameters, list)
                    or any(not isinstance(parameter, str) for parameter in parameters)
                ):
                    raise InputDataError(
                        f"Action parameters in {flow!r}/{subflow!r} must be a list of strings"
                    )
                action_key = normalize_name(action_name)
                if action_key in normalized_actions:
                    raise InputDataError(
                        f"Duplicate normalized action {action_name!r} "
                        f"in {flow!r}/{subflow!r}"
                    )
                normalized_actions[action_key] = parameters

            subflow_key = normalize_name(subflow)
            if subflow_key in normalized_subflows:
                raise InputDataError(
                    f"Duplicate normalized subflow {subflow!r} in flow {flow!r}"
                )
            normalized_subflows[subflow_key] = normalized_actions

        flow_key = normalize_name(flow)
        if flow_key in schema:
            raise InputDataError(f"Duplicate normalized action schema flow {flow!r}")
        schema[flow_key] = normalized_subflows
    return schema


def _lookup_subflow_actions(
    flow_name: str,
    subflow_name: str,
    canonical_name: str,
    action_schema: Mapping[str, Mapping[str, Mapping[str, list[str]]]],
    guideline_aliases: Mapping[str, Mapping[str, str]],
) -> Mapping[str, list[str]] | None:
    flow_key = normalize_name(flow_name)
    subflows = action_schema.get(flow_key)
    if subflows is None:
        return None

    subflow_key = normalize_name(subflow_name)
    actions = subflows.get(subflow_key)
    canonical_key = normalize_name(canonical_name)
    if actions is None:
        actions = subflows.get(canonical_key)
    if actions is None:
        flow_aliases = guideline_aliases.get(flow_key, {})
        alias_matches = [
            candidate_actions
            for schema_subflow, candidate_actions in subflows.items()
            if normalize_name(flow_aliases.get(schema_subflow, "")) == canonical_key
        ]
        if len(alias_matches) == 1:
            actions = alias_matches[0]
    return actions


def _resolve_subflow_actions(
    record: Mapping[str, Any],
    canonical_name: str,
    action_schema: Mapping[str, Mapping[str, Mapping[str, list[str]]]],
    guideline_aliases: Mapping[str, Mapping[str, str]],
    sop_guideline: Mapping[str, Any] | None = None,
) -> dict[str, list[str]] | None:
    scenario = record.get("scenario")
    if not isinstance(scenario, Mapping):
        return None

    main_actions = _lookup_subflow_actions(
        str(scenario.get("flow", "")),
        str(scenario.get("subflow", "")),
        canonical_name,
        action_schema,
        guideline_aliases,
    )
    merged: dict[str, list[str]] = dict(main_actions) if main_actions else {}

    if isinstance(sop_guideline, Mapping):
        referenced = sop_guideline.get("referenced_subflows")
        if isinstance(referenced, Mapping):
            for ref_subflow, ref_data in referenced.items():
                if not isinstance(ref_data, Mapping):
                    continue
                ref_flow = ref_data.get("flow")
                if not isinstance(ref_flow, str):
                    continue
                ref_actions = _lookup_subflow_actions(
                    ref_flow,
                    str(ref_subflow),
                    str(ref_subflow),
                    action_schema,
                    guideline_aliases,
                )
                if ref_actions:
                    for action_key, params in ref_actions.items():
                        merged.setdefault(action_key, params)

    return merged or None


def expected_action_parameters(
    record: Mapping[str, Any],
    canonical_name: str,
    function_name: str,
    action_schema: Mapping[str, Mapping[str, Mapping[str, list[str]]]],
    guideline_aliases: Mapping[str, Mapping[str, str]],
    sop_guideline: Mapping[str, Any] | None = None,
) -> list[str] | None:
    actions = _resolve_subflow_actions(
        record, canonical_name, action_schema, guideline_aliases, sop_guideline
    )
    if actions is None:
        return None
    return actions.get(normalize_name(function_name))


def load_action_display_names(path: Path) -> dict[str, str]:
    data = load_json(path)
    names: dict[str, str] = {}
    if not isinstance(data, Mapping):
        return names
    for flow, flow_data in data.items():
        if flow == "_description" or not isinstance(flow_data, Mapping):
            continue
        subflows = flow_data.get("subflows")
        if not isinstance(subflows, Mapping):
            continue
        for subflow_data in subflows.values():
            if not isinstance(subflow_data, Mapping):
                continue
            for action in subflow_data.get("actions", []):
                if isinstance(action, Mapping) and len(action) == 1:
                    action_name = next(iter(action.keys()))
                    if isinstance(action_name, str):
                        names.setdefault(normalize_name(action_name), action_name)
    return names


def create_app_context() -> AppContext:
    reviewers = load_reviewers(REVIEWERS_FILE)
    resolver, canonical_names, sop_ids = build_sop_resolver(
        GUIDELINES_FILE, DEFAULT_ALIASES
    )
    reviewer_sops = load_reviewer_sops(REVIEWER_SOPS_FILE, reviewers, sop_ids)
    source_chats = load_source_chats(DEFAULT_SOURCE, resolver)
    action_schema = load_action_schema(ACTION_SCHEMA_FILE)
    action_display_names = load_action_display_names(ACTION_SCHEMA_FILE)
    return AppContext(
        reviewers=reviewers,
        reviewer_sops=reviewer_sops,
        resolver=resolver,
        canonical_names=canonical_names,
        sop_ids=sop_ids,
        source_chats=source_chats,
        action_schema=action_schema,
        action_display_names=action_display_names,
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

    def _chat_messages_with_expected(
        self,
        record: Mapping[str, Any],
        sop_name: str,
        guideline_aliases: Mapping[str, Any],
        sop_guideline: Any,
    ) -> list[dict[str, Any]]:
        messages = message_list(record)
        for message in messages:
            if message["speaker"].casefold() != "action":
                continue
            function_name = message.get("function_name")
            if not isinstance(function_name, str):
                message["expected_parameters"] = None
                continue
            message["expected_parameters"] = expected_action_parameters(
                record,
                sop_name,
                function_name,
                self.app.action_schema,
                guideline_aliases,
                sop_guideline,
            )
        return messages

    def _reviewer_chats(self, reviewer: str) -> list[dict[str, Any]]:
        _, reviewer_chats = read_assignment_folders(
            self.app.assignments_dir, self.app.resolver
        )
        reviews = load_reviews(self.app.reviews_file)
        reviewer_reviews = reviews.get(reviewer, {})
        chats = []
        guideline_aliases = self.app.resolver[1]
        for chat in reviewer_chats.get(reviewer, []):
            sop_name = self.app.canonical_names[chat.sop_id]
            sop_guideline = resolve_chat_guideline(
                chat.record, 0, self.app.resolver
            )
            messages = self._chat_messages_with_expected(
                chat.record, sop_name, guideline_aliases, sop_guideline
            )
            chat_path = (
                self.app.assignments_dir
                / reviewer
                / f"{chat.sop_id}__{chat.chat_id}.json"
            )
            backup_path = (
                self.app.assignments_dir
                / reviewer
                / ".originals"
                / f"{chat.sop_id}__{chat.chat_id}.json"
            )
            review_entry = reviewer_reviews.get(chat.chat_id, {})
            has_original = backup_path.exists()
            original_messages = (
                self._chat_messages_with_expected(
                    load_json(backup_path),
                    sop_name,
                    guideline_aliases,
                    sop_guideline,
                )
                if has_original
                else None
            )
            edited = has_original or bool(review_entry.get("edited", False))
            edited_at = ""
            if edited:
                edited_at = str(
                    review_entry.get("edited_at")
                    or chat.record.get("edited_at")
                    or ""
                )
                if not edited_at and chat_path.exists():
                    edited_at = datetime.fromtimestamp(
                        chat_path.stat().st_mtime, tz=timezone.utc
                    ).isoformat(timespec="seconds")
            chats.append({
                "chat_id": chat.chat_id,
                "sop_id": chat.sop_id,
                "sop_name": sop_name,
                "message_count": chat.message_count,
                "has_original": has_original,
                "edited": edited,
                "edited_at": edited_at,
                "reviewed_by": str(review_entry.get("reviewed_by", "")),
                "reviewed_at": str(review_entry.get("reviewed_at", "")),
                "status": review_entry.get("status", "pending"),
                "review_reason": review_entry.get("reason", ""),
                "scenario": chat.record.get("scenario", {}),
                "sop_guideline": sop_guideline,
                "sop_actions": {
                    self.app.action_display_names.get(action_key, action_key): list(params)
                    for action_key, params in (
                        _resolve_subflow_actions(
                            chat.record,
                            sop_name,
                            self.app.action_schema,
                            guideline_aliases,
                            sop_guideline,
                        )
                        or {}
                    ).items()
                },
                "messages": messages,
                "original_messages": original_messages,
            })
        return sorted(
            chats,
            key=lambda chat: (
                0 if (chat["reviewed_at"] or chat["edited"]) else 1,
                -max(
                    parse_edited_timestamp(chat["reviewed_at"]),
                    parse_edited_timestamp(chat["edited_at"]),
                ),
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
            elif path == "/api/update-chat":
                self._update_chat(data)
            elif path == "/api/revert-chat":
                self._revert_chat(data)
            elif path == "/api/remove-chat":
                self._remove_chat(data)
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
                self.app.reviewer_sops[reviewer],
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
        review_entry: dict[str, Any] = validate_review_entry(
            data.get("status"), data.get("reason", "")
        )
        requested_by = data.get("reviewed_by", "human")
        if not isinstance(requested_by, str) or requested_by not in VALID_REVIEWED_BY:
            raise ApiError("reviewed_by must be 'ai', 'human', or 'ai_human'")
        status = review_entry["status"]
        with self.app.lock:
            chat_path, backup_path = self._find_chat_file(reviewer, chat_id)
            reviews = load_reviews(self.app.reviews_file)
            existing_entry = reviews.get(reviewer, {}).get(chat_id, {})
            existing_by = existing_entry.get("reviewed_by", "")
            if requested_by == "ai":
                reviewed_by = "ai"
            elif existing_by in {"ai", "ai_human"}:
                reviewed_by = "ai_human"
            else:
                reviewed_by = "human"
            review_entry["reviewed_by"] = reviewed_by
            review_entry["reviewed_at"] = utc_now_iso()
            existing_edited = bool(existing_entry.get("edited", False))
            if backup_path.exists() or existing_edited or bool(data.get("edited", False)):
                review_entry["edited"] = True
                chat_record = load_json(chat_path)
                edited_at = (
                    existing_entry.get("edited_at")
                    or (chat_record.get("edited_at") if isinstance(chat_record, Mapping) else "")
                    or utc_now_iso()
                )
                review_entry["edited_at"] = str(edited_at)
            reviews.setdefault(reviewer, {})[chat_id] = review_entry
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

    def _find_chat_file(self, reviewer: str, chat_id: str) -> tuple[Path, Path]:
        _, reviewer_chats = read_assignment_folders(
            self.app.assignments_dir, self.app.resolver
        )
        match = next(
            (chat for chat in reviewer_chats.get(reviewer, []) if chat.chat_id == chat_id),
            None,
        )
        if match is None:
            raise ApiError("Chat is not assigned to this reviewer", 404)
        reviewer_dir = self.app.assignments_dir / reviewer
        chat_path = reviewer_dir / f"{match.sop_id}__{match.chat_id}.json"
        if not chat_path.exists():
            raise ApiError(f"Assignment file not found for chat {chat_id}", 404)
        backup_path = reviewer_dir / ".originals" / chat_path.name
        return chat_path, backup_path

    def _update_chat(self, data: Mapping[str, Any]) -> None:
        reviewer = self._validate_reviewer(data.get("reviewer"))
        chat_id = data.get("chat_id")
        if not isinstance(chat_id, str) or not chat_id:
            raise ApiError("A chat_id is required")
        messages = validate_edited_messages(data.get("messages"))
        with self.app.lock:
            chat_path, backup_path = self._find_chat_file(reviewer, chat_id)
            record = load_json(chat_path)
            if not backup_path.exists():
                save_json_atomic(backup_path, record, prefix=".orig-")
            edited_at = utc_now_iso()
            updated_record = apply_edited_messages(record, messages, edited_at=edited_at)
            save_json_atomic(chat_path, updated_record, prefix=".chat-")
            reviews = load_reviews(self.app.reviews_file)
            if chat_id in reviews.get(reviewer, {}):
                reviews[reviewer][chat_id]["edited"] = True
                reviews[reviewer][chat_id]["edited_at"] = edited_at
                save_reviews(self.app.reviews_file, reviews)
            chats = self._reviewer_chats(reviewer)
        self._send_json(
            200,
            {"reviewer": reviewer, "chat_id": chat_id, "chats": chats},
        )

    def _revert_chat(self, data: Mapping[str, Any]) -> None:
        reviewer = self._validate_reviewer(data.get("reviewer"))
        chat_id = data.get("chat_id")
        if not isinstance(chat_id, str) or not chat_id:
            raise ApiError("A chat_id is required")
        with self.app.lock:
            chat_path, backup_path = self._find_chat_file(reviewer, chat_id)
            if not backup_path.exists():
                raise ApiError("No original backup exists for this chat", 404)
            original_record = load_json(backup_path)
            save_json_atomic(chat_path, original_record, prefix=".chat-")
            backup_path.unlink()
            reviews = load_reviews(self.app.reviews_file)
            if chat_id in reviews.get(reviewer, {}) and (
                "edited" in reviews[reviewer][chat_id]
                or "edited_at" in reviews[reviewer][chat_id]
            ):
                reviews[reviewer][chat_id].pop("edited", None)
                reviews[reviewer][chat_id].pop("edited_at", None)
                if reviews[reviewer][chat_id].get("status") == "no_error":
                    reviews[reviewer][chat_id]["reason"] = ""
                save_reviews(self.app.reviews_file, reviews)
            chats = self._reviewer_chats(reviewer)
        self._send_json(
            200,
            {"reviewer": reviewer, "chat_id": chat_id, "chats": chats},
        )

    def _remove_chat(self, data: Mapping[str, Any]) -> None:
        reviewer = self._validate_reviewer(data.get("reviewer"))
        chat_id = data.get("chat_id")
        if not isinstance(chat_id, str) or not chat_id:
            raise ApiError("A chat_id is required")
        with self.app.lock:
            chat_path, backup_path = self._find_chat_file(reviewer, chat_id)
            chat_path.unlink()
            if backup_path.exists():
                backup_path.unlink()
                if backup_path.parent.exists() and not any(backup_path.parent.iterdir()):
                    backup_path.parent.rmdir()
            reviews = load_reviews(self.app.reviews_file)
            reviewer_reviews = reviews.get(reviewer, {})
            if chat_id in reviewer_reviews:
                reviewer_reviews.pop(chat_id, None)
                if not reviewer_reviews:
                    reviews.pop(reviewer, None)
                save_reviews(self.app.reviews_file, reviews)
            chats = self._reviewer_chats(reviewer)
        self._send_json(
            200,
            {"reviewer": reviewer, "chat_id": chat_id, "chats": chats},
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
