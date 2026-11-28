#!/usr/bin/env python3
"""Assign long AB-CD chats to reviewers for manual review."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from prepare_prompts import (
    InputDataError,
    find_guideline,
    flatten_guidelines,
    load_guideline_aliases,
    load_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = PROJECT_ROOT / "data" / "source" / "abcd" / "data" / "abcd_v1.1.json"
DEFAULT_GUIDELINES = (
    PROJECT_ROOT / "data" / "source" / "abcd" / "data" / "guidelines.json"
)
DEFAULT_ALIASES = PROJECT_ROOT / "config" / "guideline_aliases.json"
DEFAULT_REVIEWERS = PROJECT_ROOT / "config" / "reviewers.json"
DEFAULT_REVIEWER_SOPS = PROJECT_ROOT / "config" / "reviewer_sops.json"
DEFAULT_ASSIGNMENTS = PROJECT_ROOT / "assignments"
INITIAL_ASSIGNMENT_COUNT = 70
INITIAL_REVIEWER_COUNT = 7
SopResolver = tuple[
    dict[str, Any],
    dict[str, dict[str, str]],
    dict[int, str],
]


class AssignmentError(ValueError):
    """Raised when source chats or assignment data are invalid."""


@dataclass(frozen=True)
class Chat:
    record: dict[str, Any]
    chat_id: str
    sop_id: str
    message_count: int
    character_count: int


def canonical_sop_id(name: str) -> str:
    """Convert a guideline's display name into a stable filename-safe ID."""
    identifier = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    if not identifier:
        raise AssignmentError(f"Unable to make an SOP ID from {name!r}")
    return identifier


def build_sop_resolver(
    guidelines_path: Path, aliases_path: Path
) -> tuple[SopResolver, dict[str, str], list[str]]:
    guidelines_data = load_json(guidelines_path)
    if not isinstance(guidelines_data, Mapping):
        raise AssignmentError(f"{guidelines_path} must contain a JSON object")

    flattened = flatten_guidelines(guidelines_data, guidelines_path)
    aliases = load_guideline_aliases(aliases_path)
    sop_ids_by_guideline: dict[int, str] = {}
    canonical_names: dict[str, str] = {}

    for definition in guidelines_data.values():
        if not isinstance(definition, Mapping):
            continue
        subflows = definition.get("subflows")
        if not isinstance(subflows, Mapping):
            continue
        for name, guideline in subflows.items():
            sop_id = canonical_sop_id(str(name))
            existing = sop_ids_by_guideline.get(id(guideline))
            if existing is not None and existing != sop_id:
                raise AssignmentError(
                    f"Guideline {name!r} maps to multiple canonical SOP IDs"
                )
            existing_name = canonical_names.get(sop_id)
            if existing_name is not None and existing_name != str(name):
                raise AssignmentError(
                    f"Guidelines {existing_name!r} and {name!r} share SOP ID {sop_id!r}"
                )
            sop_ids_by_guideline[id(guideline)] = sop_id
            canonical_names[sop_id] = str(name)

    if not sop_ids_by_guideline:
        raise AssignmentError(f"No SOP subflows were found in {guidelines_path}")
    return (
        (flattened, aliases, sop_ids_by_guideline),
        canonical_names,
        sorted(canonical_names),
    )


def chat_identifier(record: Mapping[str, Any], index: int) -> str:
    value = record.get("conversation_id", record.get("convo_id"))
    if value is None or not str(value).strip():
        raise AssignmentError(f"Chat at index {index} is missing convo_id")
    chat_id = str(value).strip()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", chat_id):
        raise AssignmentError(f"Chat at index {index} has unsafe ID {chat_id!r}")
    return chat_id


def chat_sop_id(
    record: Mapping[str, Any],
    index: int,
    resolver: SopResolver,
) -> str:
    guideline = resolve_chat_guideline(record, index, resolver)
    _, _, sop_ids_by_guideline = resolver
    sop_id = sop_ids_by_guideline.get(id(guideline))
    if sop_id is None:
        raise AssignmentError(
            f"Chat at index {index} resolves to a guideline without a canonical SOP ID"
        )
    return sop_id


def resolve_chat_guideline(
    record: Mapping[str, Any],
    index: int,
    resolver: SopResolver,
) -> Any:
    flattened, aliases, _ = resolver
    scenario = record.get("scenario")
    if not isinstance(scenario, Mapping):
        raise AssignmentError(f"Chat at index {index} has no valid scenario object")
    subflow = scenario.get("subflow")
    flow = scenario.get("flow")
    if not isinstance(subflow, str) or not subflow.strip():
        raise AssignmentError(f"Chat at index {index} is missing scenario.subflow")
    if not isinstance(flow, str) or not flow.strip():
        raise AssignmentError(f"Chat at index {index} is missing scenario.flow")

    try:
        return find_guideline(flattened, subflow, flow, aliases)
    except InputDataError as exc:
        raise AssignmentError(f"Chat at index {index}: {exc}") from exc


def message_character_count(record: Mapping[str, Any], index: int) -> tuple[int, int]:
    messages = record.get("original")
    if messages is None:
        messages = record.get("messages", record.get("turns", record.get("delexed")))
    if not isinstance(messages, list):
        raise AssignmentError(f"Chat at index {index} has no valid message list")

    character_count = 0
    for position, message in enumerate(messages, start=1):
        if isinstance(message, Mapping):
            content = message.get("text", message.get("content"))
        elif isinstance(message, (list, tuple)) and len(message) >= 2:
            content = message[1]
        else:
            raise AssignmentError(
                f"Chat at index {index} has an invalid message at position {position}"
            )
        if content is None:
            raise AssignmentError(
                f"Chat at index {index} has a message without text at position {position}"
            )
        character_count += len(str(content))
    return len(messages), character_count


def make_chat(
    record: Any,
    index: int,
    resolver: SopResolver,
) -> Chat:
    if not isinstance(record, Mapping):
        raise AssignmentError(f"Chat at index {index} is not a JSON object")
    chat_record = dict(record)
    chat_id = chat_identifier(chat_record, index)
    sop_id = chat_sop_id(chat_record, index, resolver)
    message_count, character_count = message_character_count(chat_record, index)
    return Chat(chat_record, chat_id, sop_id, message_count, character_count)


def chat_sort_key(chat: Chat) -> tuple[int, int, str]:
    return (-chat.message_count, -chat.character_count, chat.chat_id)


def load_source_chats(
    source_path: Path,
    resolver: SopResolver,
) -> list[Chat]:
    source = load_json(source_path)
    if isinstance(source, Mapping):
        records = [
            record
            for split in source.values()
            if isinstance(split, list)
            for record in split
        ]
    elif isinstance(source, list):
        records = source
    else:
        raise AssignmentError(f"{source_path} must contain a JSON list or split object")
    if not records:
        raise AssignmentError(f"No chats were found in {source_path}")

    chats = [make_chat(record, index, resolver) for index, record in enumerate(records)]
    seen: dict[str, int] = {}
    for index, chat in enumerate(chats):
        previous = seen.get(chat.chat_id)
        if previous is not None:
            raise AssignmentError(
                f"Duplicate chat ID {chat.chat_id!r} at indices {previous} and {index}"
            )
        seen[chat.chat_id] = index
    return chats


def validate_reviewer_name(name: str) -> str:
    name = name.strip()
    if not re.fullmatch(r"[A-Za-z0-9_-][A-Za-z0-9._-]*", name):
        raise AssignmentError(
            f"Invalid reviewer name {name!r}; use letters, numbers, '.', '_' or '-'."
        )
    device_name = name.split(".", 1)[0].upper()
    if name.endswith(".") or device_name in {"CON", "PRN", "AUX", "NUL"} or re.fullmatch(
        r"(COM|LPT)[1-9]", device_name
    ):
        raise AssignmentError(f"Reviewer name {name!r} is reserved on Windows")
    return name


def load_reviewers(path: Path) -> list[str]:
    data = load_json(path)
    reviewers = data.get("reviewers") if isinstance(data, Mapping) else data
    if not isinstance(reviewers, list):
        raise AssignmentError(f"{path} must contain a reviewers list")
    names = [validate_reviewer_name(name) for name in reviewers if isinstance(name, str)]
    if len(names) != len(reviewers):
        raise AssignmentError(f"Every reviewer in {path} must be a string")
    if len(names) != INITIAL_REVIEWER_COUNT:
        raise AssignmentError(
            f"{path} must list exactly {INITIAL_REVIEWER_COUNT} reviewers; found {len(names)}"
        )
    if len({name.casefold() for name in names}) != len(names):
        raise AssignmentError(f"{path} contains duplicate reviewer names")
    return names


def load_reviewer_sops(
    path: Path,
    reviewers: Sequence[str],
    sop_ids: Sequence[str],
) -> dict[str, list[str]]:
    data = load_json(path)
    if not isinstance(data, Mapping):
        raise AssignmentError(f"{path} must contain a JSON object")

    expected_reviewers = set(reviewers)
    actual_reviewers = set(data.keys())
    if actual_reviewers != expected_reviewers:
        missing = sorted(expected_reviewers - actual_reviewers)
        extra = sorted(actual_reviewers - expected_reviewers)
        raise AssignmentError(
            f"{path} reviewers mismatch (missing={missing}, extra={extra})"
        )

    valid_sops = set(sop_ids)
    seen_sops: dict[str, str] = {}
    reviewer_sops: dict[str, list[str]] = {}
    for reviewer in reviewers:
        raw_sops = data[reviewer]
        if not isinstance(raw_sops, list) or not all(
            isinstance(sop_id, str) for sop_id in raw_sops
        ):
            raise AssignmentError(
                f"{path} entry for {reviewer!r} must be a list of SOP IDs"
            )
        if not raw_sops:
            raise AssignmentError(f"{path} entry for {reviewer!r} must not be empty")
        for sop_id in raw_sops:
            if sop_id not in valid_sops:
                raise AssignmentError(
                    f"{path} assigns unknown SOP {sop_id!r} to {reviewer!r}"
                )
            previous = seen_sops.get(sop_id)
            if previous is not None:
                raise AssignmentError(
                    f"{path} assigns SOP {sop_id!r} to both {previous!r} and {reviewer!r}"
                )
            seen_sops[sop_id] = reviewer
        reviewer_sops[reviewer] = list(raw_sops)

    uncovered = sorted(valid_sops - set(seen_sops))
    if uncovered:
        raise AssignmentError(f"{path} does not cover SOP(s): {', '.join(uncovered)}")
    return reviewer_sops


def select_initial_chats(
    chats: Sequence[Chat], sop_ids: Sequence[str]
) -> list[Chat]:
    grouped: dict[str, list[Chat]] = defaultdict(list)
    for chat in chats:
        grouped[chat.sop_id].append(chat)
    for sop_id in sop_ids:
        random.shuffle(grouped[sop_id])

    missing = [sop_id for sop_id in sop_ids if not grouped[sop_id]]
    if missing:
        print(
            "Warning: no source chats are available for SOP(s): "
            + ", ".join(missing),
            file=sys.stderr,
        )

    selected = [grouped[sop_id][0] for sop_id in sop_ids if grouped[sop_id]]
    additional_slots = INITIAL_ASSIGNMENT_COUNT - len(selected)
    second_place = [group[1] for group in grouped.values() if len(group) > 1]
    random.shuffle(second_place)
    if additional_slots > len(second_place):
        raise AssignmentError(
            f"Need {INITIAL_ASSIGNMENT_COUNT} chats for the initial assignment, "
            f"but only {len(selected) + len(second_place)} qualifying chats are available"
        )
    selected.extend(second_place[:additional_slots])
    if len(selected) != INITIAL_ASSIGNMENT_COUNT:
        raise AssignmentError(
            f"Expected {INITIAL_ASSIGNMENT_COUNT} initial chats; selected {len(selected)}"
        )
    return sorted(selected, key=chat_sort_key)


def snake_order(reviewer_count: int) -> list[int]:
    forward = list(range(reviewer_count))
    backward = list(reversed(forward))
    return forward + backward


def distribute_snake(
    chats: Sequence[Chat], reviewers: Sequence[str]
) -> dict[str, list[Chat]]:
    if len(chats) != INITIAL_ASSIGNMENT_COUNT:
        raise AssignmentError(
            f"Snake distribution requires {INITIAL_ASSIGNMENT_COUNT} chats"
        )
    if len(reviewers) != INITIAL_REVIEWER_COUNT:
        raise AssignmentError(
            f"Snake distribution requires {INITIAL_REVIEWER_COUNT} reviewers"
        )

    grouped: dict[str, list[Chat]] = defaultdict(list)
    for chat in chats:
        grouped[chat.sop_id].append(chat)
    for sop_chats in grouped.values():
        sop_chats.sort(key=chat_sort_key)

    pairs = [items for items in grouped.values() if len(items) == 2]
    singles = [items[0] for items in grouped.values() if len(items) == 1]
    invalid_groups = [sop_id for sop_id, items in grouped.items() if len(items) > 2]
    if invalid_groups or len(pairs) != 15 or len(singles) != 40:
        raise AssignmentError(
            "Initial selection must contain 15 two-chat SOP bundles and 40 single-chat SOPs"
        )

    pairs.sort(
        key=lambda items: (
            -sum(chat.message_count for chat in items),
            -sum(chat.character_count for chat in items),
            items[0].sop_id,
        )
    )
    assignments = {reviewer: [] for reviewer in reviewers}
    pair_counts = Counter()
    workloads = Counter()
    for pair_index, pair in enumerate(pairs):
        pair_limit = 2 if pair_index < 14 else 3
        eligible = [
            reviewer for reviewer in reviewers if pair_counts[reviewer] < pair_limit
        ]
        reviewer = min(
            eligible,
            key=lambda name: (workloads[name], reviewers.index(name)),
        )
        assignments[reviewer].extend(pair)
        pair_counts[reviewer] += 1
        workloads[reviewer] += sum(chat.message_count for chat in pair)

    singles.sort(key=chat_sort_key)
    sequence = snake_order(len(reviewers))
    slot = 0
    for chat in singles:
        while True:
            reviewer = reviewers[sequence[slot % len(sequence)]]
            slot += 1
            if len(assignments[reviewer]) < INITIAL_ASSIGNMENT_COUNT // len(reviewers):
                assignments[reviewer].append(chat)
                break

    counts = {len(assigned) for assigned in assignments.values()}
    if counts != {INITIAL_ASSIGNMENT_COUNT // INITIAL_REVIEWER_COUNT}:
        raise AssignmentError("Snake distribution did not produce an equal chat count")
    return assignments


def read_assignment_folders(
    assignments_dir: Path,
    resolver: SopResolver,
) -> tuple[dict[str, Chat], dict[str, list[Chat]]]:
    used_chats: dict[str, Chat] = {}
    used_paths: dict[str, Path] = {}
    reviewer_chats: dict[str, list[Chat]] = {}
    if not assignments_dir.exists():
        return used_chats, reviewer_chats

    for reviewer_dir in sorted(
        path for path in assignments_dir.iterdir() if path.is_dir()
    ):
        reviewer = validate_reviewer_name(reviewer_dir.name)
        reviewer_chats[reviewer] = []
        for index, chat_path in enumerate(sorted(reviewer_dir.glob("*.json"))):
            record = load_json(chat_path)
            chat = make_chat(record, index, resolver)
            previous = used_chats.get(chat.chat_id)
            if previous is not None:
                raise AssignmentError(
                    f"Chat ID {chat.chat_id!r} is assigned more than once: "
                    f"{used_paths[chat.chat_id]} and {chat_path}"
                )
            used_chats[chat.chat_id] = chat
            used_paths[chat.chat_id] = chat_path
            reviewer_chats[reviewer].append(chat)
    return used_chats, reviewer_chats


def select_top_up_chats(
    available_chats: Sequence[Chat],
    reviewer_sop_counts: Counter[str],
    count: int,
    allowed_sops: Sequence[str] | set[str],
) -> list[Chat]:
    allowed = set(allowed_sops)
    remaining = [chat for chat in available_chats if chat.sop_id in allowed]
    selected: list[Chat] = []
    for _ in range(count):
        grouped: dict[str, list[Chat]] = defaultdict(list)
        for chat in remaining:
            grouped[chat.sop_id].append(chat)
        if not grouped:
            break

        lowest_count = min(reviewer_sop_counts[sop_id] for sop_id in grouped)
        candidate_sops = [
            sop_id
            for sop_id in grouped
            if reviewer_sop_counts[sop_id] == lowest_count
        ]
        winning_sop = random.choice(candidate_sops)
        chat = random.choice(grouped[winning_sop])
        selected.append(chat)
        remaining.remove(chat)
        reviewer_sop_counts[winning_sop] += 1
    return selected


def write_chat(chat: Chat, reviewer_dir: Path) -> Path:
    reviewer_dir.mkdir(parents=True, exist_ok=True)
    destination = reviewer_dir / f"{chat.sop_id}__{chat.chat_id}.json"
    if destination.exists():
        raise AssignmentError(f"Refusing to overwrite existing assignment {destination}")
    destination.write_text(
        json.dumps(chat.record, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return destination


def clear_existing_assignments(assignments_dir: Path) -> None:
    if not assignments_dir.exists():
        return

    for path in assignments_dir.iterdir():
        if path.is_dir() and not path.is_symlink():
            for assignment in path.iterdir():
                if assignment.is_file() and assignment.suffix.casefold() == ".json":
                    assignment.unlink()
            if not any(path.iterdir()):
                path.rmdir()
        elif path.is_file() and path.name in {"reviews.json", "coverage_report.csv"}:
            path.unlink()


def initialize_assignments(
    chats: Sequence[Chat],
    reviewers: Sequence[str],
    assignments_dir: Path,
    sop_ids: Sequence[str],
    reviewer_sops: Mapping[str, Sequence[str]],
) -> dict[str, list[Chat]]:
    per_reviewer_target = INITIAL_ASSIGNMENT_COUNT // len(reviewers)
    grouped: dict[str, list[Chat]] = defaultdict(list)
    for chat in chats:
        grouped[chat.sop_id].append(chat)
    for sop_id in sop_ids:
        random.shuffle(grouped[sop_id])

    missing = [sop_id for sop_id in sop_ids if not grouped[sop_id]]
    if missing:
        raise AssignmentError(
            "No source chats are available for SOP(s): " + ", ".join(missing)
        )

    assignments: dict[str, list[Chat]] = {}
    workloads: Counter[str] = Counter()
    for reviewer in reviewers:
        owned_sops = list(reviewer_sops[reviewer])
        if len(owned_sops) > per_reviewer_target:
            raise AssignmentError(
                f"Reviewer {reviewer!r} owns {len(owned_sops)} SOPs, "
                f"exceeding {per_reviewer_target} initial chats"
            )
        reviewer_chats = [grouped[sop_id][0] for sop_id in owned_sops]
        workloads[reviewer] = sum(chat.message_count for chat in reviewer_chats)
        assignments[reviewer] = reviewer_chats

    mean_workload = (
        sum(workloads.values()) / len(reviewers) if reviewers else 0.0
    )
    for reviewer in reviewers:
        owned_sops = list(reviewer_sops[reviewer])
        extra_needed = per_reviewer_target - len(assignments[reviewer])
        second_choices = [
            grouped[sop_id][1] for sop_id in owned_sops if len(grouped[sop_id]) > 1
        ]
        if len(second_choices) < extra_needed:
            raise AssignmentError(
                f"Reviewer {reviewer!r} needs {extra_needed} extra chats from owned SOPs, "
                f"but only {len(second_choices)} are available"
            )
        random.shuffle(second_choices)
        for _ in range(extra_needed):
            best_chat = min(
                second_choices,
                key=lambda chat: (
                    abs((workloads[reviewer] + chat.message_count) - mean_workload),
                    chat.chat_id,
                ),
            )
            assignments[reviewer].append(best_chat)
            workloads[reviewer] += best_chat.message_count
            second_choices.remove(best_chat)

    clear_existing_assignments(assignments_dir)
    for reviewer, reviewer_assignments in assignments.items():
        reviewer_dir = assignments_dir / reviewer
        for chat in reviewer_assignments:
            write_chat(chat, reviewer_dir)
    return assignments


def run_top_up(
    chats: Sequence[Chat],
    reviewer: str,
    count: int,
    assignments_dir: Path,
    resolver: SopResolver,
    allowed_sops: Sequence[str] | set[str],
) -> tuple[list[Chat], dict[str, Chat], dict[str, list[Chat]]]:
    used_chats, reviewer_chats = read_assignment_folders(assignments_dir, resolver)
    reviewer = validate_reviewer_name(reviewer)
    reviewer_chats.setdefault(reviewer, [])
    assigned_ids = set(used_chats)
    available = [chat for chat in chats if chat.chat_id not in assigned_ids]
    reviewer_sop_counts = Counter(chat.sop_id for chat in reviewer_chats[reviewer])
    selected = select_top_up_chats(
        available, reviewer_sop_counts, count, allowed_sops
    )
    reviewer_dir = assignments_dir / reviewer
    for chat in selected:
        write_chat(chat, reviewer_dir)
        reviewer_chats[reviewer].append(chat)
        used_chats[chat.chat_id] = chat
    if len(selected) < count:
        print(
            f"Warning: only {len(selected)} of {count} requested chats were available.",
            file=sys.stderr,
        )
    return selected, used_chats, reviewer_chats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=DEFAULT_SOURCE, help="AB-CD conversation JSON."
    )
    parser.add_argument("--guidelines", type=Path, default=DEFAULT_GUIDELINES)
    parser.add_argument("--guideline-aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--reviewers-file", type=Path, default=DEFAULT_REVIEWERS)
    parser.add_argument("--reviewer-sops", type=Path, default=DEFAULT_REVIEWER_SOPS)
    parser.add_argument("--assignments", type=Path, default=DEFAULT_ASSIGNMENTS)
    parser.add_argument(
        "--initialize",
        action="store_true",
        help=(
            "Replace existing assignments with the balanced initial 70-chat set "
            "and clear old review statuses."
        ),
    )
    parser.add_argument("--reviewer", help="Reviewer requesting more chats.")
    parser.add_argument("--count", type=int, default=5, help="Number of additional chats.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        if args.initialize and args.reviewer:
            raise AssignmentError("--initialize cannot be combined with --reviewer")
        if not args.initialize and not args.reviewer:
            raise AssignmentError("Use --initialize or provide --reviewer <name>")
        if args.count < 1:
            raise AssignmentError("--count must be a positive integer")

        resolver, _, sop_ids = build_sop_resolver(
            args.guidelines, args.guideline_aliases
        )
        chats = load_source_chats(args.source, resolver)
        reviewers = load_reviewers(args.reviewers_file)
        reviewer_sops = load_reviewer_sops(args.reviewer_sops, reviewers, sop_ids)
        if args.initialize:
            assignments = initialize_assignments(
                chats, reviewers, args.assignments, sop_ids, reviewer_sops
            )
            print(
                f"Assigned {sum(map(len, assignments.values()))} chats to "
                f"{len(reviewers)} reviewers."
            )
        else:
            if args.reviewer not in reviewer_sops:
                raise AssignmentError(f"Unknown reviewer {args.reviewer!r}")
            selected, _, _ = run_top_up(
                chats,
                args.reviewer,
                args.count,
                args.assignments,
                resolver,
                reviewer_sops[args.reviewer],
            )
            print(f"Assigned {len(selected)} additional chat(s) to {args.reviewer}.")
    except (AssignmentError, InputDataError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
