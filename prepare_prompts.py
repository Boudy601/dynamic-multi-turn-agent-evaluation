#!/usr/bin/env python3
"""Prepare synthetic golden-dataset injection prompts without making API calls."""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence


PROMPT_TEMPLATE_PATH = Path(__file__).with_name("prompt_template.txt")
DEFAULT_GUIDELINE_ALIASES_PATH = Path(__file__).with_name(
    "guideline_aliases.json"
)


def load_prompt_template() -> str:
    try:
        return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputDataError(
            f"Unable to read prompt template {PROMPT_TEMPLATE_PATH}: {exc}"
        ) from exc


CONVERSATION_COUNT = 10
ERROR_COUNTS = (1, 2, 3)
ERROR_COUNT_WEIGHTS = (0.5, 0.35, 0.15)
DIALOGUE_PHASES = ("early", "middle", "late")
GUIDELINE_STOP_WORDS = {"due", "to", "faq", "how", "other", "general", "one", "item"}
TARGET_JUDGES = ("intent_understanding",)


class InputDataError(ValueError):
    """Raised when an input file does not meet the supported schema."""


def load_json(path: Path) -> Any:
    try:
        with path.open(encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError as exc:
        raise InputDataError(f"Input file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise InputDataError(f"Invalid JSON in {path}: {exc}") from exc


def load_guideline_aliases(path: Path) -> dict[str, dict[str, str]]:
    data = load_json(path)
    if not isinstance(data, Mapping):
        raise InputDataError(f"{path} must contain a JSON object")

    aliases: dict[str, dict[str, str]] = {}
    for flow, flow_aliases in data.items():
        if not isinstance(flow_aliases, Mapping):
            raise InputDataError(
                f"Guideline aliases for flow {flow!r} in {path} "
                "must be an object"
            )
        normalized_flow = normalize_name(flow)
        aliases[normalized_flow] = {}
        for subflow, guideline in flow_aliases.items():
            if not isinstance(guideline, str) or not guideline.strip():
                raise InputDataError(
                    f"Guideline alias {flow!r}/{subflow!r} in {path} "
                    "must map to a non-empty string"
                )
            aliases[normalized_flow][normalize_name(subflow)] = guideline
    return aliases


def normalize_name(value: Any) -> str:
    """Make underscore, space, hyphen, and case differences irrelevant."""
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def semantic_name(value: Any) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value).casefold())
    meaningful = [
        token for token in tokens if token not in GUIDELINE_STOP_WORDS
    ]
    return "".join(meaningful).rstrip("0123456789")


def json_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def load_judges(data: Any, path: Path) -> dict[str, dict[str, Any]]:
    if isinstance(data, Mapping):
        definitions = data
    elif isinstance(data, list):
        definitions = {}
        for definition in data:
            if not isinstance(definition, Mapping):
                raise InputDataError(f"Each judge in {path} must be an object")
            name = definition.get("dimension_node_name")
            if not name:
                raise InputDataError(
                    f"Each judge in {path} needs dimension_node_name"
                )
            definitions[str(name)] = dict(definition)
    else:
        raise InputDataError(f"{path} must contain a judge object or list")

    result: dict[str, dict[str, Any]] = {}
    for name, definition in definitions.items():
        if not isinstance(definition, Mapping):
            raise InputDataError(f"Definition for judge {name!r} must be an object")
        result[str(name)] = dict(definition)
    if not result:
        raise InputDataError(f"{path} does not define any judges")
    return result


def flatten_guidelines(data: Any, path: Path) -> dict[str, Any]:
    if not isinstance(data, Mapping):
        raise InputDataError(f"{path} must contain a JSON object")
    flattened: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(value, Mapping):
            raise InputDataError(f"Guideline {key!r} in {path} must be an object")
        subflows = value.get("subflows")
        if isinstance(subflows, Mapping):
            for subflow_name, sop in subflows.items():
                flattened[normalize_name(subflow_name)] = sop
                alias = semantic_name(subflow_name)
                if alias and alias not in flattened:
                    flattened[alias] = sop
        else:
            flattened[normalize_name(key)] = value
    return flattened


def conversation_id(conversation: Mapping[str, Any], index: int) -> Any:
    value = conversation.get("conversation_id", conversation.get("convo_id"))
    if value is None:
        raise InputDataError(f"Conversation at index {index} is missing conversation_id")
    return value


def conversation_subflow(conversation: Mapping[str, Any], index: int) -> str:
    value = conversation.get("subflow", conversation.get("intent"))
    if value is None:
        scenario = conversation.get("scenario")
        if isinstance(scenario, Mapping):
            value = scenario.get("subflow", scenario.get("intent"))
    if not value:
        raise InputDataError(f"Conversation at index {index} is missing subflow/intent")
    return str(value)


def conversation_flow(conversation: Mapping[str, Any], index: int) -> str:
    value = conversation.get("flow")
    if value is None:
        scenario = conversation.get("scenario")
        if isinstance(scenario, Mapping):
            value = scenario.get("flow")
    if not value:
        raise InputDataError(f"Conversation at index {index} is missing flow")
    return str(value)


def find_guideline(
    guidelines: Mapping[str, Any],
    requested: str,
    flow: str | None = None,
    aliases: Mapping[str, Mapping[str, str]] | None = None,
) -> Any:
    exact = normalize_name(requested)
    if exact in guidelines:
        return guidelines[exact]

    if flow and aliases:
        flow_aliases = aliases.get(normalize_name(flow), {})
        mapped_name = flow_aliases.get(exact)
        if mapped_name is not None:
            mapped_key = normalize_name(mapped_name)
            if mapped_key in guidelines:
                return guidelines[mapped_key]
            raise InputDataError(
                f"Guideline alias for flow/subflow {flow!r}/{requested!r} "
                f"points to missing guideline {mapped_name!r}"
            )

    alias = semantic_name(requested)
    if alias in guidelines:
        return guidelines[alias]
    raise InputDataError(
        f"subflow/intent {requested!r} does not match any guideline"
    )


def iter_turns(conversation: Mapping[str, Any], index: int) -> Iterator[tuple[int, str, str]]:
    turns = conversation.get("turns")
    if turns is None:
        turns = conversation.get("original", conversation.get("delexed"))
    if not isinstance(turns, Sequence) or isinstance(turns, (str, bytes)):
        raise InputDataError(f"Conversation at index {index} has no valid turns list")

    for position, turn in enumerate(turns, start=1):
        if isinstance(turn, Mapping):
            speaker = turn.get("speaker")
            content = turn.get("text", turn.get("content"))
            turn_id = turn.get("turn_id", position)
        elif isinstance(turn, Sequence) and len(turn) == 2:
            speaker, content = turn
            turn_id = position
        else:
            raise InputDataError(
                f"Conversation at index {index} has an invalid turn at position {position}"
            )
        if not isinstance(speaker, str) or content is None:
            raise InputDataError(
                f"Conversation at index {index} has an incomplete turn at position {position}"
            )
        yield int(turn_id), speaker.upper(), str(content)


def format_turns(conversation: Mapping[str, Any], index: int) -> str:
    return "\n".join(
        f"{speaker}\n{content}" for _, speaker, content in iter_turns(conversation, index)
    )


def find_judge(judges: Mapping[str, dict[str, Any]], requested: str) -> str:
    requested_key = normalize_name(requested)
    for name in judges:
        if normalize_name(name) == requested_key:
            return name
    available = ", ".join(judges)
    raise InputDataError(
        f"Unknown target judge {requested!r}. Available judges: {available}"
    )


def assemble_prompt(
    conversation: Mapping[str, Any],
    index: int,
    judge_name: str,
    judge_definition: Mapping[str, Any],
    non_target_judges: Sequence[str],
    sop: Any,
    target_error_count: int,
    target_dialogue_phase: str,
) -> str:
    scenario = conversation.get("scenario", {})
    return load_prompt_template().format(
        target_judge_name=judge_name,
        target_judge_definition=json_text(judge_definition),
        non_target_judges=json_text(list(non_target_judges)),
        sop_content=json_text(sop),
        scenario_json=json_text(scenario),
        raw_conversation=format_turns(conversation, index),
        target_error_count=target_error_count,
        target_dialogue_phase=target_dialogue_phase,
    )


def prepare_records(
    conversations: Any,
    guidelines: Mapping[str, Any],
    judges: Mapping[str, dict[str, Any]],
    requested_judges: Sequence[str] | None,
    conversation_count: int,
    rng: random.Random,
    guideline_aliases: Mapping[str, Mapping[str, str]] | None = None,
) -> list[dict[str, Any]]:
    if isinstance(conversations, Mapping):
        split_records = [
            record
            for split in conversations.values()
            if isinstance(split, list)
            for record in split
        ]
        if not split_records:
            raise InputDataError(
                "Conversations object must contain non-empty list splits"
            )
        conversations = split_records
    if not isinstance(conversations, list):
        raise InputDataError("Conversations input must contain a JSON list")
    if conversation_count < 0:
        raise InputDataError("conversation_count must be zero or greater")
    if conversation_count:
        conversations = conversations[:conversation_count]
    target_judges = (
        [find_judge(judges, name) for name in requested_judges]
        if requested_judges
        else list(judges)
    )
    records: list[dict[str, Any]] = []
    errors: list[str] = []
    for index, conversation in enumerate(conversations):
        if not isinstance(conversation, Mapping):
            errors.append(f"Conversation at index {index} is not an object")
            continue
        try:
            flow = conversation_flow(conversation, index)
            subflow = conversation_subflow(conversation, index)
            sop = find_guideline(guidelines, subflow, flow, guideline_aliases)
            for judge_name in target_judges:
                error_count = rng.choices(
                    ERROR_COUNTS, weights=ERROR_COUNT_WEIGHTS, k=1
                )[0]
                phase = rng.choice(DIALOGUE_PHASES)
                non_targets = [
                    name for name in judges if name != judge_name
                ]
                records.append(
                    {
                        "conversation_id": conversation_id(conversation, index),
                        "target_judge": judge_name,
                        "sampled_params": {
                            "target_error_count": error_count,
                            "target_dialogue_phase": phase,
                        },
                        "assembled_prompt": assemble_prompt(
                            conversation,
                            index,
                            judge_name,
                            judges[judge_name],
                            non_targets,
                            sop,
                            error_count,
                            phase,
                        ),
                    }
                )
        except InputDataError as exc:
            errors.append(str(exc))
    if errors:
        raise InputDataError("Input validation failed:\n- " + "\n- ".join(errors))
    return records


def write_jsonl(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--guidelines",
        type=Path,
        default=Path(r"abcd\data\guidelines.json"),
    )
    parser.add_argument(
        "--guideline-aliases",
        type=Path,
        default=DEFAULT_GUIDELINE_ALIASES_PATH,
        help="Explicit flow/subflow-to-guideline mappings.",
    )
    parser.add_argument(
        "--conversations",
        type=Path,
        default=Path(r"abcd\data\abcd_v1.1.json"),
        help="Conversation JSON file; defaults to abcd\\data\\abcd_v1.1.json.",
    )
    parser.add_argument(
        "--judges",
        type=Path,
        default=Path("judges_definations.json"),
    )
    parser.add_argument(
        "--target-judge",
        action="append",
        help="Judge to prepare; repeat for multiple judges. Overrides TARGET_JUDGES.",
    )
    parser.add_argument(
        "--conversation-count",
        type=int,
        default=CONVERSATION_COUNT,
        help="Number of conversations to process; 0 processes all.",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("prepared_prompts.jsonl")
    )
    parser.add_argument("--seed", type=int, help="Seed the local sampler for reproducibility.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        guidelines = flatten_guidelines(load_json(args.guidelines), args.guidelines)
        guideline_aliases = load_guideline_aliases(args.guideline_aliases)
        judges = load_judges(load_json(args.judges), args.judges)
        rng = random.Random(args.seed)
        records = prepare_records(
            load_json(args.conversations),
            guidelines,
            judges,
            args.target_judge or TARGET_JUDGES,
            args.conversation_count,
            rng,
            guideline_aliases,
        )
        write_jsonl(args.output, records)
    except InputDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    print(f"Wrote {len(records)} prepared prompt(s) to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
