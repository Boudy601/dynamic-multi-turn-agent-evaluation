#!/usr/bin/env python3
"""Prepare synthetic golden-dataset injection prompts without making API calls."""

from __future__ import annotations

import argparse
import json
import random
import re
import string
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # Support both direct script execution and package imports.
    from . import factual_hallucination as factual
except ImportError:  # pragma: no cover - exercised when run as a script
    import factual_hallucination as factual

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = PROJECT_ROOT / "prompts"
DATA_JUDGES_PROMPTS_DIR = PROMPTS_DIR / "data_judges_injection"
INJECTION_PATTERNS_PATH = (
    DATA_JUDGES_PROMPTS_DIR / "data_judges_injection_patterns.json"
)

# ==============================================================================
# BATCH CONFIGURATION: Set the conversation counts and paths directly here
# ==============================================================================
BATCH_CONFIGS = [
    {
        "name": "factual_hallucination",
        "template_path": (
            PROMPTS_DIR / "factual_hallucination_injection_prompt_template.txt"
        ),
        "count": 0,
        "target_judges": ("factual_hallucination",),
    },
    {
        "name": "data_judges",
        "template_path": (
            DATA_JUDGES_PROMPTS_DIR / "data_judges_injection_prompt_template.txt"
        ),
        "count": 5,
        "target_judges": ("missing_data",),
    },
]


ERROR_COUNTS = (1, 2, 3)
ERROR_COUNT_WEIGHTS = (0.7, 0.3, 0)
DIALOGUE_PHASES = ("early", "middle", "late")
GUIDELINE_STOP_WORDS = {"due", "to", "faq", "how", "other", "general", "one", "item"}


DEFAULT_GUIDELINE_ALIASES_PATH = PROJECT_ROOT / "config" / "guideline_aliases.json"
INJECTION_MECHANISMS = (
    "mutate_sop_for_existing_ask",
    "inject_new_customer_ask",
)
MUTATION_PATTERN_KEYS = (
    "pattern_1_ambiguous_unquantified",
    "pattern_2_incomplete_branch",
    "pattern_3_truncated_rag_reference",
)


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


def load_injection_patterns(path: Path) -> dict[str, dict[str, str]]:
    data = load_json(path)
    if not isinstance(data, Mapping) or not isinstance(
        data.get("INJECTION_PATTERNS"), Mapping
    ):
        raise InputDataError(f"{path} must contain an INJECTION_PATTERNS object")

    patterns = data["INJECTION_PATTERNS"]
    for key, pattern in patterns.items():
        if not isinstance(key, str) or not isinstance(pattern, Mapping):
            raise InputDataError(f"Invalid injection pattern {key!r} in {path}")
        if not all(
            isinstance(pattern.get(field), str) and pattern[field]
            for field in ("name", "rule", "example")
        ):
            raise InputDataError(
                f"Injection pattern {key!r} in {path} needs name, rule, and example"
            )
    return {str(key): dict(pattern) for key, pattern in patterns.items()}


INJECTION_PATTERNS = load_injection_patterns(INJECTION_PATTERNS_PATH)


def resolve_prompt_template_path(template_path: Path | str) -> Path:
    """Resolve explicit file path with automatic fallback to project prompts directory."""
    path = Path(template_path)
    if path.is_file():
        return path
    fallback = PROMPTS_DIR / path.name
    if fallback.is_file():
        return fallback
    raise InputDataError(
        f"Prompt template file does not exist: {path} (also checked {fallback})"
    )


def load_prompt_template(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise InputDataError(f"Unable to read prompt template {path}: {exc}") from exc


def get_template_placeholders(template_text: str) -> set[str]:
    """Extract all top-level variable names from a format string without formatting errors."""
    formatter = string.Formatter()
    return {
        field_name
        for _, field_name, _, _ in formatter.parse(template_text)
        if field_name is not None
    }


def load_guideline_aliases(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    data = load_json(path)
    if not isinstance(data, Mapping):
        raise InputDataError(f"{path} must contain a JSON object")

    aliases: dict[str, dict[str, str]] = {}
    for flow, flow_aliases in data.items():
        if not isinstance(flow_aliases, Mapping):
            raise InputDataError(
                f"Guideline aliases for flow {flow!r} in {path} must be an object"
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
    return re.sub(r"[^a-z0-9]+", "", str(value).casefold())


def semantic_name(value: Any) -> str:
    tokens = re.findall(r"[a-z0-9]+", str(value).casefold())
    meaningful = [token for token in tokens if token not in GUIDELINE_STOP_WORDS]
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
                raise InputDataError(f"Each judge in {path} needs dimension_node_name")
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
        raise InputDataError(
            f"Conversation at index {index} is missing conversation_id"
        )
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
    raise InputDataError(f"subflow/intent {requested!r} does not match any guideline")


def iter_turns(
    conversation: Mapping[str, Any], index: int
) -> Iterator[tuple[int, str, str]]:
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
        f"{speaker}\n{content}"
        for _, speaker, content in iter_turns(conversation, index)
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
    template_text: str,
    available_placeholders: set[str],
    context: Mapping[str, Any],
) -> str:
    """Format template using only placeholders expected by the template, ignoring extraneous parameters."""
    missing = [name for name in available_placeholders if name not in context]
    if missing:
        raise InputDataError(
            f"Prompt template missing required variables: {sorted(missing)}"
        )

    # Filter context so we don't accidentally pass unexpected kwargs
    filtered_context = {name: context[name] for name in available_placeholders}
    return template_text.format(**filtered_context)


def select_injection_pattern(
    patterns: Mapping[str, Mapping[str, str]],
    judge_name: str,
    injection_mechanism: str,
    sample_index: int,
) -> Mapping[str, str]:
    if judge_name == "data_conflict":
        pattern_name = "data_conflict_contradiction"
    elif (
        judge_name == "missing_data"
        and injection_mechanism == "inject_new_customer_ask"
    ):
        pattern_name = "missing_data_new_ask"
    elif (
        judge_name == "missing_data"
        and injection_mechanism == "mutate_sop_for_existing_ask"
    ):
        pattern_name = MUTATION_PATTERN_KEYS[sample_index % len(MUTATION_PATTERN_KEYS)]
    else:
        raise InputDataError(
            f"No injection pattern is defined for judge {judge_name!r} "
            f"and mechanism {injection_mechanism!r}"
        )
    try:
        return patterns[pattern_name]
    except KeyError as exc:
        raise InputDataError(
            f"Missing injection pattern definition: {pattern_name}"
        ) from exc


def load_existing_conversation_ids(path: Path) -> set[str]:
    """Read existing conversation IDs from a JSONL file."""
    if not path.is_file():
        return set()
    ids: set[str] = set()
    try:
        with path.open(encoding="utf-8") as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    cid = record.get("conversation_id", record.get("convo_id"))
                    if cid is not None:
                        ids.add(str(cid))
                except json.JSONDecodeError:
                    continue
    except OSError:
        pass
    return ids


def prepare_factual_hallucination_record(
    conversation: Mapping[str, Any],
    judge_name: str,
    judges: Mapping[str, dict[str, Any]],
    sop: Any,
    scenario: Mapping[str, Any],
    template_text: str,
    required_placeholders: set[str],
    index: int,
) -> dict[str, Any]:
    """Assemble one factual-hallucination prompt from eligible Agent claims.

    Conversations without a claim that literally states trusted scenario
    evidence are rejected during preparation instead of being sent to the model.
    """

    turns = factual.extract_conversation_turns(conversation)
    evidence = factual.filter_supported_evidence(
        factual.build_available_evidence(scenario), scenario
    )
    selection = factual.select_factual_candidates(turns, scenario, evidence, sop=sop)
    clear_defects, review_flags = factual._assess_pre_existing_non_target_defects(
        turns, scenario, sop, evidence
    )
    if clear_defects:
        raise InputDataError(
            "Conversation at index "
            f"{index} has a clear pre-existing non-target or baseline defect: "
            + "; ".join(clear_defects)
        )
    if not selection.eligible:
        raise InputDataError(
            f"Conversation at index {index} has no eligible factual "
            f"candidate ({selection.primary_exclusion_code or 'NO_SUPPORTED_EVIDENCE'})"
        )

    screening_stage = factual.python_screening_stage(selection)

    non_targets = [name for name in judges if name != judge_name]
    prompt_context = {
        "target_judge_name": judge_name,
        "target_judge_definition": json_text(judges[judge_name]),
        "non_target_judges": json_text(non_targets),
        "sop_content": json_text(sop),
        "scenario_json": json_text(scenario),
        "available_evidence": json_text(evidence),
        "eligible_candidates": json_text(
            [candidate.as_prompt_item() for candidate in selection.eligible]
        ),
        "raw_conversation": factual.format_numbered_turns(turns),
    }
    assembled = assemble_prompt(
        template_text=template_text,
        available_placeholders=required_placeholders,
        context=prompt_context,
    )

    sampled_params: dict[str, Any] = {
        "eligible_candidate_count": len(selection.eligible),
        "screening_decision": screening_stage.decision,
        "screening_requires_llm": screening_stage.requires_llm,
        "screening_reason": screening_stage.reason,
        "candidate_id": screening_stage.candidate_id,
    }
    if review_flags:
        sampled_params["human_review_flags"] = review_flags

    return {
        "conversation_id": conversation_id(conversation, index),
        "target_judge": judge_name,
        "sampled_params": sampled_params,
        "assembled_prompt": assembled,
        "source_conversation": turns,
        "source_scenario": dict(scenario),
        "source_sop": sop,
        "available_evidence": evidence,
        "judge_definition": judges[judge_name],
        "human_review_flags": review_flags,
        "factual_selection": selection.to_dict(),
        "screening": screening_stage.to_dict(),
    }


def prepare_single_record(
    conversation: Mapping[str, Any],
    judge_name: str,
    judges: Mapping[str, dict[str, Any]],
    guidelines: Mapping[str, Any],
    guideline_aliases: Mapping[str, Mapping[str, str]] | None,
    template_text: str,
    required_placeholders: set[str],
    needs_mechanism: bool,
    mechanisms: list[str],
    weights: list[float],
    rng: random.Random,
    index: int = 0,
) -> dict[str, Any]:
    """Assemble a single prompt record for a given conversation and target judge."""
    flow = conversation_flow(conversation, index)
    subflow = conversation_subflow(conversation, index)
    sop = find_guideline(guidelines, subflow, flow, guideline_aliases)
    scenario = conversation.get("scenario", {})
    if judge_name == factual.FACTUAL_JUDGE:
        return prepare_factual_hallucination_record(
            conversation=conversation,
            judge_name=judge_name,
            judges=judges,
            sop=sop,
            scenario=scenario if isinstance(scenario, Mapping) else {},
            template_text=template_text,
            required_placeholders=required_placeholders,
            index=index,
        )
    turns_text = format_turns(conversation, index)

    error_count = rng.choices(ERROR_COUNTS, weights=ERROR_COUNT_WEIGHTS, k=1)[0]
    phase = rng.choice(DIALOGUE_PHASES)
    non_targets = [name for name in judges if name != judge_name]

    sampled_params: dict[str, Any] = {
        "target_error_count": error_count,
        "target_dialogue_phase": phase,
        "sop_content": sop,
    }

    prompt_context: dict[str, Any] = {
        "target_judge_name": judge_name,
        "target_judge_definition": json_text(judges[judge_name]),
        "non_target_judges": json_text(non_targets),
        "sop_content": json_text(sop),
        "scenario_json": json_text(scenario),
        "raw_conversation": turns_text,
        "target_error_count": error_count,
        "target_dialogue_phase": phase,
    }

    if needs_mechanism:
        injection_mechanism = rng.choices(mechanisms, weights=weights, k=1)[0]
        sampled_params["injection_mechanism"] = injection_mechanism
        prompt_context["injection_mechanism"] = injection_mechanism
        pattern = select_injection_pattern(
            INJECTION_PATTERNS, judge_name, injection_mechanism, index
        )
        sampled_params["selected_pattern"] = pattern["name"]
        prompt_context.update(
            selected_pattern_name=pattern["name"],
            pattern_execution_rule=pattern["rule"],
            pattern_contrastive_example=pattern["example"],
        )

    assembled = assemble_prompt(
        template_text=template_text,
        available_placeholders=required_placeholders,
        context=prompt_context,
    )

    return {
        "conversation_id": conversation_id(conversation, index),
        "target_judge": judge_name,
        "sampled_params": sampled_params,
        "assembled_prompt": assembled,
    }


def prepare_records(
    conversations: Any,
    guidelines: Mapping[str, Any],
    judges: Mapping[str, dict[str, Any]],
    requested_judges: Sequence[str] | None,
    conversation_count: int,
    rng: random.Random,
    prompt_template_path: Path,
    guideline_aliases: Mapping[str, Mapping[str, str]] | None = None,
    mechanism_percentages: Mapping[str, float] | None = None,
    used_conversation_ids: set[str] | None = None,
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

    target_judges = (
        [find_judge(judges, name) for name in requested_judges]
        if requested_judges
        else list(judges)
    )

    template_text = load_prompt_template(prompt_template_path)
    required_placeholders = get_template_placeholders(template_text)

    needs_mechanism = "injection_mechanism" in required_placeholders
    percentages = mechanism_percentages or {
        INJECTION_MECHANISMS[0]: 50.0,
        INJECTION_MECHANISMS[1]: 50.0,
    }
    mechanisms = list(percentages)
    weights = [percentages[mech] for mech in mechanisms]

    if used_conversation_ids is None:
        used_conversation_ids = set()

    records: list[dict[str, Any]] = []

    # Number of distinct conversations to assign to each individual target judge
    per_judge_target = (
        conversation_count
        if conversation_count > 0
        else max(1, len(conversations) // max(1, len(target_judges)))
    )

    conv_iter = iter(enumerate(conversations))

    for judge_name in target_judges:
        judge_count = 0
        for index, conversation in conv_iter:
            if not isinstance(conversation, Mapping):
                continue
            try:
                cid = str(conversation_id(conversation, index))
            except InputDataError:
                continue

            if cid in used_conversation_ids:
                continue

            try:
                rec = prepare_single_record(
                    conversation=conversation,
                    judge_name=judge_name,
                    judges=judges,
                    guidelines=guidelines,
                    guideline_aliases=guideline_aliases,
                    template_text=template_text,
                    required_placeholders=required_placeholders,
                    needs_mechanism=needs_mechanism,
                    mechanisms=mechanisms,
                    weights=weights,
                    rng=rng,
                    index=index,
                )
            except InputDataError:
                continue

            used_conversation_ids.add(cid)
            records.append(rec)
            judge_count += 1
            if judge_count >= per_judge_target:
                break

    return records


def prepare_all_batches(
    all_conversations: list[Any],
    guidelines: Mapping[str, Any],
    judges: Mapping[str, dict[str, Any]],
    rng: random.Random,
    guideline_aliases: Mapping[str, Mapping[str, str]] | None = None,
    mechanism_percentages: Mapping[str, float] | None = None,
    existing_conversation_ids: set[str] | None = None,
    batch_names: set[str] | None = None,
) -> list[dict[str, Any]]:
    """Process all configured batches sequentially with non-overlapping conversation assignment."""
    combined_records: list[dict[str, Any]] = []
    used_conversation_ids = set(existing_conversation_ids or ())

    for batch in BATCH_CONFIGS:
        if batch_names and batch["name"] not in batch_names:
            continue
        count = batch.get("count", 0)
        if count <= 0:
            if batch_names and batch["name"] in batch_names:
                count = 5
            else:
                continue

        template_path = resolve_prompt_template_path(batch["template_path"])
        batch_judges = batch.get("target_judges")

        records = prepare_records(
            conversations=all_conversations,
            guidelines=guidelines,
            judges=judges,
            requested_judges=batch_judges,
            conversation_count=count,
            rng=rng,
            prompt_template_path=template_path,
            guideline_aliases=guideline_aliases,
            mechanism_percentages=mechanism_percentages,
            used_conversation_ids=used_conversation_ids,
        )
        print(
            f"Batch '{batch['name']}': prepared {len(records)} prompt(s) "
            f"({count} per judge) using {template_path.name}"
        )
        combined_records.extend(records)

    return combined_records


def write_jsonl(
    path: Path, records: Sequence[Mapping[str, Any]], append: bool = False
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    with path.open(mode, encoding="utf-8", newline="\n") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--guidelines",
        type=Path,
        default=PROJECT_ROOT / "data" / "source" / "abcd" / "data" / "guidelines.json",
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
        default=PROJECT_ROOT / "data" / "source" / "abcd" / "data" / "abcd_v1.1.json",
        help="Conversation JSON file; defaults to the source AB-CD dataset.",
    )
    parser.add_argument(
        "--judges",
        type=Path,
        default=PROJECT_ROOT / "config" / "judges_definitions.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "generated" / "prepared_prompts.jsonl",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output file instead of appending new prompts.",
    )
    parser.add_argument(
        "--batch",
        action="append",
        choices=[batch["name"] for batch in BATCH_CONFIGS],
        help="Only prepare the named batch; repeat to select more than one.",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Allow re-selecting conversations that were already prepared or generated.",
    )
    parser.add_argument(
        "--mutate-sop-percentage",
        type=float,
        default=100,
        help="Percentage of prompts using mutate_sop_for_existing_ask (when applicable).",
    )
    parser.add_argument(
        "--inject-ask-percentage",
        type=float,
        default=0,
        help="Percentage of prompts using inject_new_customer_ask (when applicable).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Seed the local sampler for reproducibility.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        percentages = {
            INJECTION_MECHANISMS[0]: args.mutate_sop_percentage,
            INJECTION_MECHANISMS[1]: args.inject_ask_percentage,
        }
        if any(value < 0 for value in percentages.values()):
            raise InputDataError("Mechanism percentages cannot be negative")
        if abs(sum(percentages.values()) - 100.0) > 1e-9:
            raise InputDataError("Mechanism percentages must add up to 100")

        guidelines = flatten_guidelines(load_json(args.guidelines), args.guidelines)
        guideline_aliases = load_guideline_aliases(args.guideline_aliases)
        judges = load_judges(load_json(args.judges), args.judges)
        rng = random.Random(args.seed)

        raw_conversations = load_json(args.conversations)
        if isinstance(raw_conversations, Mapping):
            raw_conversations = [
                rec
                for split in raw_conversations.values()
                if isinstance(split, list)
                for rec in split
            ]

        # Shuffle conversations so every run without a fixed seed selects a different random order
        rng.shuffle(raw_conversations)

        # Collect existing conversation IDs to avoid repeating previously prepared or generated chats
        existing_ids: set[str] = set()
        if not args.allow_duplicates:
            if not args.overwrite and args.output.is_file():
                existing_ids.update(load_existing_conversation_ids(args.output))
            golden_path = (
                PROJECT_ROOT / "data" / "generated" / "golden_dataset_injected.jsonl"
            )
            if golden_path.is_file():
                existing_ids.update(load_existing_conversation_ids(golden_path))

        if existing_ids:
            print(
                f"Excluding {len(existing_ids)} previously prepared/completed conversation ID(s)."
            )

        records = prepare_all_batches(
            all_conversations=raw_conversations,
            guidelines=guidelines,
            judges=judges,
            rng=rng,
            guideline_aliases=guideline_aliases,
            mechanism_percentages=percentages,
            existing_conversation_ids=existing_ids,
            batch_names=set(args.batch) if args.batch else None,
        )

        should_append = (not args.overwrite) and args.output.is_file()
        write_jsonl(args.output, records, append=should_append)
    except InputDataError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    action_word = "Appended" if should_append else "Wrote"
    print(
        f"Done! {action_word} a total of {len(records)} prepared prompt(s) to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
