#!/usr/bin/env python3
"""Generate and validate injected golden-dataset records concurrently."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

from anthropic import (
    APIConnectionError as AnthropicConnectionError,
    APIStatusError as AnthropicAPIStatusError,
    APITimeoutError as AnthropicTimeoutError,
    AsyncAnthropic,
    RateLimitError as AnthropicRateLimitError,
)
from dotenv import load_dotenv
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator
from tqdm.asyncio import tqdm_asyncio



MAX_VALIDATION_RETRIES = 3
DEFAULT_CONCURRENCY = 1
MAX_NETWORK_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0
DEFAULT_TIMEOUT_SECONDS = 120.0
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default service and model (can also be overridden via CLI arguments --platform and --model).
# Put the selected platform's API key in your .env file.
PLATFORM = "gemini"
MODEL = "gemini-3.5-flash-lite"
PLATFORM_API_KEYS = {
    "chatgpt": "OPENAI_API_KEY",
    "claude": "ANTHROPIC_API_KEY",
    "grok": "XAI_API_KEY",
    "gemini": "GEMINI_API_KEY",
}

# These endpoints expose OpenAI-compatible chat-completions APIs. Claude uses
# its native async client because its message protocol is different.
OPENAI_COMPATIBLE_BASE_URLS = {
    "chatgpt": None,
    "grok": "https://api.x.ai/v1",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
}


class ModelConfigurationError(ValueError):
    """Raised when the provider rejects the configured model."""


def validate_platform(platform: str) -> None:
    """Ensure the selected platform is supported."""
    supported_platforms = set(PLATFORM_API_KEYS) | {"chatgpt_oauth"}
    if platform not in supported_platforms:
        supported = ", ".join(sorted(supported_platforms))
        raise ModelConfigurationError(
            f"Platform {platform!r} does not exist. Choose one of: {supported}."
        )


def is_model_not_found_error(exc: Exception) -> bool:
    """Recognize provider responses for a model that is unavailable."""
    status_code = getattr(exc, "status_code", None)
    message = str(exc).casefold()
    return status_code == 404 or (
        "model" in message
        and any(
            phrase in message
            for phrase in ("not found", "does not exist", "not available", "unknown")
        )
    )


DATA_JUDGES: set[str] = {"missing_data", "data_conflict"}


class BaseInjectionMetadata(BaseModel):
    """Base metadata describing the error injection operation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    target_judge: str
    already_defective: bool
    scenario_modified: bool
    modified_scenario_fields: dict[str, Any]
    injected_turn_indices: list[int]
    total_injected_errors: int = Field(ge=0)
    target_dialogue_phase: Literal["early", "middle", "late"]
    scope: Literal["pre_existing", "single_turn", "multi_turn_injection"]


class GeneralInjectionMetadata(BaseInjectionMetadata):
    """Metadata for general injection prompt records."""


class TaskClosureInjectionMetadata(GeneralInjectionMetadata):
    """V5.4 Task Closure metadata (supports downstream CUSTOMER coherence edits).

    The default maintains compatibility with older Task Closure datasets that
    predate customer_adjusted_turn_indices. Other judges remain strict.
    """

    customer_adjusted_turn_indices: list[int] = Field(default_factory=list)


class DataJudgesInjectionMetadata(BaseInjectionMetadata):
    """Metadata for data judges (missing_data, data_conflict) records."""

    injection_mechanism: Literal[
        "mutate_sop_for_existing_ask",
        "inject_new_customer_ask",
    ]
    sop_modified: bool
    original_sop_content: str | dict[str, Any] | list[Any] | None = None
    modified_sop_content: str | dict[str, Any] | list[Any] | None = None


# Alias for backward-compatibility or generic typing
InjectionMetadata = (
    GeneralInjectionMetadata | TaskClosureInjectionMetadata | DataJudgesInjectionMetadata
)


class Turn(BaseModel):
    """Single conversation turn."""

    model_config = ConfigDict(extra="forbid", strict=True)

    turn_id: int = Field(ge=1)
    speaker: Literal["CUSTOMER", "AGENT", "ACTION"]
    content: str


class ViolationItem(BaseModel):
    """Details of a single procedural or policy violation."""

    model_config = ConfigDict(extra="forbid", strict=True)

    turn_id: int = Field(ge=1)
    speaker: Literal["CUSTOMER", "AGENT", "ACTION"]
    explanation: str


class JudgeEvaluationDetail(BaseModel):
    """Evaluation output for the target judge."""

    model_config = ConfigDict(extra="forbid", strict=True)

    verdict: Literal["FAIL"]
    violations: list[ViolationItem]
    overall_reasoning: str


MetadataT = TypeVar("MetadataT", bound=BaseInjectionMetadata)


class BaseGoldenRecordOutput(BaseModel, Generic[MetadataT]):
    """Base schema for generated golden dataset record outputs."""

    model_config = ConfigDict(extra="forbid", strict=True)

    injection_metadata: MetadataT
    injected_conversation: list[Turn] = Field(min_length=1)
    judge_evaluations: dict[str, JudgeEvaluationDetail]

    def _validate_common_structure(
        self, metadata: BaseInjectionMetadata
    ) -> None:
        """Validate cross-field business rules and consistency."""
        target_judge = metadata.target_judge
        if target_judge not in self.judge_evaluations:
            raise ValueError(
                f"judge_evaluations is missing the exact target_judge key "
                f"{target_judge!r}"
            )

        actual_turn_ids = [turn.turn_id for turn in self.injected_conversation]
        if target_judge == "task_closure":
            # V5.4 preserves source IDs; they need not start at 1 or be contiguous.
            if any(a >= b for a, b in zip(actual_turn_ids, actual_turn_ids[1:])):
                raise ValueError(
                    "task_closure turn_ids must be strictly increasing and unique; "
                    f"got {actual_turn_ids!r}"
                )
        else:
            expected_turn_ids = list(range(1, len(self.injected_conversation) + 1))
            if actual_turn_ids != expected_turn_ids:
                raise ValueError(
                    "injected_conversation turn_ids must be strictly sequential "
                    f"starting at 1; got {actual_turn_ids!r}"
                )

        if len(metadata.injected_turn_indices) != metadata.total_injected_errors:
            raise ValueError(
                "injected_turn_indices length "
                f"{len(metadata.injected_turn_indices)} does not equal "
                f"total_injected_errors {metadata.total_injected_errors}"
            )

        conversation_ids = set(actual_turn_ids)
        missing_ids = sorted(
            set(metadata.injected_turn_indices) - conversation_ids
        )
        if missing_ids:
            raise ValueError(
                "injected_turn_indices contains turn_ids absent from "
                f"injected_conversation: {missing_ids!r}"
            )

        violations = self.judge_evaluations[target_judge].violations
        if len(violations) != metadata.total_injected_errors:
            raise ValueError(
                f"target judge {target_judge!r} has {len(violations)} violations, "
                f"expected {metadata.total_injected_errors}"
            )

        violation_ids = [violation.turn_id for violation in violations]
        if sorted(violation_ids) != sorted(metadata.injected_turn_indices):
            raise ValueError(
                "violation turn_ids "
                f"{violation_ids!r} do not match injected_turn_indices "
                f"{metadata.injected_turn_indices!r}"
            )

        # Verify each violation's speaker matches the speaker at that turn in the conversation
        turn_speaker_map = {
            turn.turn_id: turn.speaker for turn in self.injected_conversation
        }
        for violation in violations:
            expected_speaker = turn_speaker_map.get(violation.turn_id)
            if expected_speaker and violation.speaker != expected_speaker:
                raise ValueError(
                    f"Violation at turn {violation.turn_id} lists speaker {violation.speaker!r}, "
                    f"but injected_conversation has speaker {expected_speaker!r}"
                )
            if violation.speaker == "CUSTOMER":
                raise ValueError(
                    f"Violation at turn {violation.turn_id} has speaker 'CUSTOMER'; "
                    "violations must strictly reference AGENT or ACTION turns"
                )

        if metadata.scenario_modified and not metadata.modified_scenario_fields:
            raise ValueError(
                "modified_scenario_fields must not be empty when "
                "scenario_modified is true"
            )
        if not metadata.scenario_modified and metadata.modified_scenario_fields:
            raise ValueError(
                "modified_scenario_fields must be {} when "
                "scenario_modified is false"
            )


class GeneralGoldenRecordOutput(BaseGoldenRecordOutput[GeneralInjectionMetadata]):
    """Schema for records generated from the general injection prompt."""

    @model_validator(mode="after")
    def validate_structure(self) -> "GeneralGoldenRecordOutput":
        self._validate_common_structure(self.injection_metadata)
        if self.injection_metadata.target_judge in DATA_JUDGES:
            raise ValueError(
                f"target_judge {self.injection_metadata.target_judge!r} is a data judge. "
                f"Use DataJudgesGoldenRecordOutput for {sorted(DATA_JUDGES)}."
            )
        return self


class TaskClosureGoldenRecordOutput(BaseGoldenRecordOutput[TaskClosureInjectionMetadata]):
    """Task Closure-specific schema; does not relax other judges' contracts."""

    @model_validator(mode="after")
    def validate_structure(self) -> "TaskClosureGoldenRecordOutput":
        self._validate_common_structure(self.injection_metadata)
        if self.injection_metadata.target_judge != "task_closure":
            raise ValueError("TaskClosureGoldenRecordOutput requires task_closure")
        return self


class DataJudgesGoldenRecordOutput(BaseGoldenRecordOutput[DataJudgesInjectionMetadata]):
    """Schema for records generated from the data judges injection prompt."""

    @model_validator(mode="after")
    def validate_structure(self) -> "DataJudgesGoldenRecordOutput":
        self._validate_common_structure(self.injection_metadata)
        if self.injection_metadata.target_judge not in DATA_JUDGES:
            raise ValueError(
                f"target_judge {self.injection_metadata.target_judge!r} is not a valid data judge. "
                f"Expected one of: {sorted(DATA_JUDGES)}."
            )
        return self


class GoldenRecordOutput:
    """Validator factory that dispatches to General or DataJudges output schema."""

    @classmethod
    def model_validate(
        cls,
        obj: Any,
        *,
        strict: bool | None = None,
        from_attributes: bool | None = None,
        context: dict[str, Any] | None = None,
    ) -> GeneralGoldenRecordOutput | TaskClosureGoldenRecordOutput | DataJudgesGoldenRecordOutput:
        judge = None
        if isinstance(obj, dict):
            meta = obj.get("injection_metadata")
            if isinstance(meta, dict):
                judge = meta.get("target_judge")
        elif hasattr(obj, "injection_metadata"):
            meta = getattr(obj, "injection_metadata")
            judge = getattr(meta, "target_judge", None)

        if judge in DATA_JUDGES:
            return DataJudgesGoldenRecordOutput.model_validate(
                obj,
                strict=strict,
                from_attributes=from_attributes,
                context=context,
            )
        if judge == "task_closure":
            return TaskClosureGoldenRecordOutput.model_validate(
                obj,
                strict=strict,
                from_attributes=from_attributes,
                context=context,
            )
        return GeneralGoldenRecordOutput.model_validate(
            obj,
            strict=strict,
            from_attributes=from_attributes,
            context=context,
        )


def _has_sop_content(content: Any) -> bool:
    """Check if SOP snippet contains non-empty text, dict, or list content."""
    if content is None:
        return False
    if isinstance(content, str):
        return bool(content.strip())
    if isinstance(content, (dict, list)):
        return len(content) > 0
    return False


def _validate_common_sampled_rules(
    record: BaseGoldenRecordOutput,
    sampled_params: dict[str, Any],
    expected_target_judge: str | None = None,
) -> None:
    """Validate sampled parameters common to both general and data judges."""
    metadata: BaseInjectionMetadata = getattr(record, "injection_metadata")
    if expected_target_judge and metadata.target_judge != expected_target_judge:
        raise ValueError(
            f"target_judge {metadata.target_judge!r} does not match expected "
            f"{expected_target_judge!r}"
        )

    target_error_count = sampled_params.get("target_error_count")

    if metadata.already_defective:
        if metadata.scope != "pre_existing":
            raise ValueError(
                "scope must be 'pre_existing' when already_defective is true"
            )
    else:
        if (
            target_error_count is not None
            and metadata.total_injected_errors != target_error_count
        ):
            raise ValueError(
                "total_injected_errors "
                f"{metadata.total_injected_errors} does not equal sampled "
                f"target_error_count {target_error_count}"
            )
        target_phase = sampled_params.get("target_dialogue_phase")
        if target_phase == "auto":
            if metadata.target_judge != "task_closure":
                raise ValueError("auto phase is reserved for task_closure")
            # The LLM must return a concrete phase, never the input sentinel.
            # Pydantic already enforces early/middle/late in the output schema.
            # Approximate positional bands overlap because dialogue phases are
            # semantic, not fixed-size thirds of a conversation.
            total_turns = len(record.injected_conversation)
            low, high = {
                "early": (0.0, 0.45),
                "middle": (0.20, 0.80),
                "late": (0.55, 1.0),
            }[metadata.target_dialogue_phase]
            turn_positions = {
                turn.turn_id: index
                for index, turn in enumerate(record.injected_conversation)
            }
            relative_positions = [
                turn_positions[turn_id] / max(1, total_turns - 1)
                for turn_id in metadata.injected_turn_indices
            ]
            if not all(low <= position <= high for position in relative_positions):
                raise ValueError(
                    "LLM-selected phase is inconsistent with the reported "
                    "injection turn positions"
                )
        elif target_phase and metadata.target_dialogue_phase != target_phase:
            raise ValueError(
                f"target_dialogue_phase {metadata.target_dialogue_phase!r} does "
                f"not equal sampled target_dialogue_phase {target_phase!r}"
            )
        expected_scope = (
            "single_turn"
            if metadata.total_injected_errors == 1
            else "multi_turn_injection"
        )
        if metadata.scope != expected_scope:
            raise ValueError(
                f"scope must be {expected_scope!r} when total_injected_errors "
                f"is {metadata.total_injected_errors}"
            )


def validate_general_rules(
    record: GeneralGoldenRecordOutput,
    sampled_params: dict[str, Any],
    expected_target_judge: str | None = None,
) -> None:
    """Validate business rules specific to general prompt outputs."""
    _validate_common_sampled_rules(record, sampled_params, expected_target_judge)


def validate_data_judges_rules(
    record: DataJudgesGoldenRecordOutput,
    sampled_params: dict[str, Any],
    expected_target_judge: str | None = None,
) -> None:
    """Validate business rules specific to data judges prompt outputs."""
    _validate_common_sampled_rules(record, sampled_params, expected_target_judge)
    metadata = record.injection_metadata

    sampled_mechanism = sampled_params.get("injection_mechanism")
    if sampled_mechanism and metadata.injection_mechanism != sampled_mechanism:
        raise ValueError(
            f"injection_mechanism {metadata.injection_mechanism!r} does not "
            f"match sampled injection_mechanism {sampled_mechanism!r}"
        )

    if metadata.already_defective:
        if metadata.sop_modified:
            raise ValueError(
                "sop_modified must be false when already_defective is true"
            )
        if metadata.original_sop_content is not None:
            raise ValueError(
                "original_sop_content must be null when already_defective is true"
            )
        if metadata.modified_sop_content is not None:
            raise ValueError(
                "modified_sop_content must be null when already_defective is true"
            )
        if metadata.scenario_modified:
            raise ValueError(
                "scenario_modified must be false when already_defective is true"
            )
    else:
        if not metadata.sop_modified:
            if metadata.original_sop_content is not None:
                raise ValueError(
                    "original_sop_content must be null when sop_modified is false"
                )
            if metadata.modified_sop_content is not None:
                raise ValueError(
                    "modified_sop_content must be null when sop_modified is false"
                )
        else:
            if not _has_sop_content(metadata.original_sop_content):
                raise ValueError(
                    "original_sop_content must be a non-empty string, object, "
                    "or list when sop_modified is true"
                )
            if not _has_sop_content(metadata.modified_sop_content):
                raise ValueError(
                    "modified_sop_content must be a non-empty string, object, "
                    "or list when sop_modified is true"
                )
            if metadata.modified_sop_content == metadata.original_sop_content:
                raise ValueError(
                    "modified_sop_content must differ from original_sop_content "
                    "when sop_modified is true"
                )

            original_sop = sampled_params.get("sop_content")
            if (
                original_sop is not None
                and metadata.modified_sop_content == original_sop
            ):
                raise ValueError(
                    "modified_sop_content must differ from the input sop_content "
                    "when sop_modified is true"
                )

        if (
            metadata.target_judge == "data_conflict"
            and not (metadata.sop_modified or metadata.scenario_modified)
        ):
            raise ValueError(
                "data_conflict records must modify the SOP or scenario when "
                "already_defective is false"
            )

        if (
            metadata.target_judge == "missing_data"
            and metadata.injection_mechanism == "mutate_sop_for_existing_ask"
            and not metadata.sop_modified
        ):
            raise ValueError(
                "missing_data records using mutate_sop_for_existing_ask must "
                "modify the SOP when already_defective is false"
            )


def validate_task_closure_source(
    record: TaskClosureGoldenRecordOutput,
    source_turns: list[dict[str, Any]],
) -> None:
    """Validate V5.4 structural edits against original turns.

    Semantic coherence (that a rewritten CUSTOMER line only repairs a direct
    contradiction and preserves every actual request) requires review: the
    source turn diff alone cannot establish that reliably.
    """
    if not isinstance(source_turns, list) or not source_turns:
        raise ValueError("task_closure requires non-empty source_turns")
    generated = record.injected_conversation
    if len(generated) != len(source_turns):
        raise ValueError("task_closure must preserve source turn count")

    metadata = record.injection_metadata
    if metadata.scenario_modified or metadata.modified_scenario_fields:
        raise ValueError("task_closure cannot modify scenario or scenario fields")

    target_ids = metadata.injected_turn_indices
    customer_ids = metadata.customer_adjusted_turn_indices
    if len(target_ids) != len(set(target_ids)):
        raise ValueError("duplicate injected_turn_indices are not allowed")
    if len(customer_ids) != len(set(customer_ids)):
        raise ValueError("duplicate customer_adjusted_turn_indices are not allowed")
    if any(type(turn_id) is not int or turn_id < 1 for turn_id in customer_ids):
        raise ValueError("customer_adjusted_turn_indices must contain positive integer turn IDs")
    if any(type(turn_id) is not int for turn_id in target_ids):
        raise ValueError("injected_turn_indices must contain integer turn IDs")

    changed_agents: set[int] = set()
    changed_customers: set[int] = set()
    original_ids: list[int] = []
    original_speakers: dict[int, str] = {}
    for position, (original, edited) in enumerate(zip(source_turns, generated), start=1):
        original_id = original.get("turn_id")
        original_speaker = original.get("speaker")
        original_content = original.get("content")
        if type(original_id) is not int or original_id < 1:
            raise ValueError(f"invalid source turn_id at position {position}")
        if edited.turn_id != original_id:
            raise ValueError(f"task_closure changed original turn_id at position {position}")
        if edited.speaker != original_speaker:
            raise ValueError(f"task_closure changed speaker role at turn {original_id}")
        if not isinstance(original_content, str):
            raise ValueError(f"invalid source content at turn {original_id}")
        original_ids.append(original_id)
        original_speakers[original_id] = original_speaker
        if original_content == edited.content:
            continue
        if metadata.already_defective:
            raise ValueError("already_defective must preserve the complete conversation")
        if edited.speaker == "ACTION":
            raise ValueError(f"task_closure cannot alter verified ACTION turn {original_id}")
        if edited.speaker == "AGENT":
            changed_agents.add(original_id)
        elif edited.speaker == "CUSTOMER":
            changed_customers.add(original_id)

    if any(a >= b for a, b in zip(original_ids, original_ids[1:])):
        raise ValueError("source turn_ids must be strictly increasing and unique")

    if metadata.already_defective:
        if customer_ids:
            raise ValueError("already_defective requires empty customer_adjusted_turn_indices")
        # injected_turn_indices for pre-existing defects identifies existing
        # offending AGENT/ACTION turns, not edits. They were not changed above.
        for turn_id in target_ids:
            if original_speakers.get(turn_id) not in {"AGENT", "ACTION"}:
                raise ValueError("pre-existing violations must reference AGENT/ACTION turns")
        return

    if changed_agents != set(target_ids):
        raise ValueError(
            "changed AGENT turn IDs do not match injected_turn_indices: "
            f"actual={sorted(changed_agents)}, claimed={sorted(set(target_ids))}"
        )
    if changed_customers != set(customer_ids):
        raise ValueError(
            "changed CUSTOMER turn IDs do not match customer_adjusted_turn_indices: "
            f"actual={sorted(changed_customers)}, claimed={sorted(set(customer_ids))}"
        )
    for turn_id in changed_customers:
        if not any(agent_id < turn_id for agent_id in changed_agents):
            raise ValueError(
                f"CUSTOMER coherence edit at turn {turn_id} is not downstream "
                "of a modified AGENT turn"
            )



def validate_record(
    payload: Any,
    sampled_params: dict[str, Any],
    expected_target_judge: str | None = None,
) -> GeneralGoldenRecordOutput | DataJudgesGoldenRecordOutput:
    """Validate schema and sampled-parameter business rules based on judge type."""
    judge = expected_target_judge
    if not judge and isinstance(payload, dict):
        metadata = payload.get("injection_metadata")
        if isinstance(metadata, dict):
            judge = metadata.get("target_judge")

    if judge in DATA_JUDGES:
        record = DataJudgesGoldenRecordOutput.model_validate(payload)
        validate_data_judges_rules(record, sampled_params, expected_target_judge)
    elif judge == "task_closure":
        record = TaskClosureGoldenRecordOutput.model_validate(payload)
        validate_general_rules(record, sampled_params, expected_target_judge)
    else:
        record = GeneralGoldenRecordOutput.model_validate(payload)
        validate_general_rules(record, sampled_params, expected_target_judge)

    return record


def parse_json_response(raw_response: str) -> Any:
    """Extract and parse the outermost JSON object from the model response."""
    text = raw_response.strip()
    start_idx = text.find("{")
    end_idx = text.rfind("}")
    if start_idx != -1 and end_idx != -1 and end_idx >= start_idx:
        text = text[start_idx : end_idx + 1]
    return json.loads(text)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Load records from a JSONL file."""
    records: list[dict[str, Any]] = []
    try:
        with path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON in {path} line {line_number}: {exc}"
                    ) from exc
                if not isinstance(value, dict):
                    raise ValueError(
                        f"{path} line {line_number} must contain a JSON object"
                    )
                records.append(value)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input file not found: {path}") from exc
    return records


def load_completed_ids(path: Path) -> set[str]:
    """Load already processed conversation IDs to support resuming."""
    if not path.exists():
        return set()
    completed: set[str] = set()
    for record in load_jsonl(path):
        conversation_id = record.get("conversation_id")
        if conversation_id is not None:
            completed.add(str(conversation_id))
    return completed


def raw_completion_text(response: Any) -> str:
    """Extract text content from an OpenAI-compatible completion response."""
    content = response.choices[0].message.content
    if not content:
        raise ValueError("LLM response contained no message content")
    return content


def is_retryable_network_error(exc: Exception) -> bool:
    """Check if an exception is a transient network or rate-limit error."""
    if isinstance(
        exc,
        (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            AnthropicConnectionError,
            AnthropicTimeoutError,
            AnthropicRateLimitError,
            asyncio.TimeoutError,
        ),
    ):
        return True
    if isinstance(exc, (APIStatusError, AnthropicAPIStatusError)):
        return exc.status_code == 429 or exc.status_code >= 500
    return False


async def request_with_backoff(
    client: Any,
    semaphore: asyncio.Semaphore,
    platform: str,
    model: str,
    messages: list[dict[str, str]],
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> str:
    """Execute an API call with semaphore concurrency control and exponential backoff."""
    for network_attempt in range(MAX_NETWORK_RETRIES + 1):
        try:
            async with semaphore:
                if platform == "chatgpt_oauth":
                    return await client.generate(messages, model, timeout_seconds)
                if platform == "claude":
                    response = await client.messages.create(
                        model=model,
                        max_tokens=8192,
                        messages=messages,
                        timeout=timeout_seconds,
                    )
                    content = "".join(
                        block.text
                        for block in response.content
                        if getattr(block, "type", None) == "text"
                    )
                    if not content:
                        raise ValueError("Claude response contained no text content")
                    return content
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    response_format={"type": "json_object"},
                    timeout=timeout_seconds,
                )
                return raw_completion_text(response)
        except Exception as exc:
            if is_model_not_found_error(exc):
                raise ModelConfigurationError(
                    f"Model {model!r} does not exist or is unavailable on "
                    f"platform {platform!r}. Provider response: {exc}"
                ) from exc
            if not is_retryable_network_error(exc) or (
                network_attempt >= MAX_NETWORK_RETRIES
            ):
                raise
            delay = min(
                BACKOFF_MAX_SECONDS,
                BACKOFF_BASE_SECONDS * (2**network_attempt),
            )
            await asyncio.sleep(random.uniform(0, delay))
    raise RuntimeError("unreachable network retry state")


async def append_jsonl(
    path: Path, value: dict[str, Any], lock: asyncio.Lock
) -> None:
    """Safely append a single JSON record to a file using an asyncio lock."""
    line = json.dumps(value, ensure_ascii=False) + "\n"
    async with lock:
        await asyncio.to_thread(_append_text, path, line)


def _append_text(path: Path, text: str) -> None:
    """Blocking file write helper executed inside a thread."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(text)
        file.flush()


async def process_record(
    input_record: dict[str, Any],
    client: Any,
    platform: str,
    model: str,
    semaphore: asyncio.Semaphore,
    output_path: Path,
    failed_path: Path,
    output_lock: asyncio.Lock,
    failed_lock: asyncio.Lock,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> None:
    """Process a single prompt record with a self-correcting validation retry loop."""
    conversation_id = input_record.get("conversation_id")
    target_judge = input_record.get("target_judge")
    sampled_params = input_record.get("sampled_params")
    assembled_prompt = input_record.get("assembled_prompt")

    if conversation_id is None or not isinstance(sampled_params, dict):
        raise ValueError("Each input record needs conversation_id and sampled_params")
    if not isinstance(assembled_prompt, str) or not assembled_prompt:
        raise ValueError(f"{conversation_id}: assembled_prompt must be non-empty")

    messages = [{"role": "user", "content": assembled_prompt}]
    validation_errors: list[str] = []
    raw_output = ""

    for validation_attempt in range(MAX_VALIDATION_RETRIES + 1):
        try:
            raw_output = await request_with_backoff(
                client, semaphore, platform, model, messages, timeout_seconds
            )
        except ModelConfigurationError as exc:
            print(f"Model configuration error: {exc}")
            await append_jsonl(
                failed_path,
                {
                    "conversation_id": conversation_id,
                    "target_judge": target_judge,
                    "sampled_params": sampled_params,
                    "model_error": str(exc),
                },
                failed_lock,
            )
            return
        except (
            APIConnectionError,
            APITimeoutError,
            RateLimitError,
            APIStatusError,
            AnthropicConnectionError,
            AnthropicTimeoutError,
            AnthropicRateLimitError,
            AnthropicAPIStatusError,
            asyncio.TimeoutError,
        ) as exc:
            await append_jsonl(
                failed_path,
                {
                    "conversation_id": conversation_id,
                    "target_judge": target_judge,
                    "sampled_params": sampled_params,
                    "network_error": str(exc),
                    "last_raw_response": raw_output,
                },
                failed_lock,
            )
            return

        try:
            payload = parse_json_response(raw_output)
            if isinstance(payload, dict) and payload.get("error") == "NO_VALID_INJECTION":
                if target_judge != "task_closure":
                    raise ValueError("NO_VALID_INJECTION is only allowed for task_closure")
                skip_reason = payload.get("reason")
                if not isinstance(skip_reason, str) or not skip_reason.strip():
                    raise ValueError("NO_VALID_INJECTION requires a non-empty reason")
                # A legitimate non-injectable sample is NOT a malformed generation.
                await append_jsonl(
                    failed_path.with_name("skipped_generations.jsonl"),
                    {
                        "conversation_id": conversation_id,
                        "target_judge": target_judge,
                        "status": "skipped",
                        "reason": skip_reason,
                        "sampled_params": sampled_params,
                    },
                    failed_lock,
                )
                return
            record = validate_record(payload, sampled_params, target_judge)
            if target_judge == "task_closure":
                validate_task_closure_source(record, input_record.get("source_turns"))
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            validation_errors.append(str(exc))
            if validation_attempt >= MAX_VALIDATION_RETRIES:
                break
            messages.extend(
                [
                    {"role": "assistant", "content": raw_output},
                    {
                        "role": "user",
                        "content": (
                            "Your previous output failed validation with the "
                            "following error(s):\n"
                            f"{exc}\n\n"
                            "Please fix these specific issues while strictly "
                            "following all original instructions and schema "
                            "rules. Output ONLY the corrected valid JSON."
                        ),
                    },
                ]
            )
        else:
            dumped = record.model_dump(mode="json")
            await append_jsonl(
                output_path,
                {
                    "conversation_id": conversation_id,
                    "injection_metadata": dumped["injection_metadata"],
                    "injected_conversation": dumped["injected_conversation"],
                    "judge_evaluations": dumped["judge_evaluations"],
                },
                output_lock,
            )
            return

    await append_jsonl(
        failed_path,
        {
            "conversation_id": conversation_id,
            "target_judge": target_judge,
            "sampled_params": sampled_params,
            "validation_errors": validation_errors,
            "last_raw_response": raw_output,
        },
        failed_lock,
    )


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "generated" / "prepared_prompts.jsonl"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "generated" / "golden_dataset_injected.jsonl"),
    )
    parser.add_argument(
        "--failed",
        default=str(PROJECT_ROOT / "data" / "generated" / "failed_generations.jsonl"),
    )
    parser.add_argument("--platform", default=PLATFORM)
    parser.add_argument("--model", default=None)
    parser.add_argument("--only-judge", default=None, help="Only run one target judge")
    parser.add_argument("--oauth-script", default=os.getenv("CHATGPT_OAUTH_SCRIPT"),
                        help="Path to the authenticated local OAuth script")
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("MAX_CONCURRENT_REQUESTS", DEFAULT_CONCURRENCY)),
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> None:
    """Main asynchronous runner."""
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")

    platform = args.platform
    model = args.model or ("gpt-6-luna" if platform == "chatgpt_oauth" else MODEL)
    validate_platform(platform)
    if platform == "chatgpt_oauth" and model != "gpt-6-luna":
        raise ValueError("OAuth testing is locked to gpt-6-luna; no Sol fallback")

    api_key = None
    if platform != "chatgpt_oauth":
        api_key_name = PLATFORM_API_KEYS[platform]
        api_key = os.getenv(api_key_name)
        if not api_key:
            raise RuntimeError(
                f"{api_key_name} is not set. Add it to the .env file for "
                f"platform {platform!r}."
            )

    input_path = Path(args.input)
    output_path = Path(args.output)
    failed_path = Path(args.failed)
    completed_ids = load_completed_ids(output_path)
    # Avoid repeatedly regenerating non-injectable cases after resuming.
    completed_ids |= load_completed_ids(
        failed_path.with_name("skipped_generations.jsonl")
    )
    input_records = [
        record
        for record in load_jsonl(input_path)
        if str(record.get("conversation_id")) not in completed_ids
        and (args.only_judge is None or record.get("target_judge") == args.only_judge)
    ]
    if not input_records:
        print("No pending records.")
        return

    if platform == "chatgpt_oauth":
        if not args.oauth_script:
            raise RuntimeError(
                "Specify --oauth-script with the authenticated "
                "chatgpt_plus_oauth_fixed.py file. Never supply tokens."
            )
        client: Any = ChatGPTOAuthResponsesClient(args.oauth_script)
        await asyncio.to_thread(client.check_login)
    elif platform == "claude":
        client = AsyncAnthropic(
            api_key=api_key, max_retries=0, timeout=args.timeout
        )
    else:
        client_options: dict[str, Any] = {
            "api_key": api_key,
            "max_retries": 0,
            "timeout": args.timeout,
        }
        base_url = OPENAI_COMPATIBLE_BASE_URLS[platform]
        if base_url:
            client_options["base_url"] = base_url
        client = AsyncOpenAI(**client_options)

    semaphore = asyncio.Semaphore(args.concurrency)
    output_lock = asyncio.Lock()
    failed_lock = asyncio.Lock()
    tasks = [
        process_record(
            record,
            client,
            platform,
            model,
            semaphore,
            output_path,
            failed_path,
            output_lock,
            failed_lock,
            args.timeout,
        )
        for record in input_records
    ]
    try:
        await tqdm_asyncio.gather(
            *tasks, desc="Generating golden dataset", unit="record"
        )
    finally:
        await client.close()


def main() -> None:
    """CLI entrypoint."""
    load_dotenv()
    args = parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()