#!/usr/bin/env python3
"""Generate and validate injected golden-dataset records concurrently."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from collections.abc import Mapping
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

try:  # Support both direct script execution and package imports.
    from . import factual_hallucination as factual
except ImportError:  # pragma: no cover - exercised when run as a script
    import factual_hallucination as factual


MAX_VALIDATION_RETRIES = 3
DEFAULT_CONCURRENCY = 1
MAX_NETWORK_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0
BACKOFF_MAX_SECONDS = 30.0
DEFAULT_TIMEOUT_SECONDS = 60
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
    supported_platforms = set(PLATFORM_API_KEYS)
    if platform not in supported_platforms:
        supported = ", ".join(sorted(supported_platforms))
        raise ModelConfigurationError(
            f"Platform {platform!r} does not exist. Choose one of: {supported}."
        )


def resolve_api_key(platform: str) -> str:
    """Return the configured API key name/value for a supported platform."""

    validate_platform(platform)
    api_key_name = PLATFORM_API_KEYS[platform]
    api_key = os.getenv(api_key_name)
    if not api_key:
        raise RuntimeError(
            f"{api_key_name} is not set. Add it to the .env file for "
            f"platform {platform!r}."
        )
    return api_key


def create_async_client(platform: str, api_key: str, timeout: float) -> Any:
    """Create the provider client with the shared configuration policy."""

    if platform == "claude":
        return AsyncAnthropic(api_key=api_key, max_retries=0, timeout=timeout)
    client_options: dict[str, Any] = {
        "api_key": api_key,
        "max_retries": 0,
        "timeout": timeout,
    }
    base_url = OPENAI_COMPATIBLE_BASE_URLS[platform]
    if base_url:
        client_options["base_url"] = base_url
    return AsyncOpenAI(**client_options)


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


class DataJudgesInjectionMetadata(BaseInjectionMetadata):
    """Metadata for data judges (missing_data, data_conflict) records."""

    injection_mechanism: Literal[
        "mutate_sop_for_existing_ask",
        "inject_new_customer_ask",
    ]
    selected_pattern: str
    sop_modified: bool
    original_sop_content: str | dict[str, Any] | list[Any] | None = None
    modified_sop_content: str | dict[str, Any] | list[Any] | None = None


# Alias for backward-compatibility or generic typing
InjectionMetadata = GeneralInjectionMetadata | DataJudgesInjectionMetadata


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

        expected_turn_ids = list(range(1, len(self.injected_conversation) + 1))
        actual_turn_ids = [turn.turn_id for turn in self.injected_conversation]
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
    ) -> GeneralGoldenRecordOutput | DataJudgesGoldenRecordOutput:
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
        if target_phase and metadata.target_dialogue_phase != target_phase:
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
    else:
        record = GeneralGoldenRecordOutput.model_validate(payload)
        validate_general_rules(record, sampled_params, expected_target_judge)

    return record


def _source_conversation_from_record(
    record: "factual.FactualHallucinationGoldenRecordOutput",
) -> list[dict[str, Any]]:
    """Recover the source turns when the prepared record is unavailable."""

    turns = [
        turn.model_dump(mode="json") for turn in record.judge_input.conversation
    ]
    metadata = record.injection_metadata
    if metadata.injection_status == "injected":
        for turn in turns:
            if turn.get("turn_id") != metadata.injection_turn_id:
                continue
            content = str(turn.get("content", ""))
            if metadata.injected_claim in content:
                turn["content"] = content.replace(
                    metadata.injected_claim, metadata.original_agent_claim, 1
                )
            break
    return turns


def _factual_audit_context(
    record: "factual.FactualHallucinationGoldenRecordOutput",
    input_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the source and screening context needed for independent review."""

    def prepared_value(key: str, fallback: Any) -> Any:
        if isinstance(input_record, Mapping) and key in input_record:
            return input_record[key]
        return fallback

    source_conversation = prepared_value("source_conversation", None)
    if not isinstance(source_conversation, list):
        source_conversation = _source_conversation_from_record(record)

    available_evidence = prepared_value("available_evidence", None)
    if not isinstance(available_evidence, list):
        available_evidence = [
            item.model_dump(mode="json") for item in record.judge_input.evidence
        ]

    source_scenario = prepared_value(
        "source_scenario", record.judge_input.scenario
    )
    if not isinstance(source_scenario, Mapping):
        source_scenario = record.judge_input.scenario

    judge_definition = prepared_value(
        "judge_definition", record.judge_input.target_judge_definition
    )
    if not isinstance(judge_definition, Mapping):
        judge_definition = record.judge_input.target_judge_definition

    context: dict[str, Any] = {
        "source_conversation": [dict(turn) for turn in source_conversation],
        "source_scenario": dict(source_scenario),
        "source_sop": prepared_value("source_sop", record.judge_input.sop),
        "available_evidence": [dict(item) for item in available_evidence],
        "judge_definition": dict(judge_definition),
    }
    if isinstance(input_record, Mapping):
        if "screening" in input_record:
            context["screening"] = input_record["screening"]
        if "screened_candidate" in input_record:
            context["screened_candidate"] = input_record["screened_candidate"]
    return context


def build_factual_output_record(
    conversation_id: Any,
    record: "factual.FactualHallucinationGoldenRecordOutput",
    input_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Shape a validated factual record like an original framework JSONL row."""

    metadata = record.injection_metadata
    explanation = metadata.contradiction_explanation
    return {
        "conversation_id": conversation_id,
        "injection_metadata": metadata.model_dump(mode="json"),
        "injected_conversation": [
            turn.model_dump(mode="json") for turn in record.judge_input.conversation
        ],
        "judge_evaluations": {
            metadata.target_judge: {
                "verdict": "FAIL",
                "violations": [
                    {
                        "turn_id": metadata.injection_turn_id,
                        "speaker": "AGENT",
                        "explanation": explanation,
                    }
                ],
                "overall_reasoning": explanation,
            }
        },
        "audit_context": _factual_audit_context(record, input_record),
        "validation_report": record.validation_report.model_dump(mode="json"),
    }


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


def load_pending_records(
    input_path: Path, output_path: Path, rejected_path: Path
) -> list[dict[str, Any]]:
    """Return input records that are not accepted or structurally rejected.

    REVIEW rows are deliberately excluded from the completed set. A record that
    was blocked before generation must be retried after the input is corrected,
    rather than silently skipped forever.
    """

    completed_ids = load_completed_ids(output_path) | load_completed_ids(
        rejected_path
    )
    return [
        record
        for record in load_jsonl(input_path)
        if str(record.get("conversation_id")) not in completed_ids
    ]


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
    rejected_path: Path,
    output_lock: asyncio.Lock,
    failed_lock: asyncio.Lock,
    rejected_lock: asyncio.Lock,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    review_path: Path | None = None,
    review_lock: asyncio.Lock | None = None,
    allow_unscreened_factual: bool = False,
) -> None:
    """Process a single prompt record with a self-correcting validation retry loop.

    ``allow_unscreened_factual`` is retained only for source compatibility. The
    screening gate still fails closed.
    """
    conversation_id = input_record.get("conversation_id")
    target_judge = input_record.get("target_judge")
    sampled_params = input_record.get("sampled_params")
    assembled_prompt = input_record.get("assembled_prompt")

    if conversation_id is None or not isinstance(sampled_params, dict):
        raise ValueError("Each input record needs conversation_id and sampled_params")
    if not isinstance(assembled_prompt, str) or not assembled_prompt:
        raise ValueError(f"{conversation_id}: assembled_prompt must be non-empty")

    review_path = review_path or failed_path
    review_lock = review_lock or failed_lock

    if target_judge == factual.FACTUAL_JUDGE:
        gate = factual.evaluate_generation_gate(
            input_record, allow_unscreened=allow_unscreened_factual
        )
        if not gate.ready:
            if gate.decision == "REJECT":
                rejection = factual.FactualRejectionRecord(
                    conversation_id=conversation_id,
                    target_judge=factual.FACTUAL_JUDGE,
                    injection_status="INJECTION_NOT_POSSIBLE",
                    rejection_code="SCREENING_REJECTED",
                    rejection_reason=gate.reason,
                    raw_response=json.dumps(
                        input_record.get("screening", {}), ensure_ascii=False
                    ),
                    sampled_params=sampled_params,
                )
                await append_jsonl(
                    rejected_path,
                    rejection.model_dump(mode="json"),
                    rejected_lock,
                )
            else:
                screened_candidate = input_record.get("screened_candidate")
                review_claim = (
                    str(screened_candidate.get("target_claim", ""))
                    if isinstance(screened_candidate, Mapping)
                    else ""
                )
                review = factual.FactualReviewRecord(
                    conversation_id=conversation_id,
                    target_judge=factual.FACTUAL_JUDGE,
                    review_status="REVIEW",
                    stage="screening",
                    reason=gate.reason,
                    target_claim=review_claim,
                    sampled_params=sampled_params,
                )
                await append_jsonl(
                    review_path, review.model_dump(mode="json"), review_lock
                )
            return

    messages = [{"role": "user", "content": assembled_prompt}]
    validation_errors: list[str] = []
    raw_output = ""
    payload: Any = None

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
            if target_judge == factual.FACTUAL_JUDGE:
                if factual.is_injection_not_possible(payload, raw_output):
                    rejection = factual.normalize_rejection(
                        payload, raw_output, conversation_id, sampled_params
                    )
                    await append_jsonl(
                        rejected_path,
                        rejection.model_dump(mode="json"),
                        rejected_lock,
                    )
                    return
                factual_record = factual.validate_factual_golden_record(
                    payload, sampled_params, input_record
                )
                output_record = build_factual_output_record(
                    conversation_id, factual_record, input_record
                )
            else:
                record = validate_record(payload, sampled_params, target_judge)
                dumped = record.model_dump(mode="json")
                output_record = {
                    "conversation_id": conversation_id,
                    "injection_metadata": dumped["injection_metadata"],
                    "injected_conversation": dumped["injected_conversation"],
                    "judge_evaluations": dumped["judge_evaluations"],
                }
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
            await append_jsonl(output_path, output_record, output_lock)
            return

    if target_judge == factual.FACTUAL_JUDGE:
        disposition = factual.classify_post_generation_failure(
            " ".join(validation_errors)
        )
        if disposition.disposition == "REVIEW":
            target_claim, evidence_location, evidence_text = (
                factual.extract_claim_fields(payload)
            )
            review = factual.FactualReviewRecord(
                conversation_id=conversation_id,
                target_judge=factual.FACTUAL_JUDGE,
                review_status="REVIEW",
                stage="post_generation",
                reason=disposition.reason,
                target_claim=target_claim,
                evidence_location=evidence_location,
                evidence_text=evidence_text,
                sampled_params=sampled_params,
            )
            await append_jsonl(
                review_path, review.model_dump(mode="json"), review_lock
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
        default=str(
            PROJECT_ROOT
            / "data"
            / "generated"
            / "prepared_prompts_screened.jsonl"
        ),
        help="Screened prompt JSONL produced by screen_prompts.py.",
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "generated" / "golden_dataset_injected.jsonl"),
    )
    parser.add_argument(
        "--failed",
        default=str(PROJECT_ROOT / "data" / "generated" / "failed_generations.jsonl"),
    )
    parser.add_argument(
        "--rejected",
        default=str(PROJECT_ROOT / "data" / "generated" / "rejected_injections.jsonl"),
        help="Separate file for structured INJECTION_NOT_POSSIBLE rejections (factual_hallucination).",
    )
    parser.add_argument(
        "--review",
        default=str(PROJECT_ROOT / "data" / "generated" / "review_samples.jsonl"),
        help=(
            "Separate file for ambiguous screening/post-generation REVIEW "
            "samples (factual_hallucination)."
        ),
    )
    parser.add_argument("--platform", default=PLATFORM)
    parser.add_argument("--model", default=MODEL)
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
    model = args.model
    validate_platform(platform)
    api_key = resolve_api_key(platform)

    input_path = Path(args.input)
    output_path = Path(args.output)
    failed_path = Path(args.failed)
    rejected_path = Path(args.rejected)
    review_path = Path(args.review)
    input_records = load_pending_records(
        input_path, output_path, rejected_path
    )
    if not input_records:
        print("No pending records.")
        return

    client: Any = create_async_client(platform, api_key, args.timeout)

    semaphore = asyncio.Semaphore(args.concurrency)
    output_lock = asyncio.Lock()
    failed_lock = asyncio.Lock()
    rejected_lock = asyncio.Lock()
    review_lock = asyncio.Lock()
    tasks = [
        process_record(
            record,
            client,
            platform,
            model,
            semaphore,
            output_path,
            failed_path,
            rejected_path,
            output_lock,
            failed_lock,
            rejected_lock,
            args.timeout,
            review_path=review_path,
            review_lock=review_lock,
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