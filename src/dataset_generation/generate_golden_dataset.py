#!/usr/bin/env python3
"""Generate and validate injected golden-dataset records concurrently."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
from pathlib import Path
from typing import Any, Literal

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
DEFAULT_CONCURRENCY = 5
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
    supported_platforms = set(PLATFORM_API_KEYS)
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


class InjectionMetadata(BaseModel):
    """Metadata describing the error injection operation and parameters."""

    model_config = ConfigDict(extra="forbid", strict=True)

    target_judge: str
    already_defective: bool
    scenario_modified: bool
    modified_scenario_fields: dict[str, Any]
    injected_turn_indices: list[int]
    total_injected_errors: int = Field(ge=0)
    target_dialogue_phase: Literal["early", "middle", "late"]
    scope: Literal["pre_existing", "single_turn", "multi_turn_injection"]


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


class GoldenRecordOutput(BaseModel):
    """Top-level schema for the generated golden dataset record."""

    model_config = ConfigDict(extra="forbid", strict=True)

    injection_metadata: InjectionMetadata
    injected_conversation: list[Turn] = Field(min_length=1)
    judge_evaluations: dict[str, JudgeEvaluationDetail]

    @model_validator(mode="after")
    def validate_structure(self) -> "GoldenRecordOutput":
        """Validate cross-field business rules and consistency."""
        metadata = self.injection_metadata
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
        return self


def validate_record(
    payload: Any,
    sampled_params: dict[str, Any],
    expected_target_judge: str | None = None,
) -> GoldenRecordOutput:
    """Validate schema and sampled-parameter business rules."""
    record = GoldenRecordOutput.model_validate(payload)
    metadata = record.injection_metadata

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
        if metadata.total_injected_errors != target_error_count:
            raise ValueError(
                "total_injected_errors "
                f"{metadata.total_injected_errors} does not equal sampled "
                f"target_error_count {target_error_count}"
            )
        target_phase = sampled_params.get("target_dialogue_phase")
        if metadata.target_dialogue_phase != target_phase:
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
            record = validate_record(payload, sampled_params, target_judge)
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
            await append_jsonl(
                output_path,
                {
                    "conversation_id": conversation_id,
                    **record.model_dump(mode="json"),
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
    input_records = [
        record
        for record in load_jsonl(input_path)
        if str(record.get("conversation_id")) not in completed_ids
    ]
    if not input_records:
        print("No pending records.")
        return

    if platform == "claude":
        client: Any = AsyncAnthropic(
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