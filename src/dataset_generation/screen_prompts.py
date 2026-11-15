#!/usr/bin/env python3
"""Screen prepared factual-hallucination candidates before generation.

This stage combines the deterministic Python candidate filter with one LLM
screening call per candidate. Only ``ELIGIBLE`` records are written to the
screened prompt file consumed by ``generate_golden_dataset.py``. ``REJECT`` and
``REVIEW`` outcomes are written to separate audit files and can never reach
generation.

The provider configuration (platform, model, API-key variable, base URL, retry
policy) is reused from ``generate_golden_dataset.py``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # Support both direct script execution and package imports.
    from . import factual_hallucination as factual
    from .generate_golden_dataset import (
        DEFAULT_TIMEOUT_SECONDS,
        MODEL,
        PLATFORM,
        load_jsonl,
        request_with_backoff,
        resolve_api_key,
        validate_platform,
        create_async_client,
    )
except ImportError:  # pragma: no cover - exercised when run as a script
    import factual_hallucination as factual
    from generate_golden_dataset import (
        DEFAULT_TIMEOUT_SECONDS,
        MODEL,
        PLATFORM,
        load_jsonl,
        request_with_backoff,
        resolve_api_key,
        validate_platform,
        create_async_client,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ScreeningOutcome:
    """Result of screening one prepared record."""

    decision: str
    record: dict[str, Any] | None
    audit: dict[str, Any]


def _already_screened(record: Mapping[str, Any]) -> bool:
    screening = record.get("screening")
    return (
        isinstance(screening, Mapping)
        and str(screening.get("decision", "")).upper() == "ELIGIBLE"
        and isinstance(record.get("screened_candidate"), Mapping)
    )


def build_screening_request(
    record: Mapping[str, Any],
) -> list[dict[str, str]] | None:
    """Return the provider messages for a record, or None when no call is needed."""

    if _already_screened(record):
        return None
    selection = factual.selection_from_prompt_record(record)
    if not selection.eligible:
        return None
    candidate = selection.eligible[0]
    turns = record.get("source_conversation")
    scenario = record.get("source_scenario")
    return factual.build_screening_messages(
        candidate,
        [dict(turn) for turn in turns if isinstance(turn, Mapping)]
        if isinstance(turns, Sequence) and not isinstance(turns, (str, bytes))
        else [],
        dict(scenario) if isinstance(scenario, Mapping) else {},
    )


def _eligible_record(
    record: Mapping[str, Any],
    result: factual.FactualScreeningResult,
    selection: factual.FactualCandidateSelection,
) -> dict[str, Any]:
    candidate = next(
        (item for item in selection.eligible if item.candidate_id == result.candidate_id),
        selection.eligible[0],
    )
    prepared = dict(record)
    prepared["screening"] = result.to_dict()
    prepared["screened_candidate"] = {
        "candidate_id": candidate.candidate_id,
        "turn_id": candidate.turn_id,
        "attribute_kind": candidate.attribute_kind,
        "error_category": candidate.error_category,
        "target_claim": candidate.original_agent_claim,
        "evidence_source_type": candidate.evidence["source_type"],
        "evidence_location": candidate.evidence["location"],
        "evidence_text": candidate.evidence["text"],
    }
    prepared["assembled_prompt"] = (
        str(record.get("assembled_prompt", ""))
        + factual.screened_candidate_constraint(result)
    )
    sampled = dict(record.get("sampled_params") or {})
    sampled.update(
        {
            "screening_decision": "ELIGIBLE",
            "screening_requires_llm": False,
            "screening_reason": result.reason,
            "candidate_id": candidate.candidate_id,
        }
    )
    prepared["sampled_params"] = sampled
    return prepared


def screen_prepared_record(
    record: Mapping[str, Any],
    raw_response: str | None = None,
    *,
    llm_invoked: bool | None = None,
) -> ScreeningOutcome:
    """Apply Python + LLM screening to one prepared record.

    Pure and deterministic for tests: ``raw_response`` is the already-obtained
    model text. When it is omitted, the record is held as ``REVIEW`` because an
    unconfirmed candidate must not be generated.
    """

    conversation_id = record.get("conversation_id")
    if conversation_id is None:
        raise ValueError("Prepared record is missing conversation_id")
    sampled_params = record.get("sampled_params")
    sampled = dict(sampled_params) if isinstance(sampled_params, Mapping) else {}

    if _already_screened(record):
        screening = record.get("screening")
        audit = factual.build_screening_audit_record(
            factual.FactualScreeningResult(
                decision="ELIGIBLE",
                reason=str(
                    (screening or {}).get(
                        "reason", "Record was already screened ELIGIBLE."
                    )
                ),
                target_claim=str(
                    (record.get("screened_candidate") or {}).get("target_claim", "")
                ),
                evidence_location=str(
                    (record.get("screened_candidate") or {}).get(
                        "evidence_location", ""
                    )
                ),
                evidence_text=str(
                    (record.get("screened_candidate") or {}).get("evidence_text", "")
                ),
                candidate_id=(record.get("screened_candidate") or {}).get(
                    "candidate_id"
                ),
            ),
            conversation_id,
            sampled,
        ).model_dump(mode="json")
        return ScreeningOutcome(
            decision="ELIGIBLE", record=dict(record), audit=audit
        )

    selection = factual.selection_from_prompt_record(record)
    stage = factual.python_screening_stage(selection)

    if not stage.requires_llm:
        result = stage
    elif raw_response is None:
        result = factual.screen_factual_candidates(
            selection,
            None,
            llm_invoked=bool(llm_invoked),
        )
    else:
        result = factual.apply_llm_screening(stage, raw_response, selection)

    audit = factual.build_screening_audit_record(
        result, conversation_id, sampled
    ).model_dump(mode="json")

    if result.decision == "ELIGIBLE":
        return ScreeningOutcome(
            decision="ELIGIBLE",
            record=_eligible_record(record, result, selection),
            audit=audit,
        )
    return ScreeningOutcome(decision=result.decision, record=None, audit=audit)


def screening_error_outcome(
    record: Mapping[str, Any], error: Exception
) -> ScreeningOutcome:
    """Fail closed: a provider error becomes REVIEW, never ELIGIBLE."""

    conversation_id = record.get("conversation_id")
    if conversation_id is None:
        raise ValueError("Prepared record is missing conversation_id")
    sampled = record.get("sampled_params")
    result = factual.FactualScreeningResult(
        decision="REVIEW",
        reason=(
            "The screening model call failed, so the candidate is unconfirmed "
            f"and must not be generated: {type(error).__name__}: {error}"
        ),
    )
    audit = factual.build_screening_audit_record(
        result,
        conversation_id,
        dict(sampled) if isinstance(sampled, Mapping) else {},
    ).model_dump(mode="json")
    return ScreeningOutcome(decision="REVIEW", record=None, audit=audit)


async def screen_records(
    records: Sequence[Mapping[str, Any]],
    *,
    client: Any,
    platform: str,
    model: str,
    semaphore: asyncio.Semaphore,
    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
) -> list[ScreeningOutcome]:
    """Screen records concurrently, reusing the generation provider helpers."""

    async def screen_one(record: Mapping[str, Any]) -> ScreeningOutcome:
        messages = build_screening_request(record)
        if messages is None:
            return screen_prepared_record(record, llm_invoked=False)
        try:
            raw_response = await request_with_backoff(
                client,
                semaphore,
                platform,
                model,
                messages,
                timeout_seconds,
            )
        except Exception as exc:  # fail closed, keep the batch running
            return screening_error_outcome(record, exc)
        return screen_prepared_record(record, raw_response, llm_invoked=True)

    return list(
        await asyncio.gather(*(screen_one(record) for record in records))
    )


def write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write records as JSONL, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def screening_counts(outcomes: Sequence[ScreeningOutcome]) -> dict[str, int]:
    """Return decision counts for the run summary."""

    counts: dict[str, int] = {}
    for outcome in outcomes:
        counts[outcome.decision] = counts.get(outcome.decision, 0) + 1
    return counts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "generated" / "prepared_prompts.jsonl"),
        help="Prepared prompt JSONL produced by prepare_prompts.py.",
    )
    parser.add_argument(
        "--output",
        default=str(
            PROJECT_ROOT / "data" / "generated" / "prepared_prompts_screened.jsonl"
        ),
        help="Screened prompts that may be generated.",
    )
    parser.add_argument(
        "--rejected",
        default=str(PROJECT_ROOT / "data" / "generated" / "screening_rejected.jsonl"),
    )
    parser.add_argument(
        "--review",
        default=str(PROJECT_ROOT / "data" / "generated" / "screening_review.jsonl"),
    )
    parser.add_argument(
        "--audit",
        default=str(PROJECT_ROOT / "data" / "generated" / "screening_audit.jsonl"),
        help="Combined audit rows for every screening decision.",
    )
    parser.add_argument("--platform", default=PLATFORM)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--factual-only",
        action="store_true",
        help="Skip records whose target_judge is not factual_hallucination.",
    )
    return parser.parse_args()


async def run(args: argparse.Namespace) -> int:
    if args.concurrency < 1:
        raise ValueError("--concurrency must be at least 1")
    validate_platform(args.platform)

    records = load_jsonl(Path(args.input))
    if args.factual_only:
        records = [
            record
            for record in records
            if record.get("target_judge") == factual.FACTUAL_JUDGE
        ]
    if not records:
        print("No prepared records to screen.")
        return 0

    api_key = resolve_api_key(args.platform)
    client: Any = create_async_client(args.platform, api_key, args.timeout)
    semaphore = asyncio.Semaphore(args.concurrency)
    try:
        outcomes = await screen_records(
            records,
            client=client,
            platform=args.platform,
            model=args.model,
            semaphore=semaphore,
            timeout_seconds=args.timeout,
        )
    finally:
        await client.close()

    eligible = [outcome.record for outcome in outcomes if outcome.record is not None]
    rejected = [
        outcome.audit for outcome in outcomes if outcome.decision == "REJECT"
    ]
    review = [outcome.audit for outcome in outcomes if outcome.decision == "REVIEW"]
    audit = [outcome.audit for outcome in outcomes]

    write_jsonl(Path(args.output), [row for row in eligible if row is not None])
    write_jsonl(Path(args.rejected), rejected)
    write_jsonl(Path(args.review), review)
    write_jsonl(Path(args.audit), audit)

    counts = screening_counts(outcomes)
    print(
        "Screening complete: "
        f"{counts.get('ELIGIBLE', 0)} ELIGIBLE, "
        f"{counts.get('REJECT', 0)} REJECT, "
        f"{counts.get('REVIEW', 0)} REVIEW "
        f"({len(records)} input record(s))."
    )
    print(f"Wrote screened prompts to {args.output}")
    return 0


def main() -> int:
    from dotenv import load_dotenv

    load_dotenv()
    return asyncio.run(run(parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
