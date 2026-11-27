#!/usr/bin/env python3
"""Evaluate a yes/no decision model on paired original (error) vs. edited (clean) chats."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def load_dotenv_file(path: Path) -> None:
    """Populate os.environ from a .env file without overwriting existing keys."""
    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src" / "dataset_generation"))

from assign import chat_identifier, chat_sop_id  # noqa: E402
from prepare_prompts import load_json  # noqa: E402
from server import (  # noqa: E402
    ACTION_SCHEMA_FILE,
    DEFAULT_ALIASES,
    GUIDELINES_FILE,
    build_sop_resolver,
    expected_action_parameters,
    load_action_schema,
    message_list,
    resolve_chat_guideline,
)


SYSTEMONE_URL = "https://inference.baseten.co/v1/systemone"
DEFAULT_MODEL = "decision-models/baseten-bison-latest"
DEFAULT_REVIEWER = "bahaa"
DEFAULT_PAIR_COUNT = 10
DEFAULT_SEED = 42
DEFAULT_THRESHOLD = 0.5
DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_LOG_DIR = PROJECT_ROOT / "logs"

DETECTION_INSTRUCTIONS = (
    "Check this chat against the attached SOP and User Scenario to determine whether "
    "the agent committed any meaningful SOP violation. "
    "Treat the SOP as the ground truth—note that the 'Expected parameters:' lines in "
    "the chat logs are just system hints showing possible parameters, not strict SOP "
    "requirements. "
    "Important evaluation rules: "
    "(1) Use practical, human common sense. Do NOT nitpick natural conversational flow, "
    "small talk, empathy, or minor step-ordering flexibility (e.g., chatting briefly or "
    "reassuring the customer before asking for their name/Account ID is completely fine). "
    "(2) Answer YES only if there is at least one meaningful SOP violation, such as: "
    "skipping required tool calls/actions, entering wrong or blank parameters, following "
    "the wrong workflow/resolution, failing to provide required information/links/codes, "
    "or breaking explicit business rules (like issuing a refund when they shouldn't)."
)


@dataclass(frozen=True)
class EvaluationCase:
    pair_filename: str
    chat_id: str
    sop_id: str
    sop_name: str
    variant: str  # "original_error" or "edited_clean"
    expected_violation: bool
    source_path: Path
    state: str


@dataclass(frozen=True)
class CaseResult:
    pair_filename: str
    chat_id: str
    sop_id: str
    variant: str
    expected_violation: bool
    probability: float
    predicted_violation: bool
    correct: bool
    latency_seconds: float
    raw_response: dict[str, Any]


def format_chat_turns(messages: Sequence[Mapping[str, Any]]) -> str:
    """Format chat turns identically to the review UI / Gemini evaluation prompt."""
    formatted_turns: list[str] = []
    for index, message in enumerate(messages, start=1):
        speaker = str(message.get("speaker", "unknown"))
        text = str(message.get("text", ""))
        header = f"{index}. [{speaker.upper()}] {text}"
        if speaker.casefold() != "action":
            formatted_turns.append(header)
            continue

        raw_params = message.get("parameters")
        params = (
            ", ".join(str(p) for p in raw_params)
            if isinstance(raw_params, list) and raw_params
            else "(none)"
        )
        raw_expected = message.get("expected_parameters")
        if not isinstance(raw_expected, list):
            expected = "Not listed for this subflow"
        elif raw_expected:
            expected = ", ".join(str(p) for p in raw_expected)
        else:
            expected = "(none)"
        fn_name = str(message.get("function_name") or "unknown")
        formatted_turns.append(
            f"{header}\n"
            f"   Function: {fn_name}\n"
            f"   Parameters: {params}\n"
            f"   Expected parameters: {expected}"
        )
    return "\n\n".join(formatted_turns)


def build_case_state(
    record: Mapping[str, Any],
    resolver: Any,
    canonical_names: Mapping[str, str],
    action_schema: Mapping[str, Mapping[str, Mapping[str, list[str]]]],
) -> tuple[str, str, str, str]:
    """Build the System One `state` string containing SOP, User Scenario, and Chat."""
    messages = message_list(record)
    sop_guideline = resolve_chat_guideline(record, 0, resolver)
    sop_id = chat_sop_id(record, 0, resolver)
    sop_name = canonical_names[sop_id]
    chat_id = chat_identifier(record, 0)
    guideline_aliases = resolver[1]

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
            action_schema,
            guideline_aliases,
            sop_guideline,
        )

    sop_json = json.dumps(sop_guideline or {}, ensure_ascii=False, indent=2)
    scenario_json = json.dumps(record.get("scenario") or {}, ensure_ascii=False, indent=2)
    chat_lines = format_chat_turns(messages)

    state = "\n".join([
        f"SOP ({sop_name}):",
        sop_json,
        "",
        "---",
        "User Scenario:",
        scenario_json,
        "",
        "---",
        "Chat:",
        chat_lines,
    ])
    return chat_id, sop_id, sop_name, state


def load_paired_cases(
    reviewer_dir: Path,
    pair_count: int,
    seed: int,
) -> list[EvaluationCase]:
    """Sample paired original (incorrect) and edited (corrected) chats for evaluation."""
    originals_dir = reviewer_dir / ".originals"
    if not originals_dir.is_dir():
        raise FileNotFoundError(f"Originals directory not found: {originals_dir}")

    available_filenames = sorted(
        path.name
        for path in originals_dir.glob("*.json")
        if (reviewer_dir / path.name).is_file()
    )
    if pair_count <= 0:
        raise ValueError("pair_count must be positive")
    if pair_count > len(available_filenames):
        raise ValueError(
            f"Requested {pair_count} pairs, but only {len(available_filenames)} "
            f"paired chats exist in {originals_dir}"
        )

    sampled_filenames = random.Random(seed).sample(available_filenames, pair_count)
    resolver, canonical_names, _ = build_sop_resolver(GUIDELINES_FILE, DEFAULT_ALIASES)
    action_schema = load_action_schema(ACTION_SCHEMA_FILE)

    cases: list[EvaluationCase] = []
    for filename in sampled_filenames:
        variants = (
            ("original_error", True, originals_dir / filename),
            ("edited_clean", False, reviewer_dir / filename),
        )
        for variant, expected_violation, path in variants:
            record = load_json(path)
            chat_id, sop_id, sop_name, state = build_case_state(
                record, resolver, canonical_names, action_schema
            )
            cases.append(
                EvaluationCase(
                    pair_filename=filename,
                    chat_id=chat_id,
                    sop_id=sop_id,
                    sop_name=sop_name,
                    variant=variant,
                    expected_violation=expected_violation,
                    source_path=path,
                    state=state,
                )
            )
    return cases


def extract_noul_probability(payload: Mapping[str, Any], question_key: str) -> float:
    """Extract the `noul` probability from a Baseten System One response."""
    containers: list[Any] = [payload]
    if isinstance(payload.get("system_one"), Mapping):
        containers.insert(0, payload["system_one"])

    for container in containers:
        answers = container.get("answers")
        if not isinstance(answers, Mapping):
            continue
        entry = answers.get(question_key)
        if isinstance(entry, (int, float)):
            return float(entry)
        if isinstance(entry, Mapping):
            for key in ("noul", "probability", "prob", "score", "value"):
                value = entry.get(key)
                if isinstance(value, (int, float)):
                    return float(value)
            answer = entry.get("answer")
            if isinstance(answer, bool):
                return 1.0 if answer else 0.0
    raise KeyError(
        f"Could not find noul probability for {question_key!r} in response: {payload}"
    )


def evaluate_case(
    api_key: str,
    model: str,
    case: EvaluationCase,
    threshold: float,
    timeout_seconds: float,
) -> CaseResult:
    """Send a single chat evaluation request to the Baseten System One endpoint."""
    body = json.dumps(
        {
            "model": model,
            "state": case.state,
            "questions": {
                "has_sop_violation": {
                    "type": "noul",
                    "instructions": DETECTION_INSTRUCTIONS,
                }
            },
        },
        ensure_ascii=False,
    ).encode("utf-8")
    req = urllib.request.Request(
        SYSTEMONE_URL,
        data=body,
        headers={
            "Authorization": f"Api-Key {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"System One API HTTP {exc.code} for {case.pair_filename} "
            f"({case.variant}): {detail}"
        ) from exc
    latency = time.perf_counter() - started
    prob = extract_noul_probability(payload, "has_sop_violation")
    predicted = prob >= threshold
    return CaseResult(
        pair_filename=case.pair_filename,
        chat_id=case.chat_id,
        sop_id=case.sop_id,
        variant=case.variant,
        expected_violation=case.expected_violation,
        probability=prob,
        predicted_violation=predicted,
        correct=(predicted == case.expected_violation),
        latency_seconds=latency,
        raw_response=payload,
    )


def compute_metrics(results: Sequence[CaseResult]) -> dict[str, Any]:
    tp = sum(1 for r in results if r.expected_violation and r.predicted_violation)
    fn = sum(1 for r in results if r.expected_violation and not r.predicted_violation)
    fp = sum(1 for r in results if not r.expected_violation and r.predicted_violation)
    tn = sum(1 for r in results if not r.expected_violation and not r.predicted_violation)
    total = len(results)
    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    avg_latency = (
        sum(r.latency_seconds for r in results) / total if total else 0.0
    )
    return {
        "total_cases": total,
        "correct": tp + tn,
        "accuracy": accuracy,
        "tp_error_caught": tp,
        "fn_error_missed": fn,
        "fp_clean_flagged": fp,
        "tn_clean_passed": tn,
        "precision": precision,
        "recall_original_errors": recall,
        "specificity_edited_clean": specificity,
        "f1": f1,
        "avg_latency_seconds": avg_latency,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reviewer", default=DEFAULT_REVIEWER)
    parser.add_argument("--count", type=int, default=DEFAULT_PAIR_COUNT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT_SECONDS)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    load_dotenv_file(PROJECT_ROOT / ".env")
    api_key = os.environ.get("BASETEN_API_KEY", "").strip()
    if not api_key:
        print("Error: BASETEN_API_KEY is not set in environment or .env.", file=sys.stderr)
        return 2

    args = parse_args()
    reviewer_dir = PROJECT_ROOT / "assignments" / args.reviewer
    cases = load_paired_cases(reviewer_dir, args.count, args.seed)
    print(
        f"Loaded {len(cases)} test cases ({args.count} original error + "
        f"{args.count} edited clean) from {reviewer_dir} (seed={args.seed})."
    )

    results: list[CaseResult] = []
    for idx, case in enumerate(cases, start=1):
        result = evaluate_case(
            api_key=api_key,
            model=args.model,
            case=case,
            threshold=args.threshold,
            timeout_seconds=args.timeout,
        )
        results.append(result)
        mark = "PASS" if result.correct else "FAIL"
        exp_label = "YES" if result.expected_violation else "NO "
        pred_label = "YES" if result.predicted_violation else "NO "
        print(
            f"[{idx:02d}/{len(cases):02d}] {mark} | {case.pair_filename:<34} | "
            f"{case.variant:<14} | exp={exp_label} pred={pred_label} "
            f"(p={result.probability:.4f}, {result.latency_seconds:.2f}s)"
        )

    metrics = compute_metrics(results)
    print("\n--- Summary ---")
    print(
        f"Accuracy: {metrics['correct']}/{metrics['total_cases']} "
        f"({metrics['accuracy']:.1%})"
    )
    print(
        f"Originals (error=YES) caught (Recall): "
        f"{metrics['tp_error_caught']}/{metrics['tp_error_caught'] + metrics['fn_error_missed']} "
        f"({metrics['recall_original_errors']:.1%})"
    )
    print(
        f"Edited (error=NO) passed (Specificity): "
        f"{metrics['tn_clean_passed']}/{metrics['tn_clean_passed'] + metrics['fp_clean_flagged']} "
        f"({metrics['specificity_edited_clean']:.1%})"
    )
    print(
        f"Precision: {metrics['precision']:.1%} | F1: {metrics['f1']:.1%} | "
        f"Avg latency: {metrics['avg_latency_seconds']:.2f}s"
    )

    output_path = args.output or (
        DEFAULT_LOG_DIR
        / f"decision_model_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "model": args.model,
            "reviewer": args.reviewer,
            "pair_count": args.count,
            "seed": args.seed,
            "threshold": args.threshold,
            "instructions": DETECTION_INSTRUCTIONS,
        },
        "metrics": metrics,
        "results": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved detailed report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
