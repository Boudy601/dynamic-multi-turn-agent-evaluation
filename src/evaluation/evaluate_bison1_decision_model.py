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
DEFAULT_REVIEWERS = ("amr", "omar", "youssef")
DEFAULT_PAIR_COUNT = -1
DEFAULT_SEED = 42
DEFAULT_CLEAN_CONFIDENCE_THRESHOLD = 0.85
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
    "small talk, empathy, paraphrasing, or harmless step-ordering flexibility (e.g., "
    "reassuring the customer before asking for their name/Account ID is fine, as long as "
    "required gates like identity verification are not skipped before taking an action). "
    "(2) Answer YES only if there is at least one meaningful SOP violation in any of "
    "these categories: "
    "(a) Tool & parameter errors: skipping a required action, calling the wrong tool, "
    "entering wrong, blank, or mismatched parameters compared to the SOP or User Scenario, "
    "or claiming an action was taken without actually calling the tool; "
    "(b) Workflow & policy breaches: taking the wrong resolution path, violating explicit "
    "business or eligibility rules (e.g., granting a refund, credit, or exception when "
    "ineligible), or ignoring mandatory SOP conditions and prerequisites; "
    "(c) Missing required deliverables: failing to gather required information or failing "
    "to share required codes, links, policies, or next steps with the customer; "
    "(d) Overpromising & unsupported claims: fabricating details or guaranteeing specific "
    "outcomes, amounts, matches, or timelines not supported by the SOP, User Scenario, or "
    "tool outputs (e.g., promising an exact discount amount, price match, or delivery/refund "
    "timeframe when the SOP only authorizes a generic action or does not specify one)."
)


@dataclass(frozen=True)
class EvaluationCase:
    reviewer: str
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
    reviewer: str
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


def normalize_reviewers(raw_reviewers: Sequence[str]) -> list[str]:
    """Flatten space- or comma-separated reviewer names while preserving order."""
    reviewers: list[str] = []
    seen: set[str] = set()
    for item in raw_reviewers:
        for part in item.split(","):
            name = part.strip()
            if name and name not in seen:
                seen.add(name)
                reviewers.append(name)
    if not reviewers:
        raise ValueError("At least one reviewer must be specified")
    return reviewers


def load_paired_cases(
    reviewer_dirs: Path | Sequence[Path],
    pair_count: int,
    seed: int,
) -> list[EvaluationCase]:
    """Sample paired original (incorrect) and edited (corrected) chats for evaluation."""
    dirs = [reviewer_dirs] if isinstance(reviewer_dirs, Path) else list(reviewer_dirs)
    if not dirs:
        raise ValueError("At least one reviewer directory is required")

    available_pairs: list[tuple[str, Path, Path]] = []
    for reviewer_dir in dirs:
        originals_dir = reviewer_dir / ".originals"
        if not originals_dir.is_dir():
            raise FileNotFoundError(f"Originals directory not found: {originals_dir}")
        for orig_path in sorted(originals_dir.glob("*.json")):
            edited_path = reviewer_dir / orig_path.name
            if edited_path.is_file():
                available_pairs.append((reviewer_dir.name, orig_path, edited_path))

    if not available_pairs:
        dir_list = ", ".join(str(d) for d in dirs)
        raise ValueError(f"No paired chats exist across [{dir_list}]")
    if pair_count == -1:
        sampled_pairs = available_pairs
    elif pair_count <= 0:
        raise ValueError("pair_count must be positive or -1 for all paired chats")
    elif pair_count > len(available_pairs):
        dir_list = ", ".join(str(d) for d in dirs)
        raise ValueError(
            f"Requested {pair_count} pairs, but only {len(available_pairs)} "
            f"paired chats exist across [{dir_list}]"
        )
    else:
        sampled_pairs = random.Random(seed).sample(available_pairs, pair_count)
    resolver, canonical_names, _ = build_sop_resolver(GUIDELINES_FILE, DEFAULT_ALIASES)
    action_schema = load_action_schema(ACTION_SCHEMA_FILE)

    cases: list[EvaluationCase] = []
    for reviewer, orig_path, edited_path in sampled_pairs:
        variants = (
            ("original_error", True, orig_path),
            ("edited_clean", False, edited_path),
        )
        for variant, expected_violation, path in variants:
            record = load_json(path)
            chat_id, sop_id, sop_name, state = build_case_state(
                record, resolver, canonical_names, action_schema
            )
            cases.append(
                EvaluationCase(
                    reviewer=reviewer,
                    pair_filename=orig_path.name,
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
    clean_confidence_threshold: float,
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
            f"System One API HTTP {exc.code} for {case.reviewer}/{case.pair_filename} "
            f"({case.variant}): {detail}"
        ) from exc
    latency = time.perf_counter() - started
    prob = extract_noul_probability(payload, "has_sop_violation")
    # Predict clean (NO) only when confidence in NO (1 - p_violation) meets the threshold.
    clean_confidence = 1.0 - prob
    predicted = clean_confidence < clean_confidence_threshold
    return CaseResult(
        reviewer=case.reviewer,
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
    parser.add_argument(
        "--reviewer",
        "--reviewers",
        dest="reviewers",
        nargs="+",
        default=list(DEFAULT_REVIEWERS),
        help="One or more reviewer names (space- or comma-separated).",
    )
    parser.add_argument(
        "--count",
        type=int,
        default=DEFAULT_PAIR_COUNT,
        help="Number of paired chats to sample, or -1 to evaluate all paired chats.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--clean-confidence",
        "--threshold",
        dest="clean_confidence_threshold",
        type=float,
        default=DEFAULT_CLEAN_CONFIDENCE_THRESHOLD,
        help=(
            "Minimum confidence (1 - p_violation) required to predict NO (clean). "
            "Default: 0.7 (p <= 0.30 -> NO, else YES)."
        ),
    )
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
    if not 0.0 <= args.clean_confidence_threshold <= 1.0:
        print(
            "Error: --clean-confidence/--threshold must be between 0.0 and 1.0.",
            file=sys.stderr,
        )
        return 2
    reviewers = normalize_reviewers(args.reviewers)
    reviewer_dirs = [PROJECT_ROOT / "assignments" / name for name in reviewers]
    cases = load_paired_cases(reviewer_dirs, args.count, args.seed)
    pair_total = len(cases) // 2
    print(
        f"Loaded {len(cases)} test cases ({pair_total} original error + "
        f"{pair_total} edited clean) from reviewers {', '.join(reviewers)} "
        f"(seed={args.seed}, clean_confidence>={args.clean_confidence_threshold:.2f})."
    )

    results: list[CaseResult] = []
    for idx, case in enumerate(cases, start=1):
        result = evaluate_case(
            api_key=api_key,
            model=args.model,
            case=case,
            clean_confidence_threshold=args.clean_confidence_threshold,
            timeout_seconds=args.timeout,
        )
        results.append(result)
        mark = "PASS" if result.correct else "FAIL"
        exp_label = "YES" if result.expected_violation else "NO "
        pred_label = "YES" if result.predicted_violation else "NO "
        clean_conf = 1.0 - result.probability
        print(
            f"[{idx:02d}/{len(cases):02d}] {mark} | {case.reviewer:<8} | "
            f"{case.pair_filename:<34} | {case.variant:<14} | "
            f"exp={exp_label} pred={pred_label} "
            f"(p={result.probability:.4f}, clean={clean_conf:.1%}, {result.latency_seconds:.2f}s)"
        )

    metrics = compute_metrics(results)
    metrics_by_reviewer = {
        name: compute_metrics([r for r in results if r.reviewer == name])
        for name in reviewers
    }
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
    if len(reviewers) > 1:
        print("By reviewer:")
        for name in reviewers:
            rm = metrics_by_reviewer[name]
            print(
                f"  {name:<10} {rm['correct']}/{rm['total_cases']} ({rm['accuracy']:.1%}) | "
                f"recall={rm['recall_original_errors']:.1%} spec={rm['specificity_edited_clean']:.1%}"
            )

    output_path = args.output or (
        DEFAULT_LOG_DIR
        / f"decision_model_eval_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "config": {
            "model": args.model,
            "reviewers": reviewers,
            "pair_count": pair_total,
            "requested_pair_count": args.count,
            "seed": args.seed,
            "clean_confidence_threshold": args.clean_confidence_threshold,
            "violation_probability_cutoff": round(1.0 - args.clean_confidence_threshold, 6),
            "instructions": DETECTION_INSTRUCTIONS,
        },
        "metrics": metrics,
        "metrics_by_reviewer": metrics_by_reviewer,
        "results": [asdict(r) for r in results],
    }
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Saved detailed report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
