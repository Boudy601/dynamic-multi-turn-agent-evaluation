#!/usr/bin/env python3
"""Build a lightweight quality report for factual-hallucination runs.

The report only uses artefacts that actually exist on disk:

- accepted golden rows (``golden_dataset_injected.jsonl``)
- generation failures (``failed_generations.jsonl``)
- structured model rejections (``rejected_injections.jsonl``)
- screening/post-generation REVIEW samples (``review_samples.jsonl``)
- screening audit rows (``screening_audit.jsonl``)

It never invents judge executions or per-record validation results that are
not stored in those rows.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

try:  # Support both direct script execution and package imports.
    from . import factual_hallucination as factual
    from .generate_golden_dataset import load_jsonl
except ImportError:  # pragma: no cover - exercised when run as a script
    import factual_hallucination as factual
    from generate_golden_dataset import load_jsonl


PROJECT_ROOT = Path(__file__).resolve().parents[2]
GENERATED_DIR = PROJECT_ROOT / "data" / "generated"


def _decision_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        decision = str(row.get("decision", "UNKNOWN")).strip().upper()
        counts[decision] = counts.get(decision, 0) + 1
    return counts


def _load_optional(path: Path) -> list[dict[str, Any]] | None:
    """Return rows, or None when the artefact does not exist."""

    if not path.is_file():
        return None
    return load_jsonl(path)


def _count_or_none(rows: Sequence[Mapping[str, Any]] | None) -> int | None:
    return None if rows is None else len(rows)


def build_quality_report(
    *,
    golden_path: Path,
    failed_path: Path,
    rejected_path: Path,
    review_path: Path,
    screening_path: Path,
    total_input_records: int | None = None,
    candidate_filter_exclusions: int | None = None,
) -> dict[str, Any]:
    """Read run artefacts and return a JSON-serialisable report."""

    accepted_file = _load_optional(golden_path)
    failures_file = _load_optional(failed_path)
    rejections_file = _load_optional(rejected_path)
    reviews_file = _load_optional(review_path)
    screening_file = _load_optional(screening_path)

    accepted = accepted_file or []
    failures = failures_file or []
    rejections = rejections_file or []
    reviews = reviews_file or []

    metrics = factual.compute_run_metrics(
        accepted_rows=accepted,
        rejections=rejections,
        failures=failures_file,
        reviews=reviews,
        screening_records=screening_file,
        candidate_filter_exclusions=candidate_filter_exclusions,
        total_input_records=total_input_records,
    )

    accepted_by_status: dict[str, int] = {}
    for row in accepted:
        metadata = row.get("injection_metadata")
        status = (
            metadata.get("injection_status")
            if isinstance(metadata, Mapping)
            else "UNKNOWN"
        )
        accepted_by_status[str(status)] = accepted_by_status.get(str(status), 0) + 1

    review_stages: dict[str, int] = {}
    for row in reviews:
        stage = str(row.get("stage", "unknown"))
        review_stages[stage] = review_stages.get(stage, 0) + 1

    return {
        "inputs": {
            "golden": str(golden_path),
            "failed": str(failed_path),
            "rejected": str(rejected_path),
            "review": str(review_path),
            "screening": str(screening_path),
        },
        "files_present": {
            "golden": accepted_file is not None,
            "failed": failures_file is not None,
            "rejected": rejections_file is not None,
            "review": reviews_file is not None,
            "screening": screening_file is not None,
        },
        "counts": {
            "accepted": _count_or_none(accepted_file),
            "accepted_by_status": accepted_by_status,
            "rejected": _count_or_none(rejections_file),
            "failed": _count_or_none(failures_file),
            "reviewed": _count_or_none(reviews_file),
            "review_by_stage": review_stages,
            "screening": (
                _decision_counts(screening_file)
                if screening_file is not None
                else None
            ),
            "screening_records": _count_or_none(screening_file),
        },
        "metrics": metrics.model_dump(mode="json"),
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def _count(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def render_markdown(report: Mapping[str, Any]) -> str:
    counts = report["counts"]
    metrics = report["metrics"]
    screening = counts["screening"]
    screening_text = (
        "n/a (screening audit file not found)"
        if screening is None
        else (
            f"{screening.get('ELIGIBLE', 0)} ELIGIBLE, "
            f"{screening.get('REJECT', 0)} REJECT, "
            f"{screening.get('REVIEW', 0)} REVIEW"
        )
    )
    lines = [
        "# Factual Hallucination (L5) Quality Report",
        "",
        "## Counts",
        "",
        f"- Accepted golden samples: {_count(counts['accepted'])}",
        f"- Rejected samples: {_count(counts['rejected'])}",
        f"- Generation failures: {_count(counts['failed'])}",
        f"- REVIEW samples: {_count(counts['reviewed'])}",
        f"- Screening: {screening_text}",
        "",
        "## Rates",
        "",
        f"- Overall acceptance rate: {_pct(metrics.get('overall_acceptance_rate'))}",
        f"- Generation acceptance rate: "
        f"{_pct(metrics.get('generation_acceptance_rate'))}",
        f"- Validation success rate: "
        f"{_pct(metrics.get('validation_success_rate'))}",
        f"- Failure rate: {_pct(metrics.get('failure_rate'))}",
        "",
        "## Failure Reasons",
        "",
    ]
    failure_reasons = metrics.get("failure_reasons") or {}
    if failure_reasons:
        lines.extend(
            f"- `{code}`: {count}" for code, count in sorted(failure_reasons.items())
        )
    else:
        lines.append("- None recorded.")
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- Rates are computed only from artefacts present on disk.",
            "- `n/a` means the required data was not available; it is not zero.",
            "- This report is not a judge execution and contains no judge results.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--golden", default=str(GENERATED_DIR / "golden_dataset_injected.jsonl")
    )
    parser.add_argument(
        "--failed", default=str(GENERATED_DIR / "failed_generations.jsonl")
    )
    parser.add_argument(
        "--rejected", default=str(GENERATED_DIR / "rejected_injections.jsonl")
    )
    parser.add_argument(
        "--review", default=str(GENERATED_DIR / "review_samples.jsonl")
    )
    parser.add_argument(
        "--screening", default=str(GENERATED_DIR / "screening_audit.jsonl")
    )
    parser.add_argument("--output", default=str(GENERATED_DIR / "factual_quality_report.json"))
    parser.add_argument(
        "--markdown",
        default=None,
        help="Optional path for a Markdown summary.",
    )
    parser.add_argument(
        "--total-input-records",
        type=int,
        default=None,
        help="Optional true input-record denominator for acceptance rates.",
    )
    parser.add_argument(
        "--candidate-filter-exclusions",
        type=int,
        default=None,
        help="Optional count of conversations excluded before preparation.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_quality_report(
        golden_path=Path(args.golden),
        failed_path=Path(args.failed),
        rejected_path=Path(args.rejected),
        review_path=Path(args.review),
        screening_path=Path(args.screening),
        total_input_records=args.total_input_records,
        candidate_filter_exclusions=args.candidate_filter_exclusions,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if args.markdown:
        markdown_path = Path(args.markdown)
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(render_markdown(report), encoding="utf-8")

    counts = report["counts"]
    metrics = report["metrics"]
    print(
        "Factual quality: "
        f"{_count(counts['accepted'])} accepted, "
        f"{_count(counts['rejected'])} rejected, "
        f"{_count(counts['failed'])} failed, "
        f"{_count(counts['reviewed'])} review; "
        f"validation success {_pct(metrics.get('validation_success_rate'))}."
    )
    print(f"Wrote report to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
