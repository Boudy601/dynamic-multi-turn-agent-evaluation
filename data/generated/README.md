# Generated Data

This directory stores temporary and experimental artifacts produced by the dataset-generation pipeline.

## Purpose

The directory contains intermediate and final outputs generated during the `Factual Hallucination (L5)` extension development and testing.

These artifacts support pipeline validation, debugging, quality assessment, and reproducibility.

## Live Pilot Artifacts

The live pilot files document the end-to-end preparation, screening, and generation workflow.

| Artifact | Description |
|---|---|
| `live_pilot_prepared.jsonl` | Five prepared candidate prompts |
| `live_pilot_screened.jsonl` | Candidates accepted by the screening stage |
| `live_pilot_screening_audit.jsonl` | Screening decisions and audit information |
| `live_pilot_screening_rejected.jsonl` | Candidates rejected during screening |
| `live_pilot_screening_review.jsonl` | Candidates requiring manual review |
| `live_pilot_generation_input.jsonl` | Two manually approved candidates (10515 and 3064) |
| `live_pilot_golden.jsonl` | Generated samples from the live pilot |
| `live_pilot_failed.jsonl` | Generation failures |
| `live_pilot_rejected.jsonl` | Samples rejected during generation validation |
| `live_pilot_review.jsonl` | Samples requiring further review |

## Data Handling

- Generated artifacts are excluded from version control through `.gitignore`.
- Original source datasets should remain unchanged.
- Customer-support conversations may contain sensitive or identifying information and should not be published publicly without appropriate authorization and anonymization.
- Pilot outputs must be manually reviewed before being treated as trusted Golden Evaluation data.
- A successful generation run does not, by itself, establish sample quality.

## Reproducibility

The artifacts can be regenerated using the dataset-generation scripts under:

`src/dataset_generation/`

Refer to `FACTUAL_HALLUCINATION_README.md` for the extension workflow and implementation details.

## Status

The current artifacts represent an experimental live pilot, not a production-ready or fully validated Golden Dataset.

Cross-judge evaluation and final sample-quality verification remain separate validation steps.