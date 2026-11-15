# Factual Hallucination (L5) Pipeline

This document describes the completed `factual_hallucination` extension in this
repository. It covers the pre-generation screening stage, evidence verification,
controlled single-claim injection, post-generation validation, quality reporting,
and the commands for a small live pilot.

The extension is additive. It does not replace the existing dataset-generation
framework, judge definitions, non-factual prompt batches, or source datasets.

## What the pipeline guarantees

For an accepted golden row:

- Exactly one Agent statement is changed.
- The change is one replacement of one factual claim with an incompatible value
  of the same entity and attribute.
- The target claim was selected by the deterministic Python filter and confirmed
  by the LLM screening stage before generation started.
- The injected claim directly contradicts one trusted evidence item.
- The source conversation, scenario, SOP, customer turns, action turns, and
  unrelated Agent turns are preserved.
- The generated row keeps the original core JSONL keys and adds the audit
  context and validation report needed to inspect the sample independently. The
  four original keys remain unchanged in meaning.
- The target judge receives the expected `FAIL` label for the verified
  contradiction.

An accepted row is still an automatically validated sample. It is not proof of
semantic correctness, human review, or an independent judge execution.

## Pipeline

```text
prepare_prompts.py
    -> deterministic candidate selection
    -> Python screening stage (REVIEW + requires_llm=true)
    -> prepared_prompts.jsonl

screen_prompts.py
    -> Python screening stage
    -> one LLM screening call per Python-eligible candidate
    -> ELIGIBLE / REJECT / REVIEW
    -> prepared_prompts_screened.jsonl (ELIGIBLE only)

generate_golden_dataset.py
    -> hard generation gate
    -> one controlled injection
    -> post-generation validation
    -> golden_dataset_injected.jsonl

report_factual_quality.py
    -> counts, rates, failure reasons
    -> factual_quality_report.json / .md
```

### 1. Preparation: `prepare_prompts.py`

For `factual_hallucination`, preparation now performs the existing deterministic
candidate selection and records:

- `factual_selection`: eligible candidates, exclusions, and audit metadata.
- `screening`: the Python-only screening stage. This is normally `REVIEW` with
  `requires_llm: true` when a candidate exists. Preparation never calls an LLM.
- Screening fields in `sampled_params`: `screening_decision`,
  `screening_requires_llm`, `screening_reason`, and `candidate_id`.

A Python-eligible candidate is not generation-eligible at this point. The LLM
screening stage must confirm it first.

### 2. Screening: `screen_prompts.py`

This stage combines the deterministic Python filter with one LLM screening call
per Python-eligible candidate. It reuses the provider configuration and retry
helpers from `generate_golden_dataset.py`.

Decisions:

| Decision | Meaning | Generation |
|---|---|---|
| `ELIGIBLE` | Python filter found a candidate and the LLM confirmed the exact claim and evidence. | Allowed |
| `REJECT` | The candidate is clearly ineligible. | Blocked |
| `REVIEW` | The candidate is uncertain, the response is malformed, or screening did not run/failed. | Blocked |

Fail-closed rules:

- A missing LLM call never produces `ELIGIBLE`.
- A provider error becomes `REVIEW`.
- A malformed or non-JSON response becomes `REVIEW`.
- An LLM `ELIGIBLE` response must echo `target_claim` and `evidence_text` exactly
  from the Python candidate. Location may be omitted, but if present it must
  match exactly.
- A Python exclusion is `REJECT` unless its reason code is inherently ambiguous,
  in which case it is `REVIEW`.

Only `ELIGIBLE` records are written to `prepared_prompts_screened.jsonl`. They
contain a `screened_candidate` block and an appended constraint that tells the
generation model to use only that screened claim.

### 3. Generation gate

`generate_golden_dataset.py` refuses to call the model for a factual record
unless screening is `ELIGIBLE` and `screened_candidate` is present.

| Input state | Routing |
|---|---|
| No screening block | `REVIEW` in `review_samples.jsonl`; no model call |
| `ELIGIBLE` without `screened_candidate` | `REVIEW`; no model call |
| `REJECT` | `rejected_injections.jsonl` with `SCREENING_REJECTED` |
| `REVIEW` or any other non-eligible state | `review_samples.jsonl`; no model call |
| `ELIGIBLE` with `screened_candidate` | Generation proceeds |

The screening gate is unconditional. There is no CLI switch that allows an
unscreened factual record to reach the generation model.

The default `--input` is `prepared_prompts_screened.jsonl`, not the unscreened
prepared file. When resuming, accepted golden rows and structured rejections are
skipped. REVIEW rows are not treated as completed, so a record can be retried
after its screening or input is corrected. A repeated retry may append another
REVIEW row; review files are not deduplicated.

### 4. Controlled injection

The prompt template instructs the model to:

- Use only the screened candidate.
- Preserve `original_agent_claim` exactly.
- Replace it once with `injected_claim`.
- Keep every other turn and the scenario unchanged.
- Copy evidence source, location, and text exactly.
- Return a structured rejection instead of forcing an injection.

The generated top-level row now contains six keys:

```json
{
  "conversation_id": "...",
  "injection_metadata": {},
  "injected_conversation": [],
  "judge_evaluations": {},
  "audit_context": {},
  "validation_report": {}
}
```

`audit_context` preserves the source conversation, source scenario, source SOP,
available evidence, judge definition, and screening/provenance fields when they
exist in the prepared input. `validation_report` preserves the automated checks
and evidence-verification result so accepted rows can be audited independently.

`injection_metadata` gained two optional, backward-compatible provenance fields:
`candidate_id` and `screening_decision`. Older rows without them remain valid.

### 5. Evidence verification

`verify_factual_evidence` checks entity, attribute, and context before accepting
a contradiction.

| Classification | Meaning | Accepted |
|---|---|---|
| `CONTRADICTED` | The injected claim asserts an explicit incompatible value for the same entity and attribute. | Yes |
| `SUPPORTED` | The evidence confirms the claim; no contradiction exists. | No |
| `UNSUPPORTED` | The evidence neither confirms nor denies the claim. | No |
| `NOT_VERIFIABLE` | The evidence is missing, ambiguous, or refers to a different entity, attribute, or context. | No |

Only a direct contradiction is accepted. Missing evidence is never converted
into a fabricated contradiction.

### 6. Post-generation validation

`validate_factual_golden_record` validates the model response before any row is
written to the golden file.

It checks:

- **JSON and schema**: the response must parse and satisfy the Pydantic factual
  output schema.
- **Metadata shape**: `already_defective`, `injection_status`, `scope`, and
  `expected_target_verdict` must be consistent.
- **Evidence reference**: source type, location, and text must match one trusted
  evidence item exactly.
- **Conversation diff**: turn count, order, speakers, and all non-target turns
  must be preserved; only one Agent turn may change.
- **Single intended change**: the edited turn must be the exact one-time
  substitution of `original_agent_claim` with `injected_claim`.
- **Evidence contradiction**: the structured rule check and
  `verify_factual_evidence` must both establish a direct contradiction.
- **Targeted attribute checks**: email and phone substitutions are accepted when
  they are the intended controlled contradiction, including local-part or domain
  email edits, while unrelated identifiers and numbers elsewhere in the edited
  claim remain protected.
- **Replacement value checks**: replacement values must have a valid format for
  the target attribute and must not collide with another trusted evidence value
  of the same kind in the available evidence.
- **Coherence**: the injected statement must not be a question, self-
  contradictory, or incoherent.
- **Procedural side effects**: the replacement must not change a status branch,
  amount/date semantics, or another non-target defect dimension.
- **Scenario preservation**: `scenario_modified` must be false and
  `modified_scenario_fields` must be empty.
- **Screened claim match**: when `screened_candidate` exists, the generated
  original claim and evidence text must match it.

The generator allows up to four model attempts (initial plus three validation
retries). A structured `INJECTION_NOT_POSSIBLE` response is written to
`rejected_injections.jsonl`.

After the retries:

- A failure that indicates genuine ambiguity is written to
  `review_samples.jsonl` with `stage: post_generation`.
- A deterministic structural or evidence failure is written to
  `failed_generations.jsonl`.
- Neither outcome enters the golden dataset.

### 7. Judge integration

The factual row includes the existing target judge label:

```json
"judge_evaluations": {
  "factual_hallucination": {
    "verdict": "FAIL",
    "violations": [],
    "overall_reasoning": "..."
  }
}
```

This is the expected label for the verified contradiction. It is not an
independent judge execution. The current generation pipeline does not execute
the non-target judges, so this extension does not fabricate their results.

## Output artefacts

| File | Produced by | Contents |
|---|---|---|
| `prepared_prompts.jsonl` | `prepare_prompts.py` | Prepared prompts with `factual_selection` and Python screening state |
| `prepared_prompts_screened.jsonl` | `screen_prompts.py` | `ELIGIBLE` prompts with `screened_candidate` |
| `screening_rejected.jsonl` | `screen_prompts.py` | Screening audit rows with `REJECT` |
| `screening_review.jsonl` | `screen_prompts.py` | Screening audit rows with `REVIEW` |
| `screening_audit.jsonl` | `screen_prompts.py` | Every screening decision |
| `golden_dataset_injected.jsonl` | generation | Accepted factual rows |
| `rejected_injections.jsonl` | generation | Structured model rejections and screening rejects |
| `failed_generations.jsonl` | generation | Deterministic post-generation failures |
| `review_samples.jsonl` | generation | Blocked/ambiguous screening and post-generation samples |
| `factual_quality_report.json` | reporting | Machine-readable quality report |
| `factual_quality_report.md` | reporting | Human-readable summary |

## Quality report

`report_factual_quality.py` reads only artefacts that exist on disk. It reports:

- Accepted, rejected, failed, and reviewed counts.
- Screening decisions (`ELIGIBLE`, `REJECT`, `REVIEW`, `NOT_RUN`).
- Review counts by stage.
- Overall acceptance rate, generation acceptance rate, validation success rate,
  and failure rate.
- Failure reason counts.

Missing files produce `n/a`, not zero. The report does not infer judge results or
per-record validation outcomes that are not stored in the JSONL files.

Use `--total-input-records` when the true input denominator is known. Otherwise
the report derives a denominator from the artefacts it can see.

## Small live pilot

Run from the repository root. These commands intentionally use separate pilot
files so existing artefacts are not modified.

```powershell
.\.venv\Scripts\python.exe src\dataset_generation\prepare_prompts.py `
  --batch factual_hallucination `
  --overwrite `
  --seed 42 `
  --output data\generated\pilot_prepared.jsonl
```

This prepares five factual prompts. Every Python-eligible prompt is still
`REVIEW` with `requires_llm: true`.

```powershell
.\.venv\Scripts\python.exe src\dataset_generation\screen_prompts.py `
  --input data\generated\pilot_prepared.jsonl `
  --output data\generated\pilot_prepared_screened.jsonl `
  --rejected data\generated\pilot_screening_rejected.jsonl `
  --review data\generated\pilot_screening_review.jsonl `
  --audit data\generated\pilot_screening_audit.jsonl `
  --factual-only `
  --concurrency 1
```

This is the first live API step. It makes one screening call per Python-eligible
candidate. `--concurrency 1` keeps the pilot small and easy to inspect.

```powershell
.\.venv\Scripts\python.exe src\dataset_generation\generate_golden_dataset.py `
  --input data\generated\pilot_prepared_screened.jsonl `
  --output data\generated\pilot_golden.jsonl `
  --failed data\generated\pilot_failed.jsonl `
  --rejected data\generated\pilot_rejected.jsonl `
  --review data\generated\pilot_review.jsonl `
  --concurrency 1
```

Only screened `ELIGIBLE` records can reach this step.

```powershell
.\.venv\Scripts\python.exe src\dataset_generation\report_factual_quality.py `
  --golden data\generated\pilot_golden.jsonl `
  --failed data\generated\pilot_failed.jsonl `
  --rejected data\generated\pilot_rejected.jsonl `
  --review data\generated\pilot_review.jsonl `
  --screening data\generated\pilot_screening_audit.jsonl `
  --output data\generated\pilot_quality_report.json `
  --markdown data\generated\pilot_quality_report.md
```

The live steps require the provider key already configured in `.env`. Offline
preparation, tests, and reporting do not require API credits.

## Tests

Use `unittest`; `pytest` is not required.

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Current result: 62 tests pass. The tests mock provider calls, so they consume no
API credits. They cover:

- Python + LLM screening decisions and fail-closed behavior.
- Exact evidence echo requirements.
- Evidence entity/attribute/context matching and classification.
- Single-injection validation, scenario preservation, and coherence.
- Email/phone substitution acceptance without weakening unrelated identifier or
  numeric checks.
- Replacement-value format and trusted-evidence collision validation.
- Procedural side effects and non-target isolation.
- Fail-closed screening/resume behavior and screened-default generation input.
- Auditable Golden output shape with `audit_context` and `validation_report`.
- Post-generation `REJECT` versus `REVIEW` routing.
- Generation gate behavior and backward compatibility.
- Quality metrics and report rendering.

## Limitations

- No live screening or generation batch was run while implementing this change.
  Only mocked tests and offline preparation/report smoke checks were executed.
- `REVIEW` samples require human inspection. They are intentionally not
  auto-promoted.
- Evidence verification is deterministic and rule-based. It is not semantic
  proof and only covers the configured factual attributes and evidence types.
- Non-target judges are not executed by this pipeline. Only the target judge
  label is produced.
- The report's denominator is an estimate unless `--total-input-records` is
  supplied.
- A screening provider outage produces `REVIEW` and may require rerunning the
  screening stage.

## Related files

- `src/dataset_generation/factual_hallucination.py`
- `src/dataset_generation/prepare_prompts.py`
- `src/dataset_generation/screen_prompts.py`
- `src/dataset_generation/generate_golden_dataset.py`
- `src/dataset_generation/report_factual_quality.py`
- `prompts/factual_hallucination_injection_prompt_template.txt`
- `tests/test_factual_pipeline.py`
- `tests/test_factual_integration.py`
- `tests/test_factual_improvements.py`
- `FACTUAL_HALLUCINATION_IMPLEMENTATION_REPORT.md`
