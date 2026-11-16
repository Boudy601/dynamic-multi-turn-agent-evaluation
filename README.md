# Task Closure Injection — V5.4

**Status:** Pilot-ready / initial validation
**Scope:** Task Closure injection methodology, core generation-pipeline integration, and preliminary testing. This is **not** a fully validated golden dataset.

## Overview

This work generates controlled **Task Closure** failures in multi-turn customer-support conversations. A failure (**Y / FAIL**) means the conversation ends with at least one substantive customer request still unresolved **within the agent's responsibility**. A pass (**N / PASS**) means the relevant requests have reached appropriate terminal outcomes.

V5.4 applies **11 decision boundaries**, including independent customer requests, SOP steps versus customer goals, backend execution versus customer-facing confirmation, completed handoffs, adequate answers, and delivery of customer-usable outputs (such as a promo code or return label).

A valid new injection must satisfy:

```text
Original conversation: N (PASS)
            + minimal, causally responsible AGENT edit
Injected conversation: Y (FAIL)
            + all integrity constraints satisfied
```

If the source is already demonstrably defective, it is returned unchanged with `already_defective=true`. If no defensible injection is possible, the case is skipped as `NO_VALID_INJECTION`.

## Pipeline walkthrough

```text
AB-CD conversations + SOP guidelines + judge definitions
                         |
                         v
1. prepare_prompts.py
   Select conversations, exclude prior IDs, load V5.4 prompt
                         |
                         v
   prepared_prompts.jsonl
                         |
                         v
2. generate_golden_dataset.py + GPT-6 Luna
   Generate a minimal Task Closure injection (or classify/skip)
                         |
                         v
3. Schema and structural validator
   Verify declared edits, turn integrity, ACTION immutability
                         |
                         v
   golden_dataset_injected.jsonl
   + skipped_generations.jsonl / failed_generations.jsonl
                         |
                         v
4. Independent semantic review + optional V5.4 blind judge
   Verify genuine N -> Y causality before final dataset approval
```

### Main implementation changes

- **Prompt preparation — `src/dataset_generation/prepare_prompts.py`:** added Task Closure-only selection, configurable sample count and V5.4 prompt path, exclusion of previously used conversation IDs, and reproducible sampling with a seed.
- **Core generator — `src/dataset_generation/generate_golden_dataset.py`:** extended the V5.4 output schema and validator to support declared downstream CUSTOMER coherence edits through `customer_adjusted_turn_indices`, while retaining compatibility with older records.
- **Integrity rules:** ACTION contents, speaker roles, turn IDs, and turn order are immutable. AGENT changes must be declared; CUSTOMER edits are limited to downstream dialogue coherence and cannot introduce new goals or create the failure themselves.
- **Evaluation:** `tools/task_closure_v5_4_judge_only.txt` uses aligned Task Closure criteria for independent judgment; its output is supporting evidence, not ground truth.

## Quick start (PowerShell)

Run from the repository root. The AB-CD source dataset and SOP guidelines must be present under `data/source/abcd/data/` (the paths can be overridden by CLI flags).

```powershell
python -m pip install -r requirements.txt

# 1. Prepare a ten-conversation V5.4 sample.
python .\src\dataset_generation\prepare_prompts.py `
  --task-closure-only `
  --task-closure-count 10 `
  --task-closure-template .\prompts\task_closure_injection_prompt_v5_4.txt `
  --output .\results_v54_core_10\prepared_prompts.jsonl `
  --overwrite `
  --seed 20261116

# 2. Run generation using an existing, authenticated LOCAL OAuth script.
# Adjust this path for your workstation; never commit credentials.
$oauthScript = "..\task-closure-test\chatgpt_plus_oauth_fixed.py"

python .\src\dataset_generation\generate_golden_dataset.py `
  --input .\results_v54_core_10\prepared_prompts.jsonl `
  --output .\results_v54_core_10\golden_dataset_injected.jsonl `
  --failed .\results_v54_core_10\failed_generations.jsonl `
  --platform chatgpt_oauth `
  --model gpt-6-luna `
  --oauth-script $oauthScript `
  --only-judge task_closure `
  --concurrency 1

# 3. Run structural regression tests (no LLM calls).
python -m unittest discover -s tests -p "test_task_closure_v54.py" -v
```

To exclude previously reviewed cases, add `--exclude-ids-file <path-to-existing-jsonl>` to the preparation command. The OAuth adapter is `src/dataset_generation/chatgpt_oauth_bridge.py`; the local authentication script and credentials are **not** part of the handoff.

## Initial results

| Check | Result |
|---|---|
| Structural regression tests | 16 passed in the earlier local test run |
| New conversations sampled | 10 |
| New AGENT-edit injections | 4 |
| Generator-classified pre-existing defects | 3 (not counted as new injections; pending independent confirmation) |
| `NO_VALID_INJECTION` / skipped | 3 |
| Produced records passing structural checks | 7 |
| V5.4 blind-judge agreement with human labels | 39/46 (84.78%) in one full run |

The four new injections were considered plausible during initial manual review, **not independently certified**. Judge disagreement and run-to-run instability were observed; agreement is not proof of semantic correctness.

## Deliverables and acceptance boundary

- Injection prompt: `prompts/task_closure_injection_prompt_v5_4.txt`
- Preparation and generation code: `src/dataset_generation/prepare_prompts.py`, `src/dataset_generation/generate_golden_dataset.py`
- OAuth adapter: `src/dataset_generation/chatgpt_oauth_bridge.py` (no credentials)
- Unit tests: `tests/test_task_closure_v54.py`
- Aligned judge prompt: `tools/task_closure_v5_4_judge_only.txt`

**Handoff acceptance:** Ready for further pilot experiments and review. **Before golden-dataset release:** independently verify Task Closure labels, baseline N -> injected Y causality, and pre-existing-defect classifications. Experimental output folders, generated datasets, ZIP patches, and secrets should not be committed as part of this code handoff.
