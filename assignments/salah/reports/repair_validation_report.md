# Repair-Stage Validation Report

## Scope

This validation covers the second-stage copies in `assignments/salah/repair_stage/`. The stage copies were created from `assignments/salah/cleaned/` after the repairability assessment. No original parent conversation was modified. The assessment determined that no new semantic action, identity, Oracle, policy, purchase, refund, or missing-turn repair could be supported by repository evidence, so the stage copies preserve the safe first-stage corrections and the unresolved defects.

## Checks performed

For each of the 10 repair-stage JSON files, I independently checked:

- JSON parsing and required fields.
- Scenario equality with the matching record in `data/source/abcd/data/abcd_v1.1.json`.
- Equal `original`/`delexed` lengths.
- Speaker alignment between original and delexed rows.
- 1-based positional delexed `turn_count` values.
- Byte-for-byte equality with its first-stage cleaned input.
- Exact equality of every original parent JSON with its matching record in `data/source/abcd/data/abcd_v1.1.json`.

All 10 files passed these structural and preservation checks. A separate source comparison also confirmed that each original parent JSON is exactly equal to its matching ABCD source record. The semantic statuses remain those established by the independent final validation because no additional semantic repairs were justified:

| ID | Repair-stage status | Result |
|---:|---|---|
| 1594 | NEEDS_REVIEW | Partial textual repair only; missing/extra workflow actions remain |
| 4583 | NEEDS_REVIEW | Counter repair only; incomplete ending remains |
| 5334 | PASS_WITH_LIMITATIONS | Retained unchanged; workflow is supported with evidence limitations |
| 6052 | REJECT | Security verification and workflow cannot be reconstructed safely |
| 1419 | REJECT | Address/purchase action path cannot be reconstructed safely |
| 1284 | REJECT | Incompatible identities/tasks and incomplete refund |
| 1916 | REJECT | Identity and financial action outcome cannot be established |
| 1416 | REJECT | Wrong assigned subflow; no payment-method task evidence |
| 285 | REJECT | Unresolved identity and unsupported account/payment mutation |
| 1739 | PASS_WITH_LIMITATIONS | Troubleshooting path supported; no external telemetry |

## Result

The repair stage introduces no new errors and does not claim to repair records that lack evidence. Current defensibly usable candidates remain **2** (`5334` and `1739`), both `PASS_WITH_LIMITATIONS`; there are **0** strict `PASS` records. The six `REJECT` records should be replaced, and the two `NEEDS_REVIEW` records require authoritative clarification or replacement before use.
