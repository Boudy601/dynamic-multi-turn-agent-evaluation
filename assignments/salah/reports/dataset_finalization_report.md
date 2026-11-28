# Final Dataset Report

## Verdict: `READY_FOR_SUBMISSION`

`assignments/salah/final_dataset/` contains exactly 10 distinct conversation JSON files. All 10 passed final validation and have accepted statuses: 2 `PASS` and 8 `PASS_WITH_LIMITATIONS`. There are no `NEEDS_REVIEW` or `REJECT` records in the final directory. The documented limitations concern unavailable external service payloads or telemetry; they do not undermine evaluation of the observed conversation and action flows. No unresolved material contradiction, unsupported action, identity defect, or security/privacy defect was detected in the selected files.

## Final contents

| Conversation ID | Scenario / subflow | Status | Corrections |
|---:|---|---|---|
| 5334 | `purchase_dispute / mistimed_billing_already_returned` | PASS_WITH_LIMITATIONS | None in final copy |
| 1739 | `troubleshoot_site / slow_speed` | PASS_WITH_LIMITATIONS | Existing typo/counter corrections retained |
| 10486 | `purchase_dispute / bad_price_competitor` | PASS_WITH_LIMITATIONS | Two spelling/grammar fixes |
| 10143 | `manage_account / status_credit_missing` | PASS_WITH_LIMITATIONS | None |
| 6026 | `order_issue / status_payment_method` | PASS | Two typo fixes |
| 10569 | `manage_account / status_service_removed` | PASS_WITH_LIMITATIONS | One punctuation/grammar fix |
| 5307 | `account_access / reset_2fa` | PASS_WITH_LIMITATIONS | One spelling fix |
| 8208 | `order_issue / status_delivery_time` | PASS_WITH_LIMITATIONS | None |
| 6813 | `order_issue / status_payment_method` | PASS | One spelling fix |
| 10471 | `subscription_inquiry / status_due_date` | PASS_WITH_LIMITATIONS | Typo and stray punctuation fixes |

Conversation IDs 5334 and 1739 are retained from the earlier cleaned copies after revalidation. The other eight conversations are eligible replacement records selected from the train split. The selected IDs are absent from all teammate assignment files, `.originals` copies, manifests, and review records.

## Quality gate

- **Exactly 10 JSON files:** PASS.
- **All files parse:** PASS.
- **10 distinct conversation IDs:** PASS.
- **All statuses accepted:** PASS (`2 PASS`, `8 PASS_WITH_LIMITATIONS`).
- **No teammate-owned or reserved IDs:** PASS; checked against all six teammate folders and `assignments/reviews.json`.
- **Original Salah source records preserved:** PASS; all ten parent JSON files remain exactly equal to their matching records in the original ABCD dataset.
- **Corrections documented and supported:** PASS; only listed text normalizations were made, and each is recorded in `replacement_correction_log.md`.
- **Scenario, turns, delexed turns, and actions:** PASS; scenarios and action targets match source records, array lengths and speaker alignment are consistent, and all delexed turn counters match array positions.
- **Material workflow, factual, action, identity, privacy, or security defect remaining in accepted records:** none detected under the repository evidence available.
- **Manifest matches final files:** PASS; all 10 IDs and paths match the delivered directory.

## Evidence limits

The source repository contains scenario data, transcript text, delexed turns, and action records, but it does not expose live payment, account, Oracle-return, email-delivery, competitor-price, or website telemetry payloads. The accepted limited-status records document exactly which outcomes rely on transcript/action evidence rather than independent backend confirmation. This evidence limit does not compromise the intended evaluation of conversation intent, workflow selection, dialogue handling, and recorded action execution; no hidden result was inferred or added.

Detailed reports are available in `assignments/salah/reports/`: ownership exclusions, candidate selection, replacement audits, correction trace, independent validation, and the final manifest. The eight excluded original Salah records and their reasons are preserved in `assignments/salah/excluded/`.
