# Replacement Candidate Selection Report

## Search scope and exclusions

The complete source dataset contains 10,042 unique conversations across train/dev/test. After excluding all 85 teammate-assigned/reserved IDs and Salah’s original ten IDs, the eligible pool contained 9,947 records. The ownership calculation and explicit teammate ID list are in `ownership_exclusion_report.md`.

I screened the entire eligible pool rather than limiting discovery to a fixed batch. The broad screen used source IDs, scenarios/subflows, original/delexed lengths and counters, speaker alignment, action targets, and action-sequence presence against `config/subflow_actions_schema.json` and `data/source/abcd/data/kb.json`. I then read the full source conversation and scenario for the selected candidates and checked the exact applicable subflow guidance. Analogous ABCD conversations were used only to understand expected workflow patterns, never as evidence for another record’s facts or actions.

## Selection criteria

Candidates were prioritized for:

1. Source scenario and conversation intent matching one another.
2. Complete original/delexed arrays with consistent speaker and turn metadata.
3. A recorded action path matching the applicable SOP and action schema.
4. No unresolved identity, security, privacy, or task-completion problem.
5. Minimal unsupported claims and no need to fabricate missing events.
6. Coverage across multiple workflows to support a varied evaluation set.

The final eight replacement records meet the core structure and workflow requirements. Several rely on an embedded Oracle/system query whose raw backend return value is not separately stored in the repository; the conversation text reports the result and subsequent action records are consistent. Those records are marked `PASS_WITH_LIMITATIONS` in the replacement validation report. Their limitations are disclosed rather than represented as independent backend verification.

## Selected replacements

| ID | Source split | Scenario / subflow | Why selected | Corrections | Proposed final status |
|---:|---|---|---|---|---|
| 10486 | train | `purchase_dispute / bad_price_competitor` | Customer, product, account/order identifiers match the scenario; pull-up, competitor reason, identity verification, and promo-code actions follow the SOP. | None | PASS_WITH_LIMITATIONS |
| 10143 | train | `manage_account / status_credit_missing` | Customer name and reason match; pull-up, previous-purchase reason, Oracle query, and $40 promo code are recorded in order; code is provided in the response. | None | PASS_WITH_LIMITATIONS |
| 6026 | train | `order_issue / status_payment_method` | Name, account ID, order ID, in-transit status, and selected debit-card method align with the scenario and SOP; all four required actions appear. | None | PASS |
| 10569 | train | `manage_account / status_service_removed` | Name, news reason, Oracle query, `add service` update, and customer confirmation that free shipping returned align with the SOP. | None | PASS_WITH_LIMITATIONS |
| 5307 | train | `account_access / reset_2fa` | Customer email matches scenario; pull-up, email detail, and security-link actions follow reset-2FA guidance; no verification result was fabricated. | None | PASS_WITH_LIMITATIONS |
| 8208 | train | `order_issue / status_delivery_time` | Name/account/order, reason, Oracle, `order received` status, and `change time` action form a complete recorded path; customer confirms. | None | PASS_WITH_LIMITATIONS |
| 6813 | train | `order_issue / status_payment_method` | Joseph Banter’s account/order match the scenario; `in transit` allows an immediate change, and the customer-requested PayPal differs from scenario debit card and matches the update action. | Corrected “immidiately” to “immediately” | PASS |
| 10471 | train | `subscription_inquiry / status_due_date` | Name/account/order match; status query, $49 due tomorrow, requested payment, send-link, and $49 detail entry align with the flow and task. | None | PASS_WITH_LIMITATIONS |

The eight replacements are distinct from one another, Salah’s original ten, and the teammate reservation union. The source scenarios and action records were preserved; any listed text normalization is logged separately. No candidate was rewritten to fit a subflow.

## Candidates screened out during manual review

- **3011 (`status_payment_method`)** was discarded despite its near-complete action chain because the customer’s spoken account ID differs from the scenario ID; the embedded verification target silently uses the scenario ID. That identity discrepancy is material.
- **10192 (`out_of_stock_general`)** was discarded because the agent promises tomorrow’s stock availability and says it will be fixed tomorrow without an action result supporting those claims.
- Several product FAQ candidates with a matching select action were not selected because the repository does not contain the underlying FAQ article text needed to independently verify the substantive advice. Their action completeness alone was insufficient for this final set.
- ID 10057 (`status_shipping_question`) was excluded from the final selection because the guest international-shipping eligibility result has no separately stored Oracle payload; a possible false benefit statement would be material to that task. ID 9607 (`status_payment_method`) was excluded because the scenario already lists debit card, which the conversation then updates to debit card despite the customer saying the method is wrong. Additional near matches with missing actions, malformed turns, mismatched identifiers, unsupported payment/order mutations, or incomplete resolution were excluded during screening.

## Existing Salah records retained

- **5334** is included from its cleaned copy as `PASS_WITH_LIMITATIONS`: its guest-member three-day wait is consistent with the seven-day SOP and no credit is issued. The day-eight conditional promise/agent ID has no independent record.
- **1739** is included from its cleaned copy as `PASS_WITH_LIMITATIONS`: the logged troubleshooting actions match the SOP and the customer confirms that the slowdown is resolved; there is no external performance telemetry.

The other eight original Salah conversations are not accepted. They are copied unchanged into `assignments/salah/excluded/` and documented in `excluded/reasons.md`.

## Outcome

The broad eligible pool is large enough to supply replacements without using teammate-owned records or forcing repairs to the original rejected records. The final selection combines two revalidated Salah conversations with eight eligible, independently audited source conversations. No selected file required a semantic correction.
