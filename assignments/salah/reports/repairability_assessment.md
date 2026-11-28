# Second-Stage Repairability Assessment

## Scope and method

This is a continuation of the existing audit. I did not restart the audit, overwrite any original conversation, or discard the existing reports. I re-opened the eight records that did not have a final `PASS_WITH_LIMITATIONS` or better result, compared their original and cleaned copies with the matching records in `data/source/abcd/data/abcd_v1.1.json`, and rechecked the exact subflow guidance, action schema, KB action vocabulary, scenario fields, and available action rows.

The matching source record for every assigned ID is the same record already present in the assignment; no separate authoritative action log, account system, payment record, Oracle output, or missing-turn source was found. I also reviewed analogous records in the original ABCD data to verify workflow shape, but did not use another conversation's facts or actions as evidence for the assigned conversation. Examples include:

- `boots_how_3` records with `search-faq`, `search-boots`, and (where completed) `select-faq`.
- `manage_pay_bill` records with pull-up, identity verification, subscription status/details, and renewal update.
- `status` records with identity/purchase validation, Oracle query, and an order update.
- `status_credit_missing` records with reason, Oracle, and promo-code actions.
- `status_payment_method` records with shipping-status and payment-method update actions.
- `reset_2fa` records with pull-up, details, and send-link actions.
- `status_service_removed` records with reason, Oracle, and `add service` update actions.

A record is **Repairable** only if the existing turns and available evidence determine a defensible corrected record without fabricating a customer identity, verification result, action, policy/Oracle outcome, purchase/refund, or missing turn. **Partially Repairable** means limited textual/metadata corrections are safe but the conversation still cannot qualify as clean. **Not Reliably Repairable** means the whole task or a material action/outcome remains ambiguous or contradicted.

## Assessments

### 1594 — `boots_faq__1594.json`

- **Previous verdict:** `NEEDS_REVIEW`.
- **Main defect:** the assigned `single_item_query / boots_how_3` conversation contains a repairable price typo, but its action trail is incomplete and branches into unrelated FAQ/policy/pricing actions.
- **Evidence examined:** scenario price `$89`; turn 15 says `$9` and turn 19 self-corrects to `$89`; exact Boots FAQ guidance and `config/subflow_actions_schema.json` require `search-faq` → `search-boots` → `select-faq`; embedded actions contain only the first two for the boots query, then `search-policy`, `search-pricing`, and `promo-code`.
- **Repairability:** **Partially Repairable**.
- **Proposed repair:** retain the existing cleaned correction `$9` → `$89`, normalize metadata/text only, and preserve every turn and action. Do not add a synthetic `select-faq` action, delete the later actions, or invent the selected article.
- **Evidence supporting repair:** scenario and adjacent customer turn resolve the amount typo; the exact action schema identifies the missing action but does not supply its historical result.
- **Remaining risks:** the missing FAQ selection and extra actions remain material workflow defects; material/stock claims are not independently supported.
- **Recommendation:** **Retain only as a flagged partial repair; do not count as a clean candidate.** Replacement is preferable if a clean Boots FAQ record is required.

### 4583 — `manage_pay_bill__4583.json`

- **Previous verdict:** `NEEDS_REVIEW`.
- **Main defect:** metadata counters are corrupt from the middle of the delexed array, and the conversation ends with an incomplete customer fragment.
- **Evidence examined:** scenario/account/order identifiers; exact Manage Pay Bill action schema; embedded actions for pull-up, identity verification, and renewal; original array length 54; source delexed counters 30, 32, 34, …, 58 after the FAQ actions; final original turn is the fragment “You also.”
- **Repairability:** **Partially Repairable**.
- **Proposed repair:** retain the existing positional-counter repair. Preserve the troubleshooting detour, renewal action, and final fragment. Do not append the likely-sounding words “have a good day,” infer a missing customer response, or remove the detour.
- **Evidence supporting repair:** array position is definitive for the `turn_count` metadata; the embedded renewal action is already present. No source record supplies the missing ending.
- **Remaining risks:** truncated ending and unexplained pre-verification detour prevent a clean task-completion judgment; payment success is supported only by the embedded action row.
- **Recommendation:** **Retain as a partial repair for review; do not count as a clean candidate.** Replace if a complete manage-pay-bill conversation is required.

### 1419 — `shipping_status__1419.json`

- **Previous verdict:** `REJECT`.
- **Main defect:** a shipping-address confirmation is handled through an account-address update, then a membership FAQ/troubleshooting path, and finally a new purchase for an already in-transit jacket.
- **Evidence examined:** scenario address/order/status; exact Shipping Status guidance and action schema (`pull-up-account`, `verify-identity`, `validate-purchase`, `ask-the-oracle`, optional supported `update-order`); action rows for `record-reason`, `update-account`, membership FAQ, `try-again`, and `make-purchase`; customer asks for a yes/no answer and reports an in-transit order.
- **Repairability:** **Not Reliably Repairable**.
- **Proposed repair:** no semantic repair. Keep the prior text-only normalization in the cleaned copy, but retain all action rows and turns.
- **Evidence supporting non-repairability:** the repository does not provide an Oracle result, proof that the address update was authorized, or authorization for a duplicate purchase. The correct response could have been confirmation, an order-address update, or another resolution; choosing one would invent an outcome.
- **Remaining risks:** unauthorized order/account mutation and unresolved customer intent.
- **Recommendation:** **Replace.** Do not count as a clean candidate.

### 1916 — `status_credit_missing__1916.json`

- **Previous verdict:** `REJECT`.
- **Main defect:** identity values conflict with the scenario and the action path does not follow Status Credit Missing.
- **Evidence examined:** scenario Joseph Banter/silver; customer values phone `9192225566`, account `4567`, order `123`; exact SOP/schema requiring pull-up → reason → Oracle → promo-code; embedded actions show repeated verification, Oracle, FAQ search, duplicate silver membership, and `$10` offer-refund; no `record-reason` or `promo-code` action; no documented silver-member `$10` cap.
- **Repairability:** **Not Reliably Repairable**.
- **Proposed repair:** no semantic repair. Keep only the prior spelling normalization and preserve all turns/actions.
- **Evidence supporting non-repairability:** the Oracle result is absent, the supplied identifiers are not scenario identifiers, and the correct credit amount/promo code cannot be derived. Analogous source records show the expected reason/Oracle/promo-code pattern but do not prove the missing actions here.
- **Remaining risks:** an invented promo code or refund would create an unsupported financial outcome; identity cannot be safely established.
- **Recommendation:** **Replace.** Do not count as a clean candidate.

### 1416 — `status_payment_method__1416.json`

- **Previous verdict:** `REJECT`.
- **Main defect:** the assigned `order_issue / status_payment_method` scenario is not the task performed in the conversation.
- **Evidence examined:** scenario says payment method debit card and shipping `out for delivery`; exact Status Payment Method schema requires pull-up → identity verification → shipping-status → update-order; conversation instead asks subscription active/due status, performs subscription-status actions, sends a link, and never records shipping status or a payment-method update.
- **Repairability:** **Not Reliably Repairable**.
- **Proposed repair:** no relabeling or semantic rewrite; retain the prior spelling normalization only.
- **Evidence supporting non-repairability:** the conversation is internally coherent as a due-date inquiry, but no evidence identifies a missing payment-method request or desired new method. Changing the scenario or inserting actions would fabricate task intent.
- **Remaining risks:** wrong-subflow evaluation and unsupported `$55` subscription result.
- **Recommendation:** **Replace.** Do not count as a clean candidate.

### 6052 — `reset_two_factor_auth__6052.json`

- **Previous verdict:** `REJECT`.
- **Main defect:** unresolved identity verification and a reset-2FA path that conflicts with the security SOP.
- **Evidence examined:** scenario Sanya Afzal, security answer `livingston`, ZIP `6227`; exact Reset Two-Factor Auth guidance/schema requires pull-up → enter email or approved PIN/security-question detail → send link; conversation records an old phone as `record-reason`, multiple non-prescribed verification attempts, a username entry, and a “first pet” question; no email is available; no authoritative verification outcome exists.
- **Repairability:** **Not Reliably Repairable**.
- **Proposed repair:** no semantic repair. Keep only spelling/metadata normalization. Do not convert the phone to an email, treat the first-pet answer as an approved verification result, or claim the reset was authorized.
- **Evidence supporting non-repairability:** the action schema confirms the required action vocabulary, while the available records do not contain a valid email, PIN, or documented successful verification. The scenario’s stored answer alone does not establish that the conversation’s different question was valid.
- **Remaining risks:** account takeover/security risk if a fabricated verification is accepted.
- **Recommendation:** **Replace.** Do not count as a clean candidate.

### 1284 — `shirt_faq__1284.json`

- **Previous verdict:** `REJECT`.
- **Main defect:** the record contains two incompatible tasks and customer profiles: an opening Joyce Wu/Tommy Hilfiger boots price-refund request, followed by a Calvin Klein shirt FAQ that happens to resemble the assigned scenario.
- **Evidence examined:** assigned Joseph Banter/Calvin Klein shirt scenario; original opening turns and identifiers; action rows for Joyce Wu/JSEX7PPUAJ, empty account, empty refund amount; later turns 42–56 form a coherent shirt FAQ suffix; exact Shirt FAQ workflow requires search/select.
- **Repairability:** **Not Reliably Repairable**.
- **Proposed repair:** no deletion or splitting of the opening refund turns, no reassignment of the account, and no invention of the missing refund amount. Retain prior spelling/counter normalization only.
- **Evidence supporting non-repairability:** the coherent suffix does not establish that the preceding refund task was completed or that the two profiles are the same customer. Splitting the suffix would create a new conversation and violate preservation rules.
- **Remaining risks:** scenario contamination, wrong identity, incomplete financial action, and ambiguous task completion.
- **Recommendation:** **Replace.** Do not count as a clean candidate.

### 285 — `status_service_removed__285.json`

- **Previous verdict:** `REJECT`.
- **Main defect:** unresolved identity and payment activity replace the required removed-service reason/Oracle/reinstatement workflow.
- **Evidence examined:** scenario Crystal Minh/silver; exact Status Service Removed guidance/schema requiring pull-up → reason (`email`, `spouse`, or `news`) → Oracle → `add service`; record supplies misspelled name and account `2645485`, repeated failed/unknown verification, subscription-status/link/payment actions, a full card number, and final update value `14`.
- **Repairability:** **Not Reliably Repairable**.
- **Proposed repair:** no semantic repair. Preserve the prior spelling/counter normalization; do not infer the reason, Oracle result, account authorization, or `add service` outcome.
- **Evidence supporting non-repairability:** no reason-slot value or Oracle result is present, and the card/payment action cannot be safely retained as proof of a valid account operation after failed verification.
- **Remaining risks:** privacy/security exposure and unsupported financial/account mutation.
- **Recommendation:** **Replace.** Do not count as a clean candidate.

## Repair-stage execution decision

No additional semantic repair is justified for any of the eight records. The existing `assignments/salah/cleaned/` copies already contain all safe text/metadata corrections identified in stage one. To make this second-stage decision independently reviewable, `assignments/salah/repair_stage/` contains byte-for-byte copies of those cleaned files; it does not alter the originals and does not pretend that unresolved workflow defects were fixed.

The only defensible changes in this stage are therefore preservation of the prior safe corrections and explicit non-repair decisions. Adding actions, deleting turns, changing subflows, inventing results, or completing fragments would violate the evidence boundary.

## Current candidate count

- **Strict `PASS`:** 0.
- **`PASS_WITH_LIMITATIONS` candidates:** 2 (`5334`, `1739`).
- **Additional partially repairable but not clean:** 2 (`1594`, `4583`).
- **Not reliably repairable / replacement recommended:** 6 (`1419`, `1916`, `1416`, `6052`, `1284`, `285`).
- **Defensibly usable candidates currently:** **2**, both with documented limitations.
