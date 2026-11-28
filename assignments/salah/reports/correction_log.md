# Correction Log

All files under `assignments/salah/cleaned/` are copies of the 10 assigned originals. The originals in the parent directory were not edited, and no turns were removed. Changes below are limited to traceable metadata or unambiguous text normalization; unresolved workflow and evidence defects remain documented in the audit and validation reports.

## Modified conversations

### 1594 — `boots_faq__1594.json`

- **Original issue:** turn 15 said “$9 Guess boots,” while the scenario and the customer’s turn 19 correction establish the item price as `$89`.
- **Category/severity:** factual/context consistency, Low.
- **Evidence:** `scenario.product.amounts[0] = 89`; turn 19 says “$89 Guess boots.”
- **Original → corrected:** `$9 Guess boots` → `$89 Guess boots` in both `original[14]` and the corresponding delexed text.
- **Verification:** scenario/text cross-check passed. The missing Boots FAQ selection and extra actions were not fabricated or removed.

### 4583 — `manage_pay_bill__4583.json`

- **Original issue:** delexed `turn_count` values diverged from array positions from turn 29 onward.
- **Category/severity:** structural integrity, High.
- **Evidence:** `original` and `delexed` each contain 54 rows; positional counters should be 1–54 and were 30, 32, 34, …, 58 for later rows.
- **Original → corrected:** all cleaned `delexed[*].turn_count` values were normalized to their 1-based array positions.
- **Verification:** JSON parses and every counter equals its row position. The incomplete final customer fragment “You also” remains unchanged.

### 6052 — `reset_two_factor_auth__6052.json`

- **Original issues:** three unambiguous spelling errors: “forst,” “oyu,” and “securty.”
- **Category/severity:** textual integrity, Low.
- **Original → corrected:** `forst` → `first`; `oyu` → `you`; `securty` → `security` in `original` and delexed text.
- **Verification:** edits are spelling-only. The security/workflow mismatch remains unchanged and is still `REJECT`.

### 1419 — `shipping_status__1419.json`

- **Original issues:** capitalization/grammar defects in turns 3, 25, 47, and 50.
- **Category/severity:** textual integrity, Low; the underlying address-update and duplicate-purchase actions remain High-severity unresolved issues.
- **Original → corrected:** `YOu` → `You`; `i ma` → `I am`; `sent it again` → `send it again`; and the dangling phrase “ship that you to you … at” → “ship that to you at 7597 Brushwick Drive.” The delexed text was kept aligned with the corrected address sentence.
- **Evidence:** direct text context and the scenario address. No action was added or removed.
- **Verification:** JSON parses; action sequence is unchanged and remains rejected for independent workflow reasons.

### 1284 — `shirt_faq__1284.json`

- **Original issues:** action text contained “progess” three times; the customer wrote “Calving Klein.” Delexed counters were offset from turn 24 onward.
- **Category/severity:** textual/structural integrity, Medium.
- **Original → corrected:** `progess` → `progress`; `Calving Klein` → `Calvin Klein`; all delexed counters normalized to array positions.
- **Evidence:** standard action wording uses “progress”; scenario product is Calvin Klein shirt.
- **Verification:** edits do not resolve the larger account/topic collision; final status remains `REJECT`.

### 1916 — `status_credit_missing__1916.json`

- **Original issue:** action text used the spelling “progess” three times.
- **Category/severity:** textual integrity, Low.
- **Original → corrected:** `progess` → `progress` in the three action rows and delexed text.
- **Verification:** workflow, identity, and unsupported policy defects remain unchanged; final status remains `REJECT`.

### 1416 — `status_payment_method__1416.json`

- **Original issue:** action text used “progess” once.
- **Category/severity:** textual integrity, Low.
- **Original → corrected:** `progess` → `progress` in the identity-verification action and delexed text.
- **Verification:** assigned-subflow mismatch remains unchanged; final status remains `REJECT`.

### 285 — `status_service_removed__285.json`

- **Original issues:** action text used “progess” four times; delexed counters were offset from turn 22 onward.
- **Category/severity:** textual/structural integrity, Medium.
- **Original → corrected:** `progess` → `progress`; all delexed counters normalized to array positions.
- **Verification:** unresolved identity, card exposure, and wrong update action remain unchanged; final status remains `REJECT`.

### 1739 — `website_too_slow__1739.json`

- **Original issues:** “tabd” and “etra” were obvious spelling errors.
- **Category/severity:** textual integrity, Low.
- **Original → corrected:** `tabd` → `tabs`; `etra` → `extra` in `original` and delexed text.
- **Evidence:** direct sentence context; no change to meaning or action sequence.
- **Verification:** try-again, log-out/in, and instructions actions remain aligned with the slow-speed SOP; final status is `PASS_WITH_LIMITATIONS`.

## Retained unchanged

### 5334 — `mistimed_billing_already_returned__5334.json`

No supported material correction was identified. The guest-member wait path is consistent with the documented seven-day workflow, and the record is retained unchanged.

## Correction boundaries

No missing action, customer detail, policy result, purchase, refund, identity verification, or outcome was invented. The reports continue to flag records whose workflow or evidence cannot be repaired confidently.
