# Excluded Original Salah Conversations

The following eight original Salah conversations did not meet final acceptance. Each original JSON is copied unchanged into this directory. The assigned parent-directory copies remain unchanged, and no record was deleted or overwritten.

| ID | File | Exclusion reason |
|---:|---|---|
| 1594 | `boots_faq__1594.json` | The price typo is repairable, but the Boots FAQ action trail lacks `select-faq` and includes additional policy/pricing/promo actions that cannot be reconstructed safely. |
| 4583 | `manage_pay_bill__4583.json` | Counter metadata was repaired, but an unexplained workflow detour and truncated final customer turn (“You also”) remain. |
| 6052 | `reset_two_factor_auth__6052.json` | Security verification is unresolved and the recorded path conflicts with the reset-2FA procedure. |
| 1419 | `shipping_status__1419.json` | Address/account mutation and new purchase actions are unsupported by the customer’s confirmation request and cannot be safely reconstructed. |
| 1284 | `shirt_faq__1284.json` | Conflicting customer/account contexts, unrelated refund and FAQ tasks, and an empty refund amount make the intended record ambiguous. |
| 1916 | `status_credit_missing__1916.json` | Identity conflicts with the scenario, required reason/promo-code actions are absent, and the `$10` policy claim lacks evidence. |
| 1416 | `status_payment_method__1416.json` | The conversation is a subscription due-date inquiry rather than the assigned payment-method task; required shipping/payment actions are missing. |
| 285 | `status_service_removed__285.json` | Identity remains unresolved, full card data was exposed, and the required reason/Oracle/reinstate path is absent. |

These records were not accepted or added to the final dataset. The corresponding existing audit, correction, and validation reports contain the detailed evidence and limits.
