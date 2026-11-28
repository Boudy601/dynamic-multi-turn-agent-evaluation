# Replacement Correction Log

The eight replacement originals remain unchanged in `data/source/abcd/data/abcd_v1.1.json`. Final copies are in `assignments/salah/final_dataset/`. Six selected replacement files received spelling/punctuation-only normalizations. No scenario, action, identifier, financial amount, Oracle result, or turn was added or removed.

| Conversation | Original content → cleaned copy | Evidence and validation |
|---:|---|---|
| 10486 | Turn 3: “higher price then your competitor” → “higher price than your competitor”; turn 18: “certaintly” → “certainly” | Grammar/spelling correction only; scenario, speaker, customer meaning, and all actions preserved. |
| 6813 | Turn 15: “immidiately” → “immediately” | Obvious spelling correction only; customer request, scenario, and all action targets are preserved. |
| 6026 | Turn 3: “yo uwish” → “you wish”; turn 16: “immidiately” → “immediately” | Obvious spacing/spelling corrections only; matching scenario identifiers and action targets preserved. |
| 10569 | Turn 5: “Its more” → “It’s more” | Punctuation/grammar correction only; customer intent and action sequence preserved. |
| 5307 | Turn 2: “two fact authentication” → “two-factor authentication” | Standard compound spelling correction only; no verification result or identity data added. |
| 10471 | Turn 2: “passible” → “possible”; turn 22: trailing `@` removed from “Thank you so much!@” | Obvious typo and stray punctuation correction only; payment amount and recorded details remain unchanged. |

The `original` and corresponding `delexed[*].text` were kept aligned for each correction. IDs 10143, 8208, 5334, and 1739 were copied without changes from their previously cleaned/source records. All ten final files passed independent structural checks after copying. The action row remains the evidence of each recorded action; no backend response payload was fabricated.
