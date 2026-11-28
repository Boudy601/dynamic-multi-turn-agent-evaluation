# Ownership and Exclusion Report

## Method

I inspected every teammate-named directory under `assignments/`, including nested `.originals/` copies, and read each JSON assignment artifact with a `convo_id`. I also checked `assignments/reviews.json` for IDs that might be under review without an assignment file. Duplicate files and review entries were deduplicated by conversation ID. No teammate folder, assignment, original, cleaned copy, or review entry was modified.

The source has 10,042 unique conversations across the `train`, `dev`, and `test` splits. Teammate assignment files/manifests reserve 85 distinct IDs. The 55 IDs in `reviews.json` are a subset of those 85; there are no review-only IDs. Salah’s 10 original IDs are also excluded from replacement selection. These sets are disjoint, leaving 9,947 eligible source conversations before candidate quality review.

## Teammate exclusions

All IDs listed below are excluded from both replacement discovery and final selection. IDs from duplicate `.originals/` files and review records are listed once.

| Teammate folder | Excluded conversation IDs | Count |
|---|---|---:|
| `amr/` | 859, 862, 1332, 1386, 1696, 2647, 3518, 3707, 4394, 4540, 5055, 5302, 5468, 5798, 5817, 5833, 6262, 6473, 6775, 6849, 7062, 7325, 7449, 7562, 7598, 7887, 9031, 9307, 9686, 9959, 10168, 10271, 10367, 10588, 10595 | 35 |
| `bahaa/` | 125, 326, 477, 1075, 1924, 2214, 2231, 3736, 5008, 5564 | 10 |
| `islam/` | 116, 224, 365, 1627, 2310, 2544, 3477, 3740, 6475, 10540 | 10 |
| `moaz/` | 126, 151, 263, 807, 1655, 2211, 3779, 7647, 8065, 8246 | 10 |
| `omar/` | 444, 1975, 2552, 2815, 3154, 3623, 3627, 3858, 8032, 10260 | 10 |
| `youssef/` | 157, 1219, 1413, 1714, 2363, 3161, 4664, 6007, 6680, 10020 | 10 |

**Teammate-reserved IDs:** 85 unique IDs. **Review-only IDs:** 0 (none).

## Salah original exclusions from replacement pool

The following ten IDs were not considered as replacements because they were already assigned to Salah: 1594, 4583, 5334, 6052, 1419, 1284, 1916, 1416, 285, 1739. Two are retained in the final dataset after revalidation; the other eight are copied to `assignments/salah/excluded/` with individual reasons.

## Final selected replacement ownership check

Selected replacement IDs: 10486, 10143, 6026, 10569, 5307, 8208, 6813, 10471. Each is absent from the complete teammate assignment/reservation union and absent from Salah’s original ten. Their matching source records are in the original ABCD dataset.
