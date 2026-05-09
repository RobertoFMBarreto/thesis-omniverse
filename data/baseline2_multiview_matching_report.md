# Baseline 2 — Phase B (minimal): multi-view geometric matching

> **NOT multi-view fusion. NOT 3D reconstruction. NOT pose estimation.** Score-level aggregation across per-view rasterisations only.

- Run id: `b57e14e9`
- Timestamp (UTC): `2026-05-09T15:47:16.293261+00:00`

## Objective

Test ONE research question: do additional deterministic viewpoints (top + front + side) improve geometric discrimination over the single-view Baseline 1?

## Methodology

For each (piece, view), the per-view depth map is back-projected to world XYZ using the per-view intrinsics and the measured camera pose (USD convention, camera looks along local -Z). Points above the support surface are kept, centroid-centred, and rasterised via Baseline 1's `rasterise_xy_to_mask` (320x320 px @ 0.25 mm/px, with convex-hull representation-normalisation when the splat is fragmented). Each per-view mask is scored against every cavity using Baseline 1's `score_pair` (180-rotation search; `inside`/`outside` on dilated cavity, IoU on non-dilated). The three per-view best scores are combined via weighted average (renormalised when a view is missing).

## Descriptors used

Per view: `inside_ratio`, `outside_ratio`, `iou`, `best_score = W_IOU·iou + W_INSIDE·inside − W_OUTSIDE·outside` (weights inherited from Baseline 1). No additional descriptors.

## Aggregation strategy

Weighted average with hardcoded weights `top_down=0.6`, `front_oblique=0.2`, `side_oblique=0.2`. Missing-view weights are dropped and remaining weights are renormalised to sum to 1.

## Aggregate score matrix

| piece | cavity_00 | cavity_01 | cavity_02 | cavity_03 |
|---|---|---|---|---|
| rectangle | 0.7928 | 0.2705 | 0.5669 | 0.4495 |
| square | 0.7029 | 0.3833 | 0.7448 | 0.6292 |
| circle | 0.6344 | 0.4861 | 0.7010 | 0.7424 |
| triangle | 0.5346 | 0.6973 | 0.6062 | 0.6206 |

## Per-piece ranking results

| piece | rank-1 | score | rank-2 | margin | low_margin | missing_view | per_view_disagreement |
|---|---|---|---|---|---|---|---|
| rectangle | cavity_00 | 0.7928 | cavity_02 | 0.2259 | False | False | False |
| square | cavity_02 | 0.7448 | cavity_00 | 0.0418 | False | False | True |
| circle | cavity_03 | 0.7424 | cavity_02 | 0.0414 | False | False | True |
| triangle | cavity_01 | 0.6973 | cavity_03 | 0.0767 | False | False | True |

## Ambiguity indicators summary

- Rank-1 pairs with `low_margin`: 0 / 4
- Rank-1 pairs with `missing_view`: 0 / 4
- Rank-1 pairs with `per_view_disagreement`: 3 / 4

## Missing-view warnings

None.

## Limitations

- Deterministic, geometry-only by design.
- Sensitive to the choice of viewpoints; Phase A used sequential single-camera relocation, not a synchronised multi-camera rig.
- Cavity side is single-view only (Baseline 1 captures); only the piece side is multi-view.
- The convex-hull representation-normalisation fallback from Baseline 1 is inherited unchanged.
- No 3D reconstruction. No pose estimation. No multi-view fusion (only score-level aggregation).
- View weights and `MIN_VIEW_POINTS` are hardcoded; not tuned.

## Closing note

This experiment is **not** multi-view fusion, **not** 3D reconstruction, **not** pose estimation. It is a minimal score-level aggregation built only to test whether additional deterministic viewpoints reduce ambiguity in the Baseline 1 ranking on this MVP set.
