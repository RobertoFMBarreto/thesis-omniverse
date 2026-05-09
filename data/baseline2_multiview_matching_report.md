# Baseline 2 — Phase B (minimal): multi-view geometric matching

> **NOT multi-view fusion. NOT 3D reconstruction. NOT pose estimation.** Score-level aggregation across per-view rasterisations only.

- Run id: `746b5f5e`
- Timestamp (UTC): `2026-05-09T16:46:05.802232+00:00`

## Objective

Test ONE research question: do additional deterministic viewpoints (top + front + side) improve geometric discrimination over the single-view Baseline 1?

## Methodology

For each (piece, view), the per-view depth map is back-projected to world XYZ using the per-view intrinsics and the measured camera pose (USD convention, camera looks along local -Z). Points above the support surface are kept, centroid-centred, and rasterised via Baseline 1's `rasterise_xy_to_mask` (320x320 px @ 0.25 mm/px, with convex-hull representation-normalisation when the splat is fragmented). Each per-view mask is scored against every cavity using Baseline 1's `score_pair` (180-rotation search; `inside`/`outside` on dilated cavity, IoU on non-dilated). The three per-view best scores are combined via weighted average (renormalised when a view is missing). The comparison is viewpoint-symmetric: each piece view is scored only against the matching cavity view (top_down vs top_down, front_oblique vs front_oblique, side_oblique vs side_oblique). Cavity-view source policy is hybrid and deterministic: top_down cavity masks reuse the validated Baseline 1 opening point cloud (`cavity_opening_pointcloud.npy`); oblique cavity masks are derived from the multi-view depth via the local XY ROI + Z rim-band extraction. The policy is recorded per view as `cavity_source` (`baseline1_validated_opening` or `multiview_roi_z_band`). This is NOT multi-view fusion, NOT 3D reconstruction, NOT pose estimation.

## Descriptors used

Per view: `inside_ratio`, `outside_ratio`, `iou`, `best_score = W_IOU·iou + W_INSIDE·inside − W_OUTSIDE·outside` (weights inherited from Baseline 1). No additional descriptors.

## Aggregation strategy

Weighted average with hardcoded weights `top_down=0.6`, `front_oblique=0.2`, `side_oblique=0.2`. Missing-view weights are dropped and remaining weights are renormalised to sum to 1.

## Aggregate score matrix

| piece | cavity_00 | cavity_01 | cavity_02 | cavity_03 |
|---|---|---|---|---|
| rectangle | 0.4931 | 0.1236 | 0.3862 | 0.3161 |
| square | 0.3939 | 0.2233 | 0.5785 | 0.4879 |
| circle | 0.3490 | 0.2872 | 0.5339 | 0.5921 |
| triangle | 0.2927 | 0.4937 | 0.4849 | 0.4995 |

## Per-piece ranking results

| piece | rank-1 | score | rank-2 | margin | low_margin | missing_view | per_view_disagreement |
|---|---|---|---|---|---|---|---|
| rectangle | cavity_00 | 0.4931 | cavity_02 | 0.1069 | False | False | True |
| square | cavity_02 | 0.5785 | cavity_03 | 0.0906 | False | False | True |
| circle | cavity_03 | 0.5921 | cavity_02 | 0.0582 | False | False | True |
| triangle | cavity_03 | 0.4995 | cavity_01 | 0.0058 | True | False | True |

## Ambiguity indicators summary

- Rank-1 pairs with `low_margin`: 1 / 4
- Rank-1 pairs with `missing_view`: 0 / 4
- Rank-1 pairs with `per_view_disagreement`: 4 / 4

## Missing-view warnings

None.

## Limitations

- Deterministic, geometry-only by design.
- Sensitive to the choice of viewpoints; Phase A used sequential single-camera relocation, not a synchronised multi-camera rig.
- Cavity side now uses the symmetric multi-view captures from `data/multiview_captures/cavities/`.
- The convex-hull representation-normalisation fallback from Baseline 1 is inherited unchanged.
- No 3D reconstruction. No pose estimation. No multi-view fusion (only score-level aggregation).
- View weights and `MIN_VIEW_POINTS` are hardcoded; not tuned.

## Closing note

This experiment is **not** multi-view fusion, **not** 3D reconstruction, **not** pose estimation. It is a minimal score-level aggregation built only to test whether additional deterministic viewpoints reduce ambiguity in the Baseline 1 ranking on this MVP set.
