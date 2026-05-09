# Baseline 2 – Phase C: Canonical Multi-View Geometric Matching

**Run ID:** `143cf836`  
**Timestamp UTC:** 2026-05-09T20:14:48Z  
**Script:** `baseline2_phaseC_canonical_multiview.py`

---

## Objective

Exploratory canonical world-frame XY representation for geometric insertion matching. Each piece and cavity is represented as a single **canonical** 2D footprint by merging surviving views, centroid-centring once, and rasterising. Baseline 1 scoring (`score_pair`) is reused **unchanged**. This is NOT fusion in the volumetric/SLAM/TSDF sense. No learned model. No descriptors added.

## Methodology

### Piece canonical construction

1. For each view (`top_down`, `front_oblique`, `side_oblique`): back-project depth, segment points above support surface (`support_z + PIECE_HEIGHT_MIN_ABOVE_SURFACE_M`), keep UN-CENTRED world XY.
2. Discard views with fewer than `MIN_VIEW_POINTS` (50) points.
3. Merge surviving views with `np.vstack`, compute centroid, centroid-centre once.
4. Rasterise via `rasterise_xy_to_mask` (Baseline 1, unchanged).

### Cavity canonical construction

1. `top_down`: load Baseline-1-validated `cavity_opening_pointcloud.npy` (fallback: `cavity_pointcloud.npy`).
2. `front_oblique`, `side_oblique`: back-project, apply Z-band `(board_top - CAVITY_DEPTH_MAX_BELOW_SURFACE_M, board_top - CAVITY_DEPTH_MIN_BELOW_SURFACE_M)` and XY ROI `±CAVITY_VIEW_ROI_HALF_SIZE_M` around cavity centre. Keep UN-CENTRED world XY.
3. Discard views with fewer than `MIN_VIEW_POINTS` (50) points.
4. Merge and centroid-centre once. Build `(mask_dil, mask_undil)` via `build_cavity_masks`.

### Scoring

For each (piece, cavity) pair, call `score_pair(piece_canonical_xy, mask_dil, mask_undil)` from Baseline 1 unchanged. Take `best_record`. Rank cavities per piece by score descending.

### Sparsity policy

No auto-fallback to Baseline 1 when canonical points are sparse. `canonical_sparse = True` is set when merged count < 150 (3×MIN_VIEW_POINTS). `invalid = True` only when merged count < 50 OR mask area = 0 px.

## Canonical Entities

### Pieces

| Piece | Views used | Merged pts | Mask area px | Bbox px | Fill ratio | Hull fallback | Sparse | Invalid |
|-------|-----------|-----------|-------------|---------|-----------|--------------|--------|---------|
| rectangle | top_down, front_oblique, side_oblique | 13872 | 13888 | 57285 | 0.242 | yes | no | no |
| square | top_down, front_oblique, side_oblique | 10004 | 9516 | 40401 | 0.236 | yes | no | no |
| circle | top_down, front_oblique, side_oblique | 8540 | 7465 | 40000 | 0.187 | yes | no | no |
| triangle | top_down, front_oblique, side_oblique | 7371 | 5362 | 40804 | 0.131 | yes | no | no |

### Cavities

| Cavity | Sources used | Merged pts | Mask area px | Bbox px | Fill ratio | Hull fallback | Sparse | Invalid |
|--------|-------------|-----------|-------------|---------|-----------|--------------|--------|---------|
| cavity_00 | top_down, front_oblique, side_oblique | 2172 | 2082 | 59792 | 0.035 | yes | no | no |
| cavity_01 | top_down, front_oblique, side_oblique | 2266 | 656 | 28512 | 0.023 | yes | no | no |
| cavity_02 | top_down, front_oblique, side_oblique | 2264 | 1000 | 34254 | 0.029 | yes | no | no |
| cavity_03 | top_down, front_oblique, side_oblique | 2343 | 596 | 22330 | 0.027 | yes | no | no |

## 4x4 Score Matrix

| Piece | cavity_00 | cavity_01 | cavity_02 | cavity_03 |
|-------|----------|----------|----------|----------|
| rectangle | 0.6628 | 0.2146 | 0.4762 | 0.2258 |
| square | 0.6240 | 0.2220 | 0.5413 | 0.1880 |
| circle | 0.5875 | 0.2126 | 0.5747 | 0.1692 |
| triangle | 0.5241 | 0.1345 | 0.4671 | 0.1001 |

## Per-Piece Ranking

| Piece | Rank-1 Cavity | Score | IoU | Rotation ° | Margin vs Rank-2 |
|-------|--------------|-------|-----|-----------|-----------------|
| rectangle | cavity_00 | 0.6628 | 0.6897 | 20.0 | 0.1866 |
| square | cavity_00 | 0.6240 | 0.5703 | 58.0 | 0.0827 |
| circle | cavity_00 | 0.5875 | 0.4771 | 44.0 | 0.0128 |
| triangle | cavity_00 | 0.5241 | 0.3276 | 336.0 | 0.0570 |

## Comparison vs Baseline 1 and Phase B Hybrid

Reference numbers: Baseline 1 final (doc 03 §17.4), Phase B hybrid (Iteration C). Phase C scores are from this run.

| Piece | B1 cavity | B1 score | B1 margin | PhB cavity | PhB agg | PhB margin | PhC cavity | PhC score | PhC margin |
|-------|----------|---------|---------|-----------|--------|----------|-----------|---------|---------|
| rectangle | cavity_00 | 0.883 | 0.293 | cavity_00 | 0.493 | 0.107 | cavity_00 | 0.6628 | 0.1866 |
| square | cavity_02 | 0.884 | 0.168 | cavity_02 | 0.578 | 0.091 | cavity_00 | 0.6240 | 0.0827 |
| circle | cavity_03 | 0.889 | 0.114 | cavity_03 | 0.592 | 0.058 | cavity_00 | 0.5875 | 0.0128 |
| triangle | cavity_01 | 0.886 | 0.227 | cavity_03 | 0.5 | 0.006 | cavity_00 | 0.5241 | 0.0570 |

## Limitations

- Canonical representation merges views naively (vstack). No view weighting, no registration, no outlier filtering.
- Oblique views add density but may add noise from perspective distortion or mis-segmented background.
- The Z-band and ROI for cavity oblique views are fixed thresholds (Phase B constants); cavities that deviate from expected depth may contribute empty or noisy slices.
- Sparsity is flagged but not handled: if most views fail, the canonical mask degrades gracefully to a single-view approximation, which may not improve over Baseline 1 or Phase B.
- No ICP, no volumetric fusion, no SLAM — this is strictly a 2D footprint approach.

## Closing Note

This is an **exploratory** experiment. The canonical world-frame XY representation is NOT fusion in the volumetric/SLAM/TSDF sense. It is a deterministic, geometry-only aggregation of projected depth footprints. Baseline 1 scoring is reused unchanged. No learned model is present. Results are intended to inform whether multi-view aggregation of 2D footprints can improve or maintain matching quality relative to Baseline 1 and Phase B.
