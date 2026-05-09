# Phase D — 3D-extrusion Affordance Dataset

> **This dataset supports learning a geometric affordance score, not robot control.**

> **Status**: Phase D.1/D.2 (dataset generation only). No model has been trained from this data.

## Objective

Generate a procedural dataset of (piece, cavity, rotation) configurations with deterministic affordance labels, to be consumed in Phase D.3+ by an interpretable classifier (logistic regression / shallow tree). Pieces and cavities are convex prismatic shapes; the affordance label is computed from a deterministic geometric fit rule (lateral feasibility AND depth feasibility).

## Controlled 3D-extrusion assumption

Piece 3D shape ≈ vertical extrusion of the 2D footprint by `piece_height_mm`. Cavity 3D shape ≈ vertical depression of the 2D opening footprint by `cavity_depth_mm`. Valid only for **convex prismatic shapes** and a **fixed vertical insertion direction**. Side-face geometry, undercuts, internal voids, non-prismatic cross-sections, and concave outlines are out of scope.

## Procedural shape families

- `rectangle` — 20 instances
- `ellipse` — 20 instances
- `regular_polygon` — 20 instances
- `convex_irregular_polygon` — 20 instances
- `rounded_rectangle` — 20 instances
- MVP real-data hold-in: 4 pieces (rectangle, square, circle, triangle) with their CAD-nominal heights and cavity depths.

## Affordance label rule (deterministic, partial-insertion)

**Phase D.7 task definition: partial insertion through an opening, NOT full containment.** A piece may be taller than the cavity is deep; the relevant question is whether the piece cross-section fits through the opening AND the cavity is deep enough to engage the piece by a mechanically meaningful depth.

For each (piece, cavity, rotation) configuration:

- **Lateral feasibility**: `outside_ratio_raw ≤ 0.05` AND `inside_ratio_raw ≥ 0.8`.

- **Required insertion depth**:

  `insertion_required_mm = max(MIN_REQUIRED_INSERTION_MM=5.0 mm, INSERTION_FRACTION=0.25 * piece_height_mm)`

- **Depth feasibility**:

  `cavity_depth_mm ≥ insertion_required_mm − DEPTH_TOLERANCE_MM=0.5 mm`

  AND `cavity_depth_mm ≥ MIN_INSERTION_GUIDANCE_MM=5.0 mm`.

- **Affordance label** = 1 iff BOTH; 0 otherwise.

Thresholds are **fixed operating points**, not free tuning parameters.

## Feature list (identity-free)

**Piece descriptors**: `piece_area_mm2`, `piece_perimeter_mm`, `piece_compactness`, `piece_height_mm`, `piece_volume_mm3`, `piece_bbox_aspect_ratio`.

**Cavity descriptors**: `cavity_area_mm2`, `cavity_perimeter_mm`, `cavity_compactness`, `cavity_depth_mm`, `cavity_volume_mm3`, `cavity_bbox_aspect_ratio`.

**Pair / action descriptors**: `area_ratio`, `depth_offset_mm`, `insertion_required_mm`, `bbox_aspect_diff`, `compactness_diff`, `candidate_rotation_deg`, `iou`, `lateral_clearance_proxy_mm2`.

## Excluded / leakage-prone columns

**Diagnostics ONLY (NOT model features — used for label generation)**: `diag_inside_ratio_raw`, `diag_outside_ratio_raw`, `diag_p_area_px`, `diag_c_area_px`, `diag_label_reason`.

**Identifiers ONLY (NOT model features — tracing / debugging)**: `config_id`, `piece_id`, `cavity_id`, `shape_family`, `is_mvp`, `cavity_source`, `split`, `heldout_family_fold`.

Phase D training MUST exclude these columns from the classifier input.

## Dataset statistics

- Pieces total: **104** (100 procedural + 4 MVP)
- Cavities per piece: 7
- Rotations per (piece, cavity): 36
- Total configurations: **26208**
- Positive labels: **5427** (20.71% positive rate)
- Negative labels: 20781
- Any family with zero positives: **False**

### Per-family breakdown

| family | fold_id | n_configs | n_positive | positive_rate |
|---|---:|---:|---:|---:|
| `rectangle` | 0 | 5544 | 918 | 0.1656 |
| `ellipse` | 1 | 5292 | 1652 | 0.3122 |
| `regular_polygon` | 2 | 5292 | 1873 | 0.3539 |
| `convex_irregular_polygon` | 3 | 5040 | 340 | 0.0675 |
| `rounded_rectangle` | 4 | 5040 | 644 | 0.1278 |

## Limitations

- **Synthetic only**: labels are generated from raster geometry, not physical insertion trials. Any sim-to-real claim is out of scope.
- **Convex prismatic only**: the extrusion assumption breaks for non-prismatic / concave shapes. Star and other stress shapes are deferred.
- **No XY offset** in this first dataset: only rotations are swept. Adding offsets is a future extension; the current design intentionally keeps configurations interpretable.
- **No height/depth measurement noise**: a noise ablation is reserved for a future run, kept separate from the primary dataset.
- **No model trained from this data yet**.

## Closing note

This dataset supports learning a **geometric affordance score, not robot control**. The downstream classifier (Phase D.3) will rank candidate cavities by predicted affordance and output a top-1 cavity per piece, plus a rank margin. Insertion execution, grasp planning, force feedback, and any robotic action remain out of scope.
