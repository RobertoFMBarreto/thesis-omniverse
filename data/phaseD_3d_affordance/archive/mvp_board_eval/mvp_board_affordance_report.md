# Phase D.6 — MVP-Board Affordance Ranking

> **Evaluates the trained Phase D affordance model on the actual Baseline 1 board scenario.** The model predicts a geometric insertion affordance score; it does NOT control the robot.

## Objective

For each MVP piece (rectangle, square, circle, triangle), rank the four actual board cavities (cavity_00..03) by predicted affordance score and compare against the Baseline 1 deterministic reference.

## Feature extraction source

- **Pieces**: `data/pieces_detected/<piece>/piece_pointcloud.npy` + `piece_metadata.json` (`piece_height_median_m`).
- **Cavities**: `data/cavities_detected/<cavity>/cavity_opening_pointcloud.npy` + `cavity_metadata.json` (`z_depth_median_m`).
- Each captured point cloud is reduced to a **convex-hull polygon** (consistent with the analytical polygons used at training).

## Model used

Both Phase D models are reproduced by re-fitting on the same Phase D dataset with the same hyperparameters. **No tuning, no feature changes.**
- `logreg`: C=`1.0`, max_iter=`5000`, class_weight=`balanced`, scaler=`StandardScaler`
- `tree`: max_depth=`4`, class_weight=`balanced`, random_state=`0`

## Ranking procedure

For each (piece, cavity): sweep 36 rotations (0°, 10°, …, 350°), build the same 20-feature vector used at training, predict the affordance probability, and take **cavity_score = max over rotations**. Per piece, sort cavities by score descending; rank-1 is the predicted insertion target. dx = dy = 0 (no XY offset).

## Piece descriptors

| piece | hull_pts | area_mm² | perim_mm | compact | height_mm | volume_mm³ | bbox_ar |
|---|---:|---:|---:|---:|---:|---:|---:|
| `rectangle` | n/a | 3690.3 | 243.1 | 0.784 | 104.5 | 385641.0 | 1.495 |
| `square` | n/a | 2477.2 | 196.6 | 0.806 | 104.5 | 258865.7 | 1.000 |
| `circle` | n/a | 1915.1 | 155.5 | 0.995 | 104.5 | 200127.3 | 1.000 |
| `triangle` | n/a | 1239.5 | 159.4 | 0.613 | 104.5 | 129525.6 | 1.000 |

## Cavity descriptors

| cavity | area_mm² | perim_mm | compact | depth_mm | volume_mm³ | bbox_ar |
|---|---:|---:|---:|---:|---:|---:|
| `cavity_00` | 3730.9 | 248.1 | 0.762 | 14.8 | 55221.6 | 1.463 |
| `cavity_01` | 1223.4 | 157.8 | 0.617 | 16.9 | 20699.2 | 1.026 |
| `cavity_02` | 2487.8 | 199.5 | 0.785 | 75.0 | 186583.4 | 1.025 |
| `cavity_03` | 1961.4 | 157.6 | 0.993 | 16.4 | 32173.2 | 1.025 |

## Per-piece rankings — logistic regression

| piece | rank | cavity | score | best_rotation | reference | match |
|---|---:|---|---:|---:|---|---|
| `rectangle` | 1 | `cavity_00` | 0.0000 | 0° | `cavity_00` | ✓ |
| `rectangle` | 2 | `cavity_01` | 0.0000 | 310° |  |  |
| `rectangle` | 3 | `cavity_02` | 0.0000 | 90° |  |  |
| `rectangle` | 4 | `cavity_03` | 0.0000 | 310° |  |  |
| | | **margin (rank1 − rank2)** | **0.0000** | | | |
| `square` | 1 | `cavity_02` | 0.0039 | 270° | `cavity_02` | ✓ |
| `square` | 2 | `cavity_00` | 0.0000 | 180° |  |  |
| `square` | 3 | `cavity_01` | 0.0000 | 230° |  |  |
| `square` | 4 | `cavity_03` | 0.0000 | 320° |  |  |
| | | **margin (rank1 − rank2)** | **0.0039** | | | |
| `circle` | 1 | `cavity_02` | 0.0508 | 350° |  |  |
| `circle` | 2 | `cavity_00` | 0.0000 | 350° |  |  |
| `circle` | 3 | `cavity_01` | 0.0000 | 240° |  |  |
| `circle` | 4 | `cavity_03` | 0.0000 | 300° | `cavity_03` |  |
| | | **margin (rank1 − rank2)** | **0.0508** | | | |
| `triangle` | 1 | `cavity_02` | 0.0552 | 130° |  |  |
| `triangle` | 2 | `cavity_00` | 0.0000 | 310° |  |  |
| `triangle` | 3 | `cavity_01` | 0.0000 | 0° | `cavity_01` |  |
| `triangle` | 4 | `cavity_03` | 0.0000 | 340° |  |  |
| | | **margin (rank1 − rank2)** | **0.0552** | | | |

## Per-piece rankings — decision tree

| piece | rank | cavity | score | best_rotation | reference | match |
|---|---:|---|---:|---:|---|---|
| `rectangle` | 1 | `cavity_00` | 0.0000 | 0° | `cavity_00` | ✓ |
| `rectangle` | 2 | `cavity_01` | 0.0000 | 0° |  |  |
| `rectangle` | 3 | `cavity_02` | 0.0000 | 0° |  |  |
| `rectangle` | 4 | `cavity_03` | 0.0000 | 0° |  |  |
| | | **margin (rank1 − rank2)** | **0.0000** | | | |
| `square` | 1 | `cavity_00` | 0.0000 | 0° |  |  |
| `square` | 2 | `cavity_01` | 0.0000 | 0° |  |  |
| `square` | 3 | `cavity_02` | 0.0000 | 0° | `cavity_02` |  |
| `square` | 4 | `cavity_03` | 0.0000 | 0° |  |  |
| | | **margin (rank1 − rank2)** | **0.0000** | | | |
| `circle` | 1 | `cavity_00` | 0.0000 | 0° |  |  |
| `circle` | 2 | `cavity_01` | 0.0000 | 0° |  |  |
| `circle` | 3 | `cavity_02` | 0.0000 | 0° |  |  |
| `circle` | 4 | `cavity_03` | 0.0000 | 0° | `cavity_03` |  |
| | | **margin (rank1 − rank2)** | **0.0000** | | | |
| `triangle` | 1 | `cavity_00` | 0.0000 | 0° |  |  |
| `triangle` | 2 | `cavity_01` | 0.0000 | 0° | `cavity_01` |  |
| `triangle` | 3 | `cavity_02` | 0.0000 | 0° |  |  |
| `triangle` | 4 | `cavity_03` | 0.0000 | 0° |  |  |
| | | **margin (rank1 − rank2)** | **0.0000** | | | |

## Comparison vs Baseline 1 deterministic reference

Reference mapping: `rectangle → cavity_00`, `square → cavity_02`, `circle → cavity_03`, `triangle → cavity_01`.

| piece | reference | logreg rank-1 | logreg ✓ | tree rank-1 | tree ✓ |
|---|---|---|---|---|---|
| `rectangle` | `cavity_00` | `cavity_00` (rank of ref = 1) | ✓ | `cavity_00` (rank of ref = 1) | ✓ |
| `square` | `cavity_02` | `cavity_02` (rank of ref = 1) | ✓ | `cavity_00` (rank of ref = 3) | ✗ |
| `circle` | `cavity_03` | `cavity_02` (rank of ref = 4) | ✗ | `cavity_00` (rank of ref = 4) | ✗ |
| `triangle` | `cavity_01` | `cavity_02` (rank of ref = 3) | ✗ | `cavity_00` (rank of ref = 2) | ✗ |

**Top-1 accuracy on the four MVP pieces**: logreg = 50.0% (2/4); tree = 25.0% (1/4).

## Limitations

- The model was trained ONLY on procedurally generated convex prismatic shapes plus four MVP-derived **procedurally constructed** cavities; the actual Baseline 1 board cavities (`cavity_00..03`) were NOT in the training set.
- Captured point clouds are sparse; descriptors are computed from the convex hull. For non-convex captured outlines this would lose detail (the MVP pieces are convex by design).
- Cavity `depth_mm` comes from `z_depth_median_m` (Isaac Sim depth-capture estimate), not the CAD nominal 75 mm.
- Rotation sweep is 36 angles at 10° steps; no XY offset.
- The ranking is over only 4 candidates; top-2 accuracy is not informative (it is trivially 1.0 for any non-degenerate ranker that finds the correct cavity in the first two).
- The ranking selects an insertion location by learned geometric affordance score; it does not control the robot.

## Closing note

The ranking outputs a top-1 cavity per MVP piece and a rank margin. These are perception-side affordance signals only; insertion execution, grasp planning, and robot control are out of scope.
