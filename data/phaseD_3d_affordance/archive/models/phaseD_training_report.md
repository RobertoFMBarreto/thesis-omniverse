# Phase D.3/D.4 — 3D-extrusion Affordance Classifier

> **The model predicts a geometric insertion affordance score, not robot control.**

> **Status**: Phase D.3/D.4 (training + evaluation only). No hyperparameter tuning, no feature pruning, no second training pass.

## Objective

Train interpretable supervised classifiers (logistic regression, decision tree) on the Phase D.1/D.2 3D-extrusion affordance dataset, and evaluate them under (a) standard train/val/test split and (b) leave-one-family-out (LOFO) cross-validation. Report leakage diagnostics, coefficient interpretability, and MVP-scenario performance.

## Dataset source

- `data/phaseD_3d_affordance/configurations_labelled.csv` (26208 rows)
- `data/phaseD_3d_affordance/dataset_summary.json`

## Feature list

**20 identity-free features used**:
- `candidate_rotation_deg`
- `piece_area_mm2`
- `piece_perimeter_mm`
- `piece_compactness`
- `piece_height_mm`
- `piece_volume_mm3`
- `piece_bbox_aspect_ratio`
- `cavity_area_mm2`
- `cavity_perimeter_mm`
- `cavity_compactness`
- `cavity_depth_mm`
- `cavity_volume_mm3`
- `cavity_bbox_aspect_ratio`
- `area_ratio`
- `volume_ratio`
- `depth_compatibility_mm`
- `bbox_aspect_diff`
- `compactness_diff`
- `iou`
- `lateral_clearance_proxy_mm2`

## Excluded columns

Identifier and diagnostic columns are excluded from the model:
- `cavity_id`
- `cavity_source`
- `config_id`
- `diag_c_area_px`
- `diag_inside_ratio_raw`
- `diag_label_reason`
- `diag_outside_ratio_raw`
- `diag_p_area_px`
- `heldout_family_fold`
- `is_mvp`
- `piece_id`
- `shape_family`
- `split`

## Leakage check

Pearson correlation threshold for leakage warning: |ρ| > **0.95**.

No feature exceeds the threshold. ✅

Top-5 features by |Pearson ρ| with the label (informational):
- `depth_compatibility_mm`: ρ = +0.3549
- `cavity_volume_mm3`: ρ = +0.3144
- `iou`: ρ = +0.2834
- `cavity_depth_mm`: ρ = +0.2672
- `cavity_compactness`: ρ = +0.2274

## Models trained

### `logreg`

- **type**: `LogisticRegression`
- **C**: `1.0`
- **class_weight**: `balanced`
- **max_iter**: `5000`
- **scaler**: `StandardScaler`

### `tree`

- **type**: `DecisionTreeClassifier`
- **max_depth**: `4`
- **class_weight**: `balanced`
- **random_state**: `0`

## Standard split metrics

| model | split | n | accuracy | precision | recall | F1 | ROC-AUC | pos_rate_pred | degenerate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| logreg | train | 18411 | 0.9605 | 0.7142 | 0.9883 | 0.8292 | 0.9945 | 0.1344 | no |
| logreg | val | 3912 | 0.9652 | 0.7412 | 0.9895 | 0.8475 | 0.9959 | 0.1304 | no |
| logreg | test | 3885 | 0.9627 | 0.7292 | 0.9948 | 0.8415 | 0.9951 | 0.1359 | no |
| tree | train | 18411 | 0.9630 | 0.7239 | 1.0000 | 0.8398 | 0.9957 | 0.1342 | no |
| tree | val | 3912 | 0.9614 | 0.7175 | 0.9974 | 0.8346 | 0.9946 | 0.1357 | no |
| tree | test | 3885 | 0.9593 | 0.7101 | 1.0000 | 0.8305 | 0.9959 | 0.1403 | no |

## Leave-one-family-out (LOFO) results

Trained on all PROCEDURAL rows except the held-out family; tested on procedural rows of the held-out family. MVP rows are kept out of LOFO training/test to avoid mixing real-data hold-in with procedural OOD evaluation; MVP performance is reported separately below.

| held-out family | n_train | n_test | n_pos_test | model | accuracy | precision | recall | F1 | ROC-AUC | pos_rate_pred | extreme_imbalance |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| convex_irregular_polygon | 20160 | 5040 | 169 | logreg | 0.9812 | 0.6652 | 0.8817 | 0.7583 | 0.9919 | 0.0444 | no |
| convex_irregular_polygon | 20160 | 5040 | 169 | tree | 0.9736 | 0.5652 | 0.9231 | 0.7011 | 0.9933 | 0.0548 | no |
| ellipse | 20160 | 5040 | 718 | logreg | 0.9115 | 0.6181 | 0.9916 | 0.7615 | 0.9951 | 0.2286 | no |
| ellipse | 20160 | 5040 | 718 | tree | 0.9575 | 0.8000 | 0.9359 | 0.8626 | 0.9612 | 0.1667 | no |
| rectangle | 20160 | 5040 | 438 | logreg | 0.9371 | 0.6003 | 0.8265 | 0.6955 | 0.9746 | 0.1196 | no |
| rectangle | 20160 | 5040 | 438 | tree | 0.9817 | 0.8650 | 0.9361 | 0.8991 | 0.9823 | 0.0940 | no |
| regular_polygon | 20160 | 5040 | 1051 | logreg | 0.9369 | 0.7861 | 0.9581 | 0.8636 | 0.9903 | 0.2542 | no |
| regular_polygon | 20160 | 5040 | 1051 | tree | 0.9800 | 0.9123 | 1.0000 | 0.9542 | 0.9903 | 0.2286 | no |
| rounded_rectangle | 20160 | 5040 | 128 | logreg | 0.9643 | 0.3984 | 0.7969 | 0.5312 | 0.9822 | 0.0508 | no |
| rounded_rectangle | 20160 | 5040 | 128 | tree | 0.9591 | 0.3804 | 0.9688 | 0.5463 | 0.9949 | 0.0647 | no |

## MVP scenario evaluation

Trained on ALL procedural rows; tested on MVP rows.

| model | n | accuracy | precision | recall | F1 | ROC-AUC | pos_rate_true | pos_rate_pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg | 1008 | 0.9692 | 0.6310 | 1.0000 | 0.7737 | 1.0000 | 0.0526 | 0.0833 |
| tree | 1008 | 0.9960 | 0.9298 | 1.0000 | 0.9636 | 0.9979 | 0.0526 | 0.0565 |

## Logistic regression — top coefficients (standardised)

Intercept (standardised): **-21.4148**

| rank | feature | coef (standardised) | |coef| | feature_mean | feature_scale |
|---:|---|---:|---:|---:|---:|
| 1 | `volume_ratio` | -15.5600 | 15.5600 | 1.4650 | 1.4009 |
| 2 | `area_ratio` | -13.5701 | 13.5701 | 1.1423 | 0.7616 |
| 3 | `iou` | +7.8630 | 7.8630 | 0.6060 | 0.1996 |
| 4 | `piece_perimeter_mm` | -1.8937 | 1.8937 | 139.6159 | 35.6282 |
| 5 | `cavity_volume_mm3` | +1.6003 | 1.6003 | 94112.1929 | 66703.6552 |
| 6 | `cavity_perimeter_mm` | +1.4054 | 1.4054 | 139.5828 | 38.3363 |
| 7 | `piece_area_mm2` | -1.0960 | 1.0960 | 1133.3741 | 638.6322 |
| 8 | `piece_height_mm` | -1.0170 | 1.0170 | 95.4737 | 26.5443 |
| 9 | `depth_compatibility_mm` | +0.8625 | 0.8625 | -13.2956 | 34.1771 |
| 10 | `cavity_compactness` | +0.6347 | 0.6347 | 0.7040 | 0.2243 |
| 11 | `cavity_area_mm2` | -0.5896 | 0.5896 | 1138.2588 | 681.2618 |
| 12 | `lateral_clearance_proxy_mm2` | +0.5624 | 0.5624 | 4.8847 | 530.3796 |
| 13 | `cavity_bbox_aspect_ratio` | -0.5296 | 0.5296 | 1.3640 | 0.4158 |
| 14 | `compactness_diff` | -0.3304 | 0.3304 | 0.0724 | 0.1488 |
| 15 | `piece_compactness` | -0.2412 | 0.2412 | 0.7073 | 0.2241 |
| 16 | `bbox_aspect_diff` | -0.1929 | 0.1929 | 0.1140 | 0.3046 |
| 17 | `cavity_depth_mm` | +0.0961 | 0.0961 | 82.1781 | 25.8144 |
| 18 | `piece_bbox_aspect_ratio` | +0.0607 | 0.0607 | 1.3698 | 0.4144 |
| 19 | `candidate_rotation_deg` | +0.0375 | 0.0375 | 175.3001 | 103.9448 |
| 20 | `piece_volume_mm3` | +0.0167 | 0.0167 | 109359.0815 | 73148.2579 |

Coefficient interpretation: positive coefficient → feature increase pushes affordance probability toward 1; negative → toward 0. Magnitudes are comparable across features because the input was standardised.

## Decision tree — feature importances

| rank | feature | importance |
|---:|---|---:|
| 1 | `depth_compatibility_mm` | 0.6175 |
| 2 | `area_ratio` | 0.2318 |
| 3 | `iou` | 0.1311 |
| 4 | `lateral_clearance_proxy_mm2` | 0.0169 |
| 5 | `piece_perimeter_mm` | 0.0024 |
| 6 | `bbox_aspect_diff` | 0.0004 |
| 7 | `candidate_rotation_deg` | 0.0000 |
| 8 | `piece_area_mm2` | 0.0000 |
| 9 | `piece_compactness` | 0.0000 |
| 10 | `piece_height_mm` | 0.0000 |
| 11 | `piece_volume_mm3` | 0.0000 |
| 12 | `piece_bbox_aspect_ratio` | 0.0000 |
| 13 | `cavity_area_mm2` | 0.0000 |
| 14 | `cavity_perimeter_mm` | 0.0000 |
| 15 | `cavity_compactness` | 0.0000 |
| 16 | `cavity_depth_mm` | 0.0000 |
| 17 | `cavity_volume_mm3` | 0.0000 |
| 18 | `cavity_bbox_aspect_ratio` | 0.0000 |
| 19 | `volume_ratio` | 0.0000 |
| 20 | `compactness_diff` | 0.0000 |

## Limitations

- Synthetic dataset only. Labels generated by raster geometry; not physical contact.
- Convex prismatic shapes only (extrusion assumption).
- No XY offset in the dataset; rotations only.
- C, max_iter, max_depth fixed; no hyperparameter tuning in this run.
- LOFO trains on 4 families and tests on the 1 held-out family; if a held-out family has extreme positive/negative imbalance, F1 may be misleading and is flagged.
- The model predicts a geometric affordance score; transfer to non-prismatic shapes, real captures, or robot execution is out of scope.

## Closing note

This phase trains an interpretable classifier on identity-free geometric features to predict insertion **affordance**. It does NOT learn robot control, grasping, force feedback, or motion planning. Downstream insertion remains the deterministic fixed primitive defined in the Phase D design.
