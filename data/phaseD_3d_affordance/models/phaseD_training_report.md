# Phase D.3/D.4 — 3D-extrusion Affordance Classifier

> **The model predicts a geometric insertion affordance score, not robot control.**

> **Status**: Phase D.3/D.4 (training + evaluation only). No hyperparameter tuning, no feature pruning, no second training pass.

## Objective

Train interpretable supervised classifiers (logistic regression, decision tree) on the Phase D.1/D.2 3D-extrusion affordance dataset, and evaluate them under (a) standard train/val/test split and (b) leave-one-family-out (LOFO) cross-validation. Report leakage diagnostics, coefficient interpretability, and MVP-scenario performance.

## Dataset source

- `data/phaseD_3d_affordance/configurations_labelled.csv` (26208 rows)
- `data/phaseD_3d_affordance/dataset_summary.json`

## Feature list

**19 identity-free features used**:
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
- `depth_offset_mm`
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
- `diag_insertion_required_mm`
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
- `lateral_clearance_proxy_mm2`: ρ = +0.4416
- `cavity_volume_mm3`: ρ = +0.3947
- `cavity_area_mm2`: ρ = +0.3560
- `cavity_compactness`: ρ = +0.3259
- `iou`: ρ = +0.2981

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
| logreg | train | 18411 | 0.9209 | 0.7355 | 0.9662 | 0.8352 | 0.9868 | 0.2727 | no |
| logreg | val | 3912 | 0.9325 | 0.7694 | 0.9798 | 0.8619 | 0.9887 | 0.2738 | no |
| logreg | test | 3885 | 0.9205 | 0.7190 | 0.9777 | 0.8286 | 0.9866 | 0.2674 | no |
| tree | train | 18411 | 0.9491 | 0.8052 | 0.9958 | 0.8904 | 0.9796 | 0.2567 | no |
| tree | val | 3912 | 0.9542 | 0.8258 | 0.9976 | 0.9036 | 0.9776 | 0.2597 | no |
| tree | test | 3885 | 0.9418 | 0.7739 | 0.9948 | 0.8706 | 0.9755 | 0.2528 | no |

## Leave-one-family-out (LOFO) results

Trained on all PROCEDURAL rows except the held-out family; tested on procedural rows of the held-out family. MVP rows are kept out of LOFO training/test to avoid mixing real-data hold-in with procedural OOD evaluation; MVP performance is reported separately below.

| held-out family | n_train | n_test | n_pos_test | model | accuracy | precision | recall | F1 | ROC-AUC | pos_rate_pred | extreme_imbalance |
|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| convex_irregular_polygon | 20160 | 5040 | 340 | logreg | 0.9417 | 0.5391 | 0.9324 | 0.6832 | 0.9840 | 0.1167 | no |
| convex_irregular_polygon | 20160 | 5040 | 340 | tree | 0.9363 | 0.5152 | 0.9441 | 0.6667 | 0.9452 | 0.1236 | no |
| ellipse | 20160 | 5040 | 1580 | logreg | 0.8726 | 0.7250 | 0.9563 | 0.8248 | 0.9678 | 0.4135 | no |
| ellipse | 20160 | 5040 | 1580 | tree | 0.9266 | 0.8103 | 1.0000 | 0.8952 | 0.9523 | 0.3869 | no |
| rectangle | 20160 | 5040 | 872 | logreg | 0.9264 | 0.7147 | 0.9564 | 0.8180 | 0.9867 | 0.2315 | no |
| rectangle | 20160 | 5040 | 872 | tree | 0.9401 | 0.7902 | 0.8899 | 0.8371 | 0.9602 | 0.1948 | no |
| regular_polygon | 20160 | 5040 | 1858 | logreg | 0.8883 | 0.7797 | 0.9715 | 0.8651 | 0.9779 | 0.4593 | no |
| regular_polygon | 20160 | 5040 | 1858 | tree | 0.9198 | 0.8442 | 0.9596 | 0.8982 | 0.9641 | 0.4190 | no |
| rounded_rectangle | 20160 | 5040 | 644 | logreg | 0.9567 | 0.8227 | 0.8432 | 0.8328 | 0.9882 | 0.1310 | no |
| rounded_rectangle | 20160 | 5040 | 644 | tree | 0.9012 | 0.6637 | 0.4596 | 0.5431 | 0.9543 | 0.0885 | no |

## MVP scenario evaluation

Trained on ALL procedural rows; tested on MVP rows.

| model | n | accuracy | precision | recall | F1 | ROC-AUC | pos_rate_true | pos_rate_pred |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg | 1008 | 0.8343 | 0.4433 | 1.0000 | 0.6143 | 0.9918 | 0.1319 | 0.2976 |
| tree | 1008 | 0.9444 | 0.7037 | 1.0000 | 0.8261 | 0.9897 | 0.1319 | 0.1875 |

## Logistic regression — top coefficients (standardised)

Intercept (standardised): **-12.3507**

| rank | feature | coef (standardised) | |coef| | feature_mean | feature_scale |
|---:|---|---:|---:|---:|---:|
| 1 | `area_ratio` | -16.4921 | 16.4921 | 1.1423 | 0.7616 |
| 2 | `iou` | +6.6288 | 6.6288 | 0.6060 | 0.1996 |
| 3 | `piece_perimeter_mm` | -3.8173 | 3.8173 | 139.6159 | 35.6282 |
| 4 | `cavity_perimeter_mm` | +2.9699 | 2.9699 | 139.5828 | 38.3363 |
| 5 | `piece_compactness` | -1.8142 | 1.8142 | 0.7073 | 0.2241 |
| 6 | `cavity_compactness` | +1.7285 | 1.7285 | 0.7040 | 0.2243 |
| 7 | `depth_offset_mm` | +0.7535 | 0.7535 | -31.5057 | 45.9590 |
| 8 | `cavity_volume_mm3` | +0.7168 | 0.7168 | 54218.2050 | 50304.8806 |
| 9 | `cavity_depth_mm` | +0.6909 | 0.6909 | 48.1270 | 29.8722 |
| 10 | `piece_volume_mm3` | +0.6624 | 0.6624 | 92366.9191 | 74023.6093 |
| 11 | `piece_height_mm` | -0.4022 | 0.4022 | 79.6328 | 34.7892 |
| 12 | `lateral_clearance_proxy_mm2` | +0.3215 | 0.3215 | 4.8847 | 530.3796 |
| 13 | `piece_area_mm2` | -0.2586 | 0.2586 | 1133.3741 | 638.6322 |
| 14 | `bbox_aspect_diff` | -0.2549 | 0.2549 | 0.1140 | 0.3046 |
| 15 | `compactness_diff` | -0.1922 | 0.1922 | 0.0724 | 0.1488 |
| 16 | `piece_bbox_aspect_ratio` | -0.1178 | 0.1178 | 1.3698 | 0.4144 |
| 17 | `cavity_bbox_aspect_ratio` | -0.0406 | 0.0406 | 1.3640 | 0.4158 |
| 18 | `candidate_rotation_deg` | +0.0163 | 0.0163 | 175.3001 | 103.9448 |
| 19 | `cavity_area_mm2` | +0.0078 | 0.0078 | 1138.2588 | 681.2618 |

Coefficient interpretation: positive coefficient → feature increase pushes affordance probability toward 1; negative → toward 0. Magnitudes are comparable across features because the input was standardised.

## Decision tree — feature importances

| rank | feature | importance |
|---:|---|---:|
| 1 | `area_ratio` | 0.4853 |
| 2 | `iou` | 0.3426 |
| 3 | `cavity_depth_mm` | 0.1332 |
| 4 | `bbox_aspect_diff` | 0.0129 |
| 5 | `depth_offset_mm` | 0.0081 |
| 6 | `lateral_clearance_proxy_mm2` | 0.0076 |
| 7 | `compactness_diff` | 0.0058 |
| 8 | `piece_height_mm` | 0.0045 |
| 9 | `candidate_rotation_deg` | 0.0000 |
| 10 | `piece_area_mm2` | 0.0000 |
| 11 | `piece_perimeter_mm` | 0.0000 |
| 12 | `piece_compactness` | 0.0000 |
| 13 | `piece_volume_mm3` | 0.0000 |
| 14 | `piece_bbox_aspect_ratio` | 0.0000 |
| 15 | `cavity_area_mm2` | 0.0000 |
| 16 | `cavity_perimeter_mm` | 0.0000 |
| 17 | `cavity_compactness` | 0.0000 |
| 18 | `cavity_volume_mm3` | 0.0000 |
| 19 | `cavity_bbox_aspect_ratio` | 0.0000 |

## Limitations

- Synthetic dataset only. Labels generated by raster geometry; not physical contact.
- Convex prismatic shapes only (extrusion assumption).
- No XY offset in the dataset; rotations only.
- C, max_iter, max_depth fixed; no hyperparameter tuning in this run.
- LOFO trains on 4 families and tests on the 1 held-out family; if a held-out family has extreme positive/negative imbalance, F1 may be misleading and is flagged.
- The model predicts a geometric affordance score; transfer to non-prismatic shapes, real captures, or robot execution is out of scope.

## Closing note

This phase trains an interpretable classifier on identity-free geometric features to predict insertion **affordance**. It does NOT learn robot control, grasping, force feedback, or motion planning. Downstream insertion remains the deterministic fixed primitive defined in the Phase D design.
