# Phase E.2 — Siamese CNN SDF Embedding (training + evaluation)

> **This model learns a geometric compatibility embedding from SDF footprints; it does not learn robot control or visual perception.**

> Status: Phase E.2 (single training run). No hyperparameter tuning, no architecture search, no LOFO, no robustness tests.

## Objective

Test whether a minimal siamese CNN can learn a useful geometric compatibility representation from 128×128 signed distance fields of piece and cavity footprints, using the Phase E.1 dataset and Phase D.7 partial-insertion labels.

## Dataset source

- `data/phaseE_learned_embeddings/phaseE_sdf_pairs.npz` (int8-quantised SDFs)
- `data/phaseE_learned_embeddings/phaseE_pairs_metadata.csv` (split / family / piece_id / cavity_id / rotation)

## Model architecture

Shared encoder (1-channel SDF in ∈ [-1, 1]):
```
Conv2d(1,16,3,p=1) -> ReLU -> MaxPool2d(2)   # (16,64,64)
Conv2d(16,32,3,p=1) -> ReLU -> MaxPool2d(2)  # (32,32,32)
Conv2d(32,64,3,p=1) -> ReLU -> MaxPool2d(2)  # (64,16,16)
AdaptiveAvgPool2d(1) -> flatten              # (64,)
Linear(64 -> 64)            # embedding
L2-normalise embedding
```

Compatibility head:
```
features = concat(|e_p - e_c|, e_p * e_c, cosine(e_p, e_c))
Linear(2D + 1 -> 32) -> ReLU -> Linear(32 -> 1) -> logit
```

- Total parameters: **31649**
- Loss: BCEWithLogitsLoss with `pos_weight = 3.8171` (class-imbalance correction).

## Training configuration

- Seed: 0
- Batch size: 256
- Optimizer: Adam, lr = 0.001
- Max epochs: 20; early-stop on val F1, patience 3
- SDF dequantise: `sdf_mm = sdf_int8 / 6.3500`
- SDF normalisation to [-1, 1]: divide by 20.0 mm
- Device: `cuda`
- **Best epoch**: 6 (val_F1 = 0.5491)

## Standard split metrics (best model)

| split | n | acc | prec | recall | F1 | AUC | pos_rate_pred | degenerate |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| train | 18411 | 0.7786 | 0.4732 | 0.5874 | 0.5242 | — | 0.2577 | no |
| val | 3912 | 0.7843 | 0.4985 | 0.6112 | 0.5491 | — | 0.2635 | no |
| test | 3885 | 0.7786 | 0.4517 | 0.5877 | 0.5108 | — | 0.2559 | no |

## Per-family test metrics

| family | n | acc | prec | recall | F1 | AUC |
|---|---:|---:|---:|---:|---:|---:|
| `convex_irregular_polygon` | 720 | 0.8903 | 0.3239 | 0.4259 | 0.3680 | — |
| `ellipse` | 804 | 0.6654 | 0.4461 | 0.7876 | 0.5696 | — |
| `rectangle` | 804 | 0.7363 | 0.2529 | 0.3492 | 0.2933 | — |
| `regular_polygon` | 821 | 0.7150 | 0.5714 | 0.6690 | 0.6164 | — |
| `rounded_rectangle` | 736 | 0.9103 | 0.7619 | 0.2078 | 0.3265 | — |

## Ranking metrics

Ranking groups by `piece_id`, takes max-over-rotations per cavity, ranks cavities, and reports.

| scope | n_pieces | with_feasible | top-1 | MRR | mean_rank | mean_margin |
|---|---:|---:|---:|---:|---:|---:|
| test_split | 104 | 93 | 0.5054 | 0.6812 | 2.1290 | 0.1242 |
| mvp_scenario | 4 | 4 | 0.0000 | 0.3750 | 2.7500 | 0.0344 |

## Comparison vs Phase D hand-crafted models

| metric | Phase D logreg (test) | Phase D tree (test) | Phase E siamese (test) |
|---|---:|---:|---:|
| F1 | 0.829 | 0.871 | 0.5108 |

Phase D values quoted from `data/phaseD_3d_affordance/models/phaseD_training_results.json`.

## Limitations

- Single training run; no hyperparameter tuning.
- LOFO not evaluated in this turn (Phase E.3 if pursued).
- Robustness perturbations not applied (Phase E.4 if pursued).
- Synthetic dataset only; convex prismatic shapes; rotations only (no XY offset).
- Perception pipeline frozen; embeddings are learned; perception is not.
- The model learns a geometric compatibility decision boundary, not robot control or insertion physics.

## Closing note

The siamese CNN learns a 64-D embedding from SDF footprints and predicts compatibility from a small head over `(|diff|, product, cosine)` of paired embeddings. It does NOT learn robot control, insertion execution, or visual perception. Whether the embedding generalises across shape families and degrades gracefully under perturbation are separate experiments (Phase E.3, E.4).
