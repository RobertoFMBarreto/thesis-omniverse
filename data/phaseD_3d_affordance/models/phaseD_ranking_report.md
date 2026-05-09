# Phase D.5 — Affordance Ranking Evaluation

> **The ranking selects an insertion location by learned geometric affordance score; it does not control the robot.**

## Objective

Given a piece and a set of candidate cavities, rank the cavities by predicted affordance score and report top-1 / top-2 feasibility, MRR, mean rank of the first feasible cavity, and rank margin. Performed for both Phase D.3 models (logistic regression, decision tree) without any retraining beyond reproducing the exact Phase D.3 setup (C=1.0, max_depth=4, class_weight='balanced').

## Ranking procedure

For each (piece_id, cavity_id) group:
- score = max over rotations of `predict_proba(class=1)`,
- best rotation = argmax over rotations,
- ground truth feasibility = any rotation has `label=1`.

Per piece, cavities are ranked by score descending. Metrics:
- top-1 feasible accuracy: rank-1 cavity is feasible in ground truth;
- top-2 feasible accuracy: a feasible cavity is among ranks 1 or 2;
- MRR: 1 / rank of first feasible cavity, averaged;
- mean rank of first feasible cavity;
- mean rank margin = score(rank-1) − score(rank-2).

Pieces with **no feasible cavity in ground truth** are excluded from accuracy/MRR/margin aggregations and reported separately as `n_pieces_with_no_feasible_truth`.

## Ranking metrics — primary scopes

| model | scope | n_pieces | with_feasible | no_feasible | top-1 | top-2 | MRR | mean_rank | mean_margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| logreg | test_split | 104 | 93 | 11 | 0.9785 | 0.9892 | 0.9875 | 1.0323 | 0.1640 |
| logreg | mvp_scenario | 4 | 4 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0090 |
| logreg | all_procedural | 100 | 100 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0185 |
| tree | test_split | 104 | 93 | 11 | 0.9462 | 0.9892 | 0.9713 | 1.0645 | 0.1504 |
| tree | mvp_scenario | 4 | 4 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 |
| tree | all_procedural | 100 | 100 | 0 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0199 |

## Per-family ranking metrics

| model | family | n_pieces | with_feasible | top-1 | top-2 | MRR | mean_rank | mean_margin |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| logreg | convex_irregular_polygon | 20 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0229 |
| logreg | ellipse | 21 | 21 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0145 |
| logreg | rectangle | 22 | 22 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0129 |
| logreg | regular_polygon | 21 | 21 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0046 |
| logreg | rounded_rectangle | 20 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0371 |
| tree | convex_irregular_polygon | 20 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0482 |
| tree | ellipse | 21 | 21 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0437 |
| tree | rectangle | 22 | 22 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0024 |
| tree | regular_polygon | 21 | 21 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0022 |
| tree | rounded_rectangle | 20 | 20 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0002 |

## Worst failure cases

Top 5 pieces with the WORST first-feasible-rank under each model (scope: `all_procedural`).

### `logreg`

None — no piece had its first feasible cavity below rank 1.

### `tree`

None — no piece had its first feasible cavity below rank 1.

## Limitations

- Ranking uses Phase D.3 models without any tuning; hyperparameters are C=1.0 (logreg) and max_depth=4 (tree).
- Cavity pool per piece is fixed at 7 (1 matching + 6 mismatched recipes) from Phase D.1/D.2; ranking accuracy depends on this construction.
- Synthetic dataset only; convex prismatic shapes; no XY offsets in the candidate space.
- Pieces with no feasible ground-truth cavity are excluded from accuracy/MRR but reported in the count column.
- The MVP scenario reuses the dataset's MVP rows; this is an in-distribution evaluation for cavities derived from MVP pieces, NOT a true MVP-vs-board insertion scene (the board scene was Baseline 1's evaluation; reusing it requires a separate inference script outside Phase D's training scope).

## Closing note

Ranking outputs a top-1 cavity per piece and a rank margin. These are perception-side affordance signals only; insertion execution, grasp planning, and robot control are out of scope. The downstream insertion is the fixed kinematic primitive defined in the Phase D design.
