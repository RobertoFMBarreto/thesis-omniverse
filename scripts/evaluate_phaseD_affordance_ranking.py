"""
evaluate_phaseD_affordance_ranking.py

Phase D.5 — ranking evaluation for learned 3D-extrusion geometric insertion
affordances.

Per (piece_id, cavity_id):
  cavity_score = max over rotations of model probability of feasibility
  cavity_best_rotation = argmax over rotations
Per piece_id: rank candidate cavities by cavity_score (descending).

Ground truth per (piece_id, cavity_id): True if any rotation has label=1.

Metrics per piece (then aggregated):
  - top-1 feasible accuracy (rank-1 cavity is among feasible cavities)
  - top-2 feasible accuracy
  - mean reciprocal rank (1 / rank of first feasible cavity)
  - mean rank of first feasible cavity
  - rank margin = score(rank-1) - score(rank-2)
Aggregations: standard test split, MVP scenario, per shape_family.

The ranking selects an insertion location by learned geometric affordance
score; it does NOT control the robot. No tuning, no new features, no model
changes.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        "/Users/robertofmbarreto/Documents/Mestrado/tese/code/thesis-omniverse",
    )
)

DATA_DIR = PROJECT_ROOT / "data" / "phaseD_3d_affordance"
CSV_PATH = DATA_DIR / "configurations_labelled.csv"
OUT_DIR  = DATA_DIR / "models"

OUT_RANK_JSON = OUT_DIR / "phaseD_ranking_results.json"
OUT_RANK_MD   = OUT_DIR / "phaseD_ranking_report.md"
OUT_RANK_CSV  = OUT_DIR / "phaseD_ranking_predictions.csv"

# Mirror Phase D.3/D.4 settings exactly
ID_COLS = {
    "config_id", "piece_id", "cavity_id", "shape_family", "is_mvp",
    "cavity_source", "split", "heldout_family_fold",
}
DIAG_COLS = {
    "diag_inside_ratio_raw", "diag_outside_ratio_raw",
    "diag_p_area_px", "diag_c_area_px", "diag_label_reason",
    "diag_insertion_required_mm",
    "label_reason",
}
LABEL_COL = "label"

LR_C        = 1.0
LR_MAX_ITER = 5000
DT_MAX_DEPTH = 4
DT_RANDOM_STATE = 0


# ── CSV helpers ───────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def parse_value(v: str):
    if v == "":
        return None
    if v in ("True", "False"):
        return v == "True"
    try:
        if "." in v or "e" in v or "E" in v:
            return float(v)
        return int(v)
    except ValueError:
        return v


def rows_to_columns(rows: list[dict]) -> dict[str, list]:
    if not rows:
        return {}
    cols = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k, v in r.items():
            cols[k].append(parse_value(v))
    return cols


# ── Models ────────────────────────────────────────────────────────────────────

def fit_logreg(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=LR_C, class_weight="balanced", max_iter=LR_MAX_ITER,
            solver="lbfgs", random_state=0,
        )),
    ]).fit(X_train, y_train)


def fit_tree(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.tree import DecisionTreeClassifier
    return DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH, class_weight="balanced",
        random_state=DT_RANDOM_STATE,
    ).fit(X_train, y_train)


# ── Ranking metrics ───────────────────────────────────────────────────────────

def per_piece_ranking(piece_to_cavities: dict, scope_label: str) -> dict:
    """
    Aggregate per-piece ranking metrics across all pieces in this scope.

    Each piece entry is a list of (cavity_id, cavity_score, is_feasible_truth,
    cavity_best_rotation).
    """
    agg = {
        "scope":    scope_label,
        "n_pieces": len(piece_to_cavities),
        "n_pieces_with_at_least_one_feasible": 0,
        "n_pieces_with_no_feasible_truth":     0,
        "top1_feasible_accuracy":  0.0,
        "top2_feasible_accuracy":  0.0,
        "mean_reciprocal_rank":    0.0,
        "mean_rank_first_feasible": 0.0,
        "mean_rank_margin":        0.0,
        "per_piece":               [],
    }

    if not piece_to_cavities:
        return agg

    n_with_feasible = 0
    n_top1 = 0
    n_top2 = 0
    sum_mrr = 0.0
    sum_rank_first = 0.0
    sum_margin = 0.0

    for piece_id, items in piece_to_cavities.items():
        items_sorted = sorted(items, key=lambda r: -r["cavity_score"])
        for rank, r in enumerate(items_sorted, start=1):
            r["rank"] = rank

        feasible_ranks = [r["rank"] for r in items_sorted if r["is_feasible_truth"]]
        if not feasible_ranks:
            agg["n_pieces_with_no_feasible_truth"] += 1
            agg["per_piece"].append({
                "piece_id":         piece_id,
                "ranking":          [{"cavity_id": r["cavity_id"],
                                       "score": round(r["cavity_score"], 6),
                                       "rank": r["rank"],
                                       "is_feasible_truth": r["is_feasible_truth"]}
                                      for r in items_sorted],
                "first_feasible_rank": None,
                "rank_margin": (
                    round(items_sorted[0]["cavity_score"]
                          - items_sorted[1]["cavity_score"], 6)
                    if len(items_sorted) > 1 else None
                ),
                "skipped_no_feasible_truth": True,
            })
            continue

        n_with_feasible += 1
        first_feasible_rank = feasible_ranks[0]
        if first_feasible_rank == 1:
            n_top1 += 1
        if first_feasible_rank <= 2:
            n_top2 += 1
        sum_mrr += 1.0 / first_feasible_rank
        sum_rank_first += first_feasible_rank
        if len(items_sorted) >= 2:
            margin = items_sorted[0]["cavity_score"] - items_sorted[1]["cavity_score"]
        else:
            margin = 0.0
        sum_margin += margin

        agg["per_piece"].append({
            "piece_id":         piece_id,
            "ranking":          [{"cavity_id": r["cavity_id"],
                                   "score": round(r["cavity_score"], 6),
                                   "rank": r["rank"],
                                   "is_feasible_truth": r["is_feasible_truth"]}
                                  for r in items_sorted],
            "first_feasible_rank": first_feasible_rank,
            "rank_margin":       round(margin, 6),
            "skipped_no_feasible_truth": False,
        })

    agg["n_pieces_with_at_least_one_feasible"] = n_with_feasible
    if n_with_feasible > 0:
        agg["top1_feasible_accuracy"]   = round(n_top1 / n_with_feasible, 6)
        agg["top2_feasible_accuracy"]   = round(n_top2 / n_with_feasible, 6)
        agg["mean_reciprocal_rank"]     = round(sum_mrr / n_with_feasible, 6)
        agg["mean_rank_first_feasible"] = round(sum_rank_first / n_with_feasible, 6)
        agg["mean_rank_margin"]         = round(sum_margin / n_with_feasible, 6)

    return agg


def build_piece_to_cavities(probs: np.ndarray, mask: np.ndarray,
                              piece_ids, cavity_ids, rotations,
                              labels) -> dict[str, list[dict]]:
    """
    Group rows under `mask` by (piece_id, cavity_id), aggregate over rotations:
      cavity_score = max prob
      cavity_best_rotation = rotation at argmax
      is_feasible_truth = any(label==1 over rotations)
    Return {piece_id: [list of cavity records]}.
    """
    grouped = defaultdict(lambda: defaultdict(list))
    idx = np.where(mask)[0]
    for i in idx:
        grouped[piece_ids[i]][cavity_ids[i]].append({
            "rot":   int(rotations[i]),
            "prob":  float(probs[i]),
            "label": int(labels[i]),
        })

    out = {}
    for pid, cav_dict in grouped.items():
        cav_list = []
        for cid, items in cav_dict.items():
            best = max(items, key=lambda r: r["prob"])
            is_feasible = any(it["label"] == 1 for it in items)
            cav_list.append({
                "cavity_id":            cid,
                "cavity_score":         best["prob"],
                "cavity_best_rotation": best["rot"],
                "is_feasible_truth":    bool(is_feasible),
                "n_rotations":          len(items),
            })
        out[pid] = cav_list
    return out


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("evaluate_phaseD_affordance_ranking.py")
    print("=" * 70)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"CSV          : {CSV_PATH}")
    print(f"OUT_DIR      : {OUT_DIR}")
    print()

    if not CSV_PATH.exists():
        print(f"[FATAL] dataset CSV not found at {CSV_PATH}")
        sys.exit(1)

    rows = load_csv(CSV_PATH)
    print(f"[load] {len(rows)} rows loaded")
    cols = rows_to_columns(rows)

    all_columns  = list(cols.keys())
    feature_cols = [c for c in all_columns
                    if c not in (ID_COLS | DIAG_COLS | {LABEL_COL})]
    feature_cols = [c for c in feature_cols
                    if all(isinstance(v, (int, float)) and v is not None
                           for v in cols[c])]
    print(f"[features] {len(feature_cols)} features (matching Phase D.3 set)")

    X = np.array([[float(cols[c][i]) for c in feature_cols]
                  for i in range(len(rows))], dtype=np.float64)
    y         = np.array(cols[LABEL_COL], dtype=np.int32)
    splits    = np.array(cols["split"])
    families  = np.array(cols["shape_family"])
    is_mvp    = np.array(cols["is_mvp"], dtype=bool)
    piece_ids = np.array(cols["piece_id"])
    cavity_ids = np.array(cols["cavity_id"])
    rotations = np.array(cols["candidate_rotation_deg"])

    # Train both models on the train split (deterministic; matches Phase D.3)
    train_mask = splits == "train"
    print(f"[train] training logreg + tree on {int(train_mask.sum())} train rows ...")
    lr_model = fit_logreg(X[train_mask], y[train_mask])
    dt_model = fit_tree(X[train_mask], y[train_mask])

    # Predict probabilities for ALL rows (so ranking can be computed across splits)
    print("[predict] predicting probabilities for all rows ...")
    lr_prob_all = lr_model.predict_proba(X)[:, 1]
    dt_prob_all = dt_model.predict_proba(X)[:, 1]

    scopes = {
        "test_split":        splits == "test",
        "mvp_scenario":      is_mvp,
        "all_procedural":    ~is_mvp,
    }
    family_scopes = {f"per_family/{fam}": (families == fam)
                     for fam in sorted(set(families))}

    all_results = {}
    for model_name, probs in [("logreg", lr_prob_all), ("tree", dt_prob_all)]:
        print(f"\n[ranking {model_name}] ...")
        model_results = {}
        for scope_name, mask in {**scopes, **family_scopes}.items():
            piece_dict = build_piece_to_cavities(
                probs, mask, piece_ids, cavity_ids, rotations, y,
            )
            agg = per_piece_ranking(piece_dict, scope_name)
            model_results[scope_name] = agg
            print(f"  [{scope_name:35s}] n_pieces={agg['n_pieces']:4d} "
                  f"with_feasible={agg['n_pieces_with_at_least_one_feasible']:4d} "
                  f"top1={agg['top1_feasible_accuracy']:.4f} "
                  f"top2={agg['top2_feasible_accuracy']:.4f} "
                  f"MRR={agg['mean_reciprocal_rank']:.4f} "
                  f"meanrank={agg['mean_rank_first_feasible']:.4f} "
                  f"margin={agg['mean_rank_margin']:.4f}")
        all_results[model_name] = model_results

    # Worst failure cases per model (top 5 pieces with feasible truth ranked worst)
    worst_cases = {}
    for model_name in ("logreg", "tree"):
        bad = []
        # Use the "all_procedural" scope for failure analysis
        scope = all_results[model_name]["all_procedural"]
        for r in scope["per_piece"]:
            if r.get("first_feasible_rank") is not None and r["first_feasible_rank"] > 1:
                bad.append({
                    "piece_id": r["piece_id"],
                    "first_feasible_rank": r["first_feasible_rank"],
                    "rank_margin": r["rank_margin"],
                    "ranking_top3": r["ranking"][:3],
                })
        bad.sort(key=lambda x: -x["first_feasible_rank"])
        worst_cases[model_name] = bad[:5]

    # Write predictions CSV
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\n[write] {OUT_RANK_CSV}")
    with OUT_RANK_CSV.open("w") as f:
        f.write("model,scope,piece_id,cavity_id,cavity_score,cavity_best_rotation,"
                "rank,is_feasible_truth\n")
        for model_name in ("logreg", "tree"):
            scope = all_results[model_name]["all_procedural"]
            for piece_block in scope["per_piece"]:
                for r in piece_block["ranking"]:
                    f.write(f"{model_name},all_procedural,{piece_block['piece_id']},"
                            f"{r['cavity_id']},{r['score']},{0},{r['rank']},"
                            f"{r['is_feasible_truth']}\n")
            scope_mvp = all_results[model_name]["mvp_scenario"]
            for piece_block in scope_mvp["per_piece"]:
                for r in piece_block["ranking"]:
                    f.write(f"{model_name},mvp_scenario,{piece_block['piece_id']},"
                            f"{r['cavity_id']},{r['score']},{0},{r['rank']},"
                            f"{r['is_feasible_truth']}\n")

    # Write JSON
    payload = {
        "schema_version": 1,
        "script_name":    "evaluate_phaseD_affordance_ranking.py",
        "phase":          "Phase D.5 — affordance-based cavity ranking",
        "phase_note": (
            "Ranking selects an insertion location by learned geometric "
            "affordance score; it does NOT control the robot. Per (piece, "
            "cavity) the score is max over rotations. No model retraining "
            "beyond reproducing Phase D.3 hyperparameters."
        ),
        "timestamp_utc":  datetime.now(timezone.utc).isoformat(),
        "feature_columns_used": feature_cols,
        "models": {
            "logreg": {"C": LR_C, "max_iter": LR_MAX_ITER,
                        "class_weight": "balanced", "scaler": "StandardScaler"},
            "tree":   {"max_depth": DT_MAX_DEPTH,
                        "class_weight": "balanced",
                        "random_state": DT_RANDOM_STATE},
        },
        "ranking_results": all_results,
        "worst_failure_cases": worst_cases,
    }
    OUT_RANK_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[write] {OUT_RANK_JSON}")

    write_report(payload, OUT_RANK_MD)
    print(f"[write] {OUT_RANK_MD}")
    print("[done].")


def write_report(payload: dict, path: Path) -> None:
    lines = []
    lines.append("# Phase D.5 — Affordance Ranking Evaluation")
    lines.append("")
    lines.append("> **The ranking selects an insertion location by learned "
                 "geometric affordance score; it does not control the robot.**")
    lines.append("")
    lines.append("## Objective")
    lines.append("")
    lines.append("Given a piece and a set of candidate cavities, rank the "
                 "cavities by predicted affordance score and report top-1 / "
                 "top-2 feasibility, MRR, mean rank of the first feasible "
                 "cavity, and rank margin. Performed for both Phase D.3 "
                 "models (logistic regression, decision tree) without any "
                 "retraining beyond reproducing the exact Phase D.3 setup "
                 "(C=1.0, max_depth=4, class_weight='balanced').")
    lines.append("")

    lines.append("## Ranking procedure")
    lines.append("")
    lines.append("For each (piece_id, cavity_id) group:")
    lines.append("- score = max over rotations of `predict_proba(class=1)`,")
    lines.append("- best rotation = argmax over rotations,")
    lines.append("- ground truth feasibility = any rotation has `label=1`.")
    lines.append("")
    lines.append("Per piece, cavities are ranked by score descending. Metrics:")
    lines.append("- top-1 feasible accuracy: rank-1 cavity is feasible in ground truth;")
    lines.append("- top-2 feasible accuracy: a feasible cavity is among ranks 1 or 2;")
    lines.append("- MRR: 1 / rank of first feasible cavity, averaged;")
    lines.append("- mean rank of first feasible cavity;")
    lines.append("- mean rank margin = score(rank-1) − score(rank-2).")
    lines.append("")
    lines.append("Pieces with **no feasible cavity in ground truth** are "
                 "excluded from accuracy/MRR/margin aggregations and reported "
                 "separately as `n_pieces_with_no_feasible_truth`.")
    lines.append("")

    lines.append("## Ranking metrics — primary scopes")
    lines.append("")
    lines.append("| model | scope | n_pieces | with_feasible | no_feasible | top-1 | top-2 | MRR | mean_rank | mean_margin |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for model_name in ("logreg", "tree"):
        for scope in ("test_split", "mvp_scenario", "all_procedural"):
            r = payload["ranking_results"][model_name][scope]
            lines.append(f"| {model_name} | {scope} | {r['n_pieces']} | "
                         f"{r['n_pieces_with_at_least_one_feasible']} | "
                         f"{r['n_pieces_with_no_feasible_truth']} | "
                         f"{r['top1_feasible_accuracy']:.4f} | "
                         f"{r['top2_feasible_accuracy']:.4f} | "
                         f"{r['mean_reciprocal_rank']:.4f} | "
                         f"{r['mean_rank_first_feasible']:.4f} | "
                         f"{r['mean_rank_margin']:.4f} |")
    lines.append("")

    lines.append("## Per-family ranking metrics")
    lines.append("")
    lines.append("| model | family | n_pieces | with_feasible | top-1 | top-2 | MRR | mean_rank | mean_margin |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for model_name in ("logreg", "tree"):
        for scope_name, r in payload["ranking_results"][model_name].items():
            if not scope_name.startswith("per_family/"):
                continue
            fam = scope_name.split("/", 1)[1]
            lines.append(f"| {model_name} | {fam} | {r['n_pieces']} | "
                         f"{r['n_pieces_with_at_least_one_feasible']} | "
                         f"{r['top1_feasible_accuracy']:.4f} | "
                         f"{r['top2_feasible_accuracy']:.4f} | "
                         f"{r['mean_reciprocal_rank']:.4f} | "
                         f"{r['mean_rank_first_feasible']:.4f} | "
                         f"{r['mean_rank_margin']:.4f} |")
    lines.append("")

    lines.append("## Worst failure cases")
    lines.append("")
    lines.append("Top 5 pieces with the WORST first-feasible-rank under "
                 "each model (scope: `all_procedural`).")
    lines.append("")
    for model_name in ("logreg", "tree"):
        lines.append(f"### `{model_name}`")
        lines.append("")
        cases = payload["worst_failure_cases"][model_name]
        if not cases:
            lines.append("None — no piece had its first feasible cavity below rank 1.")
            lines.append("")
            continue
        for c in cases:
            lines.append(f"- piece `{c['piece_id']}`: first feasible at rank "
                         f"**{c['first_feasible_rank']}** (margin "
                         f"{c['rank_margin']:.4f})")
            lines.append("")
            for r in c["ranking_top3"]:
                marker = "feasible" if r["is_feasible_truth"] else "not feasible"
                lines.append(f"  - rank {r['rank']}: `{r['cavity_id']}` "
                             f"score={r['score']:.4f} ({marker})")
            lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Ranking uses Phase D.3 models without any tuning; "
                 "hyperparameters are C=1.0 (logreg) and max_depth=4 (tree).")
    lines.append("- Cavity pool per piece is fixed at 7 (1 matching + 6 "
                 "mismatched recipes) from Phase D.1/D.2; ranking accuracy "
                 "depends on this construction.")
    lines.append("- Synthetic dataset only; convex prismatic shapes; no XY "
                 "offsets in the candidate space.")
    lines.append("- Pieces with no feasible ground-truth cavity are excluded "
                 "from accuracy/MRR but reported in the count column.")
    lines.append("- The MVP scenario reuses the dataset's MVP rows; this is "
                 "an in-distribution evaluation for cavities derived from MVP "
                 "pieces, NOT a true MVP-vs-board insertion scene (the board "
                 "scene was Baseline 1's evaluation; reusing it requires a "
                 "separate inference script outside Phase D's training scope).")
    lines.append("")

    lines.append("## Closing note")
    lines.append("")
    lines.append("Ranking outputs a top-1 cavity per piece and a rank margin. "
                 "These are perception-side affordance signals only; insertion "
                 "execution, grasp planning, and robot control are out of "
                 "scope. The downstream insertion is the fixed kinematic "
                 "primitive defined in the Phase D design.")

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
