"""
train_phaseD_3d_affordance_model.py

Phase D.3/D.4 — train and evaluate interpretable supervised classifiers for
3D-extrusion geometric affordance prediction.

Inputs:
  data/phaseD_3d_affordance/configurations_labelled.csv
  data/phaseD_3d_affordance/dataset_summary.json

Outputs (under data/phaseD_3d_affordance/models/):
  phaseD_training_results.json
  phaseD_training_report.md
  phaseD_feature_coefficients.csv
  phaseD_predictions.csv
  phaseD_confusion_matrices.csv

Models:
  - Logistic regression (StandardScaler + LogReg, C=1.0, class_weight='balanced')
  - Decision tree (max_depth=4, class_weight='balanced', random_state=0)

Evaluation:
  - Standard train/val/test split using the dataset's `split` column
  - Leave-one-family-out (LOFO) over shape_family
  - MVP scenario evaluation (is_mvp=True rows)

The model predicts a geometric insertion affordance score, NOT robot control.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        "/Users/robertofmbarreto/Documents/Mestrado/tese/code/thesis-omniverse",
    )
)

DATA_DIR    = PROJECT_ROOT / "data" / "phaseD_3d_affordance"
CSV_PATH    = DATA_DIR / "configurations_labelled.csv"
SUMMARY_JSON = DATA_DIR / "dataset_summary.json"

OUT_DIR     = DATA_DIR / "models"
OUT_RESULTS_JSON  = OUT_DIR / "phaseD_training_results.json"
OUT_REPORT_MD     = OUT_DIR / "phaseD_training_report.md"
OUT_COEF_CSV      = OUT_DIR / "phaseD_feature_coefficients.csv"
OUT_PREDS_CSV     = OUT_DIR / "phaseD_predictions.csv"
OUT_CM_CSV        = OUT_DIR / "phaseD_confusion_matrices.csv"

# ── Excluded columns ──────────────────────────────────────────────────────────

# Identifier / metadata columns (NOT features)
ID_COLS = {
    "config_id", "piece_id", "cavity_id", "shape_family", "is_mvp",
    "cavity_source", "split", "heldout_family_fold",
}

# Diagnostic / label-only columns (label leakage if used as features)
DIAG_COLS = {
    "diag_inside_ratio_raw", "diag_outside_ratio_raw",
    "diag_p_area_px", "diag_c_area_px", "diag_label_reason",
    "diag_insertion_required_mm",
    "label_reason",
}

LABEL_COL = "label"

# Leakage detection threshold
LEAKAGE_RHO_THRESHOLD = 0.95

# Logistic regression hyperparameters
LR_C       = 1.0
LR_MAX_ITER = 5000

# Decision tree hyperparameters
DT_MAX_DEPTH = 4
DT_RANDOM_STATE = 0


# ── CSV loading ───────────────────────────────────────────────────────────────

def load_csv(path: Path) -> list[dict]:
    with path.open() as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    return rows


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


# ── Feature selection ─────────────────────────────────────────────────────────

def determine_feature_columns(all_columns: list[str]) -> list[str]:
    excluded = ID_COLS | DIAG_COLS | {LABEL_COL}
    features = [c for c in all_columns if c not in excluded]
    # Keep only numeric features (drop any unexpected non-numeric column)
    return features


# ── Leakage check ─────────────────────────────────────────────────────────────

def pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    if x.std() < 1e-12 or y.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def leakage_check(X_cols: dict[str, np.ndarray], y: np.ndarray,
                   threshold: float = LEAKAGE_RHO_THRESHOLD) -> dict[str, dict]:
    report = {}
    for name, vec in X_cols.items():
        rho = pearson_correlation(vec, y.astype(np.float64))
        report[name] = {
            "pearson_corr": round(rho, 6),
            "abs_corr":     round(abs(rho), 6),
            "leakage_flag": bool(abs(rho) > threshold),
        }
    return report


# ── Metrics ───────────────────────────────────────────────────────────────────

def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                     y_prob: np.ndarray | None = None) -> dict:
    n = len(y_true)
    cm = confusion_matrix(y_true, y_pred)
    tp, fp, tn, fn = cm["tp"], cm["fp"], cm["tn"], cm["fn"]
    accuracy  = (tp + tn) / max(n, 1)
    precision = tp / max(tp + fp, 1)
    recall    = tp / max(tp + fn, 1)
    f1        = 2 * precision * recall / max(precision + recall, 1e-12)
    pos_rate_true = (y_true == 1).sum() / max(n, 1)
    pos_rate_pred = (y_pred == 1).sum() / max(n, 1)

    out = {
        "n":                int(n),
        "accuracy":         round(float(accuracy), 6),
        "precision":        round(float(precision), 6),
        "recall":           round(float(recall), 6),
        "f1":               round(float(f1), 6),
        "positive_rate_true": round(float(pos_rate_true), 6),
        "positive_rate_pred": round(float(pos_rate_pred), 6),
        "confusion_matrix": cm,
        "roc_auc":          None,
        "degenerate_flag":  bool(pos_rate_pred < 0.01 or pos_rate_pred > 0.99),
    }

    if y_prob is not None and len(np.unique(y_true)) == 2:
        try:
            from sklearn.metrics import roc_auc_score
            out["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 6)
        except Exception:
            out["roc_auc"] = None
    return out


# ── Training helpers ──────────────────────────────────────────────────────────

def fit_logreg(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=LR_C,
            class_weight="balanced",
            max_iter=LR_MAX_ITER,
            solver="lbfgs",
            random_state=0,
        )),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def fit_tree(X_train: np.ndarray, y_train: np.ndarray):
    from sklearn.tree import DecisionTreeClassifier
    clf = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH,
        class_weight="balanced",
        random_state=DT_RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    return clf


def predict_with_prob(model, X: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    y_pred = model.predict(X).astype(int)
    try:
        y_prob = model.predict_proba(X)[:, 1]
    except Exception:
        y_prob = None
    return y_pred, y_prob


# ── Main pipeline ─────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("train_phaseD_3d_affordance_model.py")
    print("=" * 70)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATA_DIR     : {DATA_DIR}")
    print(f"OUT_DIR      : {OUT_DIR}")
    print()

    # 1. Load CSV
    if not CSV_PATH.exists():
        print(f"[FATAL] dataset CSV not found at {CSV_PATH}")
        sys.exit(1)
    rows = load_csv(CSV_PATH)
    print(f"[load] {len(rows)} rows loaded from {CSV_PATH}")
    cols = rows_to_columns(rows)

    if LABEL_COL not in cols:
        print(f"[FATAL] label column '{LABEL_COL}' missing")
        sys.exit(1)

    # 2. Determine feature list
    all_columns = list(cols.keys())
    feature_cols = determine_feature_columns(all_columns)
    # Verify each feature column is numeric
    feature_cols = [c for c in feature_cols
                    if all(isinstance(v, (int, float)) and v is not None
                           for v in cols[c])]
    print(f"[features] {len(feature_cols)} feature columns selected:")
    for f in feature_cols:
        print(f"  - {f}")

    excluded_cols = sorted(set(all_columns) - set(feature_cols) - {LABEL_COL})
    print(f"[features] {len(excluded_cols)} columns excluded from features:")
    for c in excluded_cols:
        print(f"  - {c}")

    # 3. Build numpy arrays
    X = np.array([[float(cols[c][i]) for c in feature_cols]
                  for i in range(len(rows))], dtype=np.float64)
    y = np.array(cols[LABEL_COL], dtype=np.int32)

    families  = np.array(cols["shape_family"])
    is_mvp    = np.array(cols["is_mvp"], dtype=bool) if "is_mvp" in cols else np.zeros(len(rows), dtype=bool)
    splits    = np.array(cols["split"])
    config_ids = np.array(cols["config_id"])

    # 4. Leakage check on candidate features
    X_dict = {c: np.array(cols[c], dtype=np.float64) for c in feature_cols}
    leak = leakage_check(X_dict, y)
    leak_warnings = [name for name, info in leak.items() if info["leakage_flag"]]
    if leak_warnings:
        print(f"[leakage] WARNING: {len(leak_warnings)} feature(s) above |rho|"
              f"={LEAKAGE_RHO_THRESHOLD}: {leak_warnings}")
    else:
        print(f"[leakage] OK — no feature exceeds |rho|={LEAKAGE_RHO_THRESHOLD} vs label.")

    # 5. Standard split evaluation
    print("\n[eval] standard split (using `split` column) ...")
    train_mask = splits == "train"
    val_mask   = splits == "val"
    test_mask  = splits == "test"

    standard_results = {}

    for model_name, fit_fn in [("logreg", fit_logreg), ("tree", fit_tree)]:
        print(f"  [model] {model_name} ...")
        model = fit_fn(X[train_mask], y[train_mask])
        results = {}
        for split_name, mask in [("train", train_mask), ("val", val_mask), ("test", test_mask)]:
            y_p, y_prob = predict_with_prob(model, X[mask])
            metrics = compute_metrics(y[mask], y_p, y_prob)
            results[split_name] = metrics
            print(f"    {split_name}: n={metrics['n']} acc={metrics['accuracy']:.4f} "
                  f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
                  f"f1={metrics['f1']:.4f} auc={metrics['roc_auc']}")
        standard_results[model_name] = {"metrics_per_split": results}

    # 6. Save logistic regression coefficients
    print("\n[interp] logistic regression coefficients ...")
    lr_model = fit_logreg(X[train_mask], y[train_mask])
    scaler_step = lr_model.named_steps["scaler"]
    clf_step    = lr_model.named_steps["clf"]
    coef_raw    = clf_step.coef_.flatten()  # in the standardised feature space
    intercept   = float(clf_step.intercept_[0])
    coef_table  = []
    for i, name in enumerate(feature_cols):
        coef_table.append({
            "feature":         name,
            "coef_standardised": round(float(coef_raw[i]), 6),
            "abs_coef":        round(abs(float(coef_raw[i])), 6),
            "feature_mean":    round(float(scaler_step.mean_[i]), 6),
            "feature_scale":   round(float(scaler_step.scale_[i]), 6),
        })
    coef_table.sort(key=lambda r: -r["abs_coef"])
    standard_results["logreg"]["intercept_standardised"] = round(intercept, 6)
    standard_results["logreg"]["coefficients_sorted"]    = coef_table
    print(f"  intercept (standardised) = {intercept:.4f}")
    print("  top 5 coefficients by |coef|:")
    for c in coef_table[:5]:
        print(f"    {c['feature']:35s}  coef={c['coef_standardised']:+.4f}")

    # 7. Decision tree feature importance
    print("\n[interp] decision tree feature importances ...")
    dt_model = fit_tree(X[train_mask], y[train_mask])
    importances = dt_model.feature_importances_
    importance_table = sorted(
        [{"feature": name, "importance": round(float(importances[i]), 6)}
         for i, name in enumerate(feature_cols)],
        key=lambda r: -r["importance"],
    )
    standard_results["tree"]["feature_importances_sorted"] = importance_table
    print("  top 5 features by importance:")
    for c in importance_table[:5]:
        print(f"    {c['feature']:35s}  imp={c['importance']:.4f}")

    # 8. Leave-one-family-out (LOFO)
    print("\n[lofo] leave-one-family-out evaluation ...")
    procedural_families = sorted(set(families[~is_mvp]))
    lofo_results = {}
    for held in procedural_families:
        # Train: all rows where family != held AND not is_mvp
        # (MVP rows kept out of LOFO training to avoid mixing real-data hold-in
        #  with procedural OOD evaluation; reported separately in §9)
        train_mask = (families != held) & (~is_mvp)
        test_mask  = (families == held) & (~is_mvp)

        n_train = int(train_mask.sum())
        n_test  = int(test_mask.sum())
        n_pos_test = int(y[test_mask].sum())
        if n_test == 0:
            print(f"  [lofo {held}] SKIPPED — no test rows")
            continue

        n_classes_in_train = len(np.unique(y[train_mask]))
        if n_classes_in_train < 2:
            print(f"  [lofo {held}] SKIPPED — only one class in training set")
            lofo_results[held] = {"skipped": True, "reason": "single_class_train"}
            continue

        family_results = {}
        for model_name, fit_fn in [("logreg", fit_logreg), ("tree", fit_tree)]:
            model = fit_fn(X[train_mask], y[train_mask])
            y_p, y_prob = predict_with_prob(model, X[test_mask])
            metrics = compute_metrics(y[test_mask], y_p, y_prob)
            family_results[model_name] = metrics

        lofo_results[held] = {
            "n_train": n_train,
            "n_test":  n_test,
            "n_pos_test": n_pos_test,
            "extreme_imbalance_test": bool(n_pos_test == 0 or n_pos_test == n_test),
            "models": family_results,
        }
        lr_f1   = family_results["logreg"]["f1"]
        tree_f1 = family_results["tree"]["f1"]
        print(f"  [lofo {held:30s}] n_train={n_train:5d} n_test={n_test:5d} "
              f"n_pos_test={n_pos_test:4d}  logreg_F1={lr_f1:.4f}  tree_F1={tree_f1:.4f}")

    # 9. MVP scenario evaluation
    print("\n[mvp] MVP scenario evaluation ...")
    mvp_mask  = is_mvp
    proc_mask = ~is_mvp
    n_mvp = int(mvp_mask.sum())
    print(f"  MVP rows: {n_mvp}")
    mvp_results = {"n_rows": n_mvp}
    if n_mvp > 0:
        # Train on all procedural rows, evaluate on MVP
        train_mask = proc_mask
        for model_name, fit_fn in [("logreg", fit_logreg), ("tree", fit_tree)]:
            model = fit_fn(X[train_mask], y[train_mask])
            y_p, y_prob = predict_with_prob(model, X[mvp_mask])
            metrics = compute_metrics(y[mvp_mask], y_p, y_prob)
            mvp_results[model_name] = metrics
            print(f"    {model_name}: n={metrics['n']} acc={metrics['accuracy']:.4f} "
                  f"prec={metrics['precision']:.4f} rec={metrics['recall']:.4f} "
                  f"f1={metrics['f1']:.4f} auc={metrics['roc_auc']}")

    # 10. Predictions output (test split + MVP)
    print("\n[predictions] writing per-row predictions for test split ...")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pred_rows = []
    test_mask = splits == "test"
    lr_test_model = fit_logreg(X[splits == "train"], y[splits == "train"])
    dt_test_model = fit_tree(X[splits == "train"], y[splits == "train"])
    lr_p_test, lr_prob_test = predict_with_prob(lr_test_model, X[test_mask])
    dt_p_test, _            = predict_with_prob(dt_test_model, X[test_mask])
    test_indices = np.where(test_mask)[0]
    for k, idx in enumerate(test_indices):
        pred_rows.append({
            "config_id":         int(config_ids[idx]),
            "split":             "test",
            "shape_family":      str(families[idx]),
            "is_mvp":            bool(is_mvp[idx]),
            "label":             int(y[idx]),
            "logreg_pred":       int(lr_p_test[k]),
            "logreg_prob":       round(float(lr_prob_test[k]) if lr_prob_test is not None else 0.0, 6),
            "tree_pred":         int(dt_p_test[k]),
        })

    # 11. Confusion matrices file
    cm_rows = []
    for model_name, model_block in standard_results.items():
        for split_name, m in model_block["metrics_per_split"].items():
            cm = m["confusion_matrix"]
            cm_rows.append({
                "scope":      f"standard/{split_name}",
                "model":      model_name,
                "n":          m["n"],
                "tp":         cm["tp"], "fp": cm["fp"], "tn": cm["tn"], "fn": cm["fn"],
                "accuracy":   m["accuracy"], "precision": m["precision"],
                "recall":     m["recall"], "f1": m["f1"],
            })
    for held, res in lofo_results.items():
        if res.get("skipped"):
            continue
        for model_name, m in res["models"].items():
            cm = m["confusion_matrix"]
            cm_rows.append({
                "scope":   f"lofo/{held}",
                "model":   model_name,
                "n":       m["n"],
                "tp":      cm["tp"], "fp": cm["fp"], "tn": cm["tn"], "fn": cm["fn"],
                "accuracy": m["accuracy"], "precision": m["precision"],
                "recall":  m["recall"], "f1": m["f1"],
            })
    if "logreg" in mvp_results and isinstance(mvp_results["logreg"], dict):
        for model_name in ("logreg", "tree"):
            if model_name in mvp_results:
                m = mvp_results[model_name]
                cm = m["confusion_matrix"]
                cm_rows.append({
                    "scope":   "mvp/all",
                    "model":   model_name,
                    "n":       m["n"],
                    "tp":      cm["tp"], "fp": cm["fp"], "tn": cm["tn"], "fn": cm["fn"],
                    "accuracy": m["accuracy"], "precision": m["precision"],
                    "recall":  m["recall"], "f1": m["f1"],
                })

    # 12. Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Coefficients CSV
    with OUT_COEF_CSV.open("w") as f:
        f.write("rank,feature,coef_standardised,abs_coef,feature_mean,feature_scale,tree_importance\n")
        importances_dict = {r["feature"]: r["importance"] for r in importance_table}
        for rank, c in enumerate(coef_table, start=1):
            tree_imp = importances_dict.get(c["feature"], 0.0)
            f.write(f"{rank},{c['feature']},{c['coef_standardised']},{c['abs_coef']},"
                    f"{c['feature_mean']},{c['feature_scale']},{tree_imp}\n")
    print(f"[write] {OUT_COEF_CSV}")

    # Predictions CSV
    with OUT_PREDS_CSV.open("w") as f:
        if pred_rows:
            keys = list(pred_rows[0].keys())
            f.write(",".join(keys) + "\n")
            for r in pred_rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"[write] {OUT_PREDS_CSV}")

    # Confusion matrices CSV
    with OUT_CM_CSV.open("w") as f:
        if cm_rows:
            keys = list(cm_rows[0].keys())
            f.write(",".join(keys) + "\n")
            for r in cm_rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")
    print(f"[write] {OUT_CM_CSV}")

    # Results JSON
    payload = {
        "schema_version": 1,
        "script_name":    "train_phaseD_3d_affordance_model.py",
        "phase":          "Phase D.3/D.4 — affordance classifier training/eval",
        "phase_note": (
            "The model predicts a geometric insertion affordance score, NOT "
            "robot control. Interpretable supervised classifiers only. No "
            "neural network, no reinforcement learning, no robot execution."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "feature_columns_used":      feature_cols,
        "feature_columns_excluded":  excluded_cols,
        "label_column":              LABEL_COL,
        "leakage_check": {
            "threshold_abs_corr": LEAKAGE_RHO_THRESHOLD,
            "per_feature":        leak,
            "warnings":           leak_warnings,
        },
        "models": {
            "logreg": {
                "type":         "LogisticRegression",
                "C":            LR_C,
                "class_weight": "balanced",
                "max_iter":     LR_MAX_ITER,
                "scaler":       "StandardScaler",
            },
            "tree": {
                "type":         "DecisionTreeClassifier",
                "max_depth":    DT_MAX_DEPTH,
                "class_weight": "balanced",
                "random_state": DT_RANDOM_STATE,
            },
        },
        "standard_split":   standard_results,
        "lofo":             lofo_results,
        "mvp_evaluation":   mvp_results,
    }
    OUT_RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[write] {OUT_RESULTS_JSON}")

    # Markdown report
    write_report(payload, OUT_REPORT_MD)
    print(f"[write] {OUT_REPORT_MD}")
    print("[done].")


def write_report(payload: dict, path: Path) -> None:
    lines = []
    lines.append("# Phase D.3/D.4 — 3D-extrusion Affordance Classifier")
    lines.append("")
    lines.append("> **The model predicts a geometric insertion affordance "
                 "score, not robot control.**")
    lines.append("")
    lines.append("> **Status**: Phase D.3/D.4 (training + evaluation only). "
                 "No hyperparameter tuning, no feature pruning, no second "
                 "training pass.")
    lines.append("")

    lines.append("## Objective")
    lines.append("")
    lines.append("Train interpretable supervised classifiers (logistic "
                 "regression, decision tree) on the Phase D.1/D.2 "
                 "3D-extrusion affordance dataset, and evaluate them under "
                 "(a) standard train/val/test split and (b) leave-one-family-"
                 "out (LOFO) cross-validation. Report leakage diagnostics, "
                 "coefficient interpretability, and MVP-scenario performance.")
    lines.append("")

    lines.append("## Dataset source")
    lines.append("")
    lines.append(f"- `data/phaseD_3d_affordance/configurations_labelled.csv` "
                 f"({sum(payload['standard_split']['logreg']['metrics_per_split'][s]['n'] for s in ('train','val','test'))} rows)")
    lines.append(f"- `data/phaseD_3d_affordance/dataset_summary.json`")
    lines.append("")

    lines.append("## Feature list")
    lines.append("")
    lines.append(f"**{len(payload['feature_columns_used'])} identity-free features used**:")
    for f in payload["feature_columns_used"]:
        lines.append(f"- `{f}`")
    lines.append("")

    lines.append("## Excluded columns")
    lines.append("")
    lines.append("Identifier and diagnostic columns are excluded from the model:")
    for c in payload["feature_columns_excluded"]:
        lines.append(f"- `{c}`")
    lines.append("")

    lines.append("## Leakage check")
    lines.append("")
    lines.append(f"Pearson correlation threshold for leakage warning: "
                 f"|ρ| > **{payload['leakage_check']['threshold_abs_corr']}**.")
    lines.append("")
    if payload["leakage_check"]["warnings"]:
        lines.append("**Warnings**:")
        for name in payload["leakage_check"]["warnings"]:
            info = payload["leakage_check"]["per_feature"][name]
            lines.append(f"- `{name}`: ρ = {info['pearson_corr']:+.4f} "
                         f"(|ρ| = {info['abs_corr']:.4f})")
    else:
        lines.append("No feature exceeds the threshold. ✅")
    lines.append("")
    # Top-3 absolute correlations for transparency
    sorted_corrs = sorted(
        payload["leakage_check"]["per_feature"].items(),
        key=lambda kv: -abs(kv[1]["pearson_corr"]),
    )
    lines.append("Top-5 features by |Pearson ρ| with the label (informational):")
    for name, info in sorted_corrs[:5]:
        lines.append(f"- `{name}`: ρ = {info['pearson_corr']:+.4f}")
    lines.append("")

    lines.append("## Models trained")
    lines.append("")
    for model_name, mcfg in payload["models"].items():
        lines.append(f"### `{model_name}`")
        lines.append("")
        for k, v in mcfg.items():
            lines.append(f"- **{k}**: `{v}`")
        lines.append("")

    lines.append("## Standard split metrics")
    lines.append("")
    lines.append("| model | split | n | accuracy | precision | recall | F1 | ROC-AUC | pos_rate_pred | degenerate |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for model_name in ("logreg", "tree"):
        for split_name in ("train", "val", "test"):
            m = payload["standard_split"][model_name]["metrics_per_split"][split_name]
            auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "—"
            lines.append(f"| {model_name} | {split_name} | {m['n']} | "
                         f"{m['accuracy']:.4f} | {m['precision']:.4f} | "
                         f"{m['recall']:.4f} | {m['f1']:.4f} | {auc_str} | "
                         f"{m['positive_rate_pred']:.4f} | "
                         f"{'YES' if m['degenerate_flag'] else 'no'} |")
    lines.append("")

    lines.append("## Leave-one-family-out (LOFO) results")
    lines.append("")
    lines.append("Trained on all PROCEDURAL rows except the held-out family; "
                 "tested on procedural rows of the held-out family. MVP rows "
                 "are kept out of LOFO training/test to avoid mixing real-data "
                 "hold-in with procedural OOD evaluation; MVP performance is "
                 "reported separately below.")
    lines.append("")
    lines.append("| held-out family | n_train | n_test | n_pos_test | model | accuracy | precision | recall | F1 | ROC-AUC | pos_rate_pred | extreme_imbalance |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|")
    for fam, res in payload["lofo"].items():
        if res.get("skipped"):
            lines.append(f"| {fam} | — | — | — | — | SKIPPED ({res['reason']}) | | | | | | |")
            continue
        for model_name in ("logreg", "tree"):
            m = res["models"][model_name]
            auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "—"
            extr = "YES" if res["extreme_imbalance_test"] else "no"
            lines.append(f"| {fam} | {res['n_train']} | {res['n_test']} | "
                         f"{res['n_pos_test']} | {model_name} | "
                         f"{m['accuracy']:.4f} | {m['precision']:.4f} | "
                         f"{m['recall']:.4f} | {m['f1']:.4f} | {auc_str} | "
                         f"{m['positive_rate_pred']:.4f} | {extr} |")
    lines.append("")

    lines.append("## MVP scenario evaluation")
    lines.append("")
    mvp = payload["mvp_evaluation"]
    if mvp.get("n_rows", 0) > 0:
        lines.append("Trained on ALL procedural rows; tested on MVP rows.")
        lines.append("")
        lines.append("| model | n | accuracy | precision | recall | F1 | ROC-AUC | pos_rate_true | pos_rate_pred |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
        for model_name in ("logreg", "tree"):
            if model_name not in mvp:
                continue
            m = mvp[model_name]
            auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "—"
            lines.append(f"| {model_name} | {m['n']} | {m['accuracy']:.4f} | "
                         f"{m['precision']:.4f} | {m['recall']:.4f} | "
                         f"{m['f1']:.4f} | {auc_str} | "
                         f"{m['positive_rate_true']:.4f} | "
                         f"{m['positive_rate_pred']:.4f} |")
    else:
        lines.append(f"No MVP rows in dataset (n_rows = {mvp.get('n_rows', 0)}).")
    lines.append("")

    lines.append("## Logistic regression — top coefficients (standardised)")
    lines.append("")
    coefs = payload["standard_split"]["logreg"].get("coefficients_sorted", [])
    intercept = payload["standard_split"]["logreg"].get("intercept_standardised")
    if intercept is not None:
        lines.append(f"Intercept (standardised): **{intercept:+.4f}**")
        lines.append("")
    lines.append("| rank | feature | coef (standardised) | |coef| | feature_mean | feature_scale |")
    lines.append("|---:|---|---:|---:|---:|---:|")
    for rank, c in enumerate(coefs, start=1):
        lines.append(f"| {rank} | `{c['feature']}` | {c['coef_standardised']:+.4f} | "
                     f"{c['abs_coef']:.4f} | {c['feature_mean']:.4f} | "
                     f"{c['feature_scale']:.4f} |")
    lines.append("")
    lines.append("Coefficient interpretation: positive coefficient → feature "
                 "increase pushes affordance probability toward 1; negative → "
                 "toward 0. Magnitudes are comparable across features because "
                 "the input was standardised.")
    lines.append("")

    lines.append("## Decision tree — feature importances")
    lines.append("")
    imps = payload["standard_split"]["tree"].get("feature_importances_sorted", [])
    lines.append("| rank | feature | importance |")
    lines.append("|---:|---|---:|")
    for rank, c in enumerate(imps, start=1):
        lines.append(f"| {rank} | `{c['feature']}` | {c['importance']:.4f} |")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Synthetic dataset only. Labels generated by raster geometry; not physical contact.")
    lines.append("- Convex prismatic shapes only (extrusion assumption).")
    lines.append("- No XY offset in the dataset; rotations only.")
    lines.append("- C, max_iter, max_depth fixed; no hyperparameter tuning in this run.")
    lines.append("- LOFO trains on 4 families and tests on the 1 held-out family; "
                 "if a held-out family has extreme positive/negative imbalance, "
                 "F1 may be misleading and is flagged.")
    lines.append("- The model predicts a geometric affordance score; transfer to "
                 "non-prismatic shapes, real captures, or robot execution is out of scope.")
    lines.append("")

    lines.append("## Closing note")
    lines.append("")
    lines.append("This phase trains an interpretable classifier on identity-free "
                 "geometric features to predict insertion **affordance**. "
                 "It does NOT learn robot control, grasping, force feedback, or "
                 "motion planning. Downstream insertion remains the deterministic "
                 "fixed primitive defined in the Phase D design.")

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
