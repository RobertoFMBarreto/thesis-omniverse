"""
train_phaseE_siamese_embedding.py

Phase E.2 — train a minimal Siamese CNN geometric embedding model on the
Phase E SDF dataset.

The model learns a 64-D shared embedding from piece and cavity SDF inputs,
then predicts compatibility from |diff|, product, and cosine similarity of
the two embeddings via a small MLP head.

NOT robot control. NOT learned perception. Frozen perception inputs only.
Single training run; no hyperparameter tuning.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Torch (installed locally; CPU-only is fine)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        "/Users/robertofmbarreto/Documents/Mestrado/tese/code/thesis-omniverse",
    )
)

DATA_DIR = PROJECT_ROOT / "data" / "phaseE_learned_embeddings"
NPZ_PATH = DATA_DIR / "phaseE_sdf_pairs.npz"
CSV_PATH = DATA_DIR / "phaseE_pairs_metadata.csv"

OUT_DIR  = DATA_DIR / "models"
OUT_RESULTS_JSON  = OUT_DIR / "phaseE_siamese_results.json"
OUT_REPORT_MD     = OUT_DIR / "phaseE_siamese_report.md"
OUT_PREDS_CSV     = OUT_DIR / "phaseE_siamese_predictions.csv"
OUT_MODEL_PT      = OUT_DIR / "phaseE_siamese_model.pt"
OUT_CURVES_CSV    = OUT_DIR / "phaseE_siamese_training_curves.csv"
OUT_CM_CSV        = OUT_DIR / "phaseE_siamese_confusion_matrices.csv"

# ── Config (frozen) ───────────────────────────────────────────────────────────

SEED              = 0
EMBED_DIM         = 64
BATCH_SIZE        = 256
LEARNING_RATE     = 1e-3
MAX_EPOCHS        = 20
EARLY_STOP_PATIENCE = 3        # epochs without val-F1 improvement
NUM_WORKERS       = 0           # CPU-friendly; deterministic loading
SDF_INT8_SCALE    = 127.0 / 20.0   # dequantise: sdf_mm = sdf_int8 / scale
SDF_NORM_DIV_MM   = 20.0           # normalise dequantised SDF to [-1, 1]


# ── Dataset ───────────────────────────────────────────────────────────────────

class PhaseEDataset(Dataset):
    """In-memory int8 SDF arrays; dequantise + normalise per sample."""

    def __init__(self, piece_sdf_int8: np.ndarray, cavity_sdf_int8: np.ndarray,
                 labels: np.ndarray, indices: np.ndarray):
        self.piece_sdf_int8  = piece_sdf_int8
        self.cavity_sdf_int8 = cavity_sdf_int8
        self.labels          = labels
        self.indices         = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int) -> tuple:
        idx = int(self.indices[i])
        psdf_mm = self.piece_sdf_int8[idx].astype(np.float32) / SDF_INT8_SCALE
        csdf_mm = self.cavity_sdf_int8[idx].astype(np.float32) / SDF_INT8_SCALE
        psdf = np.clip(psdf_mm / SDF_NORM_DIV_MM, -1.0, 1.0)[None, :, :]   # (1, 128, 128)
        csdf = np.clip(csdf_mm / SDF_NORM_DIV_MM, -1.0, 1.0)[None, :, :]
        label = float(self.labels[idx])
        return (
            torch.from_numpy(psdf).float(),
            torch.from_numpy(csdf).float(),
            torch.tensor(label, dtype=torch.float32),
            torch.tensor(idx, dtype=torch.long),
        )


# ── Model ─────────────────────────────────────────────────────────────────────

class SharedEncoder(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool  = nn.MaxPool2d(2)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.fc    = nn.Linear(64, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))   # (B, 16, 64, 64)
        x = self.pool(F.relu(self.conv2(x)))   # (B, 32, 32, 32)
        x = self.pool(F.relu(self.conv3(x)))   # (B, 64, 16, 16)
        x = self.gap(x).flatten(1)              # (B, 64)
        x = self.fc(x)                           # (B, EMBED_DIM)
        x = F.normalize(x, p=2, dim=1)           # L2 normalise
        return x


class SiameseCompatibility(nn.Module):
    def __init__(self, embed_dim: int = EMBED_DIM, hidden: int = 32):
        super().__init__()
        self.encoder = SharedEncoder(embed_dim)
        # input to head: |diff|(D) + product(D) + cosine(1) = 2D + 1
        self.head = nn.Sequential(
            nn.Linear(2 * embed_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_p: torch.Tensor, x_c: torch.Tensor) -> torch.Tensor:
        e_p = self.encoder(x_p)   # (B, D)
        e_c = self.encoder(x_c)   # (B, D)
        abs_diff = (e_p - e_c).abs()                     # (B, D)
        product  = e_p * e_c                              # (B, D)
        cos_sim  = (e_p * e_c).sum(dim=1, keepdim=True)   # (B, 1) -- normalised → cos
        feats = torch.cat([abs_diff, product, cos_sim], dim=1)
        logit = self.head(feats).squeeze(1)               # (B,)
        return logit


# ── Training utilities ────────────────────────────────────────────────────────

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
    pos_rate_pred = (y_pred == 1).sum() / max(n, 1)
    pos_rate_true = (y_true == 1).sum() / max(n, 1)
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


def evaluate_loader(model: nn.Module, loader: DataLoader, device: torch.device
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """Return (y_true, y_pred, y_prob, sample_idx, mean_loss)."""
    model.eval()
    all_y, all_yp, all_prob, all_idx = [], [], [], []
    loss_fn = nn.BCEWithLogitsLoss()
    total_loss = 0.0
    n_batches = 0
    with torch.no_grad():
        for xp, xc, y, idx in loader:
            xp = xp.to(device); xc = xc.to(device); y = y.to(device)
            logit = model(xp, xc)
            loss = loss_fn(logit, y)
            total_loss += loss.item()
            n_batches += 1
            prob = torch.sigmoid(logit).detach().cpu().numpy()
            pred = (prob >= 0.5).astype(np.int32)
            all_y.append(y.detach().cpu().numpy().astype(np.int32))
            all_yp.append(pred)
            all_prob.append(prob)
            all_idx.append(idx.numpy())
    return (np.concatenate(all_y), np.concatenate(all_yp),
            np.concatenate(all_prob), np.concatenate(all_idx),
            total_loss / max(n_batches, 1))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("train_phaseE_siamese_embedding.py")
    print("=" * 70)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATA_DIR     : {DATA_DIR}")
    print(f"OUT_DIR      : {OUT_DIR}")
    print()

    if not NPZ_PATH.exists() or not CSV_PATH.exists():
        print(f"[FATAL] missing dataset files: {NPZ_PATH} / {CSV_PATH}")
        sys.exit(1)

    # 1. Load
    print("[load] reading npz + metadata ...")
    data = np.load(NPZ_PATH)
    piece_sdf  = data["piece_sdf_int8"]
    cavity_sdf = data["cavity_sdf_int8"]
    labels     = data["labels"]
    sample_ids = data["sample_ids"]
    n_total    = len(labels)
    print(f"[load] n_samples = {n_total}; piece_sdf shape = {piece_sdf.shape}")

    # CSV metadata (for split, family, ranking)
    rows = list(csv.DictReader(CSV_PATH.open()))
    if len(rows) != n_total:
        print(f"[FATAL] CSV row count {len(rows)} != npz n_samples {n_total}")
        sys.exit(1)
    splits     = np.array([r["split"]        for r in rows])
    families   = np.array([r["shape_family"] for r in rows])
    is_mvp     = np.array([r["is_mvp"] == "True" for r in rows], dtype=bool)
    piece_ids  = np.array([r["piece_id"]     for r in rows])
    cavity_ids = np.array([r["cavity_id"]    for r in rows])
    rotations  = np.array([int(r["rotation_deg"]) for r in rows])

    train_idx = np.where(splits == "train")[0]
    val_idx   = np.where(splits == "val")[0]
    test_idx  = np.where(splits == "test")[0]
    print(f"[split] train={len(train_idx)} val={len(val_idx)} test={len(test_idx)}")

    # Class-imbalance pos_weight from training labels
    n_pos_train = int(labels[train_idx].sum())
    n_neg_train = len(train_idx) - n_pos_train
    pos_weight_value = float(n_neg_train) / max(n_pos_train, 1)
    print(f"[class] train pos={n_pos_train} neg={n_neg_train} → pos_weight={pos_weight_value:.4f}")

    # 2. Datasets / loaders
    train_ds = PhaseEDataset(piece_sdf, cavity_sdf, labels, train_idx)
    val_ds   = PhaseEDataset(piece_sdf, cavity_sdf, labels, val_idx)
    test_ds  = PhaseEDataset(piece_sdf, cavity_sdf, labels, test_idx)

    g = torch.Generator(); g.manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                num_workers=NUM_WORKERS, generator=g)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS)

    # 3. Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    model = SiameseCompatibility(embed_dim=EMBED_DIM).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] n_params = {n_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    pos_weight = torch.tensor([pos_weight_value], device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # 4. Train with early stopping on val F1
    print("[train] starting ...")
    curves = []
    best_val_f1 = -1.0
    best_state  = None
    best_epoch  = 0
    epochs_without_improve = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        model.train()
        train_loss_sum = 0.0
        n_batches = 0
        for xp, xc, y, _ in train_loader:
            xp = xp.to(device); xc = xc.to(device); y = y.to(device)
            optimizer.zero_grad()
            logit = model(xp, xc)
            loss = loss_fn(logit, y)
            loss.backward()
            optimizer.step()
            train_loss_sum += loss.item()
            n_batches += 1
        train_loss = train_loss_sum / max(n_batches, 1)

        y_v, yp_v, prob_v, _, val_loss = evaluate_loader(model, val_loader, device)
        val_metrics = compute_metrics(y_v, yp_v, prob_v)
        val_f1 = val_metrics["f1"]

        epoch_dt = time.time() - t0
        print(f"  [epoch {epoch:02d}] train_loss={train_loss:.4f}  "
              f"val_loss={val_loss:.4f}  val_F1={val_f1:.4f}  "
              f"val_acc={val_metrics['accuracy']:.4f}  "
              f"val_pos_rate_pred={val_metrics['positive_rate_pred']:.4f}  "
              f"({epoch_dt:.1f}s)")
        curves.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "val_f1":     round(val_f1, 6),
            "val_acc":    val_metrics["accuracy"],
            "val_pos_rate_pred": val_metrics["positive_rate_pred"],
        })

        if val_f1 > best_val_f1 + 1e-6:
            best_val_f1 = val_f1
            best_state  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch  = epoch
            epochs_without_improve = 0
        else:
            epochs_without_improve += 1
            if epochs_without_improve >= EARLY_STOP_PATIENCE:
                print(f"[early-stop] no val_F1 improvement for "
                      f"{EARLY_STOP_PATIENCE} epochs (best @ epoch {best_epoch}, "
                      f"F1={best_val_f1:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # 5. Final eval on train/val/test with the best model
    print("\n[final-eval] evaluating best model ...")
    final_metrics = {}
    pred_records  = []
    for split_name, loader in [("train", DataLoader(train_ds, batch_size=BATCH_SIZE,
                                                      shuffle=False)),
                                ("val", val_loader),
                                ("test", test_loader)]:
        y, yp, prob, idx, ls = evaluate_loader(model, loader, device)
        m = compute_metrics(y, yp, prob)
        m["mean_loss"] = round(float(ls), 6)
        final_metrics[split_name] = m
        print(f"  [{split_name}] n={m['n']} acc={m['accuracy']:.4f} "
              f"prec={m['precision']:.4f} rec={m['recall']:.4f} "
              f"F1={m['f1']:.4f} AUC={m['roc_auc']} "
              f"pos_rate_pred={m['positive_rate_pred']:.4f}")
        if split_name == "test":
            for i in range(len(y)):
                row_idx = int(idx[i])
                pred_records.append({
                    "sample_id":   int(sample_ids[row_idx]),
                    "split":       "test",
                    "shape_family": str(families[row_idx]),
                    "is_mvp":       bool(is_mvp[row_idx]),
                    "label":        int(y[i]),
                    "pred":         int(yp[i]),
                    "prob":         round(float(prob[i]), 6),
                    "piece_id":     str(piece_ids[row_idx]),
                    "cavity_id":    str(cavity_ids[row_idx]),
                    "rotation_deg": int(rotations[row_idx]),
                })

    degenerate = bool(final_metrics["test"]["degenerate_flag"])
    print(f"[degeneracy] test pos_rate_pred = "
          f"{final_metrics['test']['positive_rate_pred']:.4f} → "
          f"{'DEGENERATE' if degenerate else 'OK'}")

    # 6. Per-family on test split
    print("\n[per-family] test split ...")
    per_family_metrics = {}
    test_preds_by_idx = {p["sample_id"]: p for p in pred_records}
    for fam in sorted(set(families)):
        fam_rows = [r for r in pred_records if r["shape_family"] == fam]
        if not fam_rows:
            continue
        y_arr  = np.array([r["label"] for r in fam_rows], dtype=np.int32)
        yp_arr = np.array([r["pred"]  for r in fam_rows], dtype=np.int32)
        pr_arr = np.array([r["prob"]  for r in fam_rows], dtype=np.float64)
        m = compute_metrics(y_arr, yp_arr, pr_arr)
        per_family_metrics[fam] = m
        print(f"  [family {fam:30s}] n={m['n']:5d} acc={m['accuracy']:.4f} "
              f"prec={m['precision']:.4f} rec={m['recall']:.4f} F1={m['f1']:.4f}")

    # 7. Ranking eval on test split (and MVP scenario rows if any)
    print("\n[ranking] grouping by piece_id within test split + MVP rows ...")

    def rank_pieces(scope_name: str, scope_indices: np.ndarray) -> dict:
        """Compute ranking metrics over pieces within scope_indices."""
        # Need probs for ALL scope rows, not just test. Compute now.
        if len(scope_indices) == 0:
            return {"scope": scope_name, "n_pieces": 0, "skipped": True}
        scope_ds = PhaseEDataset(piece_sdf, cavity_sdf, labels, scope_indices)
        scope_loader = DataLoader(scope_ds, batch_size=BATCH_SIZE, shuffle=False)
        _, _, prob_arr, idx_arr, _ = evaluate_loader(model, scope_loader, device)

        # Group by piece_id, then cavity_id
        per_piece = defaultdict(lambda: defaultdict(list))
        for k, idx in enumerate(idx_arr):
            row_i = int(idx)
            per_piece[piece_ids[row_i]][cavity_ids[row_i]].append({
                "rot":   int(rotations[row_i]),
                "prob":  float(prob_arr[k]),
                "label": int(labels[row_i]),
            })

        # For each piece, rank cavities by max prob over rotations
        n_pieces      = 0
        n_with_feas   = 0
        n_top1        = 0
        sum_mrr       = 0.0
        sum_rank_first = 0.0
        sum_margin    = 0.0
        for pid, cavs in per_piece.items():
            n_pieces += 1
            ranked = []
            for cid, items in cavs.items():
                best = max(items, key=lambda r: r["prob"])
                feas = any(it["label"] == 1 for it in items)
                ranked.append({"cavity_id": cid, "score": best["prob"],
                                "is_feasible_truth": feas})
            ranked.sort(key=lambda r: -r["score"])
            for rank, r in enumerate(ranked, start=1):
                r["rank"] = rank
            feas_ranks = [r["rank"] for r in ranked if r["is_feasible_truth"]]
            if not feas_ranks:
                continue
            n_with_feas += 1
            first = feas_ranks[0]
            if first == 1:
                n_top1 += 1
            sum_mrr += 1.0 / first
            sum_rank_first += first
            margin = (ranked[0]["score"] - ranked[1]["score"]) if len(ranked) > 1 else 0.0
            sum_margin += margin

        return {
            "scope":                                 scope_name,
            "n_pieces":                              n_pieces,
            "n_pieces_with_at_least_one_feasible":   n_with_feas,
            "top1_feasible_accuracy":                round(n_top1 / max(n_with_feas, 1), 6),
            "mean_reciprocal_rank":                  round(sum_mrr / max(n_with_feas, 1), 6),
            "mean_rank_first_feasible":              round(sum_rank_first / max(n_with_feas, 1), 6),
            "mean_rank_margin":                      round(sum_margin / max(n_with_feas, 1), 6),
        }

    test_idx_with_pieces = test_idx
    mvp_idx              = np.where(is_mvp)[0]

    ranking_results = {
        "test_split":   rank_pieces("test_split", test_idx_with_pieces),
        "mvp_scenario": rank_pieces("mvp_scenario", mvp_idx),
    }
    for k, v in ranking_results.items():
        if v.get("skipped"):
            print(f"  [ranking {k}] SKIPPED")
            continue
        print(f"  [ranking {k:14s}] n_pieces={v['n_pieces']:4d} "
              f"with_feasible={v['n_pieces_with_at_least_one_feasible']:4d} "
              f"top1={v['top1_feasible_accuracy']:.4f} "
              f"MRR={v['mean_reciprocal_rank']:.4f} "
              f"meanrank={v['mean_rank_first_feasible']:.4f} "
              f"margin={v['mean_rank_margin']:.4f}")

    # 8. Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Save model
    torch.save({
        "state_dict":  model.state_dict(),
        "embed_dim":   EMBED_DIM,
        "best_epoch":  best_epoch,
        "best_val_f1": best_val_f1,
    }, OUT_MODEL_PT)
    print(f"\n[write] {OUT_MODEL_PT}")

    # Training curves CSV
    print(f"[write] {OUT_CURVES_CSV}")
    with OUT_CURVES_CSV.open("w") as f:
        if curves:
            keys = list(curves[0].keys())
            f.write(",".join(keys) + "\n")
            for r in curves:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

    # Predictions CSV
    print(f"[write] {OUT_PREDS_CSV}")
    with OUT_PREDS_CSV.open("w") as f:
        if pred_records:
            keys = list(pred_records[0].keys())
            f.write(",".join(keys) + "\n")
            for r in pred_records:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

    # Confusion matrices CSV
    print(f"[write] {OUT_CM_CSV}")
    with OUT_CM_CSV.open("w") as f:
        f.write("scope,n,tp,fp,tn,fn,accuracy,precision,recall,f1,roc_auc\n")
        for sname in ("train", "val", "test"):
            m = final_metrics[sname]
            cm = m["confusion_matrix"]
            f.write(f"standard/{sname},{m['n']},{cm['tp']},{cm['fp']},{cm['tn']},{cm['fn']},"
                    f"{m['accuracy']},{m['precision']},{m['recall']},{m['f1']},{m['roc_auc']}\n")
        for fam, m in per_family_metrics.items():
            cm = m["confusion_matrix"]
            f.write(f"per_family/{fam},{m['n']},{cm['tp']},{cm['fp']},{cm['tn']},{cm['fn']},"
                    f"{m['accuracy']},{m['precision']},{m['recall']},{m['f1']},{m['roc_auc']}\n")

    # Results JSON
    payload = {
        "schema_version": 1,
        "script_name":    "train_phaseE_siamese_embedding.py",
        "phase":          "Phase E.2 — siamese SDF embedding training",
        "phase_note": (
            "The model learns a geometric compatibility embedding from SDF "
            "footprints; it does NOT learn robot control or visual perception. "
            "Single training run; no hyperparameter tuning."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "device":        str(device),
        "n_params":      n_params,
        "config": {
            "seed":              SEED,
            "embed_dim":         EMBED_DIM,
            "batch_size":        BATCH_SIZE,
            "learning_rate":     LEARNING_RATE,
            "max_epochs":        MAX_EPOCHS,
            "early_stop_patience": EARLY_STOP_PATIENCE,
            "pos_weight_value":  pos_weight_value,
            "sdf_int8_scale":    SDF_INT8_SCALE,
            "sdf_norm_div_mm":   SDF_NORM_DIV_MM,
        },
        "best_epoch":     best_epoch,
        "best_val_f1":    round(best_val_f1, 6),
        "curves":         curves,
        "final_metrics":  final_metrics,
        "per_family_test": per_family_metrics,
        "ranking":        ranking_results,
        "phaseD_baseline_reference": {
            "logreg_test_F1": 0.829,
            "tree_test_F1":   0.871,
        },
    }
    print(f"[write] {OUT_RESULTS_JSON}")
    OUT_RESULTS_JSON.write_text(json.dumps(payload, indent=2))

    # Markdown report
    print(f"[write] {OUT_REPORT_MD}")
    write_report(payload, OUT_REPORT_MD)

    print("[done].")


def write_report(payload: dict, path: Path) -> None:
    cfg = payload["config"]
    fm  = payload["final_metrics"]
    lines = []
    lines.append("# Phase E.2 — Siamese CNN SDF Embedding (training + evaluation)")
    lines.append("")
    lines.append("> **This model learns a geometric compatibility embedding from "
                 "SDF footprints; it does not learn robot control or visual "
                 "perception.**")
    lines.append("")
    lines.append("> Status: Phase E.2 (single training run). No hyperparameter "
                 "tuning, no architecture search, no LOFO, no robustness tests.")
    lines.append("")

    lines.append("## Objective")
    lines.append("")
    lines.append("Test whether a minimal siamese CNN can learn a useful "
                 "geometric compatibility representation from 128×128 signed "
                 "distance fields of piece and cavity footprints, using the "
                 "Phase E.1 dataset and Phase D.7 partial-insertion labels.")
    lines.append("")

    lines.append("## Dataset source")
    lines.append("")
    lines.append("- `data/phaseE_learned_embeddings/phaseE_sdf_pairs.npz` (int8-quantised SDFs)")
    lines.append("- `data/phaseE_learned_embeddings/phaseE_pairs_metadata.csv` (split / family / piece_id / cavity_id / rotation)")
    lines.append("")

    lines.append("## Model architecture")
    lines.append("")
    lines.append("Shared encoder (1-channel SDF in ∈ [-1, 1]):")
    lines.append("```")
    lines.append("Conv2d(1,16,3,p=1) -> ReLU -> MaxPool2d(2)   # (16,64,64)")
    lines.append("Conv2d(16,32,3,p=1) -> ReLU -> MaxPool2d(2)  # (32,32,32)")
    lines.append("Conv2d(32,64,3,p=1) -> ReLU -> MaxPool2d(2)  # (64,16,16)")
    lines.append("AdaptiveAvgPool2d(1) -> flatten              # (64,)")
    lines.append(f"Linear(64 -> {cfg['embed_dim']})            # embedding")
    lines.append("L2-normalise embedding")
    lines.append("```")
    lines.append("")
    lines.append("Compatibility head:")
    lines.append("```")
    lines.append("features = concat(|e_p - e_c|, e_p * e_c, cosine(e_p, e_c))")
    lines.append("Linear(2D + 1 -> 32) -> ReLU -> Linear(32 -> 1) -> logit")
    lines.append("```")
    lines.append("")
    lines.append(f"- Total parameters: **{payload['n_params']}**")
    lines.append(f"- Loss: BCEWithLogitsLoss with `pos_weight = {cfg['pos_weight_value']:.4f}` "
                 f"(class-imbalance correction).")
    lines.append("")

    lines.append("## Training configuration")
    lines.append("")
    lines.append(f"- Seed: {cfg['seed']}")
    lines.append(f"- Batch size: {cfg['batch_size']}")
    lines.append(f"- Optimizer: Adam, lr = {cfg['learning_rate']}")
    lines.append(f"- Max epochs: {cfg['max_epochs']}; early-stop on val F1, "
                 f"patience {cfg['early_stop_patience']}")
    lines.append(f"- SDF dequantise: `sdf_mm = sdf_int8 / {cfg['sdf_int8_scale']:.4f}`")
    lines.append(f"- SDF normalisation to [-1, 1]: divide by {cfg['sdf_norm_div_mm']} mm")
    lines.append(f"- Device: `{payload['device']}`")
    lines.append(f"- **Best epoch**: {payload['best_epoch']} "
                 f"(val_F1 = {payload['best_val_f1']:.4f})")
    lines.append("")

    lines.append("## Standard split metrics (best model)")
    lines.append("")
    lines.append("| split | n | acc | prec | recall | F1 | AUC | pos_rate_pred | degenerate |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---|")
    for sname in ("train", "val", "test"):
        m = fm[sname]
        auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "—"
        lines.append(f"| {sname} | {m['n']} | {m['accuracy']:.4f} | "
                     f"{m['precision']:.4f} | {m['recall']:.4f} | "
                     f"{m['f1']:.4f} | {auc_str} | "
                     f"{m['positive_rate_pred']:.4f} | "
                     f"{'YES' if m['degenerate_flag'] else 'no'} |")
    lines.append("")

    lines.append("## Per-family test metrics")
    lines.append("")
    lines.append("| family | n | acc | prec | recall | F1 | AUC |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for fam, m in payload["per_family_test"].items():
        auc_str = f"{m['roc_auc']:.4f}" if m['roc_auc'] is not None else "—"
        lines.append(f"| `{fam}` | {m['n']} | {m['accuracy']:.4f} | "
                     f"{m['precision']:.4f} | {m['recall']:.4f} | "
                     f"{m['f1']:.4f} | {auc_str} |")
    lines.append("")

    lines.append("## Ranking metrics")
    lines.append("")
    lines.append("Ranking groups by `piece_id`, takes max-over-rotations per cavity, ranks cavities, and reports.")
    lines.append("")
    lines.append("| scope | n_pieces | with_feasible | top-1 | MRR | mean_rank | mean_margin |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for k, r in payload["ranking"].items():
        if r.get("skipped"):
            lines.append(f"| {k} | — | — | — | — | — | — |")
            continue
        lines.append(f"| {k} | {r['n_pieces']} | "
                     f"{r['n_pieces_with_at_least_one_feasible']} | "
                     f"{r['top1_feasible_accuracy']:.4f} | "
                     f"{r['mean_reciprocal_rank']:.4f} | "
                     f"{r['mean_rank_first_feasible']:.4f} | "
                     f"{r['mean_rank_margin']:.4f} |")
    lines.append("")

    lines.append("## Comparison vs Phase D hand-crafted models")
    lines.append("")
    lines.append("| metric | Phase D logreg (test) | Phase D tree (test) | Phase E siamese (test) |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| F1 | 0.829 | 0.871 | {fm['test']['f1']:.4f} |")
    lines.append("")
    lines.append("Phase D values quoted from `data/phaseD_3d_affordance/models/phaseD_training_results.json`.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Single training run; no hyperparameter tuning.")
    lines.append("- LOFO not evaluated in this turn (Phase E.3 if pursued).")
    lines.append("- Robustness perturbations not applied (Phase E.4 if pursued).")
    lines.append("- Synthetic dataset only; convex prismatic shapes; rotations only (no XY offset).")
    lines.append("- Perception pipeline frozen; embeddings are learned; perception is not.")
    lines.append("- The model learns a geometric compatibility decision boundary, not robot control or insertion physics.")
    lines.append("")

    lines.append("## Closing note")
    lines.append("")
    lines.append("The siamese CNN learns a 64-D embedding from SDF footprints and "
                 "predicts compatibility from a small head over `(|diff|, product, "
                 "cosine)` of paired embeddings. It does NOT learn robot control, "
                 "insertion execution, or visual perception. Whether the embedding "
                 "generalises across shape families and degrades gracefully under "
                 "perturbation are separate experiments (Phase E.3, E.4).")

    path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
