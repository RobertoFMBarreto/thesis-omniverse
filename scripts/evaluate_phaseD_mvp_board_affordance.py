"""
evaluate_phaseD_mvp_board_affordance.py

Phase D.6 — evaluate the trained Phase D affordance model on the actual
Baseline 1 board scenario.

For each MVP piece (rectangle, square, circle, triangle):
  - Compute 3D-extrusion piece descriptors from the captured top-down
    point cloud + measured piece height.
  - For each actual board cavity (cavity_00..03):
      - Compute 3D-extrusion cavity descriptors from the captured opening
        point cloud + measured cavity depth.
      - Sweep rotations 0, 10, ..., 350 degrees (dx=dy=0).
      - Compute the same pair / action features used at training.
      - Predict affordance probability with the Phase D models (logreg, tree).
      - cavity_score = max over rotations.
  - Rank cavities by cavity_score; report rank-1, margin, comparison vs
    Baseline 1 reference (rectangle->cavity_00, square->cavity_02,
    circle->cavity_03, triangle->cavity_01).

Models are reloaded by retraining on the SAME Phase D dataset with the
SAME hyperparameters (no tuning, no feature changes).
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        "/Users/robertofmbarreto/Documents/Mestrado/tese/code/thesis-omniverse",
    )
)

# Inputs
DATASET_CSV     = PROJECT_ROOT / "data" / "phaseD_3d_affordance" / "configurations_labelled.csv"
PIECES_DIR      = PROJECT_ROOT / "data" / "pieces_detected"
CAVITIES_DIR    = PROJECT_ROOT / "data" / "cavities_detected"

# Outputs
OUT_DIR         = PROJECT_ROOT / "data" / "phaseD_3d_affordance" / "mvp_board_eval"
OUT_RESULTS_JSON = OUT_DIR / "mvp_board_affordance_results.json"
OUT_RANKING_CSV  = OUT_DIR / "mvp_board_affordance_ranking.csv"
OUT_REPORT_MD    = OUT_DIR / "mvp_board_affordance_report.md"

# Phase D.3/D.4 hyperparameters (do NOT tune)
LR_C        = 1.0
LR_MAX_ITER = 5000
DT_MAX_DEPTH = 4
DT_RANDOM_STATE = 0

ID_COLS  = {"config_id", "piece_id", "cavity_id", "shape_family", "is_mvp",
             "cavity_source", "split", "heldout_family_fold"}
DIAG_COLS = {"diag_inside_ratio_raw", "diag_outside_ratio_raw",
              "diag_p_area_px", "diag_c_area_px", "diag_label_reason",
              "diag_insertion_required_mm", "label_reason"}
LABEL_COL = "label"

# MVP pieces and Baseline 1 reference mapping
MVP_PIECES = ["rectangle", "square", "circle", "triangle"]
BOARD_CAVITIES = ["cavity_00", "cavity_01", "cavity_02", "cavity_03"]
BASELINE1_REFERENCE = {
    "rectangle": "cavity_00",
    "square":    "cavity_02",
    "circle":    "cavity_03",
    "triangle":  "cavity_01",
}

# Shared rasteriser config (mirrors generate_phaseD_3d_affordance_dataset.py)
CANVAS_PX           = 320
RES_M_PER_PX        = 0.00025
WORLD_HALF_CANVAS_M = (CANVAS_PX * RES_M_PER_PX) / 2.0
CLEARANCE_PER_SIDE_MM = 0.5
CLEARANCE_PX = int(round((CLEARANCE_PER_SIDE_MM / 1000.0) / RES_M_PER_PX))
DIL_KERNEL = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (2 * CLEARANCE_PX + 1, 2 * CLEARANCE_PX + 1),
)

ROTATIONS_DEG = list(range(0, 360, 10))  # 36 angles


# ── CSV loading ───────────────────────────────────────────────────────────────

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
    cols = {k: [] for k in rows[0].keys()}
    for r in rows:
        for k, v in r.items():
            cols[k].append(parse_value(v))
    return cols


def determine_feature_columns(all_columns: list[str], cols: dict) -> list[str]:
    excluded = ID_COLS | DIAG_COLS | {LABEL_COL}
    candidates = [c for c in all_columns if c not in excluded]
    return [c for c in candidates
            if all(isinstance(v, (int, float)) and v is not None
                   for v in cols[c])]


# ── Models (deterministic re-fit) ─────────────────────────────────────────────

def fit_models(X_train: np.ndarray, y_train: np.ndarray) -> tuple:
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.pipeline import Pipeline
    lr = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=LR_C, class_weight="balanced", max_iter=LR_MAX_ITER,
            solver="lbfgs", random_state=0,
        )),
    ]).fit(X_train, y_train)
    dt = DecisionTreeClassifier(
        max_depth=DT_MAX_DEPTH, class_weight="balanced",
        random_state=DT_RANDOM_STATE,
    ).fit(X_train, y_train)
    return lr, dt


# ── Geometry helpers (mirror dataset script) ──────────────────────────────────

def shoelace_area_mm2(xy_mm: np.ndarray) -> float:
    x = xy_mm[:, 0]
    y = xy_mm[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def perimeter_mm(xy_mm: np.ndarray) -> float:
    pts = np.vstack([xy_mm, xy_mm[:1]])
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def bbox_aspect_ratio(xy_mm: np.ndarray) -> float:
    w = float(xy_mm[:, 0].max() - xy_mm[:, 0].min())
    h = float(xy_mm[:, 1].max() - xy_mm[:, 1].min())
    long_side  = max(w, h)
    short_side = max(min(w, h), 1e-9)
    return long_side / short_side


def compactness(area_mm2: float, perim_mm: float) -> float:
    if perim_mm <= 0.0:
        return 0.0
    return float(4.0 * math.pi * area_mm2 / (perim_mm ** 2))


def rotate_xy(xy_mm: np.ndarray, theta_deg: float) -> np.ndarray:
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    return xy_mm @ np.array([[c, s], [-s, c]], dtype=np.float64)


def rasterise_polygon_filled(xy_mm: np.ndarray) -> np.ndarray:
    u = (xy_mm[:, 0] / 1000.0 + WORLD_HALF_CANVAS_M) / RES_M_PER_PX
    v = (WORLD_HALF_CANVAS_M - xy_mm[:, 1] / 1000.0) / RES_M_PER_PX
    pts = np.stack([u, v], axis=1).astype(np.int32)
    mask = np.zeros((CANVAS_PX, CANVAS_PX), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def hull_polygon_from_points_m(xy_m: np.ndarray) -> np.ndarray:
    """
    Convex hull of XY samples (in metres), centroid-centred, returned in mm.

    Captured point clouds are sparse; the convex hull provides a consistent
    polygon outline analogous to the analytical polygons used in training.
    """
    xy_mm = xy_m * 1000.0
    xy_mm = xy_mm - xy_mm.mean(axis=0)   # centroid-centre
    hull = cv2.convexHull(xy_mm.astype(np.float32)).reshape(-1, 2)
    return hull.astype(np.float64)


# ── Real piece / cavity loaders ───────────────────────────────────────────────

def load_mvp_piece(name: str) -> dict:
    pc_path   = PIECES_DIR / name / "piece_pointcloud.npy"
    meta_path = PIECES_DIR / name / "piece_metadata.json"
    if not pc_path.exists():
        raise FileNotFoundError(f"piece pointcloud missing: {pc_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"piece metadata missing: {meta_path}")
    pc = np.load(str(pc_path)).astype(np.float64)
    meta = json.loads(meta_path.read_text())
    if pc.ndim != 2 or pc.shape[1] < 2:
        raise RuntimeError(f"bad piece pointcloud shape: {pc.shape}")

    xy_m = pc[:, :2]
    hull_mm = hull_polygon_from_points_m(xy_m)
    height_m = float(meta.get("piece_height_median_m"))
    if not math.isfinite(height_m) or height_m <= 0:
        raise RuntimeError(f"invalid piece height for {name}: {height_m}")
    height_mm = height_m * 1000.0

    area      = shoelace_area_mm2(hull_mm)
    perim     = perimeter_mm(hull_mm)
    compact   = compactness(area, perim)
    bbox_ar   = bbox_aspect_ratio(hull_mm)
    return {
        "name":       name,
        "hull_mm":    hull_mm,
        "area_mm2":   area,
        "perim_mm":   perim,
        "compact":    compact,
        "height_mm":  height_mm,
        "volume_mm3": area * height_mm,
        "bbox_ar":    bbox_ar,
    }


def load_board_cavity(name: str) -> dict:
    pc_path   = CAVITIES_DIR / name / "cavity_opening_pointcloud.npy"
    if not pc_path.exists():
        # Fall back to the full cavity_pointcloud.npy if opening absent
        pc_path = CAVITIES_DIR / name / "cavity_pointcloud.npy"
    meta_path = CAVITIES_DIR / name / "cavity_metadata.json"
    if not pc_path.exists():
        raise FileNotFoundError(f"cavity pointcloud missing for {name}")
    if not meta_path.exists():
        raise FileNotFoundError(f"cavity metadata missing for {name}")
    pc = np.load(str(pc_path)).astype(np.float64)
    meta = json.loads(meta_path.read_text())
    if pc.ndim != 2 or pc.shape[1] < 2:
        raise RuntimeError(f"bad cavity pointcloud shape: {pc.shape}")

    xy_m = pc[:, :2]
    hull_mm = hull_polygon_from_points_m(xy_m)

    # Phase D.7 — CAD-nominal cavity depth applied UNIFORMLY across all four
    # cavities. The captured z_depth_median_m is sensor-range-limited
    # (saturates well before reaching the cavity floor at 75 mm); using it
    # would introduce a systematic out-of-distribution input to the
    # partial-insertion classifier. Read the nominal board thickness from
    # data/expected_cad_dimensions.json once and apply to every cavity.
    cad_path = PROJECT_ROOT / "data" / "expected_cad_dimensions.json"
    try:
        cad = json.loads(cad_path.read_text())
        depth_m = float(cad.get("board", {}).get("thickness_m", 0.075))
    except Exception:
        depth_m = 0.075
    depth_source = "cad_nominal_board_thickness_uniform"
    depth_mm = depth_m * 1000.0

    area    = shoelace_area_mm2(hull_mm)
    perim   = perimeter_mm(hull_mm)
    compact = compactness(area, perim)
    bbox_ar = bbox_aspect_ratio(hull_mm)
    return {
        "name":         name,
        "hull_mm":      hull_mm,
        "area_mm2":     area,
        "perim_mm":     perim,
        "compact":      compact,
        "depth_mm":     depth_mm,
        "depth_source": depth_source,
        "volume_mm3":   area * depth_mm,
        "bbox_ar":      bbox_ar,
    }


# ── Pair feature computation ──────────────────────────────────────────────────

def compute_iou_at_rotation(piece_hull_mm: np.ndarray, cavity_hull_mm: np.ndarray,
                              rot_deg: float) -> float:
    piece_rot = rotate_xy(piece_hull_mm, rot_deg)
    mp = rasterise_polygon_filled(piece_rot)
    mc = rasterise_polygon_filled(cavity_hull_mm)
    p_bool = mp > 0
    c_bool = mc > 0
    inter  = int((p_bool & c_bool).sum())
    union  = int((p_bool | c_bool).sum())
    return inter / max(union, 1)


def build_feature_vector(piece: dict, cavity: dict, rot_deg: float,
                          feature_names: list[str]) -> np.ndarray:
    """
    Build the feature vector EXACTLY in the column order used at training.
    """
    iou_val = compute_iou_at_rotation(piece["hull_mm"], cavity["hull_mm"], rot_deg)

    feats = {
        "candidate_rotation_deg":      float(rot_deg),
        "piece_area_mm2":              piece["area_mm2"],
        "piece_perimeter_mm":          piece["perim_mm"],
        "piece_compactness":           piece["compact"],
        "piece_height_mm":             piece["height_mm"],
        "piece_volume_mm3":            piece["volume_mm3"],
        "piece_bbox_aspect_ratio":     piece["bbox_ar"],
        "cavity_area_mm2":             cavity["area_mm2"],
        "cavity_perimeter_mm":         cavity["perim_mm"],
        "cavity_compactness":          cavity["compact"],
        "cavity_depth_mm":             cavity["depth_mm"],
        "cavity_volume_mm3":           cavity["volume_mm3"],
        "cavity_bbox_aspect_ratio":    cavity["bbox_ar"],
        "area_ratio":                  piece["area_mm2"] / max(cavity["area_mm2"], 1e-6),
        "depth_offset_mm":             cavity["depth_mm"] - piece["height_mm"],
        "bbox_aspect_diff":            abs(piece["bbox_ar"] - cavity["bbox_ar"]),
        "compactness_diff":            abs(piece["compact"] - cavity["compact"]),
        "iou":                         iou_val,
        "lateral_clearance_proxy_mm2": cavity["area_mm2"] - piece["area_mm2"],
    }

    return np.array([feats[name] for name in feature_names], dtype=np.float64)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("evaluate_phaseD_mvp_board_affordance.py")
    print("=" * 70)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print()

    if not DATASET_CSV.exists():
        print(f"[FATAL] dataset CSV not found at {DATASET_CSV}")
        sys.exit(1)

    # 1. Load dataset and re-fit models (matches Phase D.3 exactly)
    rows = load_csv(DATASET_CSV)
    cols = rows_to_columns(rows)
    feature_cols = determine_feature_columns(list(cols.keys()), cols)
    print(f"[features] {len(feature_cols)} features (mirroring Phase D.3)")
    for f in feature_cols:
        print(f"  - {f}")

    X = np.array([[float(cols[c][i]) for c in feature_cols]
                  for i in range(len(rows))], dtype=np.float64)
    y      = np.array(cols[LABEL_COL], dtype=np.int32)
    splits = np.array(cols["split"])
    train_mask = splits == "train"
    print(f"[train] re-fitting logreg + tree on {int(train_mask.sum())} train rows ...")
    lr_model, dt_model = fit_models(X[train_mask], y[train_mask])

    # 2. Load MVP pieces and board cavities
    pieces = {}
    for name in MVP_PIECES:
        try:
            pieces[name] = load_mvp_piece(name)
            p = pieces[name]
            print(f"[piece {name}] hull_pts={len(p['hull_mm'])}  "
                  f"area={p['area_mm2']:.1f} mm²  perim={p['perim_mm']:.1f} mm  "
                  f"compact={p['compact']:.3f}  height={p['height_mm']:.1f} mm  "
                  f"volume={p['volume_mm3']:.1f} mm³  bbox_ar={p['bbox_ar']:.3f}")
        except Exception as exc:
            print(f"[piece {name}] FAILED to load: {exc}")
            sys.exit(1)

    cavities = {}
    for name in BOARD_CAVITIES:
        try:
            cavities[name] = load_board_cavity(name)
            c = cavities[name]
            print(f"[cavity {name}] hull_pts={len(c['hull_mm'])}  "
                  f"area={c['area_mm2']:.1f} mm²  perim={c['perim_mm']:.1f} mm  "
                  f"compact={c['compact']:.3f}  depth={c['depth_mm']:.1f} mm  "
                  f"volume={c['volume_mm3']:.1f} mm³  bbox_ar={c['bbox_ar']:.3f}")
        except Exception as exc:
            print(f"[cavity {name}] FAILED to load: {exc}")
            sys.exit(1)

    # 3. Evaluate per (piece, cavity) over the rotation sweep
    per_pair_data: dict[str, dict[str, dict]] = {}
    print("\n[scoring] sweeping rotations for each piece-cavity pair ...")
    for piece_name in MVP_PIECES:
        per_pair_data[piece_name] = {}
        for cav_name in BOARD_CAVITIES:
            piece = pieces[piece_name]
            cavity = cavities[cav_name]

            best_lr_score = -1.0
            best_lr_rot   = 0
            best_dt_score = -1.0
            best_dt_rot   = 0
            for rot_deg in ROTATIONS_DEG:
                feat_vec = build_feature_vector(piece, cavity, rot_deg, feature_cols)
                fv = feat_vec.reshape(1, -1)
                lr_prob = float(lr_model.predict_proba(fv)[0, 1])
                dt_prob = float(dt_model.predict_proba(fv)[0, 1])
                if lr_prob > best_lr_score:
                    best_lr_score, best_lr_rot = lr_prob, rot_deg
                if dt_prob > best_dt_score:
                    best_dt_score, best_dt_rot = dt_prob, rot_deg

            per_pair_data[piece_name][cav_name] = {
                "logreg_score":     round(best_lr_score, 6),
                "logreg_best_rot":  best_lr_rot,
                "tree_score":       round(best_dt_score, 6),
                "tree_best_rot":    best_dt_rot,
            }
            print(f"[score] {piece_name:10s} vs {cav_name}: "
                  f"logreg={best_lr_score:.4f}@{best_lr_rot:>3d}°  "
                  f"tree={best_dt_score:.4f}@{best_dt_rot:>3d}°")

    # 4. Rank cavities per piece for each model
    ranking = {"logreg": {}, "tree": {}}
    for piece_name in MVP_PIECES:
        pair = per_pair_data[piece_name]
        for model_name in ("logreg", "tree"):
            score_key = f"{model_name}_score"
            ranked = sorted(
                pair.items(), key=lambda kv: -kv[1][score_key],
            )
            for rank, (cav_name, vals) in enumerate(ranked, start=1):
                vals[f"{model_name}_rank"] = rank

            ref_cav = BASELINE1_REFERENCE[piece_name]
            rank_of_ref = next(r for r, (c, _) in enumerate(ranked, 1) if c == ref_cav)
            margin = (ranked[0][1][score_key] - ranked[1][1][score_key]
                      if len(ranked) > 1 else 0.0)
            ranking[model_name][piece_name] = {
                "ranking":         [(c, v[score_key]) for c, v in ranked],
                "rank_1":          ranked[0][0],
                "rank_1_score":    round(ranked[0][1][score_key], 6),
                "rank_1_best_rotation": ranked[0][1][f"{model_name}_best_rot"],
                "rank_2":          ranked[1][0],
                "rank_2_score":    round(ranked[1][1][score_key], 6),
                "margin":          round(float(margin), 6),
                "baseline1_reference":      ref_cav,
                "rank_of_baseline1_ref":    rank_of_ref,
                "matches_baseline1":        ranked[0][0] == ref_cav,
            }

    # 5. Top-1 accuracy per model
    summary = {}
    for model_name in ("logreg", "tree"):
        n_correct = sum(1 for pn in MVP_PIECES
                        if ranking[model_name][pn]["matches_baseline1"])
        summary[model_name] = {
            "n_correct_top1": n_correct,
            "n_pieces":       len(MVP_PIECES),
            "top1_accuracy":  round(n_correct / max(len(MVP_PIECES), 1), 6),
        }
        print(f"\n[summary {model_name}] top-1 = {n_correct}/{len(MVP_PIECES)} "
              f"({summary[model_name]['top1_accuracy']*100:.1f}%)")

    # 6. Write outputs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ranking CSV
    with OUT_RANKING_CSV.open("w") as f:
        f.write("piece,model,cavity,rank,score,best_rotation_deg,baseline1_reference,matches_ref\n")
        for piece_name in MVP_PIECES:
            for model_name in ("logreg", "tree"):
                ref = BASELINE1_REFERENCE[piece_name]
                for cav_name, score in ranking[model_name][piece_name]["ranking"]:
                    rank = per_pair_data[piece_name][cav_name][f"{model_name}_rank"]
                    rot  = per_pair_data[piece_name][cav_name][f"{model_name}_best_rot"]
                    matches = (rank == 1 and cav_name == ref)
                    f.write(f"{piece_name},{model_name},{cav_name},{rank},{score},{rot},{ref},{matches}\n")
    print(f"\n[write] {OUT_RANKING_CSV}")

    # JSON
    payload = {
        "schema_version": 1,
        "script_name":    "evaluate_phaseD_mvp_board_affordance.py",
        "phase":          "Phase D.6 — MVP-board affordance ranking",
        "phase_note": (
            "Evaluates the trained Phase D affordance models on the actual "
            "Baseline 1 board scenario. The model predicts a geometric "
            "insertion affordance score; it does NOT control the robot. "
            "Models are reproduced by re-fitting on the same Phase D dataset "
            "with the same hyperparameters; no tuning, no feature changes."
        ),
        "timestamp_utc":   datetime.now(timezone.utc).isoformat(),
        "feature_columns_used": feature_cols,
        "baseline1_reference": BASELINE1_REFERENCE,
        "pieces":   {k: {kk: vv for kk, vv in v.items() if kk != "hull_mm"}
                     for k, v in pieces.items()},
        "cavities": {k: {kk: vv for kk, vv in v.items() if kk != "hull_mm"}
                     for k, v in cavities.items()},
        "per_pair":  per_pair_data,
        "ranking":   ranking,
        "summary":   summary,
        "models": {
            "logreg": {"C": LR_C, "max_iter": LR_MAX_ITER,
                        "class_weight": "balanced", "scaler": "StandardScaler"},
            "tree":   {"max_depth": DT_MAX_DEPTH,
                        "class_weight": "balanced",
                        "random_state": DT_RANDOM_STATE},
        },
    }
    OUT_RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[write] {OUT_RESULTS_JSON}")

    write_report(payload, OUT_REPORT_MD)
    print(f"[write] {OUT_REPORT_MD}")
    print("[done].")


def write_report(payload: dict, path: Path) -> None:
    lines = []
    lines.append("# Phase D.6 — MVP-Board Affordance Ranking")
    lines.append("")
    lines.append("> **Evaluates the trained Phase D affordance model on the "
                 "actual Baseline 1 board scenario.** The model predicts a "
                 "geometric insertion affordance score; it does NOT control "
                 "the robot.")
    lines.append("")

    lines.append("## Objective")
    lines.append("")
    lines.append("For each MVP piece (rectangle, square, circle, triangle), "
                 "rank the four actual board cavities (cavity_00..03) by "
                 "predicted affordance score and compare against the "
                 "Baseline 1 deterministic reference.")
    lines.append("")

    lines.append("## Feature extraction source")
    lines.append("")
    lines.append("- **Pieces**: `data/pieces_detected/<piece>/piece_pointcloud.npy` "
                 "+ `piece_metadata.json` (`piece_height_median_m`).")
    lines.append("- **Cavities**: `data/cavities_detected/<cavity>/cavity_opening_pointcloud.npy` "
                 "+ `cavity_metadata.json` (`z_depth_median_m`).")
    lines.append("- Each captured point cloud is reduced to a **convex-hull polygon** "
                 "(consistent with the analytical polygons used at training).")
    lines.append("")

    lines.append("## Model used")
    lines.append("")
    lines.append("Both Phase D models are reproduced by re-fitting on the same "
                 "Phase D dataset with the same hyperparameters. **No tuning, "
                 "no feature changes.**")
    for model_name, mcfg in payload["models"].items():
        lines.append(f"- `{model_name}`: " + ", ".join(f"{k}=`{v}`" for k, v in mcfg.items()))
    lines.append("")

    lines.append("## Ranking procedure")
    lines.append("")
    lines.append("For each (piece, cavity): sweep 36 rotations (0°, 10°, …, 350°), "
                 "build the same 20-feature vector used at training, predict the "
                 "affordance probability, and take **cavity_score = max over "
                 "rotations**. Per piece, sort cavities by score descending; "
                 "rank-1 is the predicted insertion target. dx = dy = 0 (no XY offset).")
    lines.append("")

    lines.append("## Piece descriptors")
    lines.append("")
    lines.append("| piece | hull_pts | area_mm² | perim_mm | compact | height_mm | volume_mm³ | bbox_ar |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for piece_name in MVP_PIECES:
        p = payload["pieces"][piece_name]
        lines.append(f"| `{piece_name}` | n/a | {p['area_mm2']:.1f} | "
                     f"{p['perim_mm']:.1f} | {p['compact']:.3f} | "
                     f"{p['height_mm']:.1f} | {p['volume_mm3']:.1f} | "
                     f"{p['bbox_ar']:.3f} |")
    lines.append("")

    lines.append("## Cavity descriptors")
    lines.append("")
    lines.append("| cavity | area_mm² | perim_mm | compact | depth_mm | volume_mm³ | bbox_ar |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for cav_name in BOARD_CAVITIES:
        c = payload["cavities"][cav_name]
        lines.append(f"| `{cav_name}` | {c['area_mm2']:.1f} | "
                     f"{c['perim_mm']:.1f} | {c['compact']:.3f} | "
                     f"{c['depth_mm']:.1f} | {c['volume_mm3']:.1f} | "
                     f"{c['bbox_ar']:.3f} |")
    lines.append("")

    lines.append("## Per-piece rankings — logistic regression")
    lines.append("")
    write_ranking_table(lines, payload, "logreg")
    lines.append("")

    lines.append("## Per-piece rankings — decision tree")
    lines.append("")
    write_ranking_table(lines, payload, "tree")
    lines.append("")

    lines.append("## Comparison vs Baseline 1 deterministic reference")
    lines.append("")
    lines.append("Reference mapping: `rectangle → cavity_00`, "
                 "`square → cavity_02`, `circle → cavity_03`, "
                 "`triangle → cavity_01`.")
    lines.append("")
    lines.append("| piece | reference | logreg rank-1 | logreg ✓ | tree rank-1 | tree ✓ |")
    lines.append("|---|---|---|---|---|---|")
    for piece_name in MVP_PIECES:
        ref = payload["baseline1_reference"][piece_name]
        lr  = payload["ranking"]["logreg"][piece_name]
        dt  = payload["ranking"]["tree"][piece_name]
        lr_ok = "✓" if lr["matches_baseline1"] else "✗"
        dt_ok = "✓" if dt["matches_baseline1"] else "✗"
        lines.append(f"| `{piece_name}` | `{ref}` | "
                     f"`{lr['rank_1']}` (rank of ref = {lr['rank_of_baseline1_ref']}) | "
                     f"{lr_ok} | "
                     f"`{dt['rank_1']}` (rank of ref = {dt['rank_of_baseline1_ref']}) | "
                     f"{dt_ok} |")
    lines.append("")
    lr_top1 = payload["summary"]["logreg"]["top1_accuracy"]
    dt_top1 = payload["summary"]["tree"]["top1_accuracy"]
    lines.append(f"**Top-1 accuracy on the four MVP pieces**: "
                 f"logreg = {lr_top1*100:.1f}% "
                 f"({payload['summary']['logreg']['n_correct_top1']}/4); "
                 f"tree = {dt_top1*100:.1f}% "
                 f"({payload['summary']['tree']['n_correct_top1']}/4).")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- The model was trained ONLY on procedurally generated "
                 "convex prismatic shapes plus four MVP-derived "
                 "**procedurally constructed** cavities; the actual Baseline 1 "
                 "board cavities (`cavity_00..03`) were NOT in the training set.")
    lines.append("- Captured point clouds are sparse; descriptors are computed "
                 "from the convex hull. For non-convex captured outlines this "
                 "would lose detail (the MVP pieces are convex by design).")
    lines.append("- Cavity `depth_mm` comes from `z_depth_median_m` (Isaac Sim "
                 "depth-capture estimate), not the CAD nominal 75 mm.")
    lines.append("- Rotation sweep is 36 angles at 10° steps; no XY offset.")
    lines.append("- The ranking is over only 4 candidates; top-2 accuracy is "
                 "not informative (it is trivially 1.0 for any non-degenerate "
                 "ranker that finds the correct cavity in the first two).")
    lines.append("- The ranking selects an insertion location by learned "
                 "geometric affordance score; it does not control the robot.")
    lines.append("")

    lines.append("## Closing note")
    lines.append("")
    lines.append("The ranking outputs a top-1 cavity per MVP piece and a rank "
                 "margin. These are perception-side affordance signals only; "
                 "insertion execution, grasp planning, and robot control are "
                 "out of scope.")

    path.write_text("\n".join(lines) + "\n")


def write_ranking_table(lines: list[str], payload: dict, model_name: str) -> None:
    lines.append("| piece | rank | cavity | score | best_rotation | reference | match |")
    lines.append("|---|---:|---|---:|---:|---|---|")
    for piece_name in MVP_PIECES:
        ref = payload["baseline1_reference"][piece_name]
        rk  = payload["ranking"][model_name][piece_name]
        for rank, (cav_name, score) in enumerate(rk["ranking"], start=1):
            best_rot = payload["per_pair"][piece_name][cav_name][f"{model_name}_best_rot"]
            ref_str  = f"`{ref}`" if cav_name == ref else ""
            match    = "✓" if (rank == 1 and cav_name == ref) else ""
            lines.append(f"| `{piece_name}` | {rank} | `{cav_name}` | "
                         f"{score:.4f} | {best_rot}° | {ref_str} | {match} |")
        lines.append(f"| | | **margin (rank1 − rank2)** | "
                     f"**{rk['margin']:.4f}** | | | |")


if __name__ == "__main__":
    main()
