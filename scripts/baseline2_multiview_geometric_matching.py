"""
baseline2_multiview_geometric_matching.py

Baseline 2 — Phase B: deterministic multi-view geometric matching (MINIMAL).

Validation experiment for ONE research question:
    "Do additional deterministic viewpoints improve geometric discrimination?"

This is NOT multi-view fusion. It is NOT 3D reconstruction. It is NOT pose
estimation. It is score-level aggregation across per-view rasterisations.

Pipeline:
    For each piece in PIECE_NAMES:
        For each view in VIEW_NAMES:
            1. Load per-view depth + metadata.
            2. Back-project depth to world XYZ using per-view intrinsics +
               measured camera pose (USD convention: camera looks down its
               local -Z axis).
            3. Segment the piece by Z above the support surface.
            4. Centre XY on centroid; pass to Baseline 1's
               rasterise_xy_to_mask.
            5. Score against each cavity using Baseline 1's score_pair.
        Aggregate the three per-view best_scores via weighted average:
            top_down=0.6, front_oblique=0.2, side_oblique=0.2
        (Renormalise weights if a view is missing.)
    Rank cavities per piece. Compute first_vs_second_margin.
    Compute deterministic ambiguity indicators (low_margin, missing_view,
    per_view_disagreement).
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        "/Users/robertofmbarreto/Documents/Mestrado/tese/code/thesis-omniverse",
    )
)

# Make Baseline 1 helpers importable.
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from baseline1_geometric_matching import (   # noqa: E402
    rasterise_xy_to_mask,
    score_pair,
    TIE_MARGIN,
    _DIL_KERNEL,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

PIECE_NAMES = ["rectangle", "square", "circle", "triangle"]
VIEW_NAMES  = ["top_down", "front_oblique", "side_oblique"]

VIEW_WEIGHTS = {
    "top_down":      0.6,
    "front_oblique": 0.2,
    "side_oblique":  0.2,
}

MULTIVIEW_DIR = PROJECT_ROOT / "data" / "multiview_captures" / "pieces"
CAVITY_DIR    = PROJECT_ROOT / "data" / "cavities_detected"

OUT_RESULTS_JSON   = PROJECT_ROOT / "data" / "baseline2_multiview_matching_results.json"
OUT_MATRIX_CSV     = PROJECT_ROOT / "data" / "baseline2_multiview_matching_matrix.csv"
OUT_REPORT_MD      = PROJECT_ROOT / "data" / "baseline2_multiview_matching_report.md"

MIN_VIEW_POINTS                  = 50      # below this, view is marked missing
PIECE_HEIGHT_MIN_ABOVE_SURFACE_M = 0.002   # 2 mm above support surface


# ── Geometry helpers ──────────────────────────────────────────────────────────

def _quat_to_rot(q_wxyz: list) -> np.ndarray:
    """Convert (w, x, y, z) quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q_wxyz
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0.0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w/n, x/n, y/n, z/n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


def back_project_view(depth: np.ndarray, meta: dict) -> np.ndarray | None:
    """
    Back-project all valid depth pixels to world XYZ using the per-view
    intrinsics + measured camera pose.

    USD camera convention: camera local -Z is the forward (look) direction;
    +X is right; +Y is up. Pixel (u, v) at perpendicular depth d corresponds
    to camera-frame point ((u-cx)/fx * d, -(v-cy)/fy * d, -d).

    Returns Nx3 float64 in world frame, or None if no valid depth.
    """
    fx_px = float(meta["fx_px"])
    fy_px = float(meta["fy_px"])
    cx_px = float(meta["cx_px"])
    cy_px = float(meta["cy_px"])
    d_min = float(meta["depth_valid_min_m"])
    d_max = float(meta["depth_valid_max_m"])

    pose = meta["measured_pose_read_back_from_stage"]
    cam_pos = np.array(pose["position"], dtype=np.float64)
    cam_R   = _quat_to_rot(pose["quaternion"])

    H, W = depth.shape
    valid = np.isfinite(depth) & (depth > 0) & (depth >= d_min) & (depth <= d_max)
    if not valid.any():
        return None

    vs, us = np.where(valid)
    d = depth[vs, us].astype(np.float64)

    x_cam = (us.astype(np.float64) - cx_px) / fx_px * d
    y_cam = -(vs.astype(np.float64) - cy_px) / fy_px * d
    z_cam = -d

    cam_points = np.stack([x_cam, y_cam, z_cam], axis=1)
    world_points = cam_points @ cam_R.T + cam_pos
    return world_points


def segment_piece_world_xy(world_xyz: np.ndarray, meta: dict) -> np.ndarray | None:
    """
    Keep points whose world Z is above the support surface (target_bbox bottom).
    Drop the support-surface plane and any points below it.

    Returns Nx2 float32 (centroid-centred XY), or None if too few points.
    """
    target_center = meta.get("target_bbox_center_world_m")
    target_size_mm = meta.get("target_bbox_size_mm")
    if target_center is not None and target_size_mm is not None:
        # Support surface = bottom of the piece bbox
        surface_z = float(target_center[2]) - float(target_size_mm[2]) / 2.0 / 1000.0
    else:
        # Fallback: median Z of the lowest 10 % of points
        z_sorted = np.sort(world_xyz[:, 2])
        n = max(1, int(0.10 * len(z_sorted)))
        surface_z = float(np.median(z_sorted[:n]))

    above = world_xyz[:, 2] > (surface_z + PIECE_HEIGHT_MIN_ABOVE_SURFACE_M)
    pts = world_xyz[above]
    if len(pts) < MIN_VIEW_POINTS:
        return None

    xy = pts[:, :2].astype(np.float32)
    xy = xy - xy.mean(axis=0)   # centroid-centre, Baseline 1 convention
    return xy


# ── Scoring ───────────────────────────────────────────────────────────────────

def build_cavity_masks(cav_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rasterise cavity XY and dilate to produce (mask_dil, mask_undil)."""
    cav_xy_centred = cav_xy - cav_xy.mean(axis=0)
    mask_undil, _ = rasterise_xy_to_mask(cav_xy_centred)
    mask_dil = cv2.dilate(mask_undil, _DIL_KERNEL)
    return mask_dil, mask_undil


def score_view_against_cavity(
    piece_xy: np.ndarray, mask_c_dil: np.ndarray, mask_c_undil: np.ndarray
) -> dict:
    """Run the Baseline 1 rotation search; return the best record."""
    _, best = score_pair(piece_xy, mask_c_dil, mask_c_undil)
    return {
        "score":             float(best["score"]),
        "inside":            float(best["inside_ratio"]),
        "outside":           float(best["outside_ratio"]),
        "iou":               float(best["iou"]),
        "best_rotation_deg": int(best["rotation_deg"]),
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate_scores(per_view: dict) -> tuple[float | None, list[str]]:
    """
    Weighted average of per-view scores. Renormalise weights for missing views.
    Returns (aggregate_score_or_None, missing_views).
    """
    missing = [v for v in VIEW_NAMES if per_view[v]["missing"]]
    available = [v for v in VIEW_NAMES if not per_view[v]["missing"]]
    if not available:
        return None, missing

    raw_w = {v: VIEW_WEIGHTS[v] for v in available}
    total_w = sum(raw_w.values())
    norm_w = {v: w / total_w for v, w in raw_w.items()}

    agg = sum(per_view[v]["score"] * norm_w[v] for v in available)
    return float(agg), missing


# ── Pipeline driver ───────────────────────────────────────────────────────────

def load_cavities() -> dict:
    """Load all cavities from disk, rasterise once, return {cid: (mask_dil, mask_undil, cav_xy_meta)}."""
    cavities = {}
    cav_paths = sorted(CAVITY_DIR.glob("cavity_*"))
    for cav_path in cav_paths:
        if not cav_path.is_dir():
            continue
        pc_path = cav_path / "cavity_pointcloud.npy"
        if not pc_path.exists():
            print(f"[load_cavity] WARNING: missing {pc_path} — skipping {cav_path.name}")
            continue
        pc = np.load(str(pc_path)).astype(np.float32)
        if pc.ndim != 2 or pc.shape[1] < 2:
            print(f"[load_cavity] WARNING: bad shape {pc.shape} for {cav_path.name} — skipping")
            continue
        cav_xy = pc[:, :2]
        mask_dil, mask_undil = build_cavity_masks(cav_xy)
        cavities[cav_path.name] = {
            "mask_dil":   mask_dil,
            "mask_undil": mask_undil,
            "n_points":   int(len(cav_xy)),
        }
        print(f"[load_cavity] {cav_path.name}: {len(cav_xy)} pts  "
              f"undil={int((mask_undil>0).sum())}px  dil={int((mask_dil>0).sum())}px")
    return cavities


def process_view(piece_name: str, view_name: str) -> np.ndarray | None:
    """Load and segment one (piece, view). Returns centred XY or None if missing."""
    view_dir = None
    for d in (MULTIVIEW_DIR / piece_name).glob(f"view_*_{view_name}"):
        view_dir = d
        break
    if view_dir is None:
        print(f"[piece {piece_name}] [view {view_name}] MISSING: no view directory")
        return None

    depth_path = view_dir / "depth.npy"
    meta_path  = view_dir / "metadata.json"
    if not (depth_path.exists() and meta_path.exists()):
        print(f"[piece {piece_name}] [view {view_name}] MISSING: depth.npy or metadata.json absent")
        return None

    try:
        depth = np.load(str(depth_path))
        meta  = json.loads(meta_path.read_text())
    except Exception as exc:
        print(f"[piece {piece_name}] [view {view_name}] MISSING: load failed ({exc})")
        return None

    try:
        world = back_project_view(depth, meta)
        if world is None:
            print(f"[piece {piece_name}] [view {view_name}] MISSING: no valid depth pixels")
            return None
        xy = segment_piece_world_xy(world, meta)
        if xy is None:
            print(f"[piece {piece_name}] [view {view_name}] MISSING: < {MIN_VIEW_POINTS} segmented points")
            return None
        print(f"[piece {piece_name}] [view {view_name}] segmented {len(xy)} XY points")
        return xy
    except Exception as exc:
        print(f"[piece {piece_name}] [view {view_name}] MISSING: pipeline error ({exc})")
        return None


def rank_cavities_for_piece(per_cavity: dict) -> list[dict]:
    """Sort cavity records by aggregate_score desc; assign ranks; return list."""
    ranked = sorted(
        per_cavity.values(),
        key=lambda r: (r["aggregate_score"] is None, -(r["aggregate_score"] or 0.0)),
    )
    for idx, rec in enumerate(ranked, start=1):
        rec["rank"] = idx
    return ranked


def per_view_disagreement(per_view_argmax: dict) -> bool:
    """True if available views disagree on best cavity."""
    cavs = [c for c in per_view_argmax.values() if c is not None]
    return len(set(cavs)) > 1 if cavs else False


# ── Output writers ────────────────────────────────────────────────────────────

def write_results_json(results: list, run_meta: dict) -> None:
    payload = {
        "schema_version": 1,
        "script_name":    "baseline2_multiview_geometric_matching.py",
        "phase":          "Baseline 2 — Phase B (minimal validation)",
        "phase_note": (
            "Minimal validation only. NOT multi-view fusion. NOT 3D reconstruction. "
            "NOT pose estimation. Score-level aggregation across per-view rasterisations."
        ),
        "view_weights":   VIEW_WEIGHTS,
        "timestamp_utc":  run_meta["timestamp_utc"],
        "run_id":         run_meta["run_id"],
        "results":        results,
    }
    OUT_RESULTS_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_RESULTS_JSON.write_text(json.dumps(payload, indent=2))
    print(f"[write] {OUT_RESULTS_JSON}")


def write_matrix_csv(score_matrix: dict, cavity_names: list) -> None:
    OUT_MATRIX_CSV.parent.mkdir(parents=True, exist_ok=True)
    lines = ["piece," + ",".join(cavity_names)]
    for piece in PIECE_NAMES:
        row = [piece]
        for cav in cavity_names:
            v = score_matrix.get(piece, {}).get(cav)
            row.append("" if v is None else f"{v:.6f}")
        lines.append(",".join(row))
    OUT_MATRIX_CSV.write_text("\n".join(lines) + "\n")
    print(f"[write] {OUT_MATRIX_CSV}")


def write_report_md(results: list, score_matrix: dict, cavity_names: list,
                     run_meta: dict) -> None:
    lines = []
    lines.append("# Baseline 2 — Phase B (minimal): multi-view geometric matching")
    lines.append("")
    lines.append("> **NOT multi-view fusion. NOT 3D reconstruction. NOT pose "
                 "estimation.** Score-level aggregation across per-view "
                 "rasterisations only.")
    lines.append("")
    lines.append(f"- Run id: `{run_meta['run_id']}`")
    lines.append(f"- Timestamp (UTC): `{run_meta['timestamp_utc']}`")
    lines.append("")

    lines.append("## Objective")
    lines.append("")
    lines.append("Test ONE research question: do additional deterministic "
                 "viewpoints (top + front + side) improve geometric "
                 "discrimination over the single-view Baseline 1?")
    lines.append("")

    lines.append("## Methodology")
    lines.append("")
    lines.append("For each (piece, view), the per-view depth map is "
                 "back-projected to world XYZ using the per-view intrinsics "
                 "and the measured camera pose (USD convention, camera looks "
                 "along local -Z). Points above the support surface are "
                 "kept, centroid-centred, and rasterised via Baseline 1's "
                 "`rasterise_xy_to_mask` (320x320 px @ 0.25 mm/px, with "
                 "convex-hull representation-normalisation when the splat "
                 "is fragmented). Each per-view mask is scored against "
                 "every cavity using Baseline 1's `score_pair` (180-rotation "
                 "search; `inside`/`outside` on dilated cavity, IoU on "
                 "non-dilated). The three per-view best scores are combined "
                 "via weighted average (renormalised when a view is missing).")
    lines.append("")

    lines.append("## Descriptors used")
    lines.append("")
    lines.append("Per view: `inside_ratio`, `outside_ratio`, `iou`, "
                 "`best_score = W_IOU·iou + W_INSIDE·inside − W_OUTSIDE·outside` "
                 "(weights inherited from Baseline 1). No additional "
                 "descriptors.")
    lines.append("")

    lines.append("## Aggregation strategy")
    lines.append("")
    lines.append(f"Weighted average with hardcoded weights "
                 f"`top_down={VIEW_WEIGHTS['top_down']}`, "
                 f"`front_oblique={VIEW_WEIGHTS['front_oblique']}`, "
                 f"`side_oblique={VIEW_WEIGHTS['side_oblique']}`. "
                 "Missing-view weights are dropped and remaining weights are "
                 "renormalised to sum to 1.")
    lines.append("")

    lines.append("## Aggregate score matrix")
    lines.append("")
    header = "| piece | " + " | ".join(cavity_names) + " |"
    sep    = "|" + "|".join(["---"] * (len(cavity_names) + 1)) + "|"
    lines.append(header)
    lines.append(sep)
    for piece in PIECE_NAMES:
        row = [piece]
        for cav in cavity_names:
            v = score_matrix.get(piece, {}).get(cav)
            row.append("—" if v is None else f"{v:.4f}")
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    lines.append("## Per-piece ranking results")
    lines.append("")
    lines.append("| piece | rank-1 | score | rank-2 | margin | low_margin | missing_view | per_view_disagreement |")
    lines.append("|---|---|---|---|---|---|---|---|")
    by_piece = {p: [r for r in results if r["piece"] == p] for p in PIECE_NAMES}
    for piece in PIECE_NAMES:
        recs = sorted(by_piece[piece], key=lambda r: r["rank"])
        if not recs:
            lines.append(f"| {piece} | — | — | — | — | — | — | — |")
            continue
        r1 = recs[0]
        r2 = recs[1] if len(recs) > 1 else {"cavity": "—", "aggregate_score": None}
        margin = r1.get("first_vs_second_margin")
        ai = r1["ambiguity_indicators"]
        r1_score = r1["aggregate_score"]
        r1_score_s = "—" if r1_score is None else f"{r1_score:.4f}"
        margin_s   = "—" if margin   is None else f"{margin:.4f}"
        lines.append(
            f"| {piece} | {r1['cavity']} | {r1_score_s} | "
            f"{r2['cavity']} | {margin_s} | "
            f"{ai['low_margin']} | {ai['missing_view']} | {ai['per_view_disagreement']} |"
        )
    lines.append("")

    lines.append("## Ambiguity indicators summary")
    lines.append("")
    rank1_records = [r for r in results if r["rank"] == 1]
    n_low_margin = sum(1 for r in rank1_records if r["ambiguity_indicators"]["low_margin"])
    n_missing    = sum(1 for r in rank1_records if r["ambiguity_indicators"]["missing_view"])
    n_disagree   = sum(1 for r in rank1_records if r["ambiguity_indicators"]["per_view_disagreement"])
    lines.append(f"- Rank-1 pairs with `low_margin`: {n_low_margin} / {len(rank1_records)}")
    lines.append(f"- Rank-1 pairs with `missing_view`: {n_missing} / {len(rank1_records)}")
    lines.append(f"- Rank-1 pairs with `per_view_disagreement`: {n_disagree} / {len(rank1_records)}")
    lines.append("")

    lines.append("## Missing-view warnings")
    lines.append("")
    any_missing = False
    for piece in PIECE_NAMES:
        recs = by_piece.get(piece, [])
        for r in recs:
            mv = r.get("missing_views", [])
            if mv:
                any_missing = True
                lines.append(f"- `{piece}` vs `{r['cavity']}`: missing views = {mv}")
    if not any_missing:
        lines.append("None.")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- Deterministic, geometry-only by design.")
    lines.append("- Sensitive to the choice of viewpoints; Phase A used "
                 "sequential single-camera relocation, not a synchronised "
                 "multi-camera rig.")
    lines.append("- Cavity side is single-view only (Baseline 1 captures); "
                 "only the piece side is multi-view.")
    lines.append("- The convex-hull representation-normalisation fallback "
                 "from Baseline 1 is inherited unchanged.")
    lines.append("- No 3D reconstruction. No pose estimation. No multi-view "
                 "fusion (only score-level aggregation).")
    lines.append("- View weights and `MIN_VIEW_POINTS` are hardcoded; not "
                 "tuned.")
    lines.append("")

    lines.append("## Closing note")
    lines.append("")
    lines.append("This experiment is **not** multi-view fusion, **not** 3D "
                 "reconstruction, **not** pose estimation. It is a minimal "
                 "score-level aggregation built only to test whether "
                 "additional deterministic viewpoints reduce ambiguity in the "
                 "Baseline 1 ranking on this MVP set.")

    OUT_REPORT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT_MD.write_text("\n".join(lines) + "\n")
    print(f"[write] {OUT_REPORT_MD}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("baseline2_multiview_geometric_matching.py — Phase B (minimal)")
    print("=" * 70)
    print(f"PROJECT_ROOT  : {PROJECT_ROOT}")
    print(f"MULTIVIEW_DIR : {MULTIVIEW_DIR}")
    print(f"CAVITY_DIR    : {CAVITY_DIR}")
    print(f"VIEW_WEIGHTS  : {VIEW_WEIGHTS}")
    print()

    cavities = load_cavities()
    if not cavities:
        print("[main] FATAL: no cavities loaded; cannot proceed.")
        sys.exit(1)
    cavity_names = sorted(cavities.keys())

    run_meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run_id":        os.urandom(4).hex(),
    }

    results = []
    score_matrix: dict[str, dict[str, float | None]] = {}

    for piece in PIECE_NAMES:
        print(f"\n[piece {piece}] processing ...")
        # Load + segment each view ONCE per piece.
        per_view_xy: dict[str, np.ndarray | None] = {}
        for view in VIEW_NAMES:
            per_view_xy[view] = process_view(piece, view)

        score_matrix[piece] = {}
        per_cavity_records: dict[str, dict] = {}

        for cav in cavity_names:
            mask_dil   = cavities[cav]["mask_dil"]
            mask_undil = cavities[cav]["mask_undil"]

            per_view_record: dict[str, dict] = {}
            per_view_argmax_helper: dict[str, str] = {}   # filled later
            for view in VIEW_NAMES:
                xy = per_view_xy[view]
                if xy is None:
                    per_view_record[view] = {
                        "score":             None,
                        "inside":            None,
                        "outside":           None,
                        "iou":               None,
                        "best_rotation_deg": None,
                        "missing":           True,
                    }
                else:
                    res = score_view_against_cavity(xy, mask_dil, mask_undil)
                    res["missing"] = False
                    per_view_record[view] = res

            agg, missing_views = aggregate_scores(per_view_record)
            print(f"[piece {piece}] [score] vs {cav}: per-view "
                  f"top={per_view_record['top_down']['score']}  "
                  f"front={per_view_record['front_oblique']['score']}  "
                  f"side={per_view_record['side_oblique']['score']}  "
                  f"-> agg={agg}")

            per_cavity_records[cav] = {
                "piece":             piece,
                "cavity":            cav,
                "per_view":          per_view_record,
                "missing_views":     missing_views,
                "aggregate_score":   agg,
            }
            score_matrix[piece][cav] = agg

        # Compute per-view argmax cavity for disagreement check
        per_view_argmax: dict[str, str | None] = {}
        for view in VIEW_NAMES:
            best_cav = None
            best_v = -1e9
            for cav in cavity_names:
                v = per_cavity_records[cav]["per_view"][view]["score"]
                if v is not None and v > best_v:
                    best_v = v
                    best_cav = cav
            per_view_argmax[view] = best_cav
        disagree_flag = per_view_disagreement(per_view_argmax)

        ranked = rank_cavities_for_piece(per_cavity_records)
        for rec in ranked:
            if rec["rank"] == 1 and len(ranked) > 1:
                second = ranked[1]
                if rec["aggregate_score"] is None or second["aggregate_score"] is None:
                    margin = None
                else:
                    margin = rec["aggregate_score"] - second["aggregate_score"]
            else:
                margin = None
            rec["first_vs_second_margin"] = margin
            rec["ambiguity_indicators"] = {
                "low_margin":            (margin is not None and margin < TIE_MARGIN),
                "missing_view":          (len(rec["missing_views"]) > 0),
                "per_view_disagreement": bool(disagree_flag) if rec["rank"] == 1 else False,
            }
            results.append(rec)

        r1 = ranked[0]
        print(f"[rank] {piece}: rank-1 {r1['cavity']}  "
              f"score={r1['aggregate_score']}  "
              f"margin={r1.get('first_vs_second_margin')}  "
              f"ambiguity={r1['ambiguity_indicators']}")

    print("\n[summary] writing outputs ...")
    write_results_json(results, run_meta)
    write_matrix_csv(score_matrix, cavity_names)
    write_report_md(results, score_matrix, cavity_names, run_meta)
    print("[summary] done.")


if __name__ == "__main__":
    main()
