"""
baseline2_phaseC_canonical_multiview.py

Baseline 2 - Phase C: Canonical multi-view geometric matching.

Exploratory canonical world-frame XY representation.
NOT volumetric fusion, NOT SLAM, NOT TSDF, NOT 3D reconstruction.
Baseline 1 scoring reused unchanged. No ML, no descriptors added.

Runs OUTSIDE Isaac Sim. Use: python3 scripts/baseline2_phaseC_canonical_multiview.py
"""

import sys
import os
import json
import glob
import uuid
import datetime
from pathlib import Path

import numpy as np
import cv2

# ---------------------------------------------------------------------------
# Path setup — ensure project root is in sys.path so sibling imports work
# ---------------------------------------------------------------------------
_SCRIPT_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

# ---------------------------------------------------------------------------
# Imports from Phase B and Baseline 1
# ---------------------------------------------------------------------------
from baseline2_multiview_geometric_matching import (
    back_project_view,
    PIECE_NAMES,
    VIEW_NAMES,
    MULTIVIEW_DIR,
    CAVITY_MULTI_DIR,
    CAVITY_DIR,
    PIECE_HEIGHT_MIN_ABOVE_SURFACE_M,
    CAVITY_DEPTH_MIN_BELOW_SURFACE_M,
    CAVITY_DEPTH_MAX_BELOW_SURFACE_M,
    CAVITY_VIEW_ROI_HALF_SIZE_M,
    MIN_VIEW_POINTS,
    build_cavity_masks,
    PROJECT_ROOT,
)

from baseline1_geometric_matching import (
    rasterise_xy_to_mask,
    score_pair,
    TIE_MARGIN,
    _DIL_KERNEL,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
OUTPUT_DIR = _PROJECT_ROOT / "data" / "baseline2_phaseC_canonical_multiview"
DEBUG_DIR = OUTPUT_DIR / "debug"
CAVITY_NAMES = ["cavity_00", "cavity_01", "cavity_02", "cavity_03"]
MIN_CANONICAL_POINTS = MIN_VIEW_POINTS           # 50
SPARSE_THRESHOLD = 3 * MIN_VIEW_POINTS           # 150


# ---------------------------------------------------------------------------
# Helpers — locating data files
# ---------------------------------------------------------------------------

def _find_view_dir(base_dir: Path, entity_name: str, view_name: str):
    """Return the first directory matching view_*_<view_name> for the entity."""
    pattern = str(base_dir / entity_name / f"view_*_{view_name}")
    matches = sorted(glob.glob(pattern))
    if matches:
        return Path(matches[0])
    return None


def _load_depth_and_meta(view_dir: Path):
    """Load depth.npy and metadata.json from a view directory. Returns (depth, meta) or (None, None)."""
    depth_path = view_dir / "depth.npy"
    meta_path = view_dir / "metadata.json"
    if not depth_path.exists() or not meta_path.exists():
        return None, None
    depth = np.load(str(depth_path))
    with open(str(meta_path), "r") as f:
        meta = json.load(f)
    return depth, meta


# ---------------------------------------------------------------------------
# Piece extraction
# ---------------------------------------------------------------------------

def extract_piece_xy_uncentred(piece_name: str, view_name: str):
    """
    Back-project one view and return UN-CENTRED world XY for the piece
    (points above support surface). Returns (xy_Nx2, n_raw) or (None, 0).
    """
    base_dir = _PROJECT_ROOT / MULTIVIEW_DIR if not Path(MULTIVIEW_DIR).is_absolute() else Path(MULTIVIEW_DIR)
    # Resolve relative paths
    if not base_dir.is_absolute():
        base_dir = _PROJECT_ROOT / MULTIVIEW_DIR

    view_dir = _find_view_dir(base_dir, piece_name, view_name)
    if view_dir is None:
        print(f"    [piece {piece_name}] [view {view_name}] directory not found under {base_dir}")
        return None, 0

    depth, meta = _load_depth_and_meta(view_dir)
    if depth is None:
        print(f"    [piece {piece_name}] [view {view_name}] depth/meta missing in {view_dir}")
        return None, 0

    pts_world = back_project_view(depth, meta)
    if pts_world is None or len(pts_world) == 0:
        print(f"    [piece {piece_name}] [view {view_name}] back_project_view returned no points")
        return None, 0

    # Support surface Z from bounding-box bottom of target
    bbox_center_z = meta["target_bbox_center_world_m"][2]
    bbox_size_z_m = meta["target_bbox_size_mm"][2] / 1000.0
    support_z = bbox_center_z - bbox_size_z_m / 2.0

    # Segment: keep points above support surface
    mask_above = pts_world[:, 2] > (support_z + PIECE_HEIGHT_MIN_ABOVE_SURFACE_M)
    pts_seg = pts_world[mask_above]
    n_raw = int(mask_above.sum())

    if n_raw < MIN_CANONICAL_POINTS:
        print(f"    [piece {piece_name}] [view {view_name}] only {n_raw} points above surface (<{MIN_CANONICAL_POINTS}), skipping")
        return None, n_raw

    xy = pts_seg[:, :2].astype(np.float64)
    print(f"    [piece {piece_name}] [view {view_name}] {n_raw} points above surface")
    return xy, n_raw


def build_canonical_piece(name: str):
    """
    Merge all views for a piece, centroid-centre once, rasterise.
    Returns canonical diagnostics dict.
    """
    print(f"  [canonical {name}] building piece canonical")
    per_view_counts = {}
    surviving_xys = []
    views_used = []

    for view in VIEW_NAMES:
        try:
            xy, n = extract_piece_xy_uncentred(name, view)
        except Exception as exc:
            print(f"    [piece {name}] [view {view}] exception: {exc}")
            xy, n = None, 0
        per_view_counts[view] = n
        if xy is not None and n >= MIN_CANONICAL_POINTS:
            surviving_xys.append(xy)
            views_used.append(view)

    if not surviving_xys:
        print(f"  [canonical {name}] no surviving views — marking invalid")
        return {
            "source_views_used": views_used,
            "per_view_raw_point_count": per_view_counts,
            "merged_point_count": 0,
            "centroid_before_centring_world_m": None,
            "mask_area_px": 0,
            "bbox_px": None,
            "fill_ratio": None,
            "convex_hull_fallback_triggered": False,
            "canonical_sparse": True,
            "invalid": True,
            "canonical_xy": None,
            "mask": None,
            "mask_info": None,
        }

    merged_xy, centroid = merge_and_centre(surviving_xys)
    n_merged = len(merged_xy)
    canonical_sparse = n_merged < SPARSE_THRESHOLD

    print(f"  [canonical {name}] merged {n_merged} points from views {views_used}, sparse={canonical_sparse}")

    mask, info = rasterise_xy_to_mask(merged_xy)
    area_px = int(info.get("n_pixels_after_close", 0))
    bbox_px = info.get("bbox_area_px", None)
    convex_hull_fallback = bool(info.get("convex_hull_fallback", False))
    n_px_fill = info.get("n_pixels_after_close", 0)
    n_px_bbox = info.get("bbox_area_px", 1) or 1
    fill_ratio = float(n_px_fill) / float(n_px_bbox) if n_px_bbox else None

    invalid = (n_merged < MIN_CANONICAL_POINTS) or (area_px == 0)

    return {
        "source_views_used": views_used,
        "per_view_raw_point_count": per_view_counts,
        "merged_point_count": n_merged,
        "centroid_before_centring_world_m": centroid.tolist(),
        "mask_area_px": area_px,
        "bbox_px": bbox_px,
        "fill_ratio": fill_ratio,
        "convex_hull_fallback_triggered": convex_hull_fallback,
        "canonical_sparse": canonical_sparse,
        "invalid": invalid,
        "canonical_xy": merged_xy,
        "mask": mask,
        "mask_info": info,
    }


# ---------------------------------------------------------------------------
# Cavity extraction
# ---------------------------------------------------------------------------

def extract_cavity_xy_uncentred_top_down(cavity_name: str):
    """
    Load UN-CENTRED world XY from baseline-1-validated opening point cloud.
    Falls back to cavity_pointcloud.npy if opening absent.
    Returns (xy_Nx2, n_raw, source_label) or (None, 0, label).
    """
    cav_dir = _PROJECT_ROOT / CAVITY_DIR if not Path(CAVITY_DIR).is_absolute() else Path(CAVITY_DIR)
    if not cav_dir.is_absolute():
        cav_dir = _PROJECT_ROOT / CAVITY_DIR

    opening_path = cav_dir / cavity_name / "cavity_opening_pointcloud.npy"
    fallback_path = cav_dir / cavity_name / "cavity_pointcloud.npy"

    if opening_path.exists():
        npy_path = opening_path
        source_label = "baseline1_validated_opening"
    elif fallback_path.exists():
        npy_path = fallback_path
        source_label = "baseline1_cavity_pointcloud_fallback"
    else:
        print(f"    [cavity {cavity_name}] [top_down] no point cloud found in {cav_dir / cavity_name}")
        return None, 0, "missing"

    pts = np.load(str(npy_path))
    # Accept Nx2 or Nx3
    if pts.ndim != 2 or pts.shape[1] < 2:
        print(f"    [cavity {cavity_name}] [top_down] unexpected shape {pts.shape}")
        return None, 0, source_label

    xy = pts[:, :2].astype(np.float64)
    n = len(xy)
    print(f"    [cavity {cavity_name}] [top_down] {n} points ({source_label})")
    return (xy if n >= MIN_CANONICAL_POINTS else None), n, source_label


def extract_cavity_xy_uncentred_oblique(cavity_name: str, view_name: str):
    """
    Back-project an oblique view for a cavity, apply Z-band and XY ROI.
    Returns UN-CENTRED world XY (Nx2) or (None, n_raw).
    """
    base_dir = _PROJECT_ROOT / CAVITY_MULTI_DIR if not Path(CAVITY_MULTI_DIR).is_absolute() else Path(CAVITY_MULTI_DIR)
    if not base_dir.is_absolute():
        base_dir = _PROJECT_ROOT / CAVITY_MULTI_DIR

    view_dir = _find_view_dir(base_dir, cavity_name, view_name)
    if view_dir is None:
        print(f"    [cavity {cavity_name}] [view {view_name}] directory not found under {base_dir}")
        return None, 0

    depth, meta = _load_depth_and_meta(view_dir)
    if depth is None:
        print(f"    [cavity {cavity_name}] [view {view_name}] depth/meta missing")
        return None, 0

    pts_world = back_project_view(depth, meta)
    if pts_world is None or len(pts_world) == 0:
        print(f"    [cavity {cavity_name}] [view {view_name}] back_project_view returned no points")
        return None, 0

    board_top = meta["target_bbox_center_world_m"][2]
    cx, cy = meta["target_bbox_center_world_m"][0], meta["target_bbox_center_world_m"][1]

    # Z-band: points inside cavity depth below board top
    z_lo = board_top - CAVITY_DEPTH_MAX_BELOW_SURFACE_M
    z_hi = board_top - CAVITY_DEPTH_MIN_BELOW_SURFACE_M
    mask_z = (pts_world[:, 2] > z_lo) & (pts_world[:, 2] < z_hi)

    # XY ROI around cavity centre
    half = CAVITY_VIEW_ROI_HALF_SIZE_M
    mask_x = np.abs(pts_world[:, 0] - cx) <= half
    mask_y = np.abs(pts_world[:, 1] - cy) <= half

    mask_all = mask_z & mask_x & mask_y
    pts_seg = pts_world[mask_all]
    n_raw = int(mask_all.sum())

    if n_raw < MIN_CANONICAL_POINTS:
        print(f"    [cavity {cavity_name}] [view {view_name}] only {n_raw} points in Z/XY band (<{MIN_CANONICAL_POINTS}), skipping")
        return None, n_raw

    xy = pts_seg[:, :2].astype(np.float64)
    print(f"    [cavity {cavity_name}] [view {view_name}] {n_raw} points in Z/XY band")
    return xy, n_raw


def build_canonical_cavity(name: str):
    """
    Merge all sources for a cavity, centroid-centre once, build masks.
    Returns canonical diagnostics dict.
    """
    print(f"  [canonical {name}] building cavity canonical")
    per_view_counts = {}
    surviving_xys = []
    views_used = []
    source_per_view = {}

    # top_down: load from Baseline 1 validated point cloud
    try:
        xy_td, n_td, src_label = extract_cavity_xy_uncentred_top_down(name)
    except Exception as exc:
        print(f"    [cavity {name}] [top_down] exception: {exc}")
        xy_td, n_td, src_label = None, 0, "error"

    per_view_counts["top_down"] = n_td
    source_per_view["top_down"] = src_label
    if xy_td is not None and n_td >= MIN_CANONICAL_POINTS:
        surviving_xys.append(xy_td)
        views_used.append("top_down")

    # front_oblique and side_oblique: back-project
    for view in ["front_oblique", "side_oblique"]:
        try:
            xy_v, n_v = extract_cavity_xy_uncentred_oblique(name, view)
        except Exception as exc:
            print(f"    [cavity {name}] [view {view}] exception: {exc}")
            xy_v, n_v = None, 0
        per_view_counts[view] = n_v
        source_per_view[view] = "multiview_roi_z_band"
        if xy_v is not None and n_v >= MIN_CANONICAL_POINTS:
            surviving_xys.append(xy_v)
            views_used.append(view)

    if not surviving_xys:
        print(f"  [canonical {name}] no surviving sources — marking invalid")
        return {
            "source_views_used": views_used,
            "per_view_raw_point_count": per_view_counts,
            "cavity_source_per_view": source_per_view,
            "merged_point_count": 0,
            "centroid_before_centring_world_m": None,
            "mask_area_px": 0,
            "bbox_px": None,
            "fill_ratio": None,
            "convex_hull_fallback_triggered": False,
            "canonical_sparse": True,
            "invalid": True,
            "canonical_xy": None,
            "mask_dil": None,
            "mask_undil": None,
            "mask_info": None,
        }

    merged_xy, centroid = merge_and_centre(surviving_xys)
    n_merged = len(merged_xy)
    canonical_sparse = n_merged < SPARSE_THRESHOLD

    print(f"  [canonical {name}] merged {n_merged} points from sources {views_used}, sparse={canonical_sparse}")

    # Build raster for diagnostics, then build_cavity_masks for scoring
    _, info = rasterise_xy_to_mask(merged_xy)
    mask_dil, mask_undil = build_cavity_masks(merged_xy)

    area_px = int(info.get("n_pixels_after_close", 0))
    bbox_px = info.get("bbox_area_px", None)
    convex_hull_fallback = bool(info.get("convex_hull_fallback", False))
    n_px_fill = info.get("n_pixels_after_close", 0)
    n_px_bbox = info.get("bbox_area_px", 1) or 1
    fill_ratio = float(n_px_fill) / float(n_px_bbox) if n_px_bbox else None

    invalid = (n_merged < MIN_CANONICAL_POINTS) or (area_px == 0)

    return {
        "source_views_used": views_used,
        "per_view_raw_point_count": per_view_counts,
        "cavity_source_per_view": source_per_view,
        "merged_point_count": n_merged,
        "centroid_before_centring_world_m": centroid.tolist(),
        "mask_area_px": area_px,
        "bbox_px": bbox_px,
        "fill_ratio": fill_ratio,
        "convex_hull_fallback_triggered": convex_hull_fallback,
        "canonical_sparse": canonical_sparse,
        "invalid": invalid,
        "canonical_xy": merged_xy,
        "mask_dil": mask_dil,
        "mask_undil": mask_undil,
        "mask_info": info,
    }


# ---------------------------------------------------------------------------
# Merge and centre
# ---------------------------------------------------------------------------

def merge_and_centre(list_of_xy: list):
    """
    Vertically stack multiple XY arrays, compute centroid, subtract it.
    Returns (centred_xy, centroid_world).
    """
    merged = np.vstack(list_of_xy)
    centroid = merged.mean(axis=0)
    centred = merged - centroid
    return centred, centroid


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_canonical_pair(piece_name: str, cavity_name: str, piece_diag: dict, cavity_diag: dict):
    """
    Score a canonical (piece, cavity) pair using Baseline 1 score_pair unchanged.
    Returns a result record dict.
    """
    print(f"  [score {piece_name}] [vs {cavity_name}]")

    if piece_diag["invalid"] or cavity_diag["invalid"]:
        reason = []
        if piece_diag["invalid"]:
            reason.append(f"piece '{piece_name}' invalid")
        if cavity_diag["invalid"]:
            reason.append(f"cavity '{cavity_name}' invalid")
        print(f"    -> failed: {'; '.join(reason)}")
        return {
            "piece": piece_name,
            "cavity": cavity_name,
            "score": None,
            "inside_ratio": None,
            "outside_ratio": None,
            "iou": None,
            "best_rotation_deg": None,
            "rank": None,
            "first_vs_second_margin": None,
            "failed": True,
            "failure_reason": "; ".join(reason),
        }

    piece_xy = piece_diag["canonical_xy"]
    mask_dil = cavity_diag["mask_dil"]
    mask_undil = cavity_diag["mask_undil"]

    _, best = score_pair(piece_xy, mask_dil, mask_undil)

    result = {
        "piece": piece_name,
        "cavity": cavity_name,
        "score": float(best["score"]),
        "inside_ratio": float(best["inside_ratio"]),
        "outside_ratio": float(best["outside_ratio"]),
        "iou": float(best["iou"]),
        "best_rotation_deg": float(best["rotation_deg"]),
        "rank": None,
        "first_vs_second_margin": None,
        "failed": False,
        "failure_reason": None,
    }
    print(f"    -> score={result['score']:.4f} iou={result['iou']:.4f} rot={result['best_rotation_deg']:.1f}°")
    return result


# ---------------------------------------------------------------------------
# Per-piece ranking
# ---------------------------------------------------------------------------

def rank_piece_results(piece_name: str, all_results: list):
    """
    Sort results for a given piece by score desc, assign ranks and margin.
    Modifies records in-place.
    """
    piece_res = [r for r in all_results if r["piece"] == piece_name and not r["failed"]]
    piece_res_sorted = sorted(piece_res, key=lambda r: r["score"], reverse=True)

    for rank_idx, rec in enumerate(piece_res_sorted):
        rec["rank"] = rank_idx + 1

    if len(piece_res_sorted) >= 2:
        margin = piece_res_sorted[0]["score"] - piece_res_sorted[1]["score"]
        piece_res_sorted[0]["first_vs_second_margin"] = float(margin)
    elif len(piece_res_sorted) == 1:
        piece_res_sorted[0]["first_vs_second_margin"] = None

    print(f"  [rank {piece_name}] top cavity: {piece_res_sorted[0]['cavity'] if piece_res_sorted else 'none'}")


# ---------------------------------------------------------------------------
# Debug image helpers
# ---------------------------------------------------------------------------

def _save_mask_png(mask: np.ndarray, path: Path, label: str):
    """Save a binary/uint8 mask as PNG."""
    if mask is None:
        return
    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(path), mask)
        print(f"    [debug] saved mask {label} -> {path.name}")
    except Exception as exc:
        print(f"    [debug] could not save mask {label}: {exc}")


def _save_scatter_png(xy: np.ndarray, path: Path, label: str):
    """Save a scatter plot of 2D XY points. Skips if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.scatter(xy[:, 0], xy[:, 1], s=1, alpha=0.5)
        ax.set_aspect("equal")
        ax.set_title(label)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(str(path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"    [debug] saved scatter {label} -> {path.name}")
    except ImportError:
        pass  # matplotlib not available — skip silently
    except Exception as exc:
        print(f"    [debug] scatter save failed for {label}: {exc}")


# ---------------------------------------------------------------------------
# Output writing
# ---------------------------------------------------------------------------

def write_outputs(
    piece_canonicals: dict,
    cavity_canonicals: dict,
    all_results: list,
    run_id: str,
    timestamp_utc: str,
):
    """Write JSON, CSV, Markdown, and debug images."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Debug masks -------------------------------------------------------
    for name, diag in piece_canonicals.items():
        if diag.get("mask") is not None:
            _save_mask_png(diag["mask"], DEBUG_DIR / f"piece_{name}_canonical_mask.png", name)
        if diag.get("canonical_xy") is not None:
            _save_scatter_png(diag["canonical_xy"], DEBUG_DIR / f"piece_{name}_canonical_scatter.png", name)

    for name, diag in cavity_canonicals.items():
        if diag.get("mask_dil") is not None:
            _save_mask_png(diag["mask_dil"], DEBUG_DIR / f"cavity_{name}_canonical_mask_dil.png", name)
        if diag.get("mask_undil") is not None:
            _save_mask_png(diag["mask_undil"], DEBUG_DIR / f"cavity_{name}_canonical_mask_undil.png", name)
        if diag.get("canonical_xy") is not None:
            _save_scatter_png(diag["canonical_xy"], DEBUG_DIR / f"cavity_{name}_canonical_scatter.png", name)

    # ---- Serialisable diagnostics ------------------------------------------
    def _serialisable_diag(d: dict):
        skip = {"canonical_xy", "mask", "mask_dil", "mask_undil", "mask_info"}
        out = {}
        for k, v in d.items():
            if k in skip:
                continue
            if isinstance(v, np.ndarray):
                out[k] = v.tolist()
            else:
                out[k] = v
        return out

    piece_diags_serial = {n: _serialisable_diag(d) for n, d in piece_canonicals.items()}
    cavity_diags_serial = {n: _serialisable_diag(d) for n, d in cavity_canonicals.items()}

    # ---- JSON --------------------------------------------------------------
    results_serial = []
    for r in all_results:
        rec = {k: v for k, v in r.items()}
        results_serial.append(rec)

    output_json = {
        "schema_version": 1,
        "script_name": "baseline2_phaseC_canonical_multiview.py",
        "phase": "Baseline 2 - Phase C (canonical multi-view, exploratory)",
        "phase_note": (
            "Exploratory canonical world-frame XY representation. "
            "NOT volumetric fusion, NOT SLAM, NOT TSDF, NOT 3D reconstruction. "
            "Baseline 1 scoring reused unchanged. No ML, no descriptors added."
        ),
        "timestamp_utc": timestamp_utc,
        "run_id": run_id,
        "canonical_entities": {
            "pieces": piece_diags_serial,
            "cavities": cavity_diags_serial,
        },
        "results": results_serial,
    }

    json_path = OUTPUT_DIR / "phaseC_canonical_results.json"
    with open(str(json_path), "w") as f:
        json.dump(output_json, f, indent=2)
    print(f"[output] JSON -> {json_path}")

    # ---- CSV score matrix --------------------------------------------------
    csv_path = OUTPUT_DIR / "phaseC_matching_matrix.csv"
    header = "piece," + ",".join(CAVITY_NAMES)
    rows = [header]
    for piece in PIECE_NAMES:
        scores_per_cav = {}
        for r in all_results:
            if r["piece"] == piece:
                val = r["score"] if not r["failed"] else ""
                scores_per_cav[r["cavity"]] = f"{val:.4f}" if isinstance(val, float) else ""
        row_vals = [scores_per_cav.get(c, "") for c in CAVITY_NAMES]
        rows.append(f"{piece}," + ",".join(row_vals))
    with open(str(csv_path), "w") as f:
        f.write("\n".join(rows) + "\n")
    print(f"[output] CSV -> {csv_path}")

    # ---- Markdown report ---------------------------------------------------
    md_path = OUTPUT_DIR / "phaseC_report.md"
    _write_markdown_report(
        md_path, piece_canonicals, cavity_canonicals, all_results, run_id, timestamp_utc
    )
    print(f"[output] Markdown -> {md_path}")


def _write_markdown_report(
    path: Path,
    piece_canonicals: dict,
    cavity_canonicals: dict,
    all_results: list,
    run_id: str,
    timestamp_utc: str,
):
    """Write the full Markdown report."""

    lines = []
    lines.append("# Baseline 2 – Phase C: Canonical Multi-View Geometric Matching\n")
    lines.append(f"**Run ID:** `{run_id}`  ")
    lines.append(f"**Timestamp UTC:** {timestamp_utc}  ")
    lines.append(f"**Script:** `baseline2_phaseC_canonical_multiview.py`\n")

    lines.append("---\n")
    lines.append("## Objective\n")
    lines.append(
        "Exploratory canonical world-frame XY representation for geometric insertion matching. "
        "Each piece and cavity is represented as a single **canonical** 2D footprint by merging "
        "surviving views, centroid-centring once, and rasterising. "
        "Baseline 1 scoring (`score_pair`) is reused **unchanged**. "
        "This is NOT fusion in the volumetric/SLAM/TSDF sense. "
        "No learned model. No descriptors added.\n"
    )

    lines.append("## Methodology\n")
    lines.append("### Piece canonical construction\n")
    lines.append(
        "1. For each view (`top_down`, `front_oblique`, `side_oblique`): back-project depth, "
        "segment points above support surface (`support_z + PIECE_HEIGHT_MIN_ABOVE_SURFACE_M`), "
        "keep UN-CENTRED world XY.\n"
        "2. Discard views with fewer than `MIN_VIEW_POINTS` (50) points.\n"
        "3. Merge surviving views with `np.vstack`, compute centroid, centroid-centre once.\n"
        "4. Rasterise via `rasterise_xy_to_mask` (Baseline 1, unchanged).\n"
    )
    lines.append("### Cavity canonical construction\n")
    lines.append(
        "1. `top_down`: load Baseline-1-validated `cavity_opening_pointcloud.npy` (fallback: `cavity_pointcloud.npy`).\n"
        "2. `front_oblique`, `side_oblique`: back-project, apply Z-band "
        "`(board_top - CAVITY_DEPTH_MAX_BELOW_SURFACE_M, board_top - CAVITY_DEPTH_MIN_BELOW_SURFACE_M)` "
        "and XY ROI `±CAVITY_VIEW_ROI_HALF_SIZE_M` around cavity centre. Keep UN-CENTRED world XY.\n"
        "3. Discard views with fewer than `MIN_VIEW_POINTS` (50) points.\n"
        "4. Merge and centroid-centre once. Build `(mask_dil, mask_undil)` via `build_cavity_masks`.\n"
    )
    lines.append("### Scoring\n")
    lines.append(
        "For each (piece, cavity) pair, call `score_pair(piece_canonical_xy, mask_dil, mask_undil)` "
        "from Baseline 1 unchanged. Take `best_record`. Rank cavities per piece by score descending.\n"
    )
    lines.append("### Sparsity policy\n")
    lines.append(
        "No auto-fallback to Baseline 1 when canonical points are sparse. "
        "`canonical_sparse = True` is set when merged count < 150 (3×MIN_VIEW_POINTS). "
        "`invalid = True` only when merged count < 50 OR mask area = 0 px.\n"
    )

    lines.append("## Canonical Entities\n")
    lines.append("### Pieces\n")
    piece_hdr = "| Piece | Views used | Merged pts | Mask area px | Bbox px | Fill ratio | Hull fallback | Sparse | Invalid |"
    piece_sep = "|-------|-----------|-----------|-------------|---------|-----------|--------------|--------|---------|"
    lines.append(piece_hdr)
    lines.append(piece_sep)
    for name in PIECE_NAMES:
        d = piece_canonicals.get(name, {})
        vu = ", ".join(d.get("source_views_used", []))
        mp = d.get("merged_point_count", 0)
        ma = d.get("mask_area_px", 0)
        bp = d.get("bbox_px", "—")
        fr = f"{d.get('fill_ratio', 0):.3f}" if d.get("fill_ratio") is not None else "—"
        hf = "yes" if d.get("convex_hull_fallback_triggered") else "no"
        sp = "yes" if d.get("canonical_sparse") else "no"
        inv = "yes" if d.get("invalid") else "no"
        lines.append(f"| {name} | {vu} | {mp} | {ma} | {bp} | {fr} | {hf} | {sp} | {inv} |")
    lines.append("")

    lines.append("### Cavities\n")
    cav_hdr = "| Cavity | Sources used | Merged pts | Mask area px | Bbox px | Fill ratio | Hull fallback | Sparse | Invalid |"
    cav_sep = "|--------|-------------|-----------|-------------|---------|-----------|--------------|--------|---------|"
    lines.append(cav_hdr)
    lines.append(cav_sep)
    for name in CAVITY_NAMES:
        d = cavity_canonicals.get(name, {})
        vu = ", ".join(d.get("source_views_used", []))
        mp = d.get("merged_point_count", 0)
        ma = d.get("mask_area_px", 0)
        bp = d.get("bbox_px", "—")
        fr = f"{d.get('fill_ratio', 0):.3f}" if d.get("fill_ratio") is not None else "—"
        hf = "yes" if d.get("convex_hull_fallback_triggered") else "no"
        sp = "yes" if d.get("canonical_sparse") else "no"
        inv = "yes" if d.get("invalid") else "no"
        lines.append(f"| {name} | {vu} | {mp} | {ma} | {bp} | {fr} | {hf} | {sp} | {inv} |")
    lines.append("")

    lines.append("## 4x4 Score Matrix\n")
    score_hdr = "| Piece | cavity_00 | cavity_01 | cavity_02 | cavity_03 |"
    score_sep = "|-------|----------|----------|----------|----------|"
    lines.append(score_hdr)
    lines.append(score_sep)
    for piece in PIECE_NAMES:
        cells = [piece]
        for cav in CAVITY_NAMES:
            match = next((r for r in all_results if r["piece"] == piece and r["cavity"] == cav), None)
            if match and not match["failed"] and match["score"] is not None:
                cells.append(f"{match['score']:.4f}")
            else:
                cells.append("—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Per-Piece Ranking\n")
    rank_hdr = "| Piece | Rank-1 Cavity | Score | IoU | Rotation ° | Margin vs Rank-2 |"
    rank_sep = "|-------|--------------|-------|-----|-----------|-----------------|"
    lines.append(rank_hdr)
    lines.append(rank_sep)
    for piece in PIECE_NAMES:
        ranked = sorted(
            [r for r in all_results if r["piece"] == piece and not r["failed"] and r["score"] is not None],
            key=lambda r: r["score"], reverse=True
        )
        if ranked:
            r1 = ranked[0]
            margin = r1.get("first_vs_second_margin")
            margin_str = f"{margin:.4f}" if margin is not None else "—"
            lines.append(
                f"| {piece} | {r1['cavity']} | {r1['score']:.4f} | {r1['iou']:.4f} | {r1['best_rotation_deg']:.1f} | {margin_str} |"
            )
        else:
            lines.append(f"| {piece} | — | — | — | — | — |")
    lines.append("")

    lines.append("## Comparison vs Baseline 1 and Phase B Hybrid\n")
    lines.append(
        "Reference numbers: Baseline 1 final (doc 03 §17.4), Phase B hybrid (Iteration C). "
        "Phase C scores are from this run.\n"
    )
    cmp_hdr = "| Piece | B1 cavity | B1 score | B1 margin | PhB cavity | PhB agg | PhB margin | PhC cavity | PhC score | PhC margin |"
    cmp_sep = "|-------|----------|---------|---------|-----------|--------|----------|-----------|---------|---------|"
    lines.append(cmp_hdr)
    lines.append(cmp_sep)

    ref_b1 = {
        "rectangle": ("cavity_00", 0.883, 0.293),
        "square":    ("cavity_02", 0.884, 0.168),
        "circle":    ("cavity_03", 0.889, 0.114),
        "triangle":  ("cavity_01", 0.886, 0.227),
    }
    ref_pb = {
        "rectangle": ("cavity_00", 0.493, 0.107),
        "square":    ("cavity_02", 0.578, 0.091),
        "circle":    ("cavity_03", 0.592, 0.058),
        "triangle":  ("cavity_03", 0.500, 0.006),
    }

    for piece in PIECE_NAMES:
        b1_c, b1_s, b1_m = ref_b1[piece]
        pb_c, pb_s, pb_m = ref_pb[piece]

        ranked = sorted(
            [r for r in all_results if r["piece"] == piece and not r["failed"] and r["score"] is not None],
            key=lambda r: r["score"], reverse=True
        )
        if ranked:
            r1 = ranked[0]
            pc_c = r1["cavity"]
            pc_s = f"{r1['score']:.4f}"
            pc_m_raw = r1.get("first_vs_second_margin")
            pc_m = f"{pc_m_raw:.4f}" if pc_m_raw is not None else "—"
        else:
            pc_c, pc_s, pc_m = "—", "—", "—"

        lines.append(
            f"| {piece} | {b1_c} | {b1_s} | {b1_m} | {pb_c} | {pb_s} | {pb_m} | {pc_c} | {pc_s} | {pc_m} |"
        )
    lines.append("")

    lines.append("## Limitations\n")
    lines.append(
        "- Canonical representation merges views naively (vstack). "
        "No view weighting, no registration, no outlier filtering.\n"
        "- Oblique views add density but may add noise from perspective distortion or mis-segmented background.\n"
        "- The Z-band and ROI for cavity oblique views are fixed thresholds (Phase B constants); "
        "cavities that deviate from expected depth may contribute empty or noisy slices.\n"
        "- Sparsity is flagged but not handled: if most views fail, the canonical mask degrades gracefully "
        "to a single-view approximation, which may not improve over Baseline 1 or Phase B.\n"
        "- No ICP, no volumetric fusion, no SLAM — this is strictly a 2D footprint approach.\n"
    )

    lines.append("## Closing Note\n")
    lines.append(
        "This is an **exploratory** experiment. The canonical world-frame XY representation "
        "is NOT fusion in the volumetric/SLAM/TSDF sense. It is a deterministic, geometry-only "
        "aggregation of projected depth footprints. "
        "Baseline 1 scoring is reused unchanged. No learned model is present. "
        "Results are intended to inform whether multi-view aggregation of 2D footprints "
        "can improve or maintain matching quality relative to Baseline 1 and Phase B.\n"
    )

    with open(str(path), "w") as f:
        f.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    run_id = uuid.uuid4().hex[:8]
    timestamp_utc = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

    print(f"[phase C] run_id={run_id}  timestamp={timestamp_utc}")
    print(f"[phase C] project root: {_PROJECT_ROOT}")
    print(f"[phase C] output dir: {OUTPUT_DIR}")

    # --- Build canonical pieces ---
    print("\n=== Building canonical pieces ===")
    piece_canonicals = {}
    for piece in PIECE_NAMES:
        print(f"\n[canonical {piece}]")
        piece_canonicals[piece] = build_canonical_piece(piece)

    # --- Build canonical cavities ---
    print("\n=== Building canonical cavities ===")
    cavity_canonicals = {}
    for cavity in CAVITY_NAMES:
        print(f"\n[canonical {cavity}]")
        cavity_canonicals[cavity] = build_canonical_cavity(cavity)

    # --- Score all pairs ---
    print("\n=== Scoring all (piece, cavity) pairs ===")
    all_results = []
    for piece in PIECE_NAMES:
        for cavity in CAVITY_NAMES:
            rec = score_canonical_pair(piece, cavity, piece_canonicals[piece], cavity_canonicals[cavity])
            all_results.append(rec)

    # --- Rank per piece ---
    print("\n=== Ranking per piece ===")
    for piece in PIECE_NAMES:
        rank_piece_results(piece, all_results)

    # --- Write outputs ---
    print("\n=== Writing outputs ===")
    write_outputs(piece_canonicals, cavity_canonicals, all_results, run_id, timestamp_utc)

    # --- Summary to stdout ---
    print("\n=== Summary ===")
    for piece in PIECE_NAMES:
        ranked = sorted(
            [r for r in all_results if r["piece"] == piece and not r["failed"] and r["score"] is not None],
            key=lambda r: r["score"], reverse=True
        )
        if ranked:
            r1 = ranked[0]
            margin = r1.get("first_vs_second_margin")
            margin_str = f"{margin:.4f}" if margin is not None else "N/A"
            print(
                f"  {piece}: rank-1={r1['cavity']} score={r1['score']:.4f} "
                f"iou={r1['iou']:.4f} rot={r1['best_rotation_deg']:.1f}deg "
                f"margin={margin_str}"
            )
        else:
            print(f"  {piece}: no valid result")

    print("\n[phase C] done.")


if __name__ == "__main__":
    main()
