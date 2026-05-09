"""
validate_cavity_captures.py

Offline validation of per-cavity capture artefacts produced by
capture_cavity_detection.py.  Runs OUTSIDE Isaac Sim — plain Python
(NumPy + OpenCV, matplotlib optional).

Cavity subfolders are discovered dynamically via sorted(DATA_DIR.glob("cavity_*")).
No shape labels (rectangle/square/etc.) are referenced here.

For each discovered cavity_NN the script checks:
  1. cavity_metadata.json exists.
  2. cavity_pointcloud.npy exists.
  3. cavity_footprint.png exists.
  4. cavity_debug.png exists.
  5. cavity_mask.png exists.
  6. Point cloud shape: ndim==2, shape[1]==3, at least 100 points.
  7. Point cloud validity: no NaNs, no infinities.
  8. Point cloud bounds: X span > 0, Y span > 0, Z max > 0, Z span >= 0.
  9. Footprint image readable and has at least one non-zero pixel.
  10. area_px (if present in metadata) is >= 10 px.
  11. Advisory metadata fields: cavity_id, centroid_world_m, xy_span_m,
      z_depth_range_m — recorded as present/absent, not a hard failure.

Global top-level files are also checked (exists / missing), but global
file absence does not fail the overall validation.

Outputs written to DATA_DIR:
  - validation_summary.json
  - validation_summary.csv
  - footprints_grid.png

Usage:
  python3 scripts/validate_cavity_captures.py
  SHAPE_INSERTION_PROJECT_ROOT=/some/path python3 scripts/validate_cavity_captures.py
"""

import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import cv2

# ── CONFIG ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        str(Path(__file__).resolve().parent.parent),   # <repo>/scripts/ -> <repo>/
    )
)
DATA_DIR = PROJECT_ROOT / "data" / "cavities_detected"

PC_MIN_POINTS = 100       # minimum acceptable point count
PC_DEGENERATE_AREA_PX = 10  # area threshold below which cavity is flagged
FOOTPRINT_GRID_CELL = 256   # pixels per cell in the footprints grid

GLOBAL_FILES = [
    "cavities_summary.json",
    "run_log.txt",
    "rgb.png",
    "depth_vis.png",
    "raw_cavity_mask.png",
    "cavities_debug.png",
    "board_debug.png",
    "board_mask.png",
    "board_region_mask.png",
]

ADVISORY_META_FIELDS = ["cavity_id", "centroid_world_m", "xy_span_m", "z_depth_range_m"]

# ── GLOBAL FILE CHECKS ────────────────────────────────────────────────────────

def check_global_files(data_dir: Path) -> dict:
    """
    Check whether each expected top-level file exists.
    Returns a dict mapping filename -> bool.
    """
    result = {}
    for fname in GLOBAL_FILES:
        result[fname] = (data_dir / fname).exists()
    return result


def print_global_summary(global_checks: dict) -> None:
    print("\n--- Global files ---")
    col_w = max(len(k) for k in global_checks) + 2
    for fname, present in global_checks.items():
        status = "PRESENT" if present else "MISSING"
        print(f"  {fname:<{col_w}} {status}")


# ── CHECK HELPERS ─────────────────────────────────────────────────────────────

def _tick(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def check_metadata(cavity_dir: Path) -> dict:
    """
    Check 1: cavity_metadata.json exists and is valid JSON.
    Check 10: area_px >= PC_DEGENERATE_AREA_PX (if field present).
    Check 11: advisory fields present/absent.
    Returns a rich result dict.
    """
    result = {
        "exists": False,
        "area_px_ok": True,          # True by default (field may be absent)
        "area_px": None,
        "advisory_fields": {f: "absent" for f in ADVISORY_META_FIELDS},
        "reason_exists": "",
        "reason_area": "",
        "content": None,
    }
    meta_path = cavity_dir / "cavity_metadata.json"
    if not meta_path.exists():
        result["reason_exists"] = "cavity_metadata.json not found"
        return result

    result["exists"] = True
    try:
        with open(meta_path, "r") as fh:
            meta = json.load(fh)
        result["content"] = meta
    except Exception as exc:
        result["reason_exists"] = f"JSON parse error: {exc}"
        result["exists"] = False
        return result

    # Advisory fields
    for field in ADVISORY_META_FIELDS:
        result["advisory_fields"][field] = "present" if field in meta else "absent"

    # Area check (check 10)
    area_px = meta.get("area_px", None)
    result["area_px"] = area_px
    if area_px is not None:
        if area_px < PC_DEGENERATE_AREA_PX:
            result["area_px_ok"] = False
            result["reason_area"] = (
                f"area_px={area_px} < {PC_DEGENERATE_AREA_PX} (degenerate cavity)"
            )

    return result


def check_pointcloud(cavity_dir: Path, is_opening: bool = False) -> dict:
    """
    Checks 2, 6, 7, 8: .npy exists, shape correct, no NaN/Inf, bounds sane.
    Z convention: Z = depth - board_surface_z, positive into the cavity,
    so Z max must be > 0.
    """
    result = {
        "exists": False,
        "shape_ok": False,
        "no_nan": False,
        "no_inf": False,
        "n_points": 0,
        "x_span_m": 0.0,
        "y_span_m": 0.0,
        "z_span_m": 0.0,
        "x_min": 0.0, "x_max": 0.0,
        "y_min": 0.0, "y_max": 0.0,
        "z_min": 0.0, "z_max": 0.0,
        "xy_ok": False,
        "z_ok": False,
        "reason_exists": "",
        "reason_shape": "",
        "reason_bounds": "",
    }
    pc_path = cavity_dir / "cavity_pointcloud.npy"
    if not pc_path.exists():
        result["reason_exists"] = "cavity_pointcloud.npy not found"
        return result

    result["exists"] = True
    try:
        pc = np.load(str(pc_path))
    except Exception as exc:
        result["reason_exists"] = f"load error: {exc}"
        result["exists"] = False
        return result

    # Shape check (check 6)
    if pc.ndim == 2 and pc.shape[1] == 3 and pc.shape[0] >= PC_MIN_POINTS:
        result["shape_ok"] = True
        result["n_points"] = int(pc.shape[0])
    else:
        issues = []
        if pc.ndim != 2:
            issues.append(f"ndim={pc.ndim} (want 2)")
        elif pc.shape[1] != 3:
            issues.append(f"shape[1]={pc.shape[1]} (want 3)")
        if pc.ndim == 2 and pc.shape[0] < PC_MIN_POINTS:
            issues.append(f"n_points={pc.shape[0]} < {PC_MIN_POINTS}")
        result["reason_shape"] = "; ".join(issues)
        result["n_points"] = int(pc.shape[0]) if pc.ndim >= 1 else 0
        return result

    # NaN / Inf checks (check 7)
    has_nan = bool(np.any(np.isnan(pc)))
    has_inf = bool(np.any(np.isinf(pc)))
    result["no_nan"] = not has_nan
    result["no_inf"] = not has_inf

    # Bounds checks (check 8) — only meaningful when data is clean
    if not has_nan and not has_inf:
        x_span = float(pc[:, 0].max() - pc[:, 0].min())
        y_span = float(pc[:, 1].max() - pc[:, 1].min())
        z_span = float(pc[:, 2].max() - pc[:, 2].min())
        z_max  = float(pc[:, 2].max())
        z_min  = float(pc[:, 2].min())

        result["x_span_m"] = x_span
        result["y_span_m"] = y_span
        result["z_span_m"] = z_span
        result["x_min"]    = float(pc[:, 0].min())
        result["x_max"]    = float(pc[:, 0].max())
        result["y_min"]    = float(pc[:, 1].min())
        result["y_max"]    = float(pc[:, 1].max())
        result["z_min"]    = float(pc[:, 2].min())
        result["z_max"]    = z_max

        xy_issues = []
        z_issues  = []
        if x_span <= 0.0:
            xy_issues.append("X span == 0")
        if y_span <= 0.0:
            xy_issues.append("Y span == 0")
        if is_opening:
            # Opening representation lives on the board top plane: Z is
            # expected to be ~0.  Only flag impossible NEGATIVE Z extents.
            if z_min < -1e-6:
                z_issues.append(f"Z min={z_min:.6f} < 0 (opening should be at Z=0)")
            if z_max > 1e-3:
                z_issues.append(
                    f"Z max={z_max:.6f} > 1mm (opening pointcloud should be flat)")
        else:
            # Auxiliary depth representation: cavity floor below the board
            # surface — Z must be positive.
            if z_max <= 0.0:
                z_issues.append(
                    f"Z max={z_max:.6f} <= 0 (depth pointcloud must have "
                    f"positive Z = depth_px - board_surface_z)")
        if z_span < 0.0:
            z_issues.append("Z span < 0")

        result["xy_ok"] = len(xy_issues) == 0
        result["z_ok"]  = len(z_issues) == 0
        all_issues = xy_issues + z_issues
        if all_issues:
            result["reason_bounds"] = "; ".join(all_issues)
    else:
        nan_msg = "NaN values present" if has_nan else ""
        inf_msg = "Inf values present" if has_inf else ""
        result["reason_bounds"] = "; ".join(filter(None, [nan_msg, inf_msg]))
        result["xy_ok"] = False
        result["z_ok"]  = False

    return result


def _pc_bounds_ok(pc_result: dict) -> bool:
    """Return True if all point cloud checks pass."""
    return (
        pc_result["shape_ok"]
        and pc_result["no_nan"]
        and pc_result["no_inf"]
        and pc_result["xy_ok"]
        and pc_result["z_ok"]
    )


def check_image_exists(cavity_dir: Path, filename: str) -> tuple:
    """Return (ok: bool, reason: str)."""
    p = cavity_dir / filename
    if not p.exists():
        return False, f"{filename} not found"
    return True, ""


def check_depth_pointcloud(cavity_dir: Path, meta_content: dict) -> dict:
    """
    Inspect the auxiliary cavity_depth_pointcloud.npy and per-cavity depth
    metadata fields.  Reports whether a depth representation is available,
    NEVER fails the cavity (this is auxiliary).

    Returns:
      available     — bool
      reason        — short string explaining absence
      n_unique_pts  — int (deduplicated by 0.1 mm rounding; replicated samples
                            from sampling-with-replacement aren't real depth)
      depth_area_px — int from metadata (None if missing)
      z_max_m       — float or None
    """
    result = {
        "available":     False,
        "reason":        "",
        "n_unique_pts":  0,
        "depth_area_px": meta_content.get("depth_area_px"),
        "z_max_m":       meta_content.get("z_depth_max_m"),
    }

    p = cavity_dir / "cavity_depth_pointcloud.npy"
    if not p.exists():
        result["reason"] = "cavity_depth_pointcloud.npy not found"
        return result
    try:
        pc = np.load(str(p))
    except Exception as exc:
        result["reason"] = f"could not load: {exc}"
        return result

    if pc.ndim != 2 or pc.shape[1] != 3:
        result["reason"] = f"depth pointcloud has unexpected shape {pc.shape}"
        return result
    # Deduplicate at 0.1 mm to ignore the with-replacement padding.
    rounded = np.unique(np.round(pc, decimals=4), axis=0)
    n_unique = int(rounded.shape[0])
    result["n_unique_pts"] = n_unique

    z_max = float(pc[:, 2].max()) if pc.size else 0.0
    if n_unique <= 1 and z_max <= 1e-6:
        result["reason"] = ("depth pointcloud is degenerate (≤1 unique point, "
                            "Z≈0) — camera sees no pixels deeper than the "
                            "board top + tolerance")
        return result

    da = result["depth_area_px"]
    if da is not None and da == 0:
        result["reason"] = ("metadata.depth_area_px = 0 — no pixels in this "
                            "cavity's opening reach the depth-band threshold")
        return result

    result["available"] = True
    return result


# ── CAD scale check ──────────────────────────────────────────────────────────

_CAD_PATH  = PROJECT_ROOT / "data" / "expected_cad_dimensions.json"
_CAD_CACHE = None


def _load_cad_cavities() -> list:
    """Return list of CAD cavity x_span, y_span pairs in metres.  Cached."""
    global _CAD_CACHE
    if _CAD_CACHE is not None:
        return _CAD_CACHE
    if not _CAD_PATH.exists():
        _CAD_CACHE = []
        return _CAD_CACHE
    try:
        with open(_CAD_PATH) as f:
            cad = json.load(f)
    except Exception:
        _CAD_CACHE = []
        return _CAD_CACHE
    cavs = (cad.get("cavities") or {})
    out = []
    for name, entry in cavs.items():
        x = entry.get("x_span_m")
        y = entry.get("y_span_m")
        if x is not None and y is not None:
            out.append((name, float(x), float(y)))
    _CAD_CACHE = out
    return out


def check_cad_scale(meta_content: dict, tol: float = 0.10) -> dict:
    """
    Compare measured opening_xy_span_m against any CAD cavity span (any
    orientation).  No shape→cavity mapping is hardcoded — the cavity passes
    if its measured span matches at least one CAD cavity within ±`tol`.

    Returns:
      checked         — bool (False if no CAD reference available)
      ok              — bool (None if not checked, else True/False)
      best_match_name — string name of the closest CAD cavity, or ""
      max_rel_err     — float, the worst dimensional relative error vs the
                         best match (0.10 = 10%)
      warning         — short string (empty when ok)
    """
    result = {
        "checked":         False,
        "ok":              None,
        "best_match_name": "",
        "max_rel_err":     None,
        "warning":         "",
    }
    cad = _load_cad_cavities()
    if not cad:
        return result
    span = meta_content.get("opening_xy_span_m") or meta_content.get("xy_span_m")
    if not span:
        result["warning"] = "no opening_xy_span_m in metadata"
        return result
    mx = float(span.get("x", 0.0))
    my = float(span.get("y", 0.0))
    if mx <= 0 or my <= 0:
        result["warning"] = f"degenerate measured span ({mx}, {my})"
        return result

    result["checked"] = True
    best = None  # (max_rel_err, name)
    for name, cx, cy in cad:
        # Try both orientations; take the smaller worst-axis error.
        for tx, ty in ((cx, cy), (cy, cx)):
            ex = abs(mx - tx) / tx
            ey = abs(my - ty) / ty
            err = max(ex, ey)
            if best is None or err < best[0]:
                best = (err, name)
    err, name = best
    result["best_match_name"] = name
    result["max_rel_err"]     = err
    result["ok"]              = err <= tol
    if not result["ok"]:
        result["warning"] = (
            f"measured opening ({mx*1000:.2f}×{my*1000:.2f} mm) is "
            f"{err*100:.1f}% away from nearest CAD cavity "
            f"'{name}' (>{tol*100:.0f}% tolerance)")
    return result


def check_footprint_nonempty(cavity_dir: Path) -> tuple:
    """
    Check 9: cavity_footprint.png is readable and has at least one non-zero pixel.
    Returns (ok: bool, reason: str).
    """
    p = cavity_dir / "cavity_footprint.png"
    if not p.exists():
        return False, "cavity_footprint.png not found"
    img = cv2.imread(str(p))
    if img is None:
        return False, "cv2.imread returned None"
    if np.count_nonzero(img) == 0:
        return False, "image is all-zero (empty footprint)"
    return True, ""


# ── VALIDATE ONE CAVITY ───────────────────────────────────────────────────────

def validate_cavity(cavity_name: str) -> dict:
    """
    Run all checks for one cavity_NN folder.
    Returns a flat result dict suitable for JSON / CSV export.
    """
    cavity_dir = DATA_DIR / cavity_name
    folder_exists = cavity_dir.is_dir()

    result = {
        "cavity": cavity_name,
        "folder_exists": folder_exists,
        # File existence
        "metadata_ok": False,
        "pc_exists": False,
        "footprint_exists": False,
        "debug_exists": False,
        "mask_exists": False,
        # Representation type (NEW)
        "pointcloud_type": "unknown",
        "primary_matching_representation": None,
        # Point cloud
        "pc_shape_ok": False,
        "pc_no_nan": False,
        "pc_no_inf": False,
        "pc_xy_ok": False,
        "primary_z_ok": False,   # renamed from pc_z_ok
        "pc_z_ok": False,        # legacy alias kept for any downstream consumers
        "pc_n_points": 0,
        "pc_x_span_m": 0.0,
        "pc_y_span_m": 0.0,
        "pc_z_min_m": 0.0,
        "pc_z_max_m": 0.0,
        "pc_z_span_m": 0.0,
        "pc_x_min": 0.0, "pc_x_max": 0.0,
        "pc_y_min": 0.0, "pc_y_max": 0.0,
        # Opening / depth (NEW)
        "opening_xy_span_m":     None,
        "depth_available":       False,
        "depth_warning":         "",
        # CAD scale (NEW)
        "cad_scale_checked":     False,
        "cad_scale_ok":          None,
        "cad_scale_warning":     "",
        # Footprint
        "footprint_ok": False,
        # Advisory metadata
        "area_px": None,
        "area_px_ok": True,
        "advisory_fields": {},
        "metadata_excerpt": {},
        # Overall
        "reasons": [],
        "overall_pass": False,
    }

    if not folder_exists:
        result["reasons"].append(f"folder {cavity_dir} does not exist")
        return result

    # Check 1: metadata
    meta_check = check_metadata(cavity_dir)
    result["metadata_ok"]    = meta_check["exists"] and meta_check["area_px_ok"]
    result["area_px"]        = meta_check["area_px"]
    result["area_px_ok"]     = meta_check["area_px_ok"]
    result["advisory_fields"] = meta_check["advisory_fields"]

    if not meta_check["exists"]:
        result["reasons"].append(meta_check["reason_exists"])
    if meta_check["exists"] and not meta_check["area_px_ok"]:
        result["reasons"].append("area: " + meta_check["reason_area"])

    # Build a compact metadata_excerpt for JSON output
    if meta_check["content"]:
        mc = meta_check["content"]
        result["metadata_excerpt"] = {
            "cavity_id":        mc.get("cavity_id"),
            "area_px":          mc.get("area_px"),
            "centroid_world_m": mc.get("centroid_world_m"),
            "xy_span_m":        mc.get("xy_span_m"),
            "z_depth_range_m":  mc.get("z_depth_range_m"),
            "point_count":      mc.get("point_count"),
        }

    # ── Representation type from metadata ───────────────────────────────────
    mc = meta_check.get("content") or {}
    pmr = mc.get("primary_matching_representation")
    fs  = mc.get("footprint_source")
    is_opening = (pmr == "cavity_opening_pointcloud") \
        or (fs == "opening_from_board_region") \
        or (mc.get("xy_projection_depth_mode") == "board_surface_depth")
    pc_type = mc.get("pointcloud_type") or (
        "opening_on_board_plane" if is_opening else "unknown"
    )
    result["pointcloud_type"]                  = pc_type
    result["primary_matching_representation"]  = pmr
    result["opening_xy_span_m"]                = mc.get("opening_xy_span_m")

    # Checks 2, 6, 7, 8: point cloud (z rule depends on representation type)
    pc_check = check_pointcloud(cavity_dir, is_opening=is_opening)
    result["pc_exists"]   = pc_check["exists"]
    result["pc_shape_ok"] = pc_check["shape_ok"]
    result["pc_no_nan"]   = pc_check["no_nan"]
    result["pc_no_inf"]   = pc_check["no_inf"]
    result["pc_xy_ok"]    = pc_check["xy_ok"]
    result["primary_z_ok"] = pc_check["z_ok"]
    result["pc_z_ok"]      = pc_check["z_ok"]   # legacy alias
    result["pc_n_points"] = pc_check["n_points"]
    result["pc_x_span_m"] = pc_check["x_span_m"]
    result["pc_y_span_m"] = pc_check["y_span_m"]
    result["pc_z_min_m"]  = pc_check["z_min"]
    result["pc_z_max_m"]  = pc_check["z_max"]
    result["pc_z_span_m"] = pc_check["z_span_m"]
    result["pc_x_min"]    = pc_check["x_min"]
    result["pc_x_max"]    = pc_check["x_max"]
    result["pc_y_min"]    = pc_check["y_min"]
    result["pc_y_max"]    = pc_check["y_max"]

    if not pc_check["exists"]:
        result["reasons"].append(pc_check["reason_exists"])
    if pc_check["exists"] and not pc_check["shape_ok"]:
        result["reasons"].append("pc shape: " + pc_check["reason_shape"])
    if pc_check["exists"] and pc_check["shape_ok"] and not _pc_bounds_ok(pc_check):
        result["reasons"].append("pc bounds: " + pc_check["reason_bounds"])

    # Check 3: footprint exists
    fp_exists, fp_reason = check_image_exists(cavity_dir, "cavity_footprint.png")
    result["footprint_exists"] = fp_exists
    if not fp_exists:
        result["reasons"].append(fp_reason)

    # Check 9: footprint not empty
    fp_nonempty, fp_ne_reason = check_footprint_nonempty(cavity_dir)
    result["footprint_ok"] = fp_nonempty
    if not fp_nonempty and fp_exists:
        result["reasons"].append("footprint: " + fp_ne_reason)

    # Check 4: debug image
    dbg_exists, dbg_reason = check_image_exists(cavity_dir, "cavity_debug.png")
    result["debug_exists"] = dbg_exists
    if not dbg_exists:
        result["reasons"].append(dbg_reason)

    # Check 5: mask image
    mask_exists, mask_reason = check_image_exists(cavity_dir, "cavity_mask.png")
    result["mask_exists"] = mask_exists
    if not mask_exists:
        result["reasons"].append(mask_reason)

    # files_ok = all five per-cavity files present
    result["files_ok"] = (
        result["metadata_ok"]
        and result["pc_exists"]
        and result["footprint_exists"]
        and result["debug_exists"]
        and result["mask_exists"]
    )

    # ── Auxiliary depth representation (advisory, never fails) ──────────────
    depth_check = check_depth_pointcloud(cavity_dir, mc)
    result["depth_available"] = depth_check["available"]
    result["depth_warning"]   = depth_check["reason"]
    if not depth_check["available"]:
        # Surface as a non-fatal note in `reasons` for visibility but do NOT
        # affect overall_pass.
        result["reasons"].append(
            f"NOTE: depth representation unavailable or degenerate; "
            f"opening representation still valid for Baseline 1. "
            f"({depth_check['reason'] or 'no reason'})")

    # ── CAD scale check (advisory, never fails) ─────────────────────────────
    cad_check = check_cad_scale(mc, tol=0.10)
    result["cad_scale_checked"] = cad_check["checked"]
    result["cad_scale_ok"]      = cad_check["ok"]
    result["cad_scale_warning"] = cad_check["warning"]
    if cad_check["checked"] and cad_check["ok"] is False:
        result["reasons"].append("WARNING: " + cad_check["warning"])

    # Overall pass: structural checks only.  Depth availability and CAD scale
    # are advisory — opening representations with Z=0 are valid for Baseline 1.
    result["overall_pass"] = (
        result["folder_exists"]
        and result["metadata_ok"]
        and result["pc_exists"]
        and result["pc_shape_ok"]
        and _pc_bounds_ok(pc_check)
        and result["footprint_ok"]
        and result["debug_exists"]
        and result["mask_exists"]
    )

    return result


# ── DISCOVER CAVITIES ─────────────────────────────────────────────────────────

def discover_cavities(data_dir: Path) -> list:
    """Return sorted list of cavity folder names found under data_dir."""
    return sorted(p.name for p in data_dir.glob("cavity_*") if p.is_dir())


# ── CONSOLE TABLE ─────────────────────────────────────────────────────────────

def print_summary_table(results: list) -> None:
    """Print a plain-ASCII aligned summary table to stdout."""
    header = (
        f"{'cavity':<12} "
        f"{'files_ok':<10} "
        f"{'pc_shape':<10} "
        f"{'pc_no_nan':<10} "
        f"{'pc_no_inf':<10} "
        f"{'pc_xy_ok':<10} "
        f"{'pc_z_ok':<10} "
        f"{'footprint':<10} "
        f"{'metadata':<10} "
        f"{'OVERALL':<8}"
    )
    sep = "-" * len(header)

    print("\n--- Cavities ---")
    print(sep)
    print(header)
    print(sep)
    for r in results:
        files_ok = (
            r["metadata_ok"]
            and r["pc_exists"]
            and r["footprint_exists"]
            and r["debug_exists"]
            and r["mask_exists"]
        )
        print(
            f"{r['cavity']:<12} "
            f"{_tick(files_ok):<10} "
            f"{_tick(r['pc_shape_ok']):<10} "
            f"{_tick(r['pc_no_nan']):<10} "
            f"{_tick(r['pc_no_inf']):<10} "
            f"{_tick(r['pc_xy_ok']):<10} "
            f"{_tick(r['pc_z_ok']):<10} "
            f"{_tick(r['footprint_ok']):<10} "
            f"{_tick(r['metadata_ok']):<10} "
            f"{_tick(r['overall_pass']):<8}"
        )
    print(sep)

    print("\nPer-cavity spans (metres), area_px, and point counts:")
    span_hdr = (
        f"  {'cavity':<12} "
        f"{'area_px':>8} "
        f"{'xy_span_x':>10} "
        f"{'xy_span_y':>10} "
        f"{'z_min':>10} "
        f"{'z_max':>10} "
        f"{'z_span':>10} "
        f"{'n_points':>10}"
    )
    print(span_hdr)
    print("  " + "-" * (len(span_hdr) - 2))
    for r in results:
        area_str = str(r["area_px"]) if r["area_px"] is not None else "N/A"
        print(
            f"  {r['cavity']:<12} "
            f"{area_str:>8} "
            f"{r['pc_x_span_m']:>10.5f} "
            f"{r['pc_y_span_m']:>10.5f} "
            f"{r['pc_z_min_m']:>10.5f} "
            f"{r['pc_z_max_m']:>10.5f} "
            f"{r['pc_z_span_m']:>10.5f} "
            f"{r['pc_n_points']:>10}"
        )

    print("\nAdvisory metadata fields (present/absent):")
    if results:
        fields = list(results[0]["advisory_fields"].keys())
        adv_hdr = f"  {'cavity':<12} " + " ".join(f"{f:<22}" for f in fields)
        print(adv_hdr)
        for r in results:
            row = f"  {r['cavity']:<12} "
            row += " ".join(f"{r['advisory_fields'].get(f, 'absent'):<22}" for f in fields)
            print(row)

    print("\nFailure reasons:")
    any_fail = False
    for r in results:
        if r["reasons"]:
            any_fail = True
            print(f"  {r['cavity']}: {'; '.join(r['reasons'])}")
    if not any_fail:
        print("  (none — all checks passed)")


# ── SAVE JSON ─────────────────────────────────────────────────────────────────

def save_json(
    global_checks: dict,
    results: list,
    out_path: Path,
) -> None:
    n_found  = len(results)
    n_passed = sum(1 for r in results if r["overall_pass"])

    payload = {
        "globals": global_checks,
        "cavities_summary_present": global_checks.get("cavities_summary.json", False),
        "n_cavities_found": n_found,
        "n_cavities_passed": n_passed,
        "cavities": [],
    }

    for r in results:
        files_ok = (
            r["metadata_ok"]
            and r["pc_exists"]
            and r["footprint_exists"]
            and r["debug_exists"]
            and r["mask_exists"]
        )
        payload["cavities"].append({
            "cavity":         r["cavity"],
            "files_ok":       files_ok,
            "pc_shape_ok":    r["pc_shape_ok"],
            "pc_no_nan":      r["pc_no_nan"],
            "pc_no_inf":      r["pc_no_inf"],
            "pc_xy_ok":       r["pc_xy_ok"],
            "pc_z_ok":        r["pc_z_ok"],
            "footprint_ok":   r["footprint_ok"],
            "metadata_ok":    r["metadata_ok"],
            "overall_pass":   r["overall_pass"],
            "pc_bounds": {
                "x_min":   r["pc_x_min"],
                "x_max":   r["pc_x_max"],
                "y_min":   r["pc_y_min"],
                "y_max":   r["pc_y_max"],
                "z_min":   r["pc_z_min_m"],
                "z_max":   r["pc_z_max_m"],
                "x_span":  r["pc_x_span_m"],
                "y_span":  r["pc_y_span_m"],
                "z_span":  r["pc_z_span_m"],
            },
            "metadata_excerpt":  r["metadata_excerpt"],
            "advisory_fields":   r["advisory_fields"],
            "failure_reasons":   r["reasons"],
        })

    with open(str(out_path), "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"[output] validation_summary.json -> {out_path}")


# ── SAVE CSV ──────────────────────────────────────────────────────────────────

def save_csv(results: list, out_path: Path) -> None:
    fieldnames = [
        "cavity",
        "files_ok",
        "pc_shape_ok",
        "pc_no_nan",
        "pc_no_inf",
        "pc_n_points",
        "pc_x_span_m",
        "pc_y_span_m",
        "pc_z_min_m",
        "pc_z_max_m",
        "pc_z_span_m",
        "footprint_ok",
        "metadata_ok",
        "overall",
    ]
    with open(str(out_path), "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            files_ok = (
                r["metadata_ok"]
                and r["pc_exists"]
                and r["footprint_exists"]
                and r["debug_exists"]
                and r["mask_exists"]
            )
            writer.writerow({
                "cavity":       r["cavity"],
                "files_ok":     files_ok,
                "pc_shape_ok":  r["pc_shape_ok"],
                "pc_no_nan":    r["pc_no_nan"],
                "pc_no_inf":    r["pc_no_inf"],
                "pc_n_points":  r["pc_n_points"],
                "pc_x_span_m":  round(r["pc_x_span_m"], 6),
                "pc_y_span_m":  round(r["pc_y_span_m"], 6),
                "pc_z_min_m":   round(r["pc_z_min_m"], 6),
                "pc_z_max_m":   round(r["pc_z_max_m"], 6),
                "pc_z_span_m":  round(r["pc_z_span_m"], 6),
                "footprint_ok": r["footprint_ok"],
                "metadata_ok":  r["metadata_ok"],
                "overall":      r["overall_pass"],
            })
    print(f"[output] validation_summary.csv  -> {out_path}")


# ── FOOTPRINTS GRID ───────────────────────────────────────────────────────────

def _grid_layout(n: int) -> tuple:
    """Return (n_cols, n_rows) for a near-square grid holding n items."""
    n_cols = max(1, math.ceil(math.sqrt(n)))
    n_rows = max(1, math.ceil(n / n_cols))
    return n_cols, n_rows


def _make_tile_opencv(cavity_name: str, img_path: Path, cell_px: int) -> np.ndarray:
    """
    Load a footprint image, resize to cell_px x cell_px, add a label bar.
    Returns a black tile with a MISSING label when the file cannot be loaded.
    """
    label_h = 28
    cell = np.zeros((cell_px, cell_px, 3), dtype=np.uint8)

    if img_path.exists():
        img = cv2.imread(str(img_path))
        if img is not None:
            cell = cv2.resize(img, (cell_px, cell_px))
        else:
            cv2.putText(cell, "READ ERROR", (10, cell_px // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            cv2.putText(cell, cavity_name, (10, cell_px // 2 + 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)
    else:
        cv2.putText(cell, "MISSING", (10, cell_px // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
        cv2.putText(cell, cavity_name, (10, cell_px // 2 + 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 200), 1)

    label_bar = np.zeros((label_h, cell_px, 3), dtype=np.uint8)
    cv2.putText(label_bar, cavity_name, (6, label_h - 7),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.vstack([label_bar, cell])


def save_footprints_grid_opencv(results: list, out_path: Path, cell_px: int) -> None:
    """Build a dynamic grid using OpenCV only."""
    n = len(results)
    if n == 0:
        blank = np.zeros((cell_px + 28, cell_px, 3), dtype=np.uint8)
        cv2.imwrite(str(out_path), blank)
        print(f"[output] footprints_grid.png     -> {out_path}  (empty, OpenCV)")
        return

    n_cols, n_rows = _grid_layout(n)
    tile_h = cell_px + 28  # cell + label bar

    tiles = []
    for r in results:
        img_path = DATA_DIR / r["cavity"] / "cavity_footprint.png"
        tiles.append(_make_tile_opencv(r["cavity"], img_path, cell_px))

    # Pad to fill grid
    blank_tile = np.zeros((tile_h, cell_px, 3), dtype=np.uint8)
    while len(tiles) < n_cols * n_rows:
        tiles.append(blank_tile)

    rows = []
    for row_idx in range(n_rows):
        row_tiles = tiles[row_idx * n_cols : (row_idx + 1) * n_cols]
        rows.append(np.hstack(row_tiles))
    grid = np.vstack(rows)
    cv2.imwrite(str(out_path), grid)
    print(f"[output] footprints_grid.png     -> {out_path}  ({n_cols}x{n_rows}, OpenCV)")


def save_footprints_grid_matplotlib(results: list, out_path: Path, cell_px: int) -> None:
    """Build a dynamic grid using matplotlib (preferred when available)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(results)
    if n == 0:
        fig, ax = plt.subplots(1, 1, figsize=(4, 4))
        ax.text(0.5, 0.5, "No cavities found", ha="center", va="center",
                transform=ax.transAxes)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=100)
        plt.close(fig)
        print(f"[output] footprints_grid.png     -> {out_path}  (empty, matplotlib)")
        return

    n_cols, n_rows = _grid_layout(n)
    fig_w = n_cols * (cell_px / 100.0)
    fig_h = n_rows * (cell_px / 100.0) + 0.5 * n_rows  # extra height for titles
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(max(fig_w, 4), max(fig_h, 4)))

    # Normalise axes to always be a 1-D list
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes_flat = list(axes.flatten())
    else:
        axes_flat = list(axes.flatten())

    for idx, r in enumerate(results):
        ax = axes_flat[idx]
        img_path = DATA_DIR / r["cavity"] / "cavity_footprint.png"
        ax.set_title(r["cavity"], fontsize=10)
        ax.axis("off")

        if img_path.exists():
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            else:
                ax.text(0.5, 0.5, f"READ ERROR\n{r['cavity']}",
                        ha="center", va="center", transform=ax.transAxes,
                        color="red", fontsize=9)
        else:
            ax.text(0.5, 0.5, f"MISSING\n{r['cavity']}",
                    ha="center", va="center", transform=ax.transAxes,
                    color="red", fontsize=9)

    # Hide unused axes
    for idx in range(len(results), len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100)
    plt.close(fig)
    print(f"[output] footprints_grid.png     -> {out_path}  ({n_cols}x{n_rows}, matplotlib)")


def save_footprints_grid(results: list, out_path: Path, cell_px: int) -> None:
    """Try matplotlib first, fall back to OpenCV."""
    try:
        import matplotlib  # noqa: F401
        save_footprints_grid_matplotlib(results, out_path, cell_px)
    except ImportError:
        print("[output] matplotlib not available — using OpenCV for grid")
        save_footprints_grid_opencv(results, out_path, cell_px)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("validate_cavity_captures.py")
    print("=" * 70)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATA_DIR     : {DATA_DIR}")

    if not DATA_DIR.exists():
        print(f"\n[ERROR] DATA_DIR does not exist: {DATA_DIR}")
        print("Cavity capture data has not been pulled to this machine yet.")
        sys.exit(1)

    # Global file checks
    global_checks = check_global_files(DATA_DIR)
    print_global_summary(global_checks)

    # Discover cavities dynamically
    cavity_names = discover_cavities(DATA_DIR)
    print(f"\n[discover] found {len(cavity_names)} cavity folder(s): {cavity_names}")

    if not cavity_names:
        print("\n[WARNING] No cavity_* subfolders found under DATA_DIR.")
        print("Run capture_cavity_detection.py inside Isaac Sim first.")
        # Still write empty outputs so downstream tools do not choke
        save_json(global_checks, [], DATA_DIR / "validation_summary.json")
        save_csv([], DATA_DIR / "validation_summary.csv")
        save_footprints_grid([], DATA_DIR / "footprints_grid.png", FOOTPRINT_GRID_CELL)
        sys.exit(0)

    # Validate each cavity
    results = []
    for name in cavity_names:
        print(f"\n[validate] checking '{name}' ...")
        r = validate_cavity(name)
        results.append(r)
        status = "PASS" if r["overall_pass"] else "FAIL"
        print(f"  -> {status}  reasons: {r['reasons'] if r['reasons'] else '(none)'}")

    # Console table
    print_summary_table(results)

    # Save outputs
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_json(global_checks, results, DATA_DIR / "validation_summary.json")
    save_csv(results, DATA_DIR / "validation_summary.csv")
    save_footprints_grid(results, DATA_DIR / "footprints_grid.png", FOOTPRINT_GRID_CELL)

    # Final verdict
    n_pass = sum(1 for r in results if r["overall_pass"])
    n_fail = len(results) - n_pass
    print(f"\n{'='*70}")
    print(f"Overall: {n_pass}/{len(results)} cavities passed all checks.")
    if n_fail > 0:
        print(f"WARNING: {n_fail} cavity/cavities failed — inspect failure reasons above.")
    else:
        print("All cavity captures are structurally valid.")
    print(f"{'='*70}\n")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
