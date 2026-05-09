"""
inspect_cavity_scene_scale.py — Read-only USD scene scale inspector

Diagnoses why cavity_03 (circular opening) appears ~60.84 mm wide in
perception when the CAD nominal is 51 mm (Ø51 mm with 0.5 mm clearance).

This script does NOT modify the scene.  It only reads USD prims and
reports bounding boxes, xformOp:scale values and parent-chain scales.

Outputs:
  data/scene_scale_inspection/scene_scale_inspection.md   (human-readable)
  data/scene_scale_inspection/scene_scale_inspection.json (machine-readable)

Run inside Isaac Sim 5.1 Script Editor:
  Open the file and press Run, or paste into the Script Editor and execute.

NOTE: __file__ is unreliable inside the Script Editor (resolves to
/tmp/carb.../script_*.py).  PROJECT_ROOT is set explicitly with an
env-var escape hatch.
"""

import asyncio
import json
import os
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ── CONFIG ─────────────────────────────────────────────────────────────────────

SCRIPT_NAME = "inspect_cavity_scene_scale.py"

# Project root — override with SHAPE_INSERTION_PROJECT_ROOT if needed.
PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        "/workspace/Tese_Roberto/shape_insertion/thesis-omniverse",
    )
)

OUT_DIR = PROJECT_ROOT / "data" / "scene_scale_inspection"

# Case-insensitive prim-path / prim-name search terms.
SEARCH_TERMS = ["cavity", "circle", "cylinder", "board", "hole"]

# Tolerance for non-unity scale detection.
SCALE_UNITY_TOL = 1e-4

# Bbox thresholds for highlighting (metres).
BBOX_SUSPECT_CIRCLE_M   = 0.06084   # measured oversized diameter
BBOX_EXPECTED_CIRCLE_M  = 0.051     # CAD nominal diameter
BBOX_BOARD_THICKNESS_M  = 0.075     # CAD board thickness (sanity check)
BBOX_HIGHLIGHT_MARGIN_M = 0.005     # ±5 mm window for circle checks
BBOX_BOARD_MARGIN_M     = 0.010     # ±10 mm window for board thickness check

# Four prims that must always appear in the report regardless of search terms.
FORCED_INSPECTION_PATHS = [
    "/World/Circle",
    "/World/Circle/Body1",
    "/World/Board_Tese",
    "/World/Board_Tese/Body1",
]

# ── HELPERS ────────────────────────────────────────────────────────────────────

def _is_near(value_m: float, target_m: float, margin_m: float = BBOX_HIGHLIGHT_MARGIN_M) -> bool:
    return abs(value_m - target_m) <= margin_m


def _scale_is_unity(scale_tuple) -> bool:
    """Return True if all three scale components are within SCALE_UNITY_TOL of 1.0."""
    return all(abs(v - 1.0) <= SCALE_UNITY_TOL for v in scale_tuple)


def _get_local_scale(prim) -> tuple:
    """
    Return the local xformOp:scale as a (sx, sy, sz) tuple, or None if absent.
    Reads UsdGeom.Xformable ordered ops and finds the scale op.
    """
    try:
        from pxr import UsdGeom
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return None
        ops = xformable.GetOrderedXformOps()
        for op in ops:
            op_type = op.GetOpType()
            if op_type == UsdGeom.XformOp.TypeScale:
                val = op.Get()
                if val is not None:
                    return tuple(float(v) for v in val)
    except Exception:
        pass
    return None


def _get_local_matrix(prim):
    """Return the local transform matrix as a flat list[16], or None on failure."""
    try:
        from pxr import UsdGeom, Usd
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return None
        mat = xformable.GetLocalTransformation(Usd.TimeCode.Default())
        if mat is None:
            return None
        return [float(v) for row in mat for v in row]
    except Exception:
        return None


def _get_world_matrix(prim):
    """Return the world (local-to-world) matrix as a flat list[16], or None."""
    try:
        from pxr import UsdGeom, Usd
        xformable = UsdGeom.Xformable(prim)
        if not xformable:
            return None
        mat = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        if mat is None:
            return None
        return [float(v) for row in mat for v in row]
    except Exception:
        return None


def _mesh_points_world_bbox(prim):
    """
    Compute the world-space AABB of a single UsdGeom.Mesh prim by transforming
    its points attribute through its local-to-world matrix.

    Returns (mn, mx) as (Gf.Vec3d, Gf.Vec3d) or None if the prim has no mesh points.
    """
    try:
        from pxr import UsdGeom, Usd, Gf
        mesh = UsdGeom.Mesh(prim)
        if not mesh:
            return None
        points_attr = mesh.GetPointsAttr()
        if not points_attr or not points_attr.HasAuthoredValue():
            return None
        local_points = points_attr.Get()
        if local_points is None or len(local_points) == 0:
            return None
        xform = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        xs, ys, zs = [], [], []
        for p in local_points:
            wp = xform.Transform(Gf.Vec3d(float(p[0]), float(p[1]), float(p[2])))
            xs.append(wp[0]); ys.append(wp[1]); zs.append(wp[2])
        from pxr import Gf as _Gf
        mn = _Gf.Vec3d(min(xs), min(ys), min(zs))
        mx = _Gf.Vec3d(max(xs), max(ys), max(zs))
        return (mn, mx)
    except Exception:
        return None


def _union_ranges(range_list):
    """
    Union a list of (mn, mx) tuples (Gf.Vec3d pairs) into a single (mn, mx).
    Returns None if the list is empty.
    """
    if not range_list:
        return None
    all_min_x = [r[0][0] for r in range_list]
    all_min_y = [r[0][1] for r in range_list]
    all_min_z = [r[0][2] for r in range_list]
    all_max_x = [r[1][0] for r in range_list]
    all_max_y = [r[1][1] for r in range_list]
    all_max_z = [r[1][2] for r in range_list]
    from pxr import Gf
    mn = Gf.Vec3d(min(all_min_x), min(all_min_y), min(all_min_z))
    mx = Gf.Vec3d(max(all_max_x), max(all_max_y), max(all_max_z))
    return (mn, mx)


def world_bbox_from_subtree(prim):
    """
    Walk the subtree rooted at prim (including prim itself), collect all
    UsdGeom.Mesh point-cloud bboxes and return their union as (mn, mx).
    Returns None if no mesh points are found anywhere in the subtree.
    """
    ranges = []
    # Walk every prim in subtree (depth-first using USD iterator).
    from pxr import Usd
    it = iter(Usd.PrimRange(prim))
    for p in it:
        r = _mesh_points_world_bbox(p)
        if r is not None:
            ranges.append(r)
    return _union_ranges(ranges)


def _compute_world_bbox(prim, bbox_cache, meters_per_unit: float):
    """
    Two-strategy world bbox computation.

    Strategy 1: UsdGeom.BBoxCache.ComputeWorldBound.
    Strategy 2: world_bbox_from_subtree (mesh-points fallback).

    Returns a dict with keys:
        bbox_available, bbox_method, bbox_min_m, bbox_max_m,
        bbox_size_m, bbox_size_mm, bbox_highlight_flags, bbox_error
    """
    result = {
        "bbox_available": False,
        "bbox_method": "unavailable",
        "bbox_min_m": None,
        "bbox_max_m": None,
        "bbox_size_m": None,
        "bbox_size_mm": None,
        "bbox_highlight_flags": [],
        "bbox_error": None,
    }

    # ── Strategy 1: BBoxCache ─────────────────────────────────────────────────
    if bbox_cache is not None:
        try:
            world_bbox = bbox_cache.ComputeWorldBound(prim)
            range_ = world_bbox.ComputeAlignedRange()
            if not range_.IsEmpty():
                mn = range_.GetMin()
                mx = range_.GetMax()
                size = (
                    float((mx[0] - mn[0]) * meters_per_unit),
                    float((mx[1] - mn[1]) * meters_per_unit),
                    float((mx[2] - mn[2]) * meters_per_unit),
                )
                result["bbox_available"] = True
                result["bbox_method"] = "BBoxCache"
                result["bbox_min_m"] = [float(mn[0] * meters_per_unit),
                                        float(mn[1] * meters_per_unit),
                                        float(mn[2] * meters_per_unit)]
                result["bbox_max_m"] = [float(mx[0] * meters_per_unit),
                                        float(mx[1] * meters_per_unit),
                                        float(mx[2] * meters_per_unit)]
                result["bbox_size_m"] = list(size)
                result["bbox_size_mm"] = [round(v * 1000.0, 3) for v in size]
                result["bbox_highlight_flags"] = _bbox_highlight_flags(size)
                return result
            else:
                result["bbox_method"] = "empty"
                result["bbox_error"] = "BBoxCache returned empty range"
        except Exception as exc:
            result["bbox_error"] = f"BBoxCache failed: {exc}"
            # fall through to strategy 2

    # ── Strategy 2: mesh-points fallback ─────────────────────────────────────
    try:
        r = world_bbox_from_subtree(prim)
        if r is not None:
            mn, mx = r
            size = (
                float((mx[0] - mn[0]) * meters_per_unit),
                float((mx[1] - mn[1]) * meters_per_unit),
                float((mx[2] - mn[2]) * meters_per_unit),
            )
            result["bbox_available"] = True
            result["bbox_method"] = "mesh_points_fallback"
            result["bbox_min_m"] = [float(mn[0] * meters_per_unit),
                                    float(mn[1] * meters_per_unit),
                                    float(mn[2] * meters_per_unit)]
            result["bbox_max_m"] = [float(mx[0] * meters_per_unit),
                                    float(mx[1] * meters_per_unit),
                                    float(mx[2] * meters_per_unit)]
            result["bbox_size_m"] = list(size)
            result["bbox_size_mm"] = [round(v * 1000.0, 3) for v in size]
            result["bbox_highlight_flags"] = _bbox_highlight_flags(size)
            # Append to existing error (if BBoxCache failed) rather than clobber.
            fallback_note = "used mesh_points_fallback"
            if result["bbox_error"]:
                result["bbox_error"] += f"; {fallback_note}"
            return result
        else:
            fallback_note = "no mesh points found in subtree"
            if result["bbox_error"]:
                result["bbox_error"] += f"; {fallback_note}"
            else:
                result["bbox_error"] = fallback_note
    except Exception as exc2:
        note = f"mesh_points_fallback failed: {exc2}"
        if result["bbox_error"]:
            result["bbox_error"] += f"; {note}"
        else:
            result["bbox_error"] = note

    return result


def _collect_parent_non_unity_scales(prim, stage):
    """
    Walk from prim up to the root.  Return a list of dicts describing every
    ancestor that has a non-unity xformOp:scale.
    """
    results = []
    current_path = prim.GetPath()
    while True:
        parent_path = current_path.GetParentPath()
        if parent_path == current_path or str(parent_path) in ("", "/"):
            break
        current_path = parent_path
        try:
            ancestor = stage.GetPrimAtPath(current_path)
            if not ancestor or not ancestor.IsValid():
                break
            scale = _get_local_scale(ancestor)
            if scale is not None and not _scale_is_unity(scale):
                results.append({
                    "path": str(current_path),
                    "scale": list(scale),
                })
        except Exception:
            break
    return results


def _prim_matches_terms(prim) -> bool:
    """Return True if the prim path or name contains any SEARCH_TERMS (case-insensitive)."""
    prim_path_lower = str(prim.GetPath()).lower()
    prim_name_lower = prim.GetName().lower()
    for term in SEARCH_TERMS:
        if term in prim_path_lower or term in prim_name_lower:
            return True
    return False


def _flag_highlights(info: dict) -> list:
    """
    Return a list of highlight reason strings for a given prim info dict.
    Empty list means no highlights.
    """
    flags = []
    type_name = info.get("type_name", "")
    prim_path_lower = info.get("path", "").lower()

    # Type-based flags
    if type_name in ("Cylinder", "Sphere", "Disk"):
        flags.append(f"prim_type={type_name} (potential circular geometry)")
    if "circle" in prim_path_lower or "cylinder" in prim_path_lower:
        flags.append("name/path contains 'circle' or 'cylinder'")

    # Non-unity local scale
    scale = info.get("local_scale")
    if scale is not None and not _scale_is_unity(scale):
        flags.append(f"NON-UNITY local scale: {scale}")

    # Non-unity ancestor scale
    ancestor_scales = info.get("parent_non_unity_scales", [])
    if ancestor_scales:
        for anc in ancestor_scales:
            flags.append(f"NON-UNITY ancestor scale at {anc['path']}: {anc['scale']}")

    # Bbox dimension proximity
    bbox_size_m = info.get("bbox_size_m")
    if bbox_size_m:
        for axis_idx, axis_label in enumerate(["X", "Y", "Z"]):
            dim = bbox_size_m[axis_idx]
            if _is_near(dim, BBOX_SUSPECT_CIRCLE_M, BBOX_HIGHLIGHT_MARGIN_M):
                flags.append(
                    f"bbox_{axis_label} = {dim*1000:.2f} mm "
                    f"≈ {BBOX_SUSPECT_CIRCLE_M*1000:.1f} mm (suspect oversized circle)"
                )
            if _is_near(dim, BBOX_EXPECTED_CIRCLE_M, BBOX_HIGHLIGHT_MARGIN_M):
                flags.append(
                    f"bbox_{axis_label} = {dim*1000:.2f} mm "
                    f"≈ {BBOX_EXPECTED_CIRCLE_M*1000:.1f} mm (matches CAD nominal)"
                )
            if _is_near(dim, BBOX_BOARD_THICKNESS_M, BBOX_BOARD_MARGIN_M):
                flags.append(
                    f"bbox_{axis_label} = {dim*1000:.2f} mm "
                    f"≈ {BBOX_BOARD_THICKNESS_M*1000:.1f} mm (matches CAD board thickness)"
                )

    return flags


def _bbox_highlight_flags(bbox_size_m) -> list:
    """
    Return structured flag strings for the bbox_highlight_flags JSON field.
    Uses the three highlight windows defined in config.
    """
    if not bbox_size_m:
        return []
    result = []
    for dim in bbox_size_m:
        if _is_near(dim, BBOX_EXPECTED_CIRCLE_M, BBOX_HIGHLIGHT_MARGIN_M):
            if "near_51mm" not in result:
                result.append("near_51mm")
        if _is_near(dim, BBOX_SUSPECT_CIRCLE_M, BBOX_HIGHLIGHT_MARGIN_M):
            if "near_60.84mm" not in result:
                result.append("near_60.84mm")
        if _is_near(dim, BBOX_BOARD_THICKNESS_M, BBOX_BOARD_MARGIN_M):
            if "near_75mm" not in result:
                result.append("near_75mm")
    return result


# ── MAIN INSPECTION ────────────────────────────────────────────────────────────

def inspect_prim(prim, stage, bbox_cache, meters_per_unit: float = 1.0) -> dict:
    """
    Gather all inspection data for a single prim.
    Returns a dict with path, type, scale, transforms, bbox and flags.
    """
    path = str(prim.GetPath())
    info = {
        "path": path,
        "type_name": prim.GetTypeName(),
        "local_scale": None,
        "local_matrix": None,
        "world_matrix": None,
        # New bbox fields (structured).
        "bbox_available": False,
        "bbox_method": "unavailable",
        "bbox_min_m": None,
        "bbox_max_m": None,
        "bbox_size_m": None,
        "bbox_size_mm": None,
        "bbox_highlight_flags": [],
        "bbox_error": None,
        # Legacy aliases kept so existing report helpers still work.
        "bbox_m": None,
        "bbox_mm": None,
        "parent_non_unity_scales": [],
        "highlights": [],
        "errors": [],
    }

    try:
        info["local_scale"] = _get_local_scale(prim)
    except Exception as exc:
        info["errors"].append(f"local_scale: {exc}")

    try:
        info["local_matrix"] = _get_local_matrix(prim)
    except Exception as exc:
        info["errors"].append(f"local_matrix: {exc}")

    try:
        info["world_matrix"] = _get_world_matrix(prim)
    except Exception as exc:
        info["errors"].append(f"world_matrix: {exc}")

    try:
        bbox_data = _compute_world_bbox(prim, bbox_cache, meters_per_unit)
        info.update(bbox_data)
        # Keep legacy aliases in sync so existing report helpers work.
        if bbox_data["bbox_available"]:
            info["bbox_m"]  = bbox_data["bbox_size_m"]
            info["bbox_mm"] = bbox_data["bbox_size_mm"]
        if bbox_data["bbox_error"]:
            info["errors"].append(f"bbox: {bbox_data['bbox_error']}")
    except Exception as exc:
        info["errors"].append(f"bbox: {exc}")

    try:
        info["parent_non_unity_scales"] = _collect_parent_non_unity_scales(prim, stage)
    except Exception as exc:
        info["errors"].append(f"parent_chain: {exc}")

    info["highlights"] = _flag_highlights(info)

    return info


def collect_candidates(stage) -> list:
    """
    Traverse all prims in the stage and return those whose path or name
    matches SEARCH_TERMS.  Results are sorted by prim path.
    """
    candidates = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        if _prim_matches_terms(prim):
            candidates.append(prim)
    # Sort deterministically by path.
    candidates.sort(key=lambda p: str(p.GetPath()))
    return candidates


# ── REPORT BUILDERS ────────────────────────────────────────────────────────────

def _scale_str(scale) -> str:
    if scale is None:
        return "absent"
    if _scale_is_unity(scale):
        return f"({scale[0]:.6f}, {scale[1]:.6f}, {scale[2]:.6f})  [unity]"
    return f"({scale[0]:.6f}, {scale[1]:.6f}, {scale[2]:.6f})  *** NON-UNITY ***"


def _bbox_str(info: dict) -> str:
    if not info.get("bbox_available"):
        method = info.get("bbox_method", "unavailable")
        err = info.get("bbox_error", "")
        if err:
            return f"unavailable [{method}] — {err}"
        return f"unavailable [{method}]"
    size_m = info.get("bbox_size_m")
    size_mm = info.get("bbox_size_mm")
    method = info.get("bbox_method", "?")
    if size_m is None or size_mm is None:
        return f"unavailable [{method}]"
    return (
        f"X={size_m[0]:.5f} m ({size_mm[0]:.2f} mm)  "
        f"Y={size_m[1]:.5f} m ({size_mm[1]:.2f} mm)  "
        f"Z={size_m[2]:.5f} m ({size_mm[2]:.2f} mm)  "
        f"[method: {method}]"
    )


def build_markdown_report(
    all_prims_info: list,
    stage_identifier: str,
    timestamp: str,
) -> str:
    lines = []

    # Header
    lines += [
        f"# Scene Scale Inspection Report",
        f"",
        f"**Script**: `{SCRIPT_NAME}`",
        f"**Timestamp**: {timestamp}",
        f"**Stage**: `{stage_identifier}`",
        f"",
        "---",
        "",
        "## Context",
        "",
        "cavity_03 (circular opening) measures **60.84 × 60.84 mm** in perception.",
        f"CAD nominal diameter: **51 mm** (clearance-adjusted: 51 mm).",
        f"Inflation: **+9.84 mm (+19.3 %)**.",
        "Perception scale is consistent across all 4 cavities (0.749 vs 0.751 mm/px → 0.3 %),",
        "ruling out projection or segmentation errors.  This report investigates the USD scene.",
        "",
        "---",
        "",
    ]

    # ── Candidates overview ────────────────────────────────────────────────────
    lines.append(f"## Candidate Prims ({len(all_prims_info)} found)")
    lines.append("")
    lines.append("Sorted by prim path.  Prims with highlights are marked `[!]`.")
    lines.append("")

    for info in all_prims_info:
        marker = "[!]" if info["highlights"] else "   "
        lines.append(f"- {marker} `{info['path']}` ({info['type_name']})")

    lines += ["", "---", ""]

    # ── Per-prim detail ────────────────────────────────────────────────────────
    lines.append("## Per-Prim Detail")
    lines.append("")

    for info in all_prims_info:
        has_flags = bool(info["highlights"])
        header_marker = "### [!]" if has_flags else "###"
        lines.append(f"{header_marker} `{info['path']}`")
        lines.append("")
        lines.append(f"- **Type**: `{info['type_name']}`")
        lines.append(f"- **Local scale**: {_scale_str(info['local_scale'])}")
        lines.append(f"- **BBox available**: `{info.get('bbox_available', False)}`")
        lines.append(f"- **BBox method**: `{info.get('bbox_method', 'unavailable')}`")
        lines.append(f"- **BBox (world)**: {_bbox_str(info)}")
        bbox_flags = info.get("bbox_highlight_flags", [])
        if bbox_flags:
            lines.append(f"- **BBox highlight flags**: {', '.join(bbox_flags)}")

        if info["parent_non_unity_scales"]:
            lines.append("- **Non-unity ancestor scales**:")
            for anc in info["parent_non_unity_scales"]:
                lines.append(f"  - `{anc['path']}` → scale {anc['scale']}")
        else:
            lines.append("- **Non-unity ancestor scales**: none detected")

        if info["highlights"]:
            lines.append("- **Highlights**:")
            for flag in info["highlights"]:
                lines.append(f"  - {flag}")

        if info["errors"]:
            lines.append("- **Errors during inspection**:")
            for err in info["errors"]:
                lines.append(f"  - `{err}`")

        lines.append("")

    lines += ["---", ""]

    # ── Forced inspection targets ─────────────────────────────────────────────
    lines.append("## Forced Inspection Targets")
    lines.append("")
    lines.append(
        "The following four prims are always reported, regardless of search terms."
    )
    lines.append("")
    forced_path_set = set(FORCED_INSPECTION_PATHS)
    for fp in FORCED_INSPECTION_PATHS:
        matches = [p for p in all_prims_info if p["path"] == fp]
        if matches:
            info = matches[0]
            lines.append(f"### `{fp}`")
            lines.append(f"- **Found**: yes")
            lines.append(f"- **Type**: `{info['type_name']}`")
            lines.append(f"- **BBox available**: `{info.get('bbox_available', False)}`")
            lines.append(f"- **BBox method**: `{info.get('bbox_method', 'unavailable')}`")
            lines.append(f"- **BBox (world)**: {_bbox_str(info)}")
            bbox_flags = info.get("bbox_highlight_flags", [])
            if bbox_flags:
                lines.append(f"- **BBox highlight flags**: {', '.join(bbox_flags)}")
            if info.get("bbox_error"):
                lines.append(f"- **BBox error**: `{info['bbox_error']}`")
        else:
            lines.append(f"### `{fp}`")
            lines.append(f"- **Found**: no — prim does not exist in this stage")
        lines.append("")
    lines += ["---", ""]

    # ── Candidate circular cavity prims ───────────────────────────────────────
    circular_prims = [
        p for p in all_prims_info
        if "circle" in p["path"].lower()
        or "cylinder" in p["path"].lower()
        or p["type_name"] in ("Cylinder", "Sphere", "Disk")
    ]
    lines.append(f"## Candidate Circular Cavity Prims ({len(circular_prims)} found)")
    lines.append("")
    if circular_prims:
        for info in circular_prims:
            lines.append(f"- `{info['path']}` ({info['type_name']})")
            lines.append(f"  - Local scale: {_scale_str(info['local_scale'])}")
            lines.append(f"  - BBox: {_bbox_str(info)}")
            if info["highlights"]:
                for flag in info["highlights"]:
                    lines.append(f"  - **[!]** {flag}")
    else:
        lines.append("_None found matching circular geometry criteria._")
    lines += ["", "---", ""]

    # ── Candidate board prims ─────────────────────────────────────────────────
    board_prims = [p for p in all_prims_info if "board" in p["path"].lower()]
    lines.append(f"## Candidate Board Prims ({len(board_prims)} found)")
    lines.append("")
    if board_prims:
        for info in board_prims:
            lines.append(f"- `{info['path']}` ({info['type_name']})")
            lines.append(f"  - Local scale: {_scale_str(info['local_scale'])}")
            lines.append(f"  - BBox: {_bbox_str(info)}")
            if info["highlights"]:
                for flag in info["highlights"]:
                    lines.append(f"  - **[!]** {flag}")
    else:
        lines.append("_None found._")
    lines += ["", "---", ""]

    # ── Non-unity scales summary ───────────────────────────────────────────────
    non_unity_prims = [
        p for p in all_prims_info
        if (p["local_scale"] is not None and not _scale_is_unity(p["local_scale"]))
        or p["parent_non_unity_scales"]
    ]
    lines.append(f"## Non-Unity Scales Found ({len(non_unity_prims)} prims)")
    lines.append("")
    if non_unity_prims:
        for info in non_unity_prims:
            lines.append(f"- `{info['path']}`")
            if info["local_scale"] is not None and not _scale_is_unity(info["local_scale"]):
                lines.append(f"  - **Local scale**: {info['local_scale']}")
            for anc in info["parent_non_unity_scales"]:
                lines.append(f"  - **Ancestor** `{anc['path']}` scale: {anc['scale']}")
    else:
        lines.append("_No non-unity scales detected in any matching prim or its ancestors._")
    lines += ["", "---", ""]

    # ── Bbox summary table ────────────────────────────────────────────────────
    lines.append("## Measured Bounding Boxes Summary")
    lines.append("")
    lines.append("| Prim Path | Type | BBox X (mm) | BBox Y (mm) | BBox Z (mm) | Method | Flags |")
    lines.append("|---|---|---|---|---|---|---|")
    for info in all_prims_info:
        bbox_mm = info.get("bbox_size_mm")
        if bbox_mm and info.get("bbox_available"):
            bx, by, bz = f"{bbox_mm[0]:.2f}", f"{bbox_mm[1]:.2f}", f"{bbox_mm[2]:.2f}"
        else:
            bx = by = bz = "N/A"
        method = info.get("bbox_method", "unavailable")
        flag_str = "; ".join(info["highlights"]) if info["highlights"] else ""
        # Truncate long flag strings for table readability.
        if len(flag_str) > 80:
            flag_str = flag_str[:77] + "..."
        lines.append(f"| `{info['path']}` | {info['type_name']} | {bx} | {by} | {bz} | {method} | {flag_str} |")
    lines += ["", "---", ""]

    # ── Likely explanation ─────────────────────────────────────────────────────
    lines.append("## Likely Explanation")
    lines.append("")

    # Gather evidence
    circle_non_unity = [
        p for p in all_prims_info
        if ("circle" in p["path"].lower() or "cylinder" in p["path"].lower()
            or p["type_name"] in ("Cylinder", "Sphere", "Disk"))
        and (
            (p["local_scale"] is not None and not _scale_is_unity(p["local_scale"]))
            or p["parent_non_unity_scales"]
        )
    ]

    # Prims with bbox near 60.84 mm
    suspect_bbox_prims = [
        p for p in all_prims_info
        if p.get("bbox_m") and any(_is_near(d, BBOX_SUSPECT_CIRCLE_M) for d in p["bbox_m"])
    ]

    # Prims with bbox near 51 mm
    expected_bbox_prims = [
        p for p in all_prims_info
        if p.get("bbox_m") and any(_is_near(d, BBOX_EXPECTED_CIRCLE_M) for d in p["bbox_m"])
    ]

    evidence_found = False

    if circle_non_unity:
        evidence_found = True
        lines.append(
            "**Non-unity scale detected on circular/cylinder prims or their ancestors.**"
        )
        lines.append("This is the most likely cause of the observed oversize.")
        lines.append("")
        for p in circle_non_unity:
            lines.append(f"- `{p['path']}`: local_scale={p['local_scale']}, "
                         f"ancestors={p['parent_non_unity_scales']}")
        lines.append("")
        lines.append(
            "**Recommended action**: open the USD file or Fusion CAD, "
            "correct the scale to (1, 1, 1) or re-export at the correct size, "
            "then recapture."
        )

    if suspect_bbox_prims:
        evidence_found = True
        lines.append("")
        lines.append(
            f"**Prims with bounding box ≈ {BBOX_SUSPECT_CIRCLE_M*1000:.1f} mm "
            f"(matches measured oversized circle):**"
        )
        for p in suspect_bbox_prims:
            lines.append(f"- `{p['path']}` bbox_mm={p.get('bbox_mm')}")

    if expected_bbox_prims:
        evidence_found = True
        lines.append("")
        lines.append(
            f"**Prims with bounding box ≈ {BBOX_EXPECTED_CIRCLE_M*1000:.1f} mm "
            f"(matches CAD nominal):**"
        )
        for p in expected_bbox_prims:
            lines.append(f"- `{p['path']}` bbox_mm={p.get('bbox_mm')}")

    if not evidence_found:
        lines.append(
            "**Evidence is inconclusive.**  "
            "No non-unity scale was found on any matching prim or its ancestors, "
            "and no bounding box matches the suspect dimension within the ±5 mm window.  "
            "Possible causes not detectable from this script alone:"
        )
        lines.append("")
        lines.append(
            "1. The circular cavity prim path does not contain the terms "
            "'cavity', 'circle', 'cylinder', 'board', or 'hole' — "
            "inspect the stage prim tree manually."
        )
        lines.append(
            "2. The oversize is baked into the mesh geometry itself "
            "(authored at 60.84 mm in Fusion), with scale=(1,1,1)."
        )
        lines.append(
            "3. A units mismatch: the stage authoring unit may differ from "
            "what the capture pipeline assumes (check `metersPerUnit` in the USD)."
        )

    lines += ["", "---", ""]

    # ── Conclusion candidates ──────────────────────────────────────────────────
    lines.append("## Conclusion Candidates")
    lines.append("")

    # Identify circle-related meshes with bbox data.
    circle_mesh_prims = [
        p for p in all_prims_info
        if ("circle" in p["path"].lower() or "cylinder" in p["path"].lower()
            or p["type_name"] in ("Cylinder", "Sphere", "Disk", "Mesh"))
        and p.get("bbox_available")
        and p.get("bbox_size_m") is not None
    ]

    near_60_84 = any(
        _is_near(dim, BBOX_SUSPECT_CIRCLE_M, 0.002)
        for p in circle_mesh_prims
        for dim in p["bbox_size_m"]
    )
    near_51 = any(
        _is_near(dim, BBOX_EXPECTED_CIRCLE_M, 0.002)
        for p in circle_mesh_prims
        for dim in p["bbox_size_m"]
    )

    if near_60_84:
        lines.append(
            "**CONCLUSION**: Circle mesh appears authored oversized at ~60.84 mm — "
            "CAD/Fusion export issue likely."
        )
    elif near_51:
        lines.append(
            "**CONCLUSION**: Circle mesh authored at correct ~51 mm — "
            "inflation source must be elsewhere."
        )
    else:
        lines.append(
            "**CONCLUSION**: Insufficient evidence to conclude — "
            "mesh extents fell outside expected windows."
        )
        if not circle_mesh_prims:
            lines.append(
                "_(No circle-related mesh prims with available bbox data were found. "
                "BBoxCache or mesh-points fallback may have failed for all candidates.)_"
            )

    lines += ["", "---", ""]
    lines.append(
        "> **NOTE**: Baseline 1 (geometric matching) remains BLOCKED until "
        "the circular cavity is corrected in the scene and recaptured."
    )
    lines.append("")

    return "\n".join(lines)


def build_json_report(
    all_prims_info: list,
    stage_identifier: str,
    timestamp: str,
) -> dict:
    """Build the machine-readable JSON report."""

    circular_prims = [
        p["path"] for p in all_prims_info
        if "circle" in p["path"].lower()
        or "cylinder" in p["path"].lower()
        or p["type_name"] in ("Cylinder", "Sphere", "Disk")
    ]
    board_prims = [p["path"] for p in all_prims_info if "board" in p["path"].lower()]
    non_unity_paths = [
        p["path"] for p in all_prims_info
        if (p["local_scale"] is not None and not _scale_is_unity(p["local_scale"]))
        or p["parent_non_unity_scales"]
    ]
    highlighted_paths = [p["path"] for p in all_prims_info if p["highlights"]]

    # ── Forced inspection targets section ─────────────────────────────────────
    forced_targets = []
    for fp in FORCED_INSPECTION_PATHS:
        matches = [p for p in all_prims_info if p["path"] == fp]
        if matches:
            p = matches[0]
            forced_targets.append({
                "path": fp,
                "found": True,
                "type_name": p["type_name"],
                "bbox_available": p.get("bbox_available", False),
                "bbox_method": p.get("bbox_method", "unavailable"),
                "bbox_min_m": p.get("bbox_min_m"),
                "bbox_max_m": p.get("bbox_max_m"),
                "bbox_size_m": p.get("bbox_size_m"),
                "bbox_size_mm": p.get("bbox_size_mm"),
                "bbox_highlight_flags": p.get("bbox_highlight_flags", []),
                "bbox_error": p.get("bbox_error"),
            })
        else:
            forced_targets.append({
                "path": fp,
                "found": False,
                "bbox_available": False,
                "bbox_method": "unavailable",
            })

    # ── Conclusion candidates ──────────────────────────────────────────────────
    circle_mesh_prims = [
        p for p in all_prims_info
        if ("circle" in p["path"].lower() or "cylinder" in p["path"].lower()
            or p["type_name"] in ("Cylinder", "Sphere", "Disk", "Mesh"))
        and p.get("bbox_available")
        and p.get("bbox_size_m") is not None
    ]
    near_60_84 = any(
        _is_near(dim, BBOX_SUSPECT_CIRCLE_M, 0.002)
        for p in circle_mesh_prims
        for dim in p["bbox_size_m"]
    )
    near_51 = any(
        _is_near(dim, BBOX_EXPECTED_CIRCLE_M, 0.002)
        for p in circle_mesh_prims
        for dim in p["bbox_size_m"]
    )
    if near_60_84:
        conclusion = "Circle mesh appears authored oversized at ~60.84 mm — CAD/Fusion export issue likely."
    elif near_51:
        conclusion = "Circle mesh authored at correct ~51 mm — inflation source must be elsewhere."
    else:
        conclusion = "Insufficient evidence to conclude — mesh extents fell outside expected windows."

    # Strip legacy-alias fields before serialising to keep JSON clean.
    def _clean_prim(p: dict) -> dict:
        out = {k: v for k, v in p.items() if k not in ("bbox_m", "bbox_mm")}
        return out

    return {
        "script": SCRIPT_NAME,
        "timestamp": timestamp,
        "stage_identifier": stage_identifier,
        "search_terms": SEARCH_TERMS,
        "context": {
            "measured_opening_mm": 60.84,
            "cad_nominal_diameter_mm": 51.0,
            "inflation_mm": 9.84,
            "inflation_pct": 19.3,
        },
        "summary": {
            "total_candidates": len(all_prims_info),
            "highlighted": len(highlighted_paths),
            "non_unity_scale_prims": len(non_unity_paths),
            "circular_candidates": len(circular_prims),
            "board_candidates": len(board_prims),
        },
        "forced_inspection_targets": forced_targets,
        "conclusion_candidates": conclusion,
        "candidate_circular_prims": circular_prims,
        "candidate_board_prims": board_prims,
        "non_unity_scale_paths": non_unity_paths,
        "highlighted_paths": highlighted_paths,
        "candidates": [_clean_prim(p) for p in all_prims_info],
    }


# ── ASYNC ENTRY POINT ──────────────────────────────────────────────────────────

async def main():
    print(f"\n{'='*70}")
    print(f"  {SCRIPT_NAME}")
    print(f"  Read-only USD scene scale inspector")
    print(f"{'='*70}\n")

    timestamp = datetime.now(timezone.utc).isoformat()

    # ── Get stage ──────────────────────────────────────────────────────────────
    try:
        import omni.usd
        ctx = omni.usd.get_context()
        stage = ctx.get_stage()
        if stage is None:
            print("[ERROR] No USD stage is currently open.  Open the scene first.")
            return
        stage_identifier = stage.GetRootLayer().identifier
        print(f"[INFO] Stage: {stage_identifier}")
    except Exception as exc:
        print(f"[ERROR] Could not obtain USD stage: {exc}")
        traceback.print_exc()
        return

    # ── Prepare output directory ───────────────────────────────────────────────
    try:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"[INFO] Output directory: {OUT_DIR}")
    except Exception as exc:
        print(f"[ERROR] Cannot create output directory {OUT_DIR}: {exc}")
        return

    # ── Read stage units ───────────────────────────────────────────────────────
    try:
        from pxr import UsdGeom
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        if meters_per_unit is None or meters_per_unit <= 0:
            meters_per_unit = 1.0
        print(f"[INFO] Stage metersPerUnit: {meters_per_unit}")
    except Exception as exc:
        meters_per_unit = 1.0
        print(f"[WARN] Could not read metersPerUnit ({exc}), assuming 1.0 (metres).")

    # ── Build bbox cache ───────────────────────────────────────────────────────
    try:
        from pxr import UsdGeom, Usd
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        print("[INFO] BBoxCache initialised.")
    except Exception as exc:
        print(f"[WARN] Could not initialise BBoxCache: {exc}.  Bbox data will use mesh-points fallback only.")
        bbox_cache = None

    # ── Traverse and collect candidates ───────────────────────────────────────
    print("[INFO] Traversing stage for candidate prims ...")
    try:
        candidates = collect_candidates(stage)
    except Exception as exc:
        print(f"[ERROR] Stage traversal failed: {exc}")
        traceback.print_exc()
        return

    print(f"[INFO] Found {len(candidates)} candidate prim(s) matching search terms.")

    # ── Forced inspection targets: always inspect these paths if they exist ───
    forced_paths_seen = set()
    for fp in FORCED_INSPECTION_PATHS:
        try:
            fp_prim = stage.GetPrimAtPath(fp)
            if fp_prim and fp_prim.IsValid():
                forced_paths_seen.add(fp)
                print(f"[INFO] Forced inspection target found: {fp}")
            else:
                # Search case-insensitively for a partial match.
                fp_lower = fp.lower()
                # Extract the meaningful token (e.g. "circle" or "board_tese")
                token = fp.lstrip("/").split("/")[0].lower()
                for prim in stage.Traverse():
                    if token in str(prim.GetPath()).lower():
                        candidates_paths = [str(c.GetPath()) for c in candidates]
                        if str(prim.GetPath()) not in candidates_paths:
                            candidates.append(prim)
                            print(f"[INFO] Forced target '{fp}' not found; including fuzzy match: {prim.GetPath()}")
                        break
                else:
                    print(f"[INFO] Forced inspection target NOT in stage: {fp}")
        except Exception as exc:
            print(f"[WARN] Error checking forced path {fp}: {exc}")

    # Ensure forced paths that exist are in candidates list (deduplicate by path string).
    existing_candidate_paths = {str(c.GetPath()) for c in candidates}
    for fp in FORCED_INSPECTION_PATHS:
        if fp not in existing_candidate_paths:
            try:
                fp_prim = stage.GetPrimAtPath(fp)
                if fp_prim and fp_prim.IsValid():
                    candidates.append(fp_prim)
                    print(f"[INFO] Added forced target to candidate list: {fp}")
            except Exception:
                pass

    # Re-sort after additions.
    candidates.sort(key=lambda p: str(p.GetPath()))

    # ── Inspect each candidate ─────────────────────────────────────────────────
    all_prims_info = []
    for prim in candidates:
        try:
            info = inspect_prim(prim, stage, bbox_cache, meters_per_unit)
            all_prims_info.append(info)

            # Console summary line
            scale_tag = ""
            if info["local_scale"] is not None and not _scale_is_unity(info["local_scale"]):
                scale_tag = "  *** NON-UNITY SCALE ***"
            bbox_tag = ""
            if info.get("bbox_size_mm"):
                b = info["bbox_size_mm"]
                bbox_tag = f"  bbox=({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}) mm [{info['bbox_method']}]"
            flag_count = len(info["highlights"])
            flag_tag = f"  [{flag_count} flag(s)]" if flag_count else ""
            print(
                f"  {'[!]' if info['highlights'] else '   '} "
                f"{info['path']}  ({info['type_name']})"
                f"{scale_tag}{bbox_tag}{flag_tag}"
            )
        except Exception as exc:
            print(f"  [ERROR] Failed to inspect {prim.GetPath()}: {exc}")
            all_prims_info.append({
                "path": str(prim.GetPath()),
                "type_name": prim.GetTypeName(),
                "local_scale": None,
                "local_matrix": None,
                "world_matrix": None,
                "bbox_available": False,
                "bbox_method": "unavailable",
                "bbox_min_m": None,
                "bbox_max_m": None,
                "bbox_size_m": None,
                "bbox_size_mm": None,
                "bbox_highlight_flags": [],
                "bbox_error": str(exc),
                "bbox_m": None,
                "bbox_mm": None,
                "parent_non_unity_scales": [],
                "highlights": [],
                "errors": [str(exc)],
            })

    # ── Build reports ──────────────────────────────────────────────────────────
    print("\n[INFO] Building reports ...")

    md_report = build_markdown_report(all_prims_info, stage_identifier, timestamp)
    json_report = build_json_report(all_prims_info, stage_identifier, timestamp)

    md_path   = OUT_DIR / "scene_scale_inspection.md"
    json_path = OUT_DIR / "scene_scale_inspection.json"

    try:
        md_path.write_text(md_report, encoding="utf-8")
        print(f"[INFO] Markdown report saved: {md_path}")
    except Exception as exc:
        print(f"[ERROR] Could not write Markdown report: {exc}")

    try:
        json_path.write_text(
            json.dumps(json_report, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(f"[INFO] JSON report saved:     {json_path}")
    except Exception as exc:
        print(f"[ERROR] Could not write JSON report: {exc}")

    # ── Final console summary ──────────────────────────────────────────────────
    s = json_report["summary"]
    print(f"\n{'='*70}")
    print(f"  INSPECTION COMPLETE")
    print(f"  Candidates inspected : {s['total_candidates']}")
    print(f"  Highlighted (flags)  : {s['highlighted']}")
    print(f"  Non-unity scales     : {s['non_unity_scale_prims']}")
    print(f"  Circular candidates  : {s['circular_candidates']}")
    print(f"  Board candidates     : {s['board_candidates']}")
    print(f"  Markdown report      : {md_path}")
    print(f"  JSON report          : {json_path}")
    print(f"{'='*70}\n")

    if s["non_unity_scale_prims"] > 0:
        print("[RESULT] Non-unity scale(s) FOUND — see 'Likely Explanation' in the report.")
    elif s["circular_candidates"] == 0:
        print(
            "[RESULT] No circular/cylinder prims detected by path/type. "
            "The circular cavity prim may use a different naming convention. "
            "Inspect the stage prim tree manually."
        )
    else:
        print(
            "[RESULT] No non-unity scales detected.  "
            "The oversize is likely baked into mesh geometry or a units mismatch. "
            "See 'Likely Explanation' in the Markdown report."
        )

    print(
        "\n[BLOCKED] Baseline 1 remains BLOCKED until the circular cavity "
        "is corrected in the scene and recaptured.\n"
    )


asyncio.ensure_future(main())
