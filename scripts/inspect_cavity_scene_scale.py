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
BBOX_HIGHLIGHT_MARGIN_M = 0.005     # ±5 mm window

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


def _get_world_bbox(prim, bbox_cache):
    """
    Return (x_m, y_m, z_m) world-space bounding-box dimensions in metres,
    or None on failure.
    """
    try:
        bbox = bbox_cache.ComputeWorldBound(prim)
        if bbox is None:
            return None
        rng = bbox.GetRange()
        if rng.IsEmpty():
            return None
        mn = rng.GetMin()
        mx = rng.GetMax()
        dims = (
            float(mx[0] - mn[0]),
            float(mx[1] - mn[1]),
            float(mx[2] - mn[2]),
        )
        return dims
    except Exception:
        return None


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
    bbox_m = info.get("bbox_m")
    if bbox_m:
        for axis_idx, axis_label in enumerate(["X", "Y", "Z"]):
            dim = bbox_m[axis_idx]
            if _is_near(dim, BBOX_SUSPECT_CIRCLE_M):
                flags.append(
                    f"bbox_{axis_label} = {dim*1000:.2f} mm "
                    f"≈ {BBOX_SUSPECT_CIRCLE_M*1000:.1f} mm (suspect oversized circle)"
                )
            if _is_near(dim, BBOX_EXPECTED_CIRCLE_M):
                flags.append(
                    f"bbox_{axis_label} = {dim*1000:.2f} mm "
                    f"≈ {BBOX_EXPECTED_CIRCLE_M*1000:.1f} mm (matches CAD nominal)"
                )

    return flags


# ── MAIN INSPECTION ────────────────────────────────────────────────────────────

def inspect_prim(prim, stage, bbox_cache) -> dict:
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
        dims = _get_world_bbox(prim, bbox_cache)
        if dims is not None:
            info["bbox_m"] = list(dims)
            info["bbox_mm"] = [round(d * 1000.0, 3) for d in dims]
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
    bbox_m = info.get("bbox_m")
    bbox_mm = info.get("bbox_mm")
    if bbox_m is None:
        return "unavailable"
    return (
        f"X={bbox_m[0]:.5f} m ({bbox_mm[0]:.2f} mm)  "
        f"Y={bbox_m[1]:.5f} m ({bbox_mm[1]:.2f} mm)  "
        f"Z={bbox_m[2]:.5f} m ({bbox_mm[2]:.2f} mm)"
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
        lines.append(f"- **BBox (world)**: {_bbox_str(info)}")

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
    lines.append("| Prim Path | Type | BBox X (mm) | BBox Y (mm) | BBox Z (mm) | Flags |")
    lines.append("|---|---|---|---|---|---|")
    for info in all_prims_info:
        bbox_mm = info.get("bbox_mm")
        if bbox_mm:
            bx, by, bz = f"{bbox_mm[0]:.2f}", f"{bbox_mm[1]:.2f}", f"{bbox_mm[2]:.2f}"
        else:
            bx = by = bz = "N/A"
        flag_str = "; ".join(info["highlights"]) if info["highlights"] else ""
        # Truncate long flag strings for table readability
        if len(flag_str) > 80:
            flag_str = flag_str[:77] + "..."
        lines.append(f"| `{info['path']}` | {info['type_name']} | {bx} | {by} | {bz} | {flag_str} |")
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
        "candidate_circular_prims": circular_prims,
        "candidate_board_prims": board_prims,
        "non_unity_scale_paths": non_unity_paths,
        "highlighted_paths": highlighted_paths,
        "candidates": all_prims_info,
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

    # ── Build bbox cache ───────────────────────────────────────────────────────
    try:
        from pxr import UsdGeom
        bbox_cache = UsdGeom.BBoxCache(
            UsdGeom.Tokens.default_,
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
        print("[INFO] BBoxCache initialised.")
    except Exception as exc:
        print(f"[WARN] Could not initialise BBoxCache: {exc}.  Bbox data will be unavailable.")
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

    # ── Inspect each candidate ─────────────────────────────────────────────────
    all_prims_info = []
    for prim in candidates:
        try:
            info = inspect_prim(prim, stage, bbox_cache)
            all_prims_info.append(info)

            # Console summary line
            scale_tag = ""
            if info["local_scale"] is not None and not _scale_is_unity(info["local_scale"]):
                scale_tag = "  *** NON-UNITY SCALE ***"
            bbox_tag = ""
            if info.get("bbox_mm"):
                b = info["bbox_mm"]
                bbox_tag = f"  bbox=({b[0]:.1f}, {b[1]:.1f}, {b[2]:.1f}) mm"
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
