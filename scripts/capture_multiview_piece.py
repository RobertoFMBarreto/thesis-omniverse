"""
capture_multiview_piece.py — Phase A: Multi-view Capture Proof-of-Life

Capture multiple RGB-D views of a single geometric piece (rectangle) by
sequentially relocating one camera prim.  This is a PROOF-OF-LIFE only:
it validates the per-view capture loop and output layout before committing
to a static multi-camera USD architecture.

Phase A note:
    Sequential-camera relocation is used here instead of multiple camera
    prims authored in USD.  The final architecture will replace this with
    three static cameras in the scene, so no costly stage edits happen at
    capture time.

Outputs per view (under OUT_ROOT/view_NN_<name>/):
    rgb.png           — raw RGB frame (BGR, OpenCV)
    depth.npy         — raw float32 depth in metres
    depth_vis.png     — colourised depth (viridis, matplotlib)
    metadata.json     — pose, intrinsics, depth window, timestamps,
                        target-resolution fields

Global outputs (under OUT_ROOT/):
    views_contact_sheet.png    — 2 x N grid: RGB (top) / depth (bottom)
    multiview_capture_summary.json
    run_log.txt

Run inside Isaac Sim 5.1 Script Editor.

NOTE: __file__ is unreliable when pasted into the Script Editor — it
resolves to a temporary path.  PROJECT_ROOT is therefore set explicitly
via env-var override.
"""

import asyncio
import json
import math
import os
import shutil
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Project root.  Override with SHAPE_INSERTION_PROJECT_ROOT on machines with
# a different layout (e.g. Mac dev workflow mounting the container repo).
PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        "/workspace/Tese_Roberto/shape_insertion/thesis-omniverse",
    )
)

# All multi-view outputs land here.  The folder for the current piece is
# completely replaced at the start of every run (siblings are not touched).
OUT_ROOT = PROJECT_ROOT / "data" / "multiview_captures" / "pieces" / "rectangle"

# Camera USD prim path.
# Phase A reuses the single existing camera prim and moves it sequentially
# between views.  Phase B will replace this with three static prims in USD.
CAMERA_PRIM_PATH = "/World/Camera"

# Render resolution — matches the validated single-view capture.
IMG_WIDTH  = 640
IMG_HEIGHT = 480

# Replicator subframes per step (higher = more stable rendering, slower).
RT_SUBFRAMES = 8

# Camera intrinsics — must match the Isaac Sim camera prim settings.
FOCAL_MM    = 24.0
APERTURE_MM = 36.0

# Label written into every metadata JSON and the global summary so any
# consumer of the outputs knows this is not the final multi-camera setup.
PHASE_A_NOTE = (
    "sequential-camera proof-of-life; not final static multi-camera architecture"
)

# ── TARGET-RESOLUTION CONFIG ──────────────────────────────────────────────────
#
# TARGET_MODE = "auto_prim_bbox"
#   Walk the stage, find the rectangle prim by name hint, compute its
#   world-space bounding-box centre, and use that as the look-at point.
#   Fails loudly with a RuntimeError if no matching prim is found.
#
# TARGET_MODE = "manual"
#   Use MANUAL_TARGET_LOOK_AT directly.  No stage traversal is performed.

TARGET_MODE = "auto_prim_bbox"   # "auto_prim_bbox" | "manual"

# Case-insensitive substrings searched in every prim's path AND GetName().
TARGET_PRIM_NAME_HINTS = ["rectangle", "Rectangle", "Rect", "rect"]

# Used when TARGET_MODE == "manual", or as emergency documentation of the
# world origin that the previous Phase A run aimed at.
MANUAL_TARGET_LOOK_AT = (0.0, 0.0, 0.0)

# ── CAMERA PLACEMENT (relative to resolved target centre) ─────────────────────
#
# Geometry rationale:
#   top_down:      camera directly above the target centre.
#     elevation from vertical = 0°.
#
#   front_oblique: camera shifted −OBLIQUE_OFFSET = 0.30 m along Y and raised
#     OBLIQUE_HEIGHT = 0.40 m above the target centre.
#     Provides a Y-axis view of the piece.
#
#   side_oblique:  camera shifted +OBLIQUE_OFFSET = 0.30 m along X and raised
#     OBLIQUE_HEIGHT = 0.40 m above the target centre.
#     Provides an X-axis view of the piece.
#
#   Together the three views give one Z-axis (top), one Y-axis (front), and
#   one X-axis (side) viewpoint — better 3D coverage than two opposing
#   X-axis obliques (oblique_left / oblique_right).
#
#   Oblique angle from vertical: atan(OBLIQUE_OFFSET / OBLIQUE_HEIGHT)
#     = atan(0.30 / 0.40) ≈ 36.9°  (~37°) — within the 35–45° target range.
#
#   Top-down uses a larger height (0.50 m) so the full rectangle footprint
#   is captured within the 640×480 FOV.

TOP_DOWN_HEIGHT = 0.50   # m above target centre (z+)
OBLIQUE_HEIGHT  = 0.40   # m above target centre (z+); gives ~37° from vertical
OBLIQUE_OFFSET  = 0.30   # m lateral offset for oblique views (front: −Y, side: +X)
                          # atan(0.30/0.40) ≈ 36.9° from vertical — within 35–45° spec

# ── VIEW CONFIGS ──────────────────────────────────────────────────────────────
#
# Positions and look-ats are placeholders here; they are overwritten in
# main() after resolve_target_look_at() returns the actual target centre.
# The static fields (name, up_axis) remain as authored.
#
# up_axis choice:
#   top_down       → (0, 1, 0)  — Y-up keeps the image right-way-up.
#   front_oblique  → (0, 1, 0)  — consistent Y-up.
#   side_oblique   → (0, 1, 0)  — consistent Y-up.

VIEWS = [
    {
        "name":         "top_down",
        "position_m":   (0.0, 0.0, 0.50),    # placeholder; recomputed in main()
        "look_at_m":    (0.0, 0.0, 0.0),      # placeholder; recomputed in main()
        "up_axis":      (0, 1, 0),
    },
    {
        "name":         "front_oblique",
        "position_m":   (0.0, -0.30, 0.40),   # placeholder; recomputed in main()
        "look_at_m":    (0.0, 0.0, 0.0),       # placeholder; recomputed in main()
        "up_axis":      (0, 1, 0),
    },
    {
        "name":         "side_oblique",
        "position_m":   (0.30, 0.0, 0.40),    # placeholder; recomputed in main()
        "look_at_m":    (0.0, 0.0, 0.0),       # placeholder; recomputed in main()
        "up_axis":      (0, 1, 0),
    },
]

# ── END CONFIG ────────────────────────────────────────────────────────────────


# ── RUN LOG (tee stdout/stderr to a file) ─────────────────────────────────────

class _TeeStream:
    """Write to both the original stream and a file.  Marked with
    `_is_run_logger` so repeated calls to setup_run_logging() do not stack
    wrappers across consecutive runs in the same Script Editor process."""
    _is_run_logger = True

    def __init__(self, original, fileobj):
        self.original = original
        self.fileobj  = fileobj

    def write(self, data):
        try:
            self.original.write(data)
        except Exception:
            pass
        try:
            self.fileobj.write(data)
            self.fileobj.flush()
        except Exception:
            pass

    def flush(self):
        try: self.original.flush()
        except Exception: pass
        try: self.fileobj.flush()
        except Exception: pass

    def isatty(self):
        return getattr(self.original, "isatty", lambda: False)()


_RUN_LOG_STATE = {"file": None}


def teardown_run_logging():
    """Restore original sys.stdout/sys.stderr and close the log file.
    Idempotent: safe to call when no logging is active."""
    if getattr(sys.stdout, "_is_run_logger", False):
        sys.stdout = sys.stdout.original
    if getattr(sys.stderr, "_is_run_logger", False):
        sys.stderr = sys.stderr.original
    f = _RUN_LOG_STATE.get("file")
    if f is not None:
        try: f.flush()
        except Exception: pass
        try: f.close()
        except Exception: pass
        _RUN_LOG_STATE["file"] = None


def setup_run_logging(out_dir: Path) -> Path:
    """Tee stdout and stderr to <out_dir>/run_log.txt.  The file is
    overwritten each run.  Always tears down any previous run-logger first
    to avoid stacking wrappers when the script is re-run in the same
    Script Editor process."""
    teardown_run_logging()
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"
    f = open(str(log_path), "w", buffering=1)   # text-mode, line-buffered

    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    f.write("# capture_multiview_piece.py — run log\n")
    f.write(f"# timestamp_utc: {ts}\n")
    f.write(f"# output_dir:    {out_dir}\n")
    f.write(f"# phase_note:    {PHASE_A_NOTE}\n")
    f.write("# This file is OVERWRITTEN at the beginning of every run.\n")
    f.write("=" * 60 + "\n")
    f.flush()

    _RUN_LOG_STATE["file"] = f
    sys.stdout = _TeeStream(sys.stdout, f)
    sys.stderr = _TeeStream(sys.stderr, f)
    return log_path


# ── INTRINSICS ────────────────────────────────────────────────────────────────

def compute_intrinsics() -> dict:
    """
    Pinhole intrinsics derived from FOCAL_MM, APERTURE_MM and image size.

    Square-pixel correction: tan_half_fov_y = tan_half_fov_x * (H/W), which
    makes fy_px == fx_px for a square-pixel sensor.  This matches the pattern
    used in capture_piece_detection.py (intrinsics_model =
    "pinhole_square_pixels").

    Returns a dict ready for JSON serialisation.
    """
    fov_h          = 2.0 * math.atan((APERTURE_MM / 2.0) / FOCAL_MM)
    tan_half_fov_x = math.tan(fov_h / 2.0)
    tan_half_fov_y = tan_half_fov_x * (IMG_HEIGHT / IMG_WIDTH)  # square pixels
    fov_v          = 2.0 * math.atan(tan_half_fov_y)
    fx_px          = (IMG_WIDTH  / 2.0) / tan_half_fov_x
    fy_px          = (IMG_HEIGHT / 2.0) / tan_half_fov_y
    cx_px          = IMG_WIDTH  / 2.0
    cy_px          = IMG_HEIGHT / 2.0

    return {
        "intrinsics_model": "pinhole_square_pixels",
        "focal_mm":         FOCAL_MM,
        "aperture_mm":      APERTURE_MM,
        "fx_px":            round(fx_px, 4),
        "fy_px":            round(fy_px, 4),
        "cx_px":            round(cx_px, 4),
        "cy_px":            round(cy_px, 4),
        "fov_h_rad":        round(fov_h, 6),
        "fov_v_rad":        round(fov_v, 6),
        "image_width":      IMG_WIDTH,
        "image_height":     IMG_HEIGHT,
    }


# ── TARGET RESOLUTION ─────────────────────────────────────────────────────────

def _mesh_points_world_bbox_subtree(prim):
    """
    Walk the subtree rooted at `prim` (inclusive), collect world-space points
    from every UsdGeom.Mesh, and return the union AABB as (mn, mx) where both
    are Gf.Vec3d.  Returns None if no mesh points are found.

    This is a local copy of the pattern from inspect_cavity_scene_scale.py —
    do NOT import that script.
    """
    try:
        from pxr import UsdGeom, Usd, Gf

        xs, ys, zs = [], [], []
        for p in Usd.PrimRange(prim):
            mesh = UsdGeom.Mesh(p)
            if not mesh:
                continue
            pts_attr = mesh.GetPointsAttr()
            if not pts_attr or not pts_attr.HasAuthoredValue():
                continue
            local_pts = pts_attr.Get()
            if local_pts is None or len(local_pts) == 0:
                continue
            xform = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(
                Usd.TimeCode.Default()
            )
            for pt in local_pts:
                wp = xform.Transform(Gf.Vec3d(float(pt[0]), float(pt[1]), float(pt[2])))
                xs.append(wp[0])
                ys.append(wp[1])
                zs.append(wp[2])

        if not xs:
            return None
        mn = Gf.Vec3d(min(xs), min(ys), min(zs))
        mx = Gf.Vec3d(max(xs), max(ys), max(zs))
        return (mn, mx)
    except Exception:
        return None


def resolve_target_look_at(stage) -> dict:
    """
    Read-only USD stage traversal to find the rectangle prim and return its
    world-space bounding-box centre as the camera look-at point.

    Strategy
    --------
    1. Traverse all prims; collect candidates whose path or GetName() contains
       any string in TARGET_PRIM_NAME_HINTS (case-insensitive substring).
    2. Filter out prims whose type name contains any of:
       'Material', 'Shader', 'Light', 'Camera', 'Scope'  (case-insensitive).
    3. For each surviving candidate, attempt bbox computation:
       a) UsdGeom.BBoxCache.ComputeWorldBound — primary strategy.
       b) mesh-points subtree walk — fallback if (a) returns an empty range.
    4. From candidates with a valid bbox, select ONE:
       - prefer exact name-level match over partial path match;
       - then prefer the largest bbox volume;
       - then prefer shorter prim-path depth (closer to root).
    5. Return a result dict.  Raise RuntimeError if no valid candidate exists.

    Returns
    -------
    dict with keys:
        selected_prim_path        str
        prim_type_name            str
        bbox_min_m                [x, y, z]
        bbox_max_m                [x, y, z]
        bbox_center_m             [x, y, z]
        bbox_size_m               [dx, dy, dz]
        bbox_size_mm              [dx, dy, dz]
        bbox_method               "BBoxCache" | "mesh_points_fallback"
        n_candidates_examined     int
    """
    from pxr import UsdGeom, Usd

    hints_lower = [h.lower() for h in TARGET_PRIM_NAME_HINTS]

    # Types to exclude (case-insensitive substring in GetTypeName()).
    EXCLUDED_TYPE_SUBSTRINGS = ["material", "shader", "light", "camera", "scope"]

    # ── Stage units ───────────────────────────────────────────────────────────
    try:
        meters_per_unit = UsdGeom.GetStageMetersPerUnit(stage)
        if meters_per_unit is None or meters_per_unit <= 0.0:
            meters_per_unit = 1.0
    except Exception:
        meters_per_unit = 1.0
    print(f"[target_resolve] stage metersPerUnit = {meters_per_unit}")

    # ── Build BBoxCache ───────────────────────────────────────────────────────
    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
            useExtentsHint=True,
        )
    except Exception as exc:
        print(f"[target_resolve] WARNING: BBoxCache init failed ({exc}); "
              "will rely on mesh-points fallback only.")
        bbox_cache = None

    # ── Collect raw candidates ────────────────────────────────────────────────
    raw_candidates = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        prim_path_lower = str(prim.GetPath()).lower()
        prim_name_lower = prim.GetName().lower()
        if any(h in prim_path_lower or h in prim_name_lower for h in hints_lower):
            raw_candidates.append(prim)

    print(f"[target_resolve] raw candidates matching hints: {len(raw_candidates)}")

    # ── Filter out excluded types ─────────────────────────────────────────────
    filtered = []
    for prim in raw_candidates:
        type_lower = prim.GetTypeName().lower()
        if any(ex in type_lower for ex in EXCLUDED_TYPE_SUBSTRINGS):
            print(f"[target_resolve]   excluded by type '{prim.GetTypeName()}': "
                  f"{prim.GetPath()}")
            continue
        filtered.append(prim)

    print(f"[target_resolve] candidates after type filter: {len(filtered)}")

    if not filtered:
        raise RuntimeError(
            f"[target_resolve] Could not auto-resolve a target prim from hints: "
            f"{TARGET_PRIM_NAME_HINTS}.\n"
            f"Tried {len(raw_candidates)} candidates, all excluded by type filter. "
            f"Either:\n"
            f"  - adjust TARGET_PRIM_NAME_HINTS at the top of the script, or\n"
            f"  - set TARGET_MODE = \"manual\" and update MANUAL_TARGET_LOOK_AT."
        )

    # ── Compute bbox for each surviving candidate ─────────────────────────────
    valid_candidates = []   # list of (prim, bbox_min_m, bbox_max_m, method)
    n_examined = 0

    for prim in filtered:
        n_examined += 1
        path_str = str(prim.GetPath())
        bbox_min_m = bbox_max_m = None
        method = None

        # Strategy 1: BBoxCache
        if bbox_cache is not None:
            try:
                world_bbox = bbox_cache.ComputeWorldBound(prim)
                rng = world_bbox.ComputeAlignedRange()
                if not rng.IsEmpty():
                    mn = rng.GetMin()
                    mx = rng.GetMax()
                    bbox_min_m = [float(mn[i] * meters_per_unit) for i in range(3)]
                    bbox_max_m = [float(mx[i] * meters_per_unit) for i in range(3)]
                    method = "BBoxCache"
            except Exception as exc:
                print(f"[target_resolve]   BBoxCache failed for {path_str}: {exc}")

        # Strategy 2: mesh-points subtree fallback
        if method is None:
            r = _mesh_points_world_bbox_subtree(prim)
            if r is not None:
                mn, mx = r
                bbox_min_m = [float(mn[i] * meters_per_unit) for i in range(3)]
                bbox_max_m = [float(mx[i] * meters_per_unit) for i in range(3)]
                method = "mesh_points_fallback"

        if bbox_min_m is None:
            print(f"[target_resolve]   both bbox strategies failed for {path_str} "
                  f"— skipping")
            continue

        valid_candidates.append((prim, bbox_min_m, bbox_max_m, method))
        sz = [bbox_max_m[i] - bbox_min_m[i] for i in range(3)]
        print(f"[target_resolve]   valid: {path_str}  type={prim.GetTypeName()}  "
              f"method={method}  "
              f"size=({sz[0]*1000:.1f}, {sz[1]*1000:.1f}, {sz[2]*1000:.1f}) mm")

    if not valid_candidates:
        raise RuntimeError(
            f"[target_resolve] Could not auto-resolve a target prim from hints: "
            f"{TARGET_PRIM_NAME_HINTS}.\n"
            f"Tried {n_examined} candidates. All failed bbox computation. Either:\n"
            f"  - adjust TARGET_PRIM_NAME_HINTS at the top of the script, or\n"
            f"  - set TARGET_MODE = \"manual\" and update MANUAL_TARGET_LOOK_AT."
        )

    # ── Select the best candidate ─────────────────────────────────────────────
    # Scoring (higher = better, applied lexicographically):
    #   1. exact_name_match: 1 if any hint == prim.GetName().lower(), else 0
    #   2. bbox_volume (larger is better, we negate for min-sort)
    #   3. path depth (fewer components = closer to root = better; negate)

    def _candidate_sort_key(entry):
        prim, bmn, bmx, _ = entry
        name_lower = prim.GetName().lower()
        exact = 1 if any(h == name_lower for h in hints_lower) else 0
        vol = ((bmx[0] - bmn[0]) * (bmx[1] - bmn[1]) * (bmx[2] - bmn[2]))
        depth = len(str(prim.GetPath()).strip("/").split("/"))
        # We want (exact DESC, volume DESC, depth ASC) so we negate exact and vol.
        return (-exact, -vol, depth)

    valid_candidates.sort(key=_candidate_sort_key)
    selected_prim, bbox_min_m, bbox_max_m, method = valid_candidates[0]

    # ── Build result dict ─────────────────────────────────────────────────────
    bbox_size_m  = [bbox_max_m[i] - bbox_min_m[i] for i in range(3)]
    bbox_size_mm = [round(v * 1000.0, 3) for v in bbox_size_m]
    bbox_center_m = [
        round((bbox_min_m[i] + bbox_max_m[i]) / 2.0, 6) for i in range(3)
    ]

    return {
        "selected_prim_path":    str(selected_prim.GetPath()),
        "prim_type_name":        selected_prim.GetTypeName(),
        "bbox_min_m":            [round(v, 6) for v in bbox_min_m],
        "bbox_max_m":            [round(v, 6) for v in bbox_max_m],
        "bbox_center_m":         bbox_center_m,
        "bbox_size_m":           [round(v, 6) for v in bbox_size_m],
        "bbox_size_mm":          bbox_size_mm,
        "bbox_method":           method,
        "n_candidates_examined": n_examined,
    }


# ── CAMERA POSE HELPERS ───────────────────────────────────────────────────────

def _look_at_quaternion(position: tuple, look_at: tuple, up_axis: tuple):
    """
    Build an orientation quaternion so the camera at `position` points toward
    `look_at` with the given `up_axis`.

    Isaac Sim cameras look along their local -Z axis (USD / OpenGL convention).
    We build the rotation matrix from the forward/right/up triad, then convert
    to a quaternion suitable for xformOp:orient (Gf.Quatd, w-first).

    Returns a Gf.Quatd.
    """
    import numpy as np
    from pxr import Gf

    pos = np.array(position, dtype=float)
    tgt = np.array(look_at,  dtype=float)
    up  = np.array(up_axis,  dtype=float)

    forward = tgt - pos
    norm_f  = np.linalg.norm(forward)
    if norm_f < 1e-9:
        raise ValueError(
            f"[set_camera_pose] position and look_at are too close: "
            f"position={position}, look_at={look_at}"
        )
    forward = forward / norm_f

    norm_u = np.linalg.norm(up)
    if norm_u < 1e-9:
        raise ValueError(f"[set_camera_pose] up_axis has zero length: {up_axis}")
    up = up / norm_u

    # right = forward x up  (then re-orthogonalise up)
    right = np.cross(forward, up)
    norm_r = np.linalg.norm(right)
    if norm_r < 1e-9:
        # forward and up are parallel — pick an arbitrary perpendicular
        alt = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(forward, alt)) > 0.9:
            alt = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, alt)
        right /= np.linalg.norm(right)
    else:
        right = right / norm_r

    up_ortho = np.cross(right, forward)  # re-orthogonalised up

    # Rotation matrix: columns are right, up_ortho, -forward
    # (camera looks along local -Z in OpenGL convention)
    R = np.eye(3)
    R[:, 0] = right
    R[:, 1] = up_ortho
    R[:, 2] = -forward   # local -Z points in world-forward direction

    # Rotation matrix → quaternion (w, x, y, z)
    trace = R[0, 0] + R[1, 1] + R[2, 2]
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    return Gf.Quatd(w, x, y, z)


def set_camera_pose(prim, position: tuple, look_at: tuple, up_axis: tuple) -> None:
    """
    Apply a world-space pose to a camera prim by:
      1. Setting xformOp:translate to `position`.
      2. Setting xformOp:orient to the quaternion that makes the camera face
         `look_at` with the given `up_axis`.

    Falls back to xformOp:rotateXYZ / xformOp:rotateZ if orient is absent,
    mirroring the pattern in capture_piece_detection.py.  For oblique views
    the quaternion path is required — the Z-only-rotation fallback will log a
    warning and produce incorrect orientation.
    """
    from pxr import UsdGeom, Gf

    xformable = UsdGeom.Xformable(prim)
    ops_dict  = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}

    if "xformOp:translate" not in ops_dict:
        raise RuntimeError(
            f"[set_camera_pose] Camera prim has no xformOp:translate: "
            f"{prim.GetPath()}"
        )

    ops_dict["xformOp:translate"].Set(Gf.Vec3d(*position))

    quat = _look_at_quaternion(position, look_at, up_axis)

    if "xformOp:orient" in ops_dict:
        ops_dict["xformOp:orient"].Set(quat)
    elif "xformOp:rotateXYZ" in ops_dict:
        print(
            "[set_camera_pose] WARNING: xformOp:orient not found — falling back "
            "to xformOp:rotateXYZ, which cannot represent arbitrary look-at "
            "orientations correctly.  Oblique views will be wrong."
        )
        ops_dict["xformOp:rotateXYZ"].Set(Gf.Vec3f(0.0, 0.0, 0.0))
    elif "xformOp:rotateZ" in ops_dict:
        print(
            "[set_camera_pose] WARNING: xformOp:orient not found — falling back "
            "to xformOp:rotateZ, which only supports Z-axis rotation.  "
            "Oblique views will be wrong."
        )
        ops_dict["xformOp:rotateZ"].Set(0.0)
    else:
        print(
            "[set_camera_pose] WARNING: no rotation op found on camera prim — "
            "orientation not changed."
        )


def get_camera_world_pose(prim) -> dict:
    """
    Read back the world-space translate and orientation of a camera prim.

    Returns a dict with:
        position   — [x, y, z] in metres
        quaternion — [w, x, y, z] (Gf.Quatd components)

    This is the same read-back pattern used in capture_piece_detection.py and
    capture_cavity_detection.py: ComputeLocalToWorldTransform at time 0.
    """
    from pxr import UsdGeom

    xformable = UsdGeom.Xformable(prim)
    world_xf  = xformable.ComputeLocalToWorldTransform(0)

    t    = world_xf.ExtractTranslation()
    quat = world_xf.ExtractRotationQuat()
    img  = quat.GetImaginary()

    return {
        "position":   [round(float(t[0]), 6),
                        round(float(t[1]), 6),
                        round(float(t[2]), 6)],
        "quaternion": [round(float(quat.GetReal()), 6),
                        round(float(img[0]),         6),
                        round(float(img[1]),         6),
                        round(float(img[2]),         6)],
    }


# ── REPLICATOR SETUP ──────────────────────────────────────────────────────────

def create_render_product_and_annotators():
    """
    Create the Replicator render product and attach rgb + distance_to_image_plane
    annotators once.  The same render product is reused across all views.

    Returns (render_product, rgb_annotator, depth_annotator).
    """
    import omni.replicator.core as rep

    print(f"[multiview] creating render product {IMG_WIDTH}x{IMG_HEIGHT} "
          f"on {CAMERA_PRIM_PATH}")
    rp = rep.create.render_product(CAMERA_PRIM_PATH, (IMG_WIDTH, IMG_HEIGHT))

    rgb_an   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_an = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_an.attach([rp])
    depth_an.attach([rp])

    print("[multiview] annotators attached: rgb, distance_to_image_plane")
    return rp, rgb_an, depth_an


# ── PER-VIEW CAPTURE ──────────────────────────────────────────────────────────

async def capture_view(view_cfg: dict, rgb_an, depth_an,
                       run_id: str, timestamp_utc: str,
                       intrinsics: dict) -> dict:
    """
    Apply the requested camera pose, step the renderer once, and return a
    result dict with all per-view data.

    The returned dict has:
        ok             — bool
        view_name      — str
        rgb            — H x W x 3 uint8 ndarray (or None on failure)
        depth          — H x W float32 ndarray, metres (or None on failure)
        requested_pose — dict
        measured_pose  — dict
        depth_valid_min_m, depth_valid_max_m, n_valid_depth_pixels
        error_message  — str or None
    """
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep

    view_name = view_cfg["name"]
    position  = view_cfg["position_m"]
    look_at   = view_cfg["look_at_m"]
    up_axis   = view_cfg["up_axis"]

    result = {
        "ok":              False,
        "view_name":       view_name,
        "rgb":             None,
        "depth":           None,
        "requested_pose":  {
            "position_m": list(position),
            "look_at_m":  list(look_at),
            "up_axis":    list(up_axis),
        },
        "measured_pose":   None,
        "depth_valid_min_m":    None,
        "depth_valid_max_m":    None,
        "n_valid_depth_pixels": 0,
        "error_message":   None,
    }

    print(f"\n[view_{view_name}] --- starting capture ---")
    print(f"[view_{view_name}] requested pos={position}  look_at={look_at}  "
          f"up={up_axis}")

    # ── Resolve camera prim ───────────────────────────────────────────────────
    stage    = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
    if not cam_prim.IsValid():
        raise RuntimeError(
            f"[view_{view_name}] Camera prim not found at {CAMERA_PRIM_PATH}"
        )

    # ── Apply camera pose ─────────────────────────────────────────────────────
    set_camera_pose(cam_prim, position, look_at, up_axis)

    # ── Render ────────────────────────────────────────────────────────────────
    print(f"[view_{view_name}] stepping renderer ({RT_SUBFRAMES} rt_subframes) ...")
    await rep.orchestrator.step_async(rt_subframes=RT_SUBFRAMES)

    # ── Read back actual pose ─────────────────────────────────────────────────
    measured_pose = get_camera_world_pose(cam_prim)
    result["measured_pose"] = measured_pose
    print(f"[view_{view_name}] measured pos={measured_pose['position']}  "
          f"quat={measured_pose['quaternion']}")

    # ── Read RGB ──────────────────────────────────────────────────────────────
    raw_rgb = rgb_an.get_data()
    if raw_rgb is None:
        raise RuntimeError(f"[view_{view_name}] rgb annotator returned None")

    print(f"[view_{view_name}] rgb type={type(raw_rgb)}")
    if isinstance(raw_rgb, dict):
        raw_rgb = raw_rgb["data"]
    raw_rgb = __import__("numpy").asarray(raw_rgb)
    raw_rgb = raw_rgb.reshape(IMG_HEIGHT, IMG_WIDTH, -1)
    rgb = raw_rgb[:, :, :3].astype(np.uint8)
    print(f"[view_{view_name}] rgb shape={rgb.shape}")

    # ── Read depth ────────────────────────────────────────────────────────────
    raw_depth = depth_an.get_data()
    if raw_depth is None:
        raise RuntimeError(f"[view_{view_name}] depth annotator returned None")

    print(f"[view_{view_name}] depth type={type(raw_depth)}")
    if isinstance(raw_depth, dict):
        raw_depth = raw_depth["data"]
    raw_depth = np.asarray(raw_depth, dtype=np.float32).reshape(IMG_HEIGHT, IMG_WIDTH)
    depth = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[view_{view_name}] depth shape={depth.shape}")

    valid_mask = (depth > 0.0) & np.isfinite(depth)
    n_valid    = int(valid_mask.sum())
    if n_valid > 0:
        d_min = float(depth[valid_mask].min())
        d_max = float(depth[valid_mask].max())
        print(f"[view_{view_name}] depth valid range "
              f"[{d_min:.4f}, {d_max:.4f}] m  ({n_valid} px)")
    else:
        d_min, d_max = 0.0, 0.0
        print(f"[view_{view_name}] WARNING: no valid depth pixels")

    result.update({
        "ok":                   True,
        "rgb":                  rgb,
        "depth":                depth,
        "depth_valid_min_m":    round(d_min, 4),
        "depth_valid_max_m":    round(d_max, 4),
        "n_valid_depth_pixels": n_valid,
    })
    return result


# ── SAVE PER-VIEW OUTPUTS ─────────────────────────────────────────────────────

def save_view_outputs(view_result: dict, view_idx: int,
                      intrinsics: dict, run_id: str, timestamp_utc: str,
                      target_info: dict) -> Path:
    """
    Save rgb.png, depth.npy, depth_vis.png, and metadata.json for one view.

    `target_info` is the dict returned by resolve_target_look_at() (or the
    synthetic manual dict built in main()).  Its fields are written into
    metadata.json under the target-resolution block.

    Returns the view directory path.
    """
    import numpy as np
    import cv2

    view_name = view_result["view_name"]
    view_dir  = OUT_ROOT / f"view_{view_idx:02d}_{view_name}"
    view_dir.mkdir(parents=True, exist_ok=True)

    if not view_result["ok"]:
        print(f"[view_{view_name}] view failed — skipping image save, "
              f"directory created at {view_dir}")
        return view_dir

    rgb   = view_result["rgb"]
    depth = view_result["depth"]

    # ── RGB ───────────────────────────────────────────────────────────────────
    rgb_path = view_dir / "rgb.png"
    try:
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(rgb_path), bgr)
        print(f"[view_{view_name}] saved rgb.png  ({rgb.shape[1]}x{rgb.shape[0]})")
    except Exception as exc:
        print(f"[view_{view_name}] WARNING: could not save rgb.png — {exc}")

    # ── Depth NPY ─────────────────────────────────────────────────────────────
    depth_npy_path = view_dir / "depth.npy"
    try:
        import numpy as np
        np.save(str(depth_npy_path), depth)
        print(f"[view_{view_name}] saved depth.npy  (float32, metres)")
    except Exception as exc:
        print(f"[view_{view_name}] WARNING: could not save depth.npy — {exc}")

    # ── Depth visualisation ───────────────────────────────────────────────────
    depth_vis_path = view_dir / "depth_vis.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
        valid_mask = (depth > 0.0) & __import__("numpy").isfinite(depth)
        if valid_mask.any():
            vmin = float(depth[valid_mask].min())
            vmax = float(depth[valid_mask].max())
        else:
            vmin, vmax = 0.0, 1.0
        im = ax.imshow(depth, cmap="viridis", vmin=vmin, vmax=vmax)
        ax.set_title(f"{view_name}  depth (m)", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="depth (m)")
        fig.savefig(str(depth_vis_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[view_{view_name}] saved depth_vis.png")
    except Exception as exc:
        print(f"[view_{view_name}] WARNING: could not save depth_vis.png — {exc}")

    # ── Per-view offset from target ───────────────────────────────────────────
    req_pos  = view_result["requested_pose"]["position_m"]   # list[3]
    tgt_ctr  = target_info.get("bbox_center_m", [0.0, 0.0, 0.0])
    cam_offset = [round(req_pos[i] - tgt_ctr[i], 6) for i in range(3)]

    # ── Metadata JSON ─────────────────────────────────────────────────────────
    meta = {
        "view_name":             view_name,
        "camera_prim_path":      CAMERA_PRIM_PATH,
        "requested_pose":        view_result["requested_pose"],
        "measured_pose_read_back_from_stage": view_result["measured_pose"],
        "image_width":           IMG_WIDTH,
        "image_height":          IMG_HEIGHT,
        "focal_mm":              intrinsics["focal_mm"],
        "aperture_mm":           intrinsics["aperture_mm"],
        "fx_px":                 intrinsics["fx_px"],
        "fy_px":                 intrinsics["fy_px"],
        "cx_px":                 intrinsics["cx_px"],
        "cy_px":                 intrinsics["cy_px"],
        "intrinsics_model":      intrinsics["intrinsics_model"],
        "depth_valid_min_m":     view_result["depth_valid_min_m"],
        "depth_valid_max_m":     view_result["depth_valid_max_m"],
        "n_valid_depth_pixels":  view_result["n_valid_depth_pixels"],
        "timestamp_utc":         timestamp_utc,
        "run_id":                run_id,
        "phase_note":            PHASE_A_NOTE,
        # ── Target-resolution fields ──────────────────────────────────────────
        "target_mode":                   TARGET_MODE,
        "selected_target_prim_path":     target_info.get("selected_prim_path"),
        "target_bbox_center_world_m":    target_info.get("bbox_center_m"),
        "target_bbox_size_mm":           target_info.get("bbox_size_mm"),
        "target_bbox_method":            target_info.get("bbox_method"),
        "requested_look_at":             view_result["requested_pose"]["look_at_m"],
        "camera_offset_from_target":     cam_offset,
    }
    meta_path = view_dir / "metadata.json"
    try:
        with open(str(meta_path), "w") as fp:
            json.dump(meta, fp, indent=2)
        print(f"[view_{view_name}] saved metadata.json")
    except Exception as exc:
        print(f"[view_{view_name}] WARNING: could not save metadata.json — {exc}")

    return view_dir


# ── CONTACT SHEET ─────────────────────────────────────────────────────────────

def build_contact_sheet(view_results: list, view_dirs: list) -> None:
    """
    Build a 2 x N grid image:
      top row    — RGB per view
      bottom row — depth_vis per view (loaded from disk)
    Column titles are the view names.

    Saved to OUT_ROOT/views_contact_sheet.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg
        import numpy as np

        n_views = len(view_results)
        fig, axes = plt.subplots(
            2, n_views,
            figsize=(5 * n_views, 8),
            dpi=100,
        )
        if n_views == 1:
            # matplotlib returns 1-D array when n_views==1
            axes = axes.reshape(2, 1)

        for col, (vr, vd) in enumerate(zip(view_results, view_dirs)):
            view_name = vr["view_name"]

            # Top row: RGB
            ax_rgb = axes[0, col]
            if vr["ok"] and vr["rgb"] is not None:
                ax_rgb.imshow(vr["rgb"])
            else:
                ax_rgb.text(0.5, 0.5, "FAILED", ha="center", va="center",
                            transform=ax_rgb.transAxes, color="red", fontsize=14)
            ax_rgb.set_title(view_name, fontsize=10, fontweight="bold")
            ax_rgb.axis("off")

            # Bottom row: depth_vis (load from disk to avoid matplotlib colour
            # state issues from the per-view save)
            ax_dep = axes[1, col]
            depth_vis_path = vd / "depth_vis.png"
            if depth_vis_path.exists():
                try:
                    img = mpimg.imread(str(depth_vis_path))
                    ax_dep.imshow(img)
                except Exception as exc:
                    ax_dep.text(0.5, 0.5, f"load err\n{exc}",
                                ha="center", va="center",
                                transform=ax_dep.transAxes, fontsize=8, color="red")
            else:
                ax_dep.text(0.5, 0.5, "no depth_vis",
                            ha="center", va="center",
                            transform=ax_dep.transAxes, fontsize=8)
            ax_dep.axis("off")

        axes[0, 0].set_ylabel("RGB", fontsize=11)
        axes[1, 0].set_ylabel("Depth", fontsize=11)

        fig.suptitle(
            f"Multi-view capture — rectangle  ({n_views} views)\n"
            f"Phase A: {PHASE_A_NOTE}",
            fontsize=10,
        )
        fig.tight_layout()

        sheet_path = OUT_ROOT / "views_contact_sheet.png"
        fig.savefig(str(sheet_path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[contact_sheet] saved {sheet_path}")

    except Exception as exc:
        print(f"[contact_sheet] WARNING: could not build contact sheet — {exc}")
        traceback.print_exc()


# ── GLOBAL SUMMARY ────────────────────────────────────────────────────────────

def build_global_summary(view_results: list, view_dirs: list,
                          run_id: str, timestamp_utc: str,
                          intrinsics: dict,
                          target_info: dict) -> None:
    """
    Write multiview_capture_summary.json with per-view records, top-level
    success/failure status, and a 'target_resolution' block.

    `target_info` is the dict returned by resolve_target_look_at() (or the
    synthetic manual dict built in main()).
    """
    n_requested = len(view_results)
    n_succeeded = sum(1 for vr in view_results if vr["ok"])
    overall_ok  = (n_succeeded == n_requested)

    view_records = []
    for idx, (vr, vd) in enumerate(zip(view_results, view_dirs)):
        req_pos  = vr["requested_pose"]["position_m"]
        tgt_ctr  = target_info.get("bbox_center_m", [0.0, 0.0, 0.0])
        cam_offset = [round(req_pos[i] - tgt_ctr[i], 6) for i in range(3)]

        rec = {
            "view_index":       idx,
            "name":             vr["view_name"],
            "prim_path":        CAMERA_PRIM_PATH,
            "requested_pose":   vr["requested_pose"],
            "measured_pose":    vr["measured_pose"],
            "image_size":       [IMG_WIDTH, IMG_HEIGHT],
            "intrinsics_summary": {
                "fx_px":    intrinsics["fx_px"],
                "fy_px":    intrinsics["fy_px"],
                "cx_px":    intrinsics["cx_px"],
                "cy_px":    intrinsics["cy_px"],
                "model":    intrinsics["intrinsics_model"],
            },
            "depth_valid_min_m":    vr["depth_valid_min_m"],
            "depth_valid_max_m":    vr["depth_valid_max_m"],
            "n_valid_depth_pixels": vr["n_valid_depth_pixels"],
            "save_path":             str(vd),
            "ok":                    vr["ok"],
            "error_message":         vr.get("error_message"),
            "camera_offset_from_target": cam_offset,
        }
        view_records.append(rec)

    # Target-resolution block — single source of truth for the run.
    target_resolution_block = {
        "target_mode":               TARGET_MODE,
        "selected_target_prim_path": target_info.get("selected_prim_path"),
        "target_prim_type_name":     target_info.get("prim_type_name"),
        "bbox_min_m":                target_info.get("bbox_min_m"),
        "bbox_max_m":                target_info.get("bbox_max_m"),
        "bbox_center_m":             target_info.get("bbox_center_m"),
        "bbox_size_m":               target_info.get("bbox_size_m"),
        "bbox_size_mm":              target_info.get("bbox_size_mm"),
        "bbox_method":               target_info.get("bbox_method"),
        "n_candidates_examined":     target_info.get("n_candidates_examined"),
    }

    summary = {
        "script_name":       "capture_multiview_piece.py",
        "phase":             "A",
        "phase_note":        PHASE_A_NOTE,
        "run_id":            run_id,
        "timestamp_utc":     timestamp_utc,
        "piece":             "rectangle",
        "n_views_requested": n_requested,
        "n_views_succeeded": n_succeeded,
        "success":           overall_ok,
        "target_resolution": target_resolution_block,
        "views":             view_records,
        "inputs_dir":        CAMERA_PRIM_PATH,
        "output_dir":        str(OUT_ROOT),
    }

    summary_path = OUT_ROOT / "multiview_capture_summary.json"
    try:
        with open(str(summary_path), "w") as fp:
            json.dump(summary, fp, indent=2)
        print(f"[summary] saved {summary_path}")
    except Exception as exc:
        print(f"[summary] WARNING: could not write summary JSON — {exc}")

    print("\n" + "=" * 60)
    print(f"  success={overall_ok}  "
          f"views_succeeded={n_succeeded}/{n_requested}")
    if not overall_ok:
        for vr in view_results:
            if not vr["ok"]:
                print(f"  [view_FAIL] {vr['view_name']}: "
                      f"{vr.get('error_message', 'unknown error')}")
    print("=" * 60)


# ── MAIN ──────────────────────────────────────────────────────────────────────

async def main():
    import numpy as np
    import omni.usd

    # ── Resolve output directory ───────────────────────────────────────────────
    # Remove ONLY this script's own target directory, then recreate it.
    # Sibling directories (e.g. other pieces) are never touched.
    try:
        if OUT_ROOT.exists():
            shutil.rmtree(str(OUT_ROOT))
            print(f"[multiview] cleared existing output dir: {OUT_ROOT}")
    except Exception as exc:
        print(f"[multiview] WARNING: could not remove {OUT_ROOT}: {exc}")

    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # ── Tee logging ───────────────────────────────────────────────────────────
    log_path = setup_run_logging(OUT_ROOT)

    # ── Banner ────────────────────────────────────────────────────────────────
    run_id        = uuid.uuid4().hex[:8]
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    print("=" * 60)
    print("capture_multiview_piece.py — Phase A: Multi-view Proof-of-Life")
    print("=" * 60)
    print(f"[multiview] phase_note     = {PHASE_A_NOTE}")
    print(f"[multiview] camera_prim    = {CAMERA_PRIM_PATH}")
    print(f"[multiview] output_dir     = {OUT_ROOT}")
    print(f"[multiview] n_views        = {len(VIEWS)}")
    print(f"[multiview] run_id         = {run_id}")
    print(f"[multiview] timestamp_utc  = {timestamp_utc}")
    print(f"[multiview] run_log        = {log_path}")
    print(f"[multiview] target_mode    = {TARGET_MODE}")
    print(f"[multiview] name_hints     = {TARGET_PRIM_NAME_HINTS}")

    # ── Intrinsics (same for all views — one camera, one sensor) ─────────────
    intrinsics = compute_intrinsics()
    print(f"[multiview] intrinsics: fx={intrinsics['fx_px']:.2f}  "
          f"fy={intrinsics['fy_px']:.2f}  "
          f"cx={intrinsics['cx_px']:.2f}  cy={intrinsics['cy_px']:.2f}  "
          f"model={intrinsics['intrinsics_model']}")

    # ── Resolve target look-at ────────────────────────────────────────────────
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[multiview] FATAL: no USD stage is open. Open the scene first.")
        teardown_run_logging()
        return

    if TARGET_MODE == "auto_prim_bbox":
        print(f"\n[target_resolve] mode=auto_prim_bbox  "
              f"hints={TARGET_PRIM_NAME_HINTS}")
        # resolve_target_look_at raises RuntimeError on failure — let it propagate.
        target_info = resolve_target_look_at(stage)

        print(f"[target_resolve] selected prim : {target_info['selected_prim_path']}")
        print(f"[target_resolve] prim type     : {target_info['prim_type_name']}")
        print(f"[target_resolve] bbox method   : {target_info['bbox_method']}")
        print(f"[target_resolve] bbox min (m)  : {target_info['bbox_min_m']}")
        print(f"[target_resolve] bbox max (m)  : {target_info['bbox_max_m']}")
        sz_mm = target_info['bbox_size_mm']
        print(f"[target_resolve] bbox size (mm): "
              f"({sz_mm[0]:.1f}, {sz_mm[1]:.1f}, {sz_mm[2]:.1f})")
        ctr = target_info['bbox_center_m']
        print(f"[target_resolve] bbox centre (m): "
              f"({ctr[0]:.4f}, {ctr[1]:.4f}, {ctr[2]:.4f})")
        print(f"[target_resolve] candidates examined: "
              f"{target_info['n_candidates_examined']}")

    else:
        # Manual mode — build a synthetic info dict; no stage traversal.
        print(f"[target_resolve] mode=manual  look_at={MANUAL_TARGET_LOOK_AT}")
        ctr = list(MANUAL_TARGET_LOOK_AT)
        target_info = {
            "selected_prim_path":    None,
            "prim_type_name":        None,
            "bbox_min_m":            None,
            "bbox_max_m":            None,
            "bbox_center_m":         ctr,
            "bbox_size_m":           None,
            "bbox_size_mm":          None,
            "bbox_method":           "manual",
            "n_candidates_examined": 0,
        }

    # ── Compute per-view positions from resolved target centre ─────────────────
    cx, cy, cz = target_info["bbox_center_m"]

    # top_down:      straight above the target (Z-axis view).
    # front_oblique: −Y offset from target at oblique height (Y-axis view).
    # side_oblique:  +X offset from target at oblique height (X-axis view).
    # Angle from vertical for oblique views:
    #   atan(OBLIQUE_OFFSET / OBLIQUE_HEIGHT) = atan(0.30/0.40) ≈ 36.9°

    view_positions = {
        "top_down":      (cx,                  cy - 0,              cz + TOP_DOWN_HEIGHT),
        "front_oblique": (cx,                  cy - OBLIQUE_OFFSET, cz + OBLIQUE_HEIGHT),
        "side_oblique":  (cx + OBLIQUE_OFFSET, cy,                  cz + OBLIQUE_HEIGHT),
    }
    look_at_point = (cx, cy, cz)

    # Update VIEWS list in-place with the computed positions and look-at.
    for view_cfg in VIEWS:
        name = view_cfg["name"]
        if name in view_positions:
            view_cfg["position_m"] = view_positions[name]
            view_cfg["look_at_m"]  = look_at_point
            offset = (
                round(view_positions[name][0] - cx, 4),
                round(view_positions[name][1] - cy, 4),
                round(view_positions[name][2] - cz, 4),
            )
            print(f"[multiview] requested view {name}: "
                  f"position={view_positions[name]}  "
                  f"look_at={look_at_point}  "
                  f"offset_from_target={offset}")

    # ── View-config summary log ───────────────────────────────────────────────
    print("\n[view_config] layout (offsets are relative to target centre):")
    print(f"[view_config] top_down       offset = "
          f"( 0.00,  0.00, +{TOP_DOWN_HEIGHT:.2f}) m")
    print(f"[view_config] front_oblique  offset = "
          f"( 0.00, -{OBLIQUE_OFFSET:.2f}, +{OBLIQUE_HEIGHT:.2f}) m"
          f"   (~37 deg from vertical)")
    print(f"[view_config] side_oblique   offset = "
          f"(+{OBLIQUE_OFFSET:.2f},  0.00, +{OBLIQUE_HEIGHT:.2f}) m"
          f"   (~37 deg from vertical)")

    # ── Create render product and annotators ONCE ─────────────────────────────
    try:
        rp, rgb_an, depth_an = create_render_product_and_annotators()
    except Exception as exc:
        print(f"[multiview] FATAL: could not create render product — {exc}")
        traceback.print_exc()
        teardown_run_logging()
        return

    # ── Per-view capture loop ─────────────────────────────────────────────────
    view_results = []
    view_dirs    = []

    for idx, view_cfg in enumerate(VIEWS):
        view_name = view_cfg["name"]
        try:
            vr = await capture_view(
                view_cfg, rgb_an, depth_an,
                run_id=run_id, timestamp_utc=timestamp_utc,
                intrinsics=intrinsics,
            )
        except Exception as exc:
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[view_FAIL] {view_name}: {msg}")
            traceback.print_exc()
            vr = {
                "ok":              False,
                "view_name":       view_name,
                "rgb":             None,
                "depth":           None,
                "requested_pose":  {
                    "position_m": list(view_cfg["position_m"]),
                    "look_at_m":  list(view_cfg["look_at_m"]),
                    "up_axis":    list(view_cfg["up_axis"]),
                },
                "measured_pose":          None,
                "depth_valid_min_m":      None,
                "depth_valid_max_m":      None,
                "n_valid_depth_pixels":   0,
                "error_message":          msg,
            }

        vd = save_view_outputs(
            vr, view_idx=idx,
            intrinsics=intrinsics,
            run_id=run_id,
            timestamp_utc=timestamp_utc,
            target_info=target_info,
        )
        view_results.append(vr)
        view_dirs.append(vd)

        # One summary line per view
        d_window = (
            f"[{vr['depth_valid_min_m']:.4f}, {vr['depth_valid_max_m']:.4f}] m"
            if vr["ok"] else "n/a"
        )
        req_pos = view_cfg["position_m"]
        meas_pos = (
            vr["measured_pose"]["position"] if vr["measured_pose"] else "n/a"
        )
        print(
            f"[view_{view_name}] ok={vr['ok']}  "
            f"req_pos=({req_pos[0]:.4f},{req_pos[1]:.4f},{req_pos[2]:.4f})  "
            f"meas_pos={meas_pos}  "
            f"depth={d_window}  "
            f"n_valid={vr['n_valid_depth_pixels']}  "
            f"save={vd}"
        )

    # ── Contact sheet ─────────────────────────────────────────────────────────
    print("\n[contact_sheet] building contact sheet ...")
    build_contact_sheet(view_results, view_dirs)

    # ── Global summary ────────────────────────────────────────────────────────
    print("\n[summary] writing global summary ...")
    build_global_summary(
        view_results, view_dirs,
        run_id=run_id,
        timestamp_utc=timestamp_utc,
        intrinsics=intrinsics,
        target_info=target_info,
    )

    teardown_run_logging()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

asyncio.ensure_future(main())
