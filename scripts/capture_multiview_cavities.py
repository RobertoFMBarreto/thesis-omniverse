"""
capture_multiview_cavities.py — Phase A: Multi-view Cavity Capture Proof-of-Life

Sequentially relocates the single camera prim to three viewpoints (top_down,
front_oblique, side_oblique) for each of the four board cavities.  Piece prims
are hidden before capture so they cannot occlude cavity openings from oblique
angles.  Cavity identity is used only for scene setup and output labelling —
it is never used for matching, classification, or scoring.

Cavity world centres are read from:
    data/cavities_detected/<cavity>/cavity_metadata.json

Phase A note:
    Sequential-camera relocation is used here instead of multiple camera
    prims authored in USD.  The final architecture will replace this with
    three static cameras in the scene, so no costly stage edits happen at
    capture time.

Outputs per view (under CAVITIES_OUT_ROOT/<cavity>/view_NN_<name>/):
    rgb.png           — raw RGB frame (BGR, OpenCV)
    depth.npy         — raw float32 depth in metres
    depth_vis.png     — colourised depth (viridis, matplotlib)
    metadata.json     — pose, intrinsics, depth window, timestamps,
                        cavity-specific fields, visibility control fields

Global outputs (under CAVITIES_OUT_ROOT/<cavity>/):
    views_contact_sheet.png
    multiview_capture_summary.json
    run_log.txt

Global all-cavities summary (under CAVITIES_OUT_ROOT/):
    multiview_phaseA_all_cavities_summary.json

Run inside Isaac Sim 5.1 Script Editor.

NOTE: __file__ is unreliable when pasted into the Script Editor — it
resolves to a temporary path.  PROJECT_ROOT is therefore set explicitly
via env-var override.

IMPORTANT — use of cavity identity in this script:
    Cavity names (cavity_00, cavity_01, …) are used ONLY for:
      - driving scene setup automation (which metadata file to read);
      - labelling output directories and metadata files.
    They are NEVER used for matching, classification, or scoring logic.
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

# ── Cavity capture order ──────────────────────────────────────────────────────
CAVITY_CAPTURE_ORDER = ["cavity_00", "cavity_01", "cavity_02", "cavity_03"]

# Where the single-view cavity metadata JSON files live.
CAVITIES_INPUT_DIR = PROJECT_ROOT / "data" / "cavities_detected"

# Root directory for all multi-view cavity outputs.
CAVITIES_OUT_ROOT = PROJECT_ROOT / "data" / "multiview_captures" / "cavities"

# Piece prims to hide BEFORE cavity captures so they cannot occlude openings.
PIECE_PRIM_PATHS_TO_HIDE = [
    "/World/Rectangle",
    "/World/Square",
    "/World/Circle",
    "/World/Triangle",
]

# When True, piece prims are hidden for the full cavity capture run and
# restored in the finally block.
HIDE_PIECES_DURING_CAVITY_CAPTURE = True

# Camera height used by the original single-view cavity capture script when
# computing board_surface_depth_m.  Used here to reconstruct board top Z.
# from scripts/capture_cavity_detection.py:69
SINGLE_VIEW_CAVITY_CAM_Z = 1.00   # metres

# ── Camera USD prim path ──────────────────────────────────────────────────────
CAMERA_PRIM_PATH = "/World/Camera"

# Render resolution — matches the validated single-view capture.
IMG_WIDTH  = 640
IMG_HEIGHT = 480

# Replicator subframes per step (higher = more stable rendering, slower).
RT_SUBFRAMES = 8

# Camera intrinsics — must match the Isaac Sim camera prim settings.
FOCAL_MM    = 24.0
APERTURE_MM = 36.0

# Label written into every metadata JSON so any consumer knows this is not the
# final multi-camera setup.
PHASE_A_NOTE = (
    "sequential-camera proof-of-life; not final static multi-camera architecture"
)

# Disclaimer written into every metadata file — makes the role of cavity
# identity explicit so downstream consumers are not misled.
CAVITY_IDENTITY_NOTE = (
    "Cavity identity used only for experimental scene setup, not for "
    "perception/matching."
)

# ── CAMERA PLACEMENT (relative to resolved target centre) ─────────────────────
#
# top_down:      camera directly above target centre (cx, cy, cz+0.50).
# front_oblique: camera shifted −OBLIQUE_OFFSET along Y and +OBLIQUE_HEIGHT.
# side_oblique:  camera shifted +OBLIQUE_OFFSET along X and +OBLIQUE_HEIGHT.
# up_axis = (0, 1, 0) for all three.

TOP_DOWN_HEIGHT = 0.50   # m above target centre (z+)
OBLIQUE_HEIGHT  = 0.40   # m above target centre (z+); gives ~37° from vertical
OBLIQUE_OFFSET  = 0.30   # m lateral offset: front = −Y, side = +X

# Correction applied after the look-at quaternion is built, rotating the camera
# around its local Z axis to match the piece multi-view camera convention.
CAMERA_Z_ROT_CORRECTION_DEG = -90.0

# ── VIEW CONFIGS ──────────────────────────────────────────────────────────────
#
# Positions and look-ats are placeholders; they are overwritten per cavity
# inside the main loop after the target centre is resolved.
VIEWS = [
    {
        "name":       "top_down",
        "position_m": (0.0, 0.0, 0.50),     # placeholder; recomputed in loop
        "look_at_m":  (0.0, 0.0, 0.0),      # placeholder; recomputed in loop
        "up_axis":    (0, 1, 0),
    },
    {
        "name":       "front_oblique",
        "position_m": (0.0, -0.30, 0.40),   # placeholder; recomputed in loop
        "look_at_m":  (0.0, 0.0, 0.0),      # placeholder; recomputed in loop
        "up_axis":    (0, 1, 0),
    },
    {
        "name":       "side_oblique",
        "position_m": (0.30, 0.0, 0.40),    # placeholder; recomputed in loop
        "look_at_m":  (0.0, 0.0, 0.0),      # placeholder; recomputed in loop
        "up_axis":    (0, 1, 0),
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
    f.write("# capture_multiview_cavities.py — run log\n")
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
    makes fy_px == fx_px for a square-pixel sensor.
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


# ── VISIBILITY CONTROL HELPERS ────────────────────────────────────────────────

def set_piece_visibility(stage, piece_paths: list, visible: bool) -> list:
    """
    For each prim path, toggle UsdGeom.Imageable visibility via
    MakeVisible() or MakeInvisible().  Returns the list of paths that were
    actually toggled (skips invalid prims and Material/Shader/Light/Camera/Scope
    prims defensively).
    """
    from pxr import UsdGeom

    EXCLUDED_TYPE_SUBSTRINGS = ["material", "shader", "light", "camera", "scope"]
    toggled = []

    for path_str in piece_paths:
        try:
            prim = stage.GetPrimAtPath(path_str)
            if not prim.IsValid():
                print(f"[set_visibility] WARNING: prim not valid at {path_str} "
                      "— skipping")
                continue
            type_lower = prim.GetTypeName().lower()
            if any(ex in type_lower for ex in EXCLUDED_TYPE_SUBSTRINGS):
                print(f"[set_visibility] WARNING: skipping excluded type "
                      f"'{prim.GetTypeName()}' at {path_str}")
                continue
            imageable = UsdGeom.Imageable(prim)
            if not imageable:
                print(f"[set_visibility] WARNING: prim not Imageable at {path_str} "
                      "— skipping")
                continue
            if visible:
                imageable.MakeVisible()
            else:
                imageable.MakeInvisible()
            toggled.append(path_str)
        except Exception as exc:
            print(f"[set_visibility] WARNING: could not set visibility on "
                  f"{path_str}: {exc}")

    return toggled


def snapshot_visibility(stage, prim_paths: list) -> dict:
    """
    Record the current visibility token for every prim path in the list.
    Returns {prim_path_str: visibility_token_or_None}.
    """
    from pxr import UsdGeom

    snapshot = {}
    for path_str in prim_paths:
        try:
            prim = stage.GetPrimAtPath(path_str)
            if prim.IsValid():
                img  = UsdGeom.Imageable(prim)
                attr = img.GetVisibilityAttr()
                snapshot[path_str] = attr.Get() if attr else None
            else:
                snapshot[path_str] = None
        except Exception:
            snapshot[path_str] = None
    return snapshot


def restore_visibility(stage, snapshot: dict) -> None:
    """
    Restore visibility tokens captured by snapshot_visibility().
    Logs each failure but does not raise — restore must always complete.
    """
    from pxr import UsdGeom

    for path_str, token in snapshot.items():
        try:
            prim = stage.GetPrimAtPath(path_str)
            if not prim.IsValid():
                print(f"[restore_visibility] prim not valid at {path_str} — skipping")
                continue
            img = UsdGeom.Imageable(prim)
            if token is None:
                # No prior attr value — make visible (inherited) as safe default.
                img.MakeVisible()
            elif str(token) == "invisible":
                img.MakeInvisible()
            else:
                img.MakeVisible()
        except Exception as exc:
            print(f"[restore_visibility] WARNING: could not restore {path_str}: {exc}")


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
    R = __import__("numpy").eye(3)
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

    Falls back to xformOp:rotateXYZ / xformOp:rotateZ if orient is absent.
    For oblique views the quaternion path is required — the Z-only-rotation
    fallback will log a warning and produce incorrect orientation.
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

    # Compose with a local-Z roll to match the piece multi-view camera convention.
    # q_z_rot = (cos(θ/2), 0, 0, sin(θ/2))  in (w, x, y, z) for a Z-axis rotation.
    _theta = math.radians(CAMERA_Z_ROT_CORRECTION_DEG)
    _cz    = math.cos(_theta / 2.0)
    _sz    = math.sin(_theta / 2.0)
    from pxr import Gf as _Gf_local
    q_z_rot = _Gf_local.Quatd(_cz, 0.0, 0.0, _sz)
    # q_final = q_look_at * q_z_rot  (local-frame rotation applied after look-at)
    quat = quat * q_z_rot

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


# ── CAVITY METADATA READER ────────────────────────────────────────────────────

def read_cavity_metadata(cavity_name: str) -> dict:
    """
    Read data/cavities_detected/<cavity_name>/cavity_metadata.json.

    Expected fields:
        centroid_world_m: {x: float, y: float}
        board_surface_depth_m: float

    Returns the parsed dict, or raises RuntimeError if the file is missing or
    malformed.
    """
    meta_path = CAVITIES_INPUT_DIR / cavity_name / "cavity_metadata.json"
    if not meta_path.exists():
        raise RuntimeError(
            f"[target_resolve] cavity_metadata.json not found at {meta_path}"
        )
    try:
        with open(str(meta_path), "r") as fp:
            data = json.load(fp)
    except Exception as exc:
        raise RuntimeError(
            f"[target_resolve] could not parse {meta_path}: {exc}"
        ) from exc

    # Validate required fields.
    cw = data.get("centroid_world_m")
    if not isinstance(cw, dict) or "x" not in cw or "y" not in cw:
        raise RuntimeError(
            f"[target_resolve] {meta_path} missing centroid_world_m.x/y"
        )
    if "board_surface_depth_m" not in data:
        raise RuntimeError(
            f"[target_resolve] {meta_path} missing board_surface_depth_m"
        )

    return data


def compute_cavity_target_centre(cavity_meta: dict) -> tuple:
    """
    Compute the 3-D target centre for a cavity from its metadata.

    X, Y  = centroid_world_m.x, centroid_world_m.y
    Z     = SINGLE_VIEW_CAVITY_CAM_Z − board_surface_depth_m

    Returns (cx, cy, cz) as floats.
    """
    cw = cavity_meta["centroid_world_m"]
    cx = float(cw["x"])
    cy = float(cw["y"])
    board_depth = float(cavity_meta["board_surface_depth_m"])
    cz = SINGLE_VIEW_CAVITY_CAM_Z - board_depth
    return (cx, cy, cz)


# ── REPLICATOR SETUP ──────────────────────────────────────────────────────────

def create_render_product_and_annotators():
    """
    Create the Replicator render product and attach rgb + distance_to_image_plane
    annotators once.  The same render product is reused across all cavities and
    all views.

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
                      cavity_name: str,
                      cavity_meta_source: str,
                      target_centre: tuple,
                      cav_out_root: Path) -> Path:
    """
    Save rgb.png, depth.npy, depth_vis.png, and metadata.json for one view.

    Parameters
    ----------
    view_result        : dict returned by capture_view()
    view_idx           : int index (0-based)
    intrinsics         : dict from compute_intrinsics()
    run_id             : str
    timestamp_utc      : str
    cavity_name        : str, e.g. "cavity_00"
    cavity_meta_source : str path of the source cavity_metadata.json
    target_centre      : (cx, cy, cz) — world-space centre aimed at
    cav_out_root       : Path — per-cavity output directory

    Returns the view directory path.
    """
    import numpy as np
    import cv2

    view_name = view_result["view_name"]
    view_dir  = cav_out_root / f"view_{view_idx:02d}_{view_name}"
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
        ax.set_title(f"{cavity_name} / {view_name}  depth (m)", fontsize=10)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="depth (m)")
        fig.savefig(str(depth_vis_path), dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[view_{view_name}] saved depth_vis.png")
    except Exception as exc:
        print(f"[view_{view_name}] WARNING: could not save depth_vis.png — {exc}")

    # ── Per-view offset from target centre ────────────────────────────────────
    req_pos    = view_result["requested_pose"]["position_m"]   # list[3]
    cam_offset = [round(req_pos[i] - target_centre[i], 6) for i in range(3)]

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
        # ── Cavity-specific fields ─────────────────────────────────────────────
        "target_cavity_name":           cavity_name,
        "cavity_centroid_source":       cavity_meta_source,
        "target_bbox_center_world_m":   list(target_centre),
        "requested_look_at":            view_result["requested_pose"]["look_at_m"],
        "camera_offset_from_target":    cam_offset,
        # ── Visibility control fields ─────────────────────────────────────────
        "visibility_control_enabled":   True,
        "hidden_piece_prim_paths":      PIECE_PRIM_PATHS_TO_HIDE,
        "camera_z_rotation_correction_deg": CAMERA_Z_ROT_CORRECTION_DEG,
        "note":  CAVITY_IDENTITY_NOTE,
        "phase_note_detail": (
            "sequential-camera proof-of-life; not final static multi-camera architecture"
        ),
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

def build_contact_sheet(view_results: list, view_dirs: list,
                        cavity_name: str, cav_out_root: Path) -> None:
    """
    Build a 2 x N grid image:
      top row    — RGB per view
      bottom row — depth_vis per view (loaded from disk)
    Column titles are the view names.

    Saved to cav_out_root/views_contact_sheet.png.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.image as mpimg

        n_views = len(view_results)
        fig, axes = plt.subplots(
            2, n_views,
            figsize=(5 * n_views, 8),
            dpi=100,
        )
        if n_views == 1:
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

            # Bottom row: depth_vis (load from disk)
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
            f"Multi-view capture — {cavity_name}  ({n_views} views)\n"
            f"Phase A: {PHASE_A_NOTE}",
            fontsize=10,
        )
        fig.tight_layout()

        sheet_path = cav_out_root / "views_contact_sheet.png"
        fig.savefig(str(sheet_path), dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"[contact_sheet] saved {sheet_path}")

    except Exception as exc:
        print(f"[contact_sheet] WARNING: could not build contact sheet — {exc}")
        traceback.print_exc()


# ── PER-CAVITY SUMMARY ────────────────────────────────────────────────────────

def build_cavity_summary(view_results: list, view_dirs: list,
                          run_id: str, timestamp_utc: str,
                          intrinsics: dict,
                          cavity_name: str,
                          cavity_meta_source: str,
                          target_centre: tuple,
                          cav_out_root: Path) -> None:
    """
    Write multiview_capture_summary.json with per-view records and top-level
    success/failure status for one cavity.
    """
    n_requested = len(view_results)
    n_succeeded = sum(1 for vr in view_results if vr["ok"])
    overall_ok  = (n_succeeded == n_requested)

    view_records = []
    for idx, (vr, vd) in enumerate(zip(view_results, view_dirs)):
        req_pos    = vr["requested_pose"]["position_m"]
        cam_offset = [round(req_pos[i] - target_centre[i], 6) for i in range(3)]

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
            "depth_valid_min_m":     vr["depth_valid_min_m"],
            "depth_valid_max_m":     vr["depth_valid_max_m"],
            "n_valid_depth_pixels":  vr["n_valid_depth_pixels"],
            "save_path":             str(vd),
            "ok":                    vr["ok"],
            "error_message":         vr.get("error_message"),
            "camera_offset_from_target": cam_offset,
        }
        view_records.append(rec)

    summary = {
        "script_name":              "capture_multiview_cavities.py",
        "phase":                    "A",
        "phase_note":               PHASE_A_NOTE,
        "run_id":                   run_id,
        "timestamp_utc":            timestamp_utc,
        "cavity":                   cavity_name,
        "cavity_centroid_source":   cavity_meta_source,
        "target_bbox_center_world_m": list(target_centre),
        "n_views_requested":        n_requested,
        "n_views_succeeded":        n_succeeded,
        "success":                  overall_ok,
        "views":                    view_records,
        "output_dir":               str(cav_out_root),
        # Visibility control fields.
        "visibility_control_enabled": True,
        "hidden_piece_prim_paths":    PIECE_PRIM_PATHS_TO_HIDE,
        "note":                       CAVITY_IDENTITY_NOTE,
    }

    summary_path = cav_out_root / "multiview_capture_summary.json"
    try:
        with open(str(summary_path), "w") as fp:
            json.dump(summary, fp, indent=2)
        print(f"[summary] saved {summary_path}")
    except Exception as exc:
        print(f"[summary] WARNING: could not write summary JSON — {exc}")

    print("\n" + "=" * 60)
    print(f"  cavity={cavity_name}  success={overall_ok}  "
          f"views_succeeded={n_succeeded}/{n_requested}")
    if not overall_ok:
        for vr in view_results:
            if not vr["ok"]:
                print(f"  [view_FAIL] {vr['view_name']}: "
                      f"{vr.get('error_message', 'unknown error')}")
    print("=" * 60)


# ── GLOBAL ALL-CAVITIES SUMMARY ───────────────────────────────────────────────

def build_all_cavities_summary(all_cavity_records: list,
                                run_id: str, timestamp_utc: str) -> None:
    """
    Write multiview_phaseA_all_cavities_summary.json under CAVITIES_OUT_ROOT.
    """
    n_attempted = len(all_cavity_records)
    n_succeeded = sum(1 for r in all_cavity_records if r.get("success", False))
    overall_ok  = (n_succeeded == n_attempted) and (n_attempted > 0)

    summary = {
        "script_name":              "capture_multiview_cavities.py",
        "phase":                    "A",
        "phase_note":               PHASE_A_NOTE,
        "run_id":                   run_id,
        "timestamp_utc":            timestamp_utc,
        "n_cavities_attempted":     n_attempted,
        "n_cavities_succeeded":     n_succeeded,
        "success_overall":          overall_ok,
        "cavities":                 all_cavity_records,
        "cavities_output_root":     str(CAVITIES_OUT_ROOT),
        "cavities_input_dir":       str(CAVITIES_INPUT_DIR),
        "visibility_control_enabled": True,
        "hidden_piece_prim_paths":  PIECE_PRIM_PATHS_TO_HIDE,
        "note":                     CAVITY_IDENTITY_NOTE,
    }

    summary_path = CAVITIES_OUT_ROOT / "multiview_phaseA_all_cavities_summary.json"
    try:
        CAVITIES_OUT_ROOT.mkdir(parents=True, exist_ok=True)
        with open(str(summary_path), "w") as fp:
            json.dump(summary, fp, indent=2)
        print(f"[all_cavities_summary] saved {summary_path}")
    except Exception as exc:
        print(f"[all_cavities_summary] WARNING: could not write summary — {exc}")


# ── RESULTS TABLE ─────────────────────────────────────────────────────────────

def print_results_table(all_cavity_records: list) -> None:
    """
    Print a fixed-width console table with one row per cavity.
    Columns: name, target_centre_m, views_succeeded/total, success.
    """
    print("\n" + "=" * 90)
    print("PHASE A CAVITY RESULTS TABLE")
    print("=" * 90)

    col_w = {
        "name":   12,
        "centre": 36,
        "views":   8,
        "ok":      8,
    }

    header = (
        f"{'cavity':<{col_w['name']}} "
        f"{'target_centre_m (x,y,z)':<{col_w['centre']}} "
        f"{'views':<{col_w['views']}} "
        f"{'success':<{col_w['ok']}}"
    )
    print(header)
    print("-" * 90)

    for rec in all_cavity_records:
        name = rec.get("cavity_name", "?")

        ctr = rec.get("target_centre_m")
        if ctr and len(ctr) == 3:
            ctr_str = f"({ctr[0]:.4f}, {ctr[1]:.4f}, {ctr[2]:.4f})"
        else:
            ctr_str = "n/a"

        views_ok    = rec.get("views_succeeded", 0)
        views_total = rec.get("views_total", 0)
        views_str   = f"{views_ok}/{views_total}"

        ok_str = "YES" if rec.get("success", False) else "NO"

        row = (
            f"{name:<{col_w['name']}} "
            f"{ctr_str:<{col_w['centre']}} "
            f"{views_str:<{col_w['views']}} "
            f"{ok_str:<{col_w['ok']}}"
        )
        print(row)

    print("=" * 90)


# ── MAIN ──────────────────────────────────────────────────────────────────────

async def main():
    import omni.usd

    # ── Run identifiers ────────────────────────────────────────────────────────
    run_id        = uuid.uuid4().hex[:8]
    timestamp_utc = datetime.now(timezone.utc).isoformat()

    # ── Stage ─────────────────────────────────────────────────────────────────
    stage = omni.usd.get_context().get_stage()
    if stage is None:
        print("[multiview] FATAL: no USD stage is open. Open the scene first.")
        return

    # ── Global output root ─────────────────────────────────────────────────────
    CAVITIES_OUT_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = setup_run_logging(CAVITIES_OUT_ROOT)

    print("=" * 80)
    print("capture_multiview_cavities.py — Phase A: Multi-view Cavity Capture")
    print("=" * 80)
    print(f"[multiview] phase_note          = {PHASE_A_NOTE}")
    print(f"[multiview] camera_prim         = {CAMERA_PRIM_PATH}")
    print(f"[multiview] cavities_out_root   = {CAVITIES_OUT_ROOT}")
    print(f"[multiview] cavities_input_dir  = {CAVITIES_INPUT_DIR}")
    print(f"[multiview] cavity_order        = {CAVITY_CAPTURE_ORDER}")
    print(f"[multiview] n_views_per_cavity  = {len(VIEWS)}")
    print(f"[multiview] run_id              = {run_id}")
    print(f"[multiview] timestamp_utc       = {timestamp_utc}")
    print(f"[multiview] run_log             = {log_path}")

    # ── Intrinsics (fixed across all cavities — one camera, one sensor) ────────
    intrinsics = compute_intrinsics()
    print(f"[multiview] intrinsics: fx={intrinsics['fx_px']:.2f}  "
          f"fy={intrinsics['fy_px']:.2f}  "
          f"cx={intrinsics['cx_px']:.2f}  cy={intrinsics['cy_px']:.2f}  "
          f"model={intrinsics['intrinsics_model']}")

    # ── Snapshot visibility of piece prims ─────────────────────────────────────
    visibility_snapshot = snapshot_visibility(stage, PIECE_PRIM_PATHS_TO_HIDE)
    print(f"[visibility] captured original visibility for "
          f"{len(visibility_snapshot)} piece prims")

    # ── Create render product once (reused across all cavities and views) ───────
    try:
        rp, rgb_an, depth_an = create_render_product_and_annotators()
    except Exception as exc:
        print(f"[multiview] FATAL: could not create render product — {exc}")
        traceback.print_exc()
        teardown_run_logging()
        return

    all_cavity_records = []

    try:
        # ── Hide all piece prims once before the cavity loop ───────────────────
        if HIDE_PIECES_DURING_CAVITY_CAPTURE:
            toggled = set_piece_visibility(stage, PIECE_PRIM_PATHS_TO_HIDE,
                                           visible=False)
            print(f"[visibility] hidden prims = {toggled}")
        else:
            print("[visibility] HIDE_PIECES_DURING_CAVITY_CAPTURE = False "
                  "— piece prims NOT hidden")

        # ── Per-cavity capture loop ────────────────────────────────────────────
        for cavity_name in CAVITY_CAPTURE_ORDER:

            print(f"\n{'#'*70}")
            print(f"[cavity {cavity_name}] starting multi-view capture")
            print(f"{'#'*70}")

            # Per-cavity try/except — one failure must not stop the run.
            try:
                # ── Read cavity metadata ───────────────────────────────────────
                cavity_meta_source = str(
                    CAVITIES_INPUT_DIR / cavity_name / "cavity_metadata.json"
                )
                try:
                    cavity_meta = read_cavity_metadata(cavity_name)
                except RuntimeError as meta_exc:
                    print(f"[cavity {cavity_name}] no metadata, skipping — "
                          f"{meta_exc}")
                    all_cavity_records.append({
                        "cavity_name":      cavity_name,
                        "target_centre_m":  None,
                        "views_succeeded":  0,
                        "views_total":      len(VIEWS),
                        "success":          False,
                        "output_dir":       None,
                        "error_message":    str(meta_exc),
                    })
                    continue

                # ── Compute target centre ──────────────────────────────────────
                target_centre = compute_cavity_target_centre(cavity_meta)
                cx, cy, cz = target_centre
                print(f"[cavity {cavity_name}] target centre = "
                      f"({cx:.4f}, {cy:.4f}, {cz:.4f}) m")
                print(f"[cavity {cavity_name}] board_surface_depth_m = "
                      f"{cavity_meta['board_surface_depth_m']:.4f} m")
                print(f"[cavity {cavity_name}] metadata source = "
                      f"{cavity_meta_source}")

                # ── Compute per-view camera positions ──────────────────────────
                view_positions = {
                    "top_down":      (cx,                  cy,                  cz + TOP_DOWN_HEIGHT),
                    "front_oblique": (cx,                  cy - OBLIQUE_OFFSET, cz + OBLIQUE_HEIGHT),
                    "side_oblique":  (cx + OBLIQUE_OFFSET, cy,                  cz + OBLIQUE_HEIGHT),
                }
                look_at_point = (cx, cy, cz)

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
                        print(f"[view_config] {name}: "
                              f"position={view_positions[name]}  "
                              f"look_at={look_at_point}  "
                              f"offset_from_target={offset}")

                # ── Per-cavity output directory ────────────────────────────────
                cav_out_root = CAVITIES_OUT_ROOT / cavity_name
                shutil.rmtree(str(cav_out_root), ignore_errors=True)
                cav_out_root.mkdir(parents=True, exist_ok=True)
                print(f"[cavity {cavity_name}] output dir = {cav_out_root}")

                # ── Per-cavity run-log tee ─────────────────────────────────────
                # The global run_log continues; we also touch a per-cavity copy.
                cav_log_path = cav_out_root / "run_log.txt"
                try:
                    with open(str(cav_log_path), "w") as _f:
                        _f.write(f"# capture_multiview_cavities.py — "
                                 f"per-cavity log\n")
                        _f.write(f"# cavity: {cavity_name}\n")
                        _f.write(f"# run_id: {run_id}\n")
                        _f.write(f"# timestamp_utc: {timestamp_utc}\n")
                        _f.write(f"# target_centre_m: {list(target_centre)}\n")
                        _f.write(f"# See global run_log.txt in {CAVITIES_OUT_ROOT}\n")
                        _f.write("=" * 60 + "\n")
                except Exception as log_exc:
                    print(f"[cavity {cavity_name}] WARNING: could not write "
                          f"per-cavity run_log.txt — {log_exc}")

                # ── Per-view capture ───────────────────────────────────────────
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
                        print(f"[view_FAIL] {cavity_name}/{view_name}: {msg}")
                        traceback.print_exc()
                        vr = {
                            "ok": False, "view_name": view_name,
                            "rgb": None, "depth": None,
                            "requested_pose": {
                                "position_m": list(view_cfg["position_m"]),
                                "look_at_m":  list(view_cfg["look_at_m"]),
                                "up_axis":    list(view_cfg["up_axis"]),
                            },
                            "measured_pose": None,
                            "depth_valid_min_m": None, "depth_valid_max_m": None,
                            "n_valid_depth_pixels": 0, "error_message": msg,
                        }

                    vd = save_view_outputs(
                        vr, view_idx=idx, intrinsics=intrinsics,
                        run_id=run_id, timestamp_utc=timestamp_utc,
                        cavity_name=cavity_name,
                        cavity_meta_source=cavity_meta_source,
                        target_centre=target_centre,
                        cav_out_root=cav_out_root,
                    )
                    view_results.append(vr)
                    view_dirs.append(vd)

                    # One summary line per view.
                    d_window = (
                        f"[{vr['depth_valid_min_m']:.4f}, "
                        f"{vr['depth_valid_max_m']:.4f}] m"
                        if vr["ok"] else "n/a"
                    )
                    req_pos  = view_cfg["position_m"]
                    meas_pos = (
                        vr["measured_pose"]["position"]
                        if vr["measured_pose"] else "n/a"
                    )
                    print(
                        f"[view_{view_name}] ok={vr['ok']}  "
                        f"req_pos=({req_pos[0]:.4f},{req_pos[1]:.4f},"
                        f"{req_pos[2]:.4f})  "
                        f"meas_pos={meas_pos}  "
                        f"depth={d_window}  "
                        f"n_valid={vr['n_valid_depth_pixels']}  "
                        f"save={vd}"
                    )

                # ── Contact sheet ──────────────────────────────────────────────
                print(f"\n[contact_sheet] building contact sheet for "
                      f"{cavity_name} ...")
                build_contact_sheet(view_results, view_dirs,
                                    cavity_name=cavity_name,
                                    cav_out_root=cav_out_root)

                # ── Per-cavity summary JSON ────────────────────────────────────
                print(f"\n[summary] writing per-cavity summary for "
                      f"{cavity_name} ...")
                build_cavity_summary(
                    view_results, view_dirs,
                    run_id=run_id, timestamp_utc=timestamp_utc,
                    intrinsics=intrinsics,
                    cavity_name=cavity_name,
                    cavity_meta_source=cavity_meta_source,
                    target_centre=target_centre,
                    cav_out_root=cav_out_root,
                )

                # ── Append global record ───────────────────────────────────────
                n_succeeded = sum(1 for vr in view_results if vr["ok"])
                all_cavity_records.append({
                    "cavity_name":      cavity_name,
                    "target_centre_m":  list(target_centre),
                    "views_succeeded":  n_succeeded,
                    "views_total":      len(VIEWS),
                    "success":          (n_succeeded == len(VIEWS)),
                    "output_dir":       str(cav_out_root),
                    "error_message":    None,
                })

            except Exception as cav_exc:
                msg = f"{type(cav_exc).__name__}: {cav_exc}"
                print(f"[cavity_FAIL] {cavity_name}: {msg}")
                traceback.print_exc()
                all_cavity_records.append({
                    "cavity_name":      cavity_name,
                    "target_centre_m":  None,
                    "views_succeeded":  0,
                    "views_total":      len(VIEWS),
                    "success":          False,
                    "output_dir":       None,
                    "error_message":    msg,
                })
                # Continue to the next cavity — do not abort the loop.
                continue

    finally:
        # ── Restore original piece visibility regardless of success/failure ────
        print("\n[visibility] restoring original piece visibility states ...")
        restore_visibility(stage, visibility_snapshot)
        print("[visibility] restore complete")

    # ── Global all-cavities summary ────────────────────────────────────────────
    print("\n[all_cavities_summary] writing global summary ...")
    build_all_cavities_summary(all_cavity_records, run_id, timestamp_utc)

    # ── Results table ──────────────────────────────────────────────────────────
    print_results_table(all_cavity_records)

    teardown_run_logging()


# ── ENTRY POINT ───────────────────────────────────────────────────────────────

asyncio.ensure_future(main())
