"""
capture_piece_detection.py — Phase 1

Detect one visible geometric piece from RGB-D, segment it from the support
surface, build a real-scale 3D point cloud, generate a 2D footprint, and
save all debug artifacts.

Run inside Isaac Sim 5.1 Script Editor.
"""

import asyncio
import math
import json
import os
import time
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Tune these constants for your scene BEFORE running.

# Camera USD path in the stage
CAMERA_PRIM_PATH = "/World/Camera"

# Camera pose override.
#
# By default we DO NOT move the camera: the script uses whatever pose is
# already authored on the stage.  Set SET_CAMERA_POSE = True to programmatically
# override the camera pose using the (CAM_X, CAM_Y, CAM_Z, CAM_ROT_Z_DEG)
# constants below.  This mirrors the convention used by
# scripts/capture_cavity_detection.py.
SET_CAMERA_POSE = False

# Camera pose: X, Y, Z in metres (world frame, Z = up).
# Used ONLY when SET_CAMERA_POSE = True.
CAM_X = -0.25
CAM_Y =  0.45
CAM_Z =  0.58   # height above world origin

# Camera rotation around Z-axis in degrees (0 = looking in -Y, no rotation).
# Used ONLY when SET_CAMERA_POSE = True.
CAM_ROT_Z_DEG = 0.0

# Render resolution
IMG_W = 640
IMG_H = 480

# Replicator subframes (higher = more stable rendering, slower)
RT_SUBFRAMES = 8

# Camera intrinsics — must match the Isaac Sim camera settings.
# focal_mm / aperture_mm define horizontal FOV.
FOCAL_MM    = 24.0
APERTURE_MM = 36.0

# Surface estimation: broad safety bounds [m] within which depth is searched
# for the local support plane.  These are deliberately wide because the active
# estimator (auto_depth_layers) discriminates among multiple peaks rather than
# relying on a tight bracket around the support.
#
# Recommended camera heights (when SET_CAMERA_POSE=False, the camera pose
# is set in the Isaac Sim stage):
#   - z ≈ 0.7 m : preferred for piece capture — narrower field of view,
#                 reduces the chance of capturing the side table as a false
#                 piece.  Local support lands near 0.7 m depth; a 105 mm
#                 tall piece top near 0.595 m.
#   - z ≈ 0.8 m : working setting for board/cavity capture (different
#                 script, different aperture trade-off).
SURFACE_DEPTH_MIN  = 0.40
SURFACE_DEPTH_MAX  = 0.90
SURFACE_HIST_BIN_M = 0.001   # 1 mm histogram bin width
SURFACE_HIST_BIN   = SURFACE_HIST_BIN_M   # alias kept for backward compatibility
# Margin used by the surface estimator's bound-saturation warning:
# warns if the estimated surface depth is within this many metres of the
# search bounds (likely indicating a pinned histogram / wrong window).
SURFACE_BOUND_WARN_M = 0.005   # 5 mm

# ── AUTO DEPTH-LAYER SUPPORT ESTIMATION ──────────────────────────────────────
# Strategy: build a 1 mm depth histogram inside the piece ROI, extract local
# peaks (each merged within SURFACE_PEAK_MERGE_DISTANCE_M of its neighbours),
# sort by depth ascending (closest first), then pick the closest LARGE peak
# as the local support plane.  This avoids the "background-wall wins" failure
# mode of plain dominant-mode estimation in scenes with several depth tiers.
SURFACE_ESTIMATION_MODE       = "auto_depth_layers"   # or "dominant_depth"
SURFACE_PEAK_MERGE_DISTANCE_M = 0.004    # peaks closer than this are merged
SUPPORT_MIN_PEAK_FRACTION     = 0.10     # peak fraction needed to qualify as support
PIECE_MAX_PEAK_FRACTION       = 0.08     # closer peaks up to this fraction are
                                          # tolerated as "piece top" before support
MIN_PIECE_ABOVE_SURFACE_M     = 0.004    # min depth gap between piece-top peak
                                          # and support peak (sanity check)

# Segmentation: a pixel belongs to the piece if its depth is MORE than this
# margin below the surface estimate.  Too small → table noise bleeds in.
# Too large → thin/flat pieces disappear.
SURFACE_TOLERANCE = 0.004   # 4 mm

# Minimum depth for any valid measurement (clips near-field noise)
DEPTH_MIN_VALID = 0.02

# Connected-component filters
CC_MIN_AREA_PX =  300   # discard blobs smaller than this
CC_MAX_AREA_PX = 50000  # discard blobs suspiciously large (probably table leak)

# ── PIECE ROI ────────────────────────────────────────────────────────────────
# Restrict surface estimation (and optionally segmentation) to a region of
# interest around the piece capture area.  This avoids unrelated surfaces in
# the scene (e.g. another table in the corner) hijacking the depth histogram.
#
# Three modes when PIECE_ROI_ENABLED=True:
#   1. If all four PIECE_ROI_X1/Y1/X2/Y2 are not None → explicit pixel ROI.
#   2. Else if PIECE_ROI_MODE == "center_fraction" → centred box covering
#      PIECE_ROI_FRACTION of the image width and height.
#   3. Otherwise → full image (same as PIECE_ROI_ENABLED=False).
PIECE_ROI_ENABLED  = True
PIECE_ROI_MODE     = "center_fraction"
PIECE_ROI_FRACTION = 0.60

# Explicit pixel ROI fallback (used only if all four are not None).
PIECE_ROI_X1 = None
PIECE_ROI_Y1 = None
PIECE_ROI_X2 = None
PIECE_ROI_Y2 = None

# When True, the raw piece mask is also restricted to the (expanded) ROI so
# that connected components from objects outside the ROI cannot enter the
# selection at all.  EXPAND_PX widens the ROI for segmentation only, leaving
# the surface-estimation ROI tighter.
RESTRICT_PIECE_MASK_TO_ROI = True
PIECE_MASK_ROI_EXPAND_PX   = 20

# Piece selection mode — used after connected-component filtering.
#   "largest"            — pick the valid component with largest area (default)
#   "closest_to_center"  — pick the valid component whose centroid is nearest
#                           to the image centre. Use this with the camera
#                           re-aimed at a specific piece to capture it
#                           deterministically without classifying.
#   "manual_index"       — pick MANUAL_COMPONENT_INDEX from the deterministic
#                           sort order (area DESC, ties broken by centroid_x ASC).
PIECE_SELECTION_MODE   = "largest"
MANUAL_COMPONENT_INDEX = 0

# Point cloud sampling target
N_POINTS = 2048

# Output directory.
# Outputs are intentionally saved inside the repository (not /tmp) so they
# can be inspected, committed, or pulled to a developer machine from the
# Isaac Sim container without docker cp + scp.
#
# Do NOT derive PROJECT_ROOT from __file__: when this script is pasted into
# the Isaac Sim Script Editor, __file__ resolves to a temporary path such as
# /tmp/carb.../script_*.py, which would put OUT_DIR under /tmp. Use an
# explicit container path, with an environment variable escape hatch for
# other machines (e.g. a Mac developer workflow).
PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        "/workspace/Tese_Roberto/shape_insertion/thesis-omniverse",
    )
)
_OUT_BASE = PROJECT_ROOT / "data" / "pieces_detected"

# Per-capture subfolder.
# Phase 1 assumes only ONE piece is visible in the capture area at a time
# (the others are manually hidden in Isaac Sim). To avoid overwriting the
# previous capture, set USE_CAPTURE_SUBDIR=True and change CAPTURE_NAME
# between runs.
#
# CAPTURE_NAME is an EXPERIMENTAL RUN LABEL ONLY — never used by detection,
# segmentation, classification, selection, or matching logic.
CAPTURE_NAME       = "piece_test"
USE_CAPTURE_SUBDIR = True

OUT_DIR = (_OUT_BASE / CAPTURE_NAME) if USE_CAPTURE_SUBDIR else _OUT_BASE

# ── END CONFIG ────────────────────────────────────────────────────────────────


# ── CAMERA SETUP ─────────────────────────────────────────────────────────────

def setup_camera(x: float, y: float, z: float, rot_z_deg: float = 0.0) -> None:
    """
    Set camera world-space translate and orientation (Z-axis rotation only).

    The camera in Isaac Sim is typically oriented so that +Z is up and the lens
    points in the -Y direction by default.  A rotation around the world Z-axis
    spins the view in the XY plane.

    Supports both xformOp:orient (quaternion) and xformOp:rotateXYZ ops,
    whichever is present on the camera prim.
    """
    import omni.usd
    from pxr import UsdGeom, Gf

    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
    if not cam_prim.IsValid():
        raise RuntimeError(f"[setup_camera] Camera prim not found: {CAMERA_PRIM_PATH}")

    xformable = UsdGeom.Xformable(cam_prim)
    ops_dict = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}

    if "xformOp:translate" not in ops_dict:
        raise RuntimeError("[setup_camera] Camera prim has no xformOp:translate op")

    ops_dict["xformOp:translate"].Set(Gf.Vec3d(x, y, z))

    half_rad = math.radians(rot_z_deg) / 2.0
    quat = Gf.Quatd(math.cos(half_rad), 0.0, 0.0, math.sin(half_rad))

    if "xformOp:orient" in ops_dict:
        ops_dict["xformOp:orient"].Set(quat)
    elif "xformOp:rotateXYZ" in ops_dict:
        ops_dict["xformOp:rotateXYZ"].Set(Gf.Vec3f(0.0, 0.0, rot_z_deg))
    elif "xformOp:rotateZ" in ops_dict:
        ops_dict["xformOp:rotateZ"].Set(rot_z_deg)
    else:
        print("[setup_camera] WARNING: no rotation op found on camera prim — "
              "orientation not changed")

    print(f"[setup_camera] pos=({x}, {y}, {z})  rotZ={rot_z_deg}°")


def get_camera_world_pose():
    """
    Read the current world translate of the camera prim.  Returns (x, y, z)
    in metres.  Used when SET_CAMERA_POSE=False so that downstream
    back-projection uses the actual stage pose, not the config constants.
    """
    import omni.usd
    from pxr import UsdGeom

    stage    = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(CAMERA_PRIM_PATH)
    if not cam_prim.IsValid():
        raise RuntimeError(f"[camera] prim not found: {CAMERA_PRIM_PATH}")

    xformable = UsdGeom.Xformable(cam_prim)
    world_xf  = xformable.ComputeLocalToWorldTransform(0)
    t         = world_xf.ExtractTranslation()
    return float(t[0]), float(t[1]), float(t[2])


# ── CAPTURE ───────────────────────────────────────────────────────────────────

async def capture_rgb_depth():
    """
    Create a Replicator render product, attach rgb and distance_to_image_plane
    annotators, step the simulation, and return (rgb_hwc_uint8, depth_hw_float32).

    depth values are in metres (distance to image plane, not ray length).
    NaN / Inf are replaced with 0.
    """
    import omni.replicator.core as rep
    import numpy as np

    print(f"[capture] creating render product {IMG_W}x{IMG_H} on {CAMERA_PRIM_PATH}")
    rp = rep.create.render_product(CAMERA_PRIM_PATH, (IMG_W, IMG_H))

    depth_an = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_an   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_an.attach([rp])
    rgb_an.attach([rp])

    print(f"[capture] stepping simulation ({RT_SUBFRAMES} rt_subframes) ...")
    await rep.orchestrator.step_async(rt_subframes=RT_SUBFRAMES)

    raw_depth = depth_an.get_data()
    raw_rgb   = rgb_an.get_data()

    if raw_depth is None or raw_rgb is None:
        raise RuntimeError("[capture] Annotator returned None — check camera prim "
                           "and render product")

    # rgb: may be ndarray or dict{"data": flat array}
    print(f"[capture] rgb returned {type(raw_rgb)}")
    if isinstance(raw_rgb, dict):
        raw_rgb = np.asarray(raw_rgb["data"]).reshape(IMG_H, IMG_W, -1)
    print(f"[capture] rgb shape={raw_rgb.shape}")
    rgb = raw_rgb[:, :, :3]   # drop alpha if present; keep H×W×3

    # depth: may be ndarray or dict{"data": flat array}
    print(f"[capture] depth returned {type(raw_depth)}")
    if isinstance(raw_depth, dict):
        raw_depth = np.asarray(raw_depth["data"]).reshape(IMG_H, IMG_W)
    raw_depth = raw_depth.astype(np.float32)
    print(f"[capture] depth shape={raw_depth.shape}")
    depth = np.nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)

    valid_d = depth[depth > DEPTH_MIN_VALID]
    if valid_d.size > 0:
        print(f"[capture] depth valid range [{valid_d.min():.4f}, {valid_d.max():.4f}] m")
    else:
        print("[capture] WARNING: no valid depth pixels above DEPTH_MIN_VALID")

    return rgb, depth


# ── SURFACE ESTIMATION ────────────────────────────────────────────────────────

def compute_piece_roi(img_w: int, img_h: int) -> tuple:
    """
    Resolve the piece ROI based on config.  Returns (x1, y1, x2, y2, source)
    where source is one of {"explicit", "center_fraction", "full"}.

    The returned box uses pixel coords with x2/y2 EXCLUSIVE (slice convention).
    """
    if not PIECE_ROI_ENABLED:
        return 0, 0, img_w, img_h, "full"

    if (PIECE_ROI_X1 is not None and PIECE_ROI_Y1 is not None and
            PIECE_ROI_X2 is not None and PIECE_ROI_Y2 is not None):
        x1 = max(0, int(PIECE_ROI_X1));  y1 = max(0, int(PIECE_ROI_Y1))
        x2 = min(img_w, int(PIECE_ROI_X2));  y2 = min(img_h, int(PIECE_ROI_Y2))
        return x1, y1, x2, y2, "explicit"

    if PIECE_ROI_MODE == "center_fraction":
        f = max(0.05, min(1.0, float(PIECE_ROI_FRACTION)))
        rw = int(round(img_w * f));  rh = int(round(img_h * f))
        x1 = (img_w - rw) // 2;      y1 = (img_h - rh) // 2
        return x1, y1, x1 + rw, y1 + rh, "center_fraction"

    # Unknown mode → full image
    return 0, 0, img_w, img_h, "full"


def expand_roi(roi: tuple, expand_px: int, img_w: int, img_h: int) -> tuple:
    """Expand (x1, y1, x2, y2) by `expand_px` pixels on every side, clipped
    to the image."""
    x1, y1, x2, y2 = roi[:4]
    return (max(0, x1 - expand_px),
            max(0, y1 - expand_px),
            min(img_w, x2 + expand_px),
            min(img_h, y2 + expand_px))


def _extract_depth_peaks(valid_depths, hist_bin_m: float,
                          merge_distance_m: float):
    """
    Extract local maxima of the depth histogram, then merge peaks that fall
    within `merge_distance_m` of an already-accepted peak (keeping the larger).
    Returns a list of dicts sorted by depth ascending (closest first):
        {"depth_m", "count_px", "fraction"}
    """
    import numpy as np

    if valid_depths.size == 0:
        return []

    d_min = float(valid_depths.min())
    d_max = float(valid_depths.max()) + hist_bin_m
    bins  = np.arange(d_min, d_max + hist_bin_m, hist_bin_m)
    hist, edges = np.histogram(valid_depths, bins=bins)
    centres     = edges[:-1] + hist_bin_m / 2.0

    if hist.size < 3:
        # Trivially small histogram: treat the global max as the single peak.
        i = int(np.argmax(hist))
        return [{
            "depth_m":  float(centres[i]),
            "count_px": int(hist[i]),
            "fraction": float(hist[i]) / float(valid_depths.size),
        }]

    # Local maxima: bin > both neighbours (strict; ties broken by left).
    is_peak = np.zeros_like(hist, dtype=bool)
    is_peak[1:-1] = (hist[1:-1] > hist[:-2]) & (hist[1:-1] >= hist[2:])
    raw = sorted(
        [(float(centres[i]), int(hist[i])) for i in np.where(is_peak)[0]],
        key=lambda x: x[0],
    )

    # Merge peaks within merge_distance_m, keeping the larger count.
    merged = []
    for d, c in raw:
        if merged and (d - merged[-1][0]) <= merge_distance_m:
            if c > merged[-1][1]:
                merged[-1] = (d, c)
        else:
            merged.append((d, c))

    total = float(valid_depths.size)
    return [
        {"depth_m": d, "count_px": c, "fraction": c / total}
        for (d, c) in merged
    ]


def _select_support_peak(peaks, camera_z: float = None):
    """
    Pick the local support peak from a list of depth peaks (closest first).
    Returns (selected_peak_dict, rank_in_peaks_list, reason_str,
             is_farthest_major_peak_bool).

    Decision rules (deterministic, in order):
      1. If a peak is closer than another but smaller than PIECE_MAX_PEAK_FRACTION,
         skip it as the piece itself.
      2. The first peak (closest first) with fraction >= SUPPORT_MIN_PEAK_FRACTION
         that wasn't skipped is the support.
      3. If no peak qualifies, fall back to the largest-fraction peak (and
         flag the reason).
    """
    if not peaks:
        return None, -1, "no peaks found", False

    n = len(peaks)
    largest_idx = max(range(n), key=lambda i: peaks[i]["fraction"])

    selected_rank = None
    skipped_close_small = []
    for i, p in enumerate(peaks):
        if p["fraction"] >= SUPPORT_MIN_PEAK_FRACTION:
            selected_rank = i
            break
        if p["fraction"] <= PIECE_MAX_PEAK_FRACTION:
            skipped_close_small.append(i)
            continue
        # Mid-sized peak we can't confidently skip as piece — accept it.
        selected_rank = i
        break

    if selected_rank is None:
        selected_rank = largest_idx
        reason = (f"no peak >= SUPPORT_MIN_PEAK_FRACTION "
                  f"({SUPPORT_MIN_PEAK_FRACTION:.2f}); fell back to largest")
    else:
        if skipped_close_small:
            reason = (f"closest large peak (fraction "
                      f"{peaks[selected_rank]['fraction']:.3f}); "
                      f"skipped {len(skipped_close_small)} closer small peak(s) "
                      f"as likely piece")
        else:
            reason = (f"first peak with fraction >= SUPPORT_MIN_PEAK_FRACTION "
                      f"({SUPPORT_MIN_PEAK_FRACTION:.2f}); "
                      f"fraction={peaks[selected_rank]['fraction']:.3f}")

    is_farthest_major = (selected_rank == largest_idx) and (selected_rank == n - 1)
    return peaks[selected_rank], selected_rank, reason, is_farthest_major


def estimate_support_surface_depth(depth, roi: tuple = None):
    """
    Estimate the depth (distance to camera) of the LOCAL support surface inside
    the piece ROI.

    SURFACE_ESTIMATION_MODE controls the algorithm:
      "auto_depth_layers" (default): histogram → local-maxima peaks → choose
        the closest peak large enough to be a support plane (skipping closer
        small peaks that are probably the piece top, and avoiding a far-only
        background peak when a nearer large peak exists).
      "dominant_depth": legacy single-mode pick (retained for diagnostics).

    Returns (surface_z_m, info_dict).  The dict has:
      mode, peaks, selected_rank, reason, n_valid_pixels, is_farthest_major
    """
    import numpy as np

    if roi is not None:
        x1, y1, x2, y2 = roi[:4]
        depth_region = depth[y1:y2, x1:x2]
        print(f"[surface_est] using piece ROI for support surface estimation")
    else:
        depth_region = depth

    valid = depth_region[(depth_region >= SURFACE_DEPTH_MIN) &
                         (depth_region <= SURFACE_DEPTH_MAX)]
    if valid.size == 0:
        raise RuntimeError(
            f"[surface_est] No valid depth pixels in range "
            f"[{SURFACE_DEPTH_MIN}, {SURFACE_DEPTH_MAX}] m within the ROI. "
            f"Check camera Z, SURFACE_DEPTH_MIN/MAX, and PIECE_ROI_*.")

    print(f"[surface_est] mode={SURFACE_ESTIMATION_MODE}")

    # Always extract peaks for diagnostic logging, even in legacy mode.
    peaks = _extract_depth_peaks(valid, SURFACE_HIST_BIN_M,
                                  SURFACE_PEAK_MERGE_DISTANCE_M)

    print(f"[surface_est] {len(peaks)} depth peak(s) inside ROI:")
    for k, p in enumerate(peaks, start=1):
        print(f"[surface_est] peak {k}: depth={p['depth_m']:.4f} m  "
              f"count={p['count_px']} px  fraction={p['fraction']*100:.1f}%")

    if SURFACE_ESTIMATION_MODE == "auto_depth_layers":
        sel, sel_rank, reason, is_farthest = _select_support_peak(peaks)
        if sel is None:
            raise RuntimeError("[surface_est] auto_depth_layers found no peaks")
        surface_d = sel["depth_m"]
        print(f"[surface_est] selected support peak: rank={sel_rank+1}, "
              f"depth={surface_d:.4f} m, reason={reason}")
        if is_farthest:
            print("[WARNING] selected support appears to be the farthest "
                  "layer. This may be background, not the local support surface.")
    else:  # legacy "dominant_depth"
        if not peaks:
            raise RuntimeError("[surface_est] no peaks; cannot pick dominant")
        sel_rank = max(range(len(peaks)), key=lambda i: peaks[i]["fraction"])
        sel = peaks[sel_rank]
        surface_d = sel["depth_m"]
        reason = "legacy dominant_depth mode (largest histogram fraction)"
        is_farthest = (sel_rank == len(peaks) - 1)
        print(f"[surface_est] dominant depth = {surface_d:.4f} m  "
              f"({sel['fraction']*100:.1f}% of valid pixels)")

    if sel["fraction"] < 0.05:
        print("[surface_est] WARNING: selected-peak fraction < 5% — depth "
              "histogram is noisy; surface estimate may be unreliable. "
              "Inspect depth_vis.png.")

    # Bound-saturation warnings (kept from prior version).
    if (surface_d - SURFACE_DEPTH_MIN) < SURFACE_BOUND_WARN_M:
        print(f"[surface_est] WARNING: estimate {surface_d:.4f} m is within "
              f"{SURFACE_BOUND_WARN_M*1000:.0f} mm of SURFACE_DEPTH_MIN "
              f"({SURFACE_DEPTH_MIN}). Likely a near-field intruder is winning "
              f"the histogram; widen SURFACE_DEPTH_MIN or tighten the piece ROI.")
    if (SURFACE_DEPTH_MAX - surface_d) < SURFACE_BOUND_WARN_M:
        print(f"[surface_est] WARNING: estimate {surface_d:.4f} m is within "
              f"{SURFACE_BOUND_WARN_M*1000:.0f} mm of SURFACE_DEPTH_MAX "
              f"({SURFACE_DEPTH_MAX}). The true surface may be beyond the "
              f"search window; widen SURFACE_DEPTH_MAX.")

    # Scene-specific defensive guard for the current camera-at-z≈0.7 m setup.
    # Local board-top is expected near 0.62-0.63 m; if we land closer to the
    # 0.7 m background, the auto-layer estimator probably picked the wrong peak.
    if surface_d > 0.68:
        print(f"[WARNING] selected support depth is close to background. "
              f"Expected support around 0.62–0.63 m for current camera pose.")

    info = {
        "mode":              SURFACE_ESTIMATION_MODE,
        "peaks":             peaks,
        "selected_rank":     int(sel_rank),
        "reason":            reason,
        "n_valid_pixels":    int(valid.size),
        "is_farthest_major": bool(is_farthest),
    }
    return surface_d, info


# ── SEGMENTATION ──────────────────────────────────────────────────────────────

def segment_piece(depth, surface_z, roi: tuple = None):
    """
    Return a raw boolean mask of pixels that are above the support surface by
    more than SURFACE_TOLERANCE metres.  If `roi` is provided as
    (x1, y1, x2, y2), pixels outside the ROI are forced to False so that
    unrelated objects elsewhere in the frame cannot enter the connected-
    component selection.

    Pixels closer to the camera than DEPTH_MIN_VALID are ignored (near-field
    sensor noise in Isaac Sim).
    """
    import numpy as np

    threshold = surface_z - SURFACE_TOLERANCE
    mask = (depth > DEPTH_MIN_VALID) & (depth < threshold)

    if roi is not None:
        x1, y1, x2, y2 = roi[:4]
        roi_mask = np.zeros_like(mask, dtype=bool)
        roi_mask[y1:y2, x1:x2] = True
        before = int(mask.sum())
        mask = mask & roi_mask
        after = int(mask.sum())
        print(f"[segment] restricted to ROI [{x1}:{x2}, {y1}:{y2}]: "
              f"{before} → {after} pixels")

    n_pixels = int(mask.sum())
    print(f"[segment] surface_z={surface_z:.4f}m  "
          f"threshold={threshold:.4f}m (= surface_z - {SURFACE_TOLERANCE*1000:.1f}mm)  "
          f"pixels above surface: {n_pixels}")

    if n_pixels == 0:
        print("[segment] WARNING: zero pixels above surface. "
              "Possible causes: piece is flush with table, camera too high, "
              "SURFACE_TOLERANCE too small, depth units mismatch, "
              "or PIECE_ROI excludes the piece.")

    # Mask-area-vs-ROI ratio guard.  If the mask covers >50% of the segmentation
    # ROI, the support surface is almost certainly wrong (e.g. the entire local
    # support is being classified as "above surface" because the histogram
    # locked onto a deeper background plane).
    expanded_roi_area = 0
    if roi is not None:
        x1, y1, x2, y2 = roi[:4]
        expanded_roi_area = max(0, (x2 - x1) * (y2 - y1))
    if expanded_roi_area > 0:
        ratio = n_pixels / float(expanded_roi_area)
        print(f"[segment] raw_piece_mask area={n_pixels} px / "
              f"expanded ROI area={expanded_roi_area} px  → ratio={ratio*100:.1f}%")
        if ratio > 0.50:
            print("[WARNING] raw piece mask covers too much of ROI. "
                  "Support surface may be wrong.")

    return mask


# ── CONNECTED COMPONENTS ──────────────────────────────────────────────────────

def select_best_component(raw_mask):
    """
    Run connected-components analysis on raw_mask, filter blobs by area, and
    return (best_mask, sorted_blobs, selected_rank).

    Sorting is deterministic: area DESC, ties broken by centroid_x ASC.

    Selection is controlled by PIECE_SELECTION_MODE:
      "largest"           — rank 0 of the sorted list (largest area).
      "closest_to_center" — component whose centroid is nearest to (IMG_W/2, IMG_H/2).
      "manual_index"      — MANUAL_COMPONENT_INDEX in the sorted list.

    Returns (None, sorted_blobs, -1) if no valid blob is found.
    """
    import math as _math
    import numpy as np
    import cv2

    binary = (raw_mask.astype(np.uint8)) * 255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    blobs = []
    for i in range(1, n):   # 0 is background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if CC_MIN_AREA_PX <= area <= CC_MAX_AREA_PX:
            blobs.append({
                "label":    i,
                "area_px":  area,
                "centroid": (float(centroids[i][0]), float(centroids[i][1])),
                "bbox":     (int(stats[i, cv2.CC_STAT_LEFT]),
                             int(stats[i, cv2.CC_STAT_TOP]),
                             int(stats[i, cv2.CC_STAT_WIDTH]),
                             int(stats[i, cv2.CC_STAT_HEIGHT])),
            })

    # Deterministic sort: area DESC, ties broken by centroid_x ASC
    blobs.sort(key=lambda b: (-b["area_px"], b["centroid"][0]))

    print(f"[cc] total components (excl. background): {n - 1}  "
          f"valid [{CC_MIN_AREA_PX}–{CC_MAX_AREA_PX} px]: {len(blobs)}")

    if not blobs:
        print("[cc] WARNING: no valid blob found. "
              "Check CC_MIN_AREA_PX, SURFACE_TOLERANCE, and inspect "
              "raw_piece_mask.png.")
        return None, blobs, -1

    # ── Console listing ───────────────────────────────────────────────────────
    print(f"[select] mode = {PIECE_SELECTION_MODE}")
    print("[select] valid components (sorted area DESC, centroid_x ASC):")
    for k, b in enumerate(blobs):
        cx, cy = b["centroid"]
        bx, by, bw, bh = b["bbox"]
        print(f"  [{k}] area={b['area_px']}  "
              f"centroid=({cx:.1f}, {cy:.1f})  "
              f"bbox=({bx}, {by}, {bw}, {bh})")

    # ── Selection ─────────────────────────────────────────────────────────────
    img_cx = IMG_W / 2.0
    img_cy = IMG_H / 2.0

    if PIECE_SELECTION_MODE == "largest":
        selected_rank = 0
        reason = "largest area"

    elif PIECE_SELECTION_MODE == "closest_to_center":
        best_dist = float("inf")
        selected_rank = 0
        for k, b in enumerate(blobs):
            cx, cy = b["centroid"]
            dist = _math.hypot(cx - img_cx, cy - img_cy)
            if dist < best_dist:
                best_dist = dist
                selected_rank = k
        reason = f"closest to image centre (dist={best_dist:.1f}px)"

    elif PIECE_SELECTION_MODE == "manual_index":
        if MANUAL_COMPONENT_INDEX >= len(blobs):
            listing = "  ".join(
                f"[{k}] area={b['area_px']} centroid=({b['centroid'][0]:.1f}, "
                f"{b['centroid'][1]:.1f}) bbox={b['bbox']}"
                for k, b in enumerate(blobs)
            )
            raise IndexError(
                f"MANUAL_COMPONENT_INDEX={MANUAL_COMPONENT_INDEX} is out of range "
                f"for {len(blobs)} valid component(s). Valid entries:  {listing}"
            )
        selected_rank = MANUAL_COMPONENT_INDEX
        reason = f"MANUAL_COMPONENT_INDEX={MANUAL_COMPONENT_INDEX}"

    else:
        raise ValueError(
            f"Unknown PIECE_SELECTION_MODE={PIECE_SELECTION_MODE!r}. "
            f"Supported: 'largest', 'closest_to_center', 'manual_index'."
        )

    best = blobs[selected_rank]
    best_mask = (labels == best["label"])
    print(f"[select] picked rank={selected_rank}  reason={reason}")

    return best_mask, blobs, selected_rank


# ── CAMERA INTRINSICS ─────────────────────────────────────────────────────────

def compute_intrinsics(cam_z: float):
    """
    Compute pixel-space intrinsics from the camera configuration constants.

    Returns a dict with: fx, fy, cx_px, cy_px, mpp_x, mpp_y
    (metres-per-pixel at distance cam_z, and principal point in pixels).

    Note: Isaac Sim's distance_to_image_plane annotator measures the
    perpendicular distance, so pinhole backprojection is correct.
    """
    fov_h = 2.0 * math.atan((APERTURE_MM / 2.0) / FOCAL_MM)
    fov_v = fov_h * (IMG_H / IMG_W)
    mpp_x = (2.0 * cam_z * math.tan(fov_h / 2.0)) / IMG_W
    mpp_y = (2.0 * cam_z * math.tan(fov_v / 2.0)) / IMG_H
    fx    = cam_z / mpp_x   # pixels per metre at distance cam_z → focal length
    fy    = cam_z / mpp_y
    return {
        "fx":    fx,
        "fy":    fy,
        "cx_px": IMG_W / 2.0,
        "cy_px": IMG_H / 2.0,
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "cam_z": cam_z,
    }


# ── DEPTH TO POINT CLOUD ─────────────────────────────────────────────────────

def depth_to_pointcloud(depth, mask, intrinsics, surface_z, cam_xy: tuple,
                         n_samples=N_POINTS):
    """
    Back-project masked depth pixels to 3D world points.

    Coordinate convention (output):
      X, Y: world-plane position centred on the piece centroid (metres)
      Z:    height above the support surface (metres, always >= 0)

    Real-world scale is preserved. Points are only centred, not unit-scaled.

    depth:      H×W float32, metres
    mask:       H×W bool
    intrinsics: dict from compute_intrinsics()
    surface_z:  estimated surface depth (metres, in camera space)
    n_samples:  target point count; padded with replacement if needed

    Returns Nx3 float32 array, or raises if mask is empty.
    """
    import numpy as np

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("[pointcloud] mask is empty — cannot build point cloud")

    mpp_x  = intrinsics["mpp_x"]
    mpp_y  = intrinsics["mpp_y"]
    cx_px  = intrinsics["cx_px"]
    cy_px  = intrinsics["cy_px"]
    cam_z  = intrinsics["cam_z"]

    # Back-project to world XY using the pinhole model.
    # Camera X/Y world position is cam_xy (the actually-active stage pose);
    # pixel offset scales by mpp.
    cam_x, cam_y = cam_xy
    world_x = cam_x + (xs.astype(np.float64) - cx_px) * mpp_x
    world_y = cam_y - (ys.astype(np.float64) - cy_px) * mpp_y  # image V flips Y

    # Z = height of piece top above support surface
    # depth[y,x] is distance to camera; surface is at surface_z from camera.
    # Points closer to camera → smaller depth value → larger height.
    world_z = surface_z - depth[ys, xs].astype(np.float64)
    world_z = np.clip(world_z, 0.0, None)   # clip numerical noise below surface

    # Centre XY around piece centroid
    cx_obj = float(world_x.mean())
    cy_obj = float(world_y.mean())
    world_x -= cx_obj
    world_y -= cy_obj

    points = np.stack([world_x, world_y, world_z], axis=1).astype(np.float32)

    # Sample / pad to exactly n_samples
    n_raw = len(points)
    replace = n_raw < n_samples
    rng = np.random.default_rng(0)
    idx = rng.choice(n_raw, size=n_samples, replace=replace)
    points = points[idx]

    print(f"[pointcloud] raw pixels={n_raw}  sampled={n_samples}  "
          f"(replace={replace})")
    print(f"  X=[{points[:,0].min():.4f}, {points[:,0].max():.4f}] m")
    print(f"  Y=[{points[:,1].min():.4f}, {points[:,1].max():.4f}] m")
    print(f"  Z=[{points[:,2].min():.4f}, {points[:,2].max():.4f}] m  "
          f"(height above surface)")
    print(f"  centroid_world=({cx_obj:.4f}, {cy_obj:.4f})")

    return points, (cx_obj, cy_obj)


# ── FOOTPRINT IMAGE ───────────────────────────────────────────────────────────

def make_footprint_image(points, resolution_m=0.0005, canvas_px=256):
    """
    Project the Nx3 point cloud onto the XY plane (top-down view) and render
    as a grayscale image where intensity encodes point density.

    resolution_m: world metres per image pixel (smaller = more detail, bigger
                  canvas needed).  Default 0.5 mm/px.
    canvas_px:    output image size in pixels (square).

    Returns a 3-channel uint8 BGR image suitable for cv2.imwrite.

    The image is centred on the piece centroid (points are already centred).
    """
    import numpy as np
    import cv2

    half_world = (canvas_px / 2.0) * resolution_m

    # Map world XY → pixel UV
    u = ((points[:, 0] + half_world) / resolution_m).astype(np.int32)
    v = ((half_world - points[:, 1]) / resolution_m).astype(np.int32)   # Y flips

    # Clip to canvas
    valid = (u >= 0) & (u < canvas_px) & (v >= 0) & (v < canvas_px)
    u = u[valid]
    v = v[valid]

    canvas = np.zeros((canvas_px, canvas_px), dtype=np.uint8)
    for uu, vv in zip(u, v):
        canvas[vv, uu] = min(255, int(canvas[vv, uu]) + 8)

    # Normalise and apply colourmap for legibility
    if canvas.max() > 0:
        vis = cv2.normalize(canvas, None, 0, 255, cv2.NORM_MINMAX)
    else:
        vis = canvas

    footprint_bgr = cv2.applyColorMap(vis.astype(np.uint8), cv2.COLORMAP_HOT)

    # Draw crosshair at centre (= piece centroid)
    mid = canvas_px // 2
    cv2.line(footprint_bgr, (mid - 10, mid), (mid + 10, mid), (0, 255, 0), 1)
    cv2.line(footprint_bgr, (mid, mid - 10), (mid, mid + 10), (0, 255, 0), 1)

    print(f"[footprint] canvas={canvas_px}px  resolution={resolution_m*1000:.1f}mm/px  "
          f"world span={half_world*2*100:.1f}cm  "
          f"projected {valid.sum()}/{len(points)} points")

    if valid.sum() < int(len(points) * 0.5):
        print(f"[footprint] WARNING: fewer than 50% of points fit in the canvas. "
              f"Consider increasing canvas_px or resolution_m.")

    return footprint_bgr


# ── DEBUG OUTPUTS ─────────────────────────────────────────────────────────────

def save_depth_layers_debug(out_dir, rgb, depth, roi: tuple,
                             expanded_roi: tuple, surface_info: dict,
                             surface_z: float):
    """Render depth_layers_debug.png: RGB with the piece ROI (cyan), the
    expanded segmentation ROI (yellow), and a text panel listing all detected
    depth peaks plus the selected support depth.  Helps the operator see at a
    glance whether the local support, not the background, was selected."""
    if rgb is None or roi is None:
        return
    import cv2
    import numpy as np

    debug = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    h, w  = debug.shape[:2]

    x1, y1, x2, y2 = roi[:4]
    cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 0), 2)   # cyan
    if expanded_roi is not None:
        ex1, ey1, ex2, ey2 = expanded_roi
        cv2.rectangle(debug, (ex1, ey1), (ex2 - 1, ey2 - 1), (0, 255, 255), 1)

    # Side panel for textual info — drawn over a translucent dark strip so
    # it's readable on any background.
    panel_h = 22 + 18 * (len(surface_info.get("peaks", [])) + 4)
    panel_h = min(panel_h, h - 10)
    panel_w = 360
    overlay = debug.copy()
    cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h),
                  (0, 0, 0), thickness=cv2.FILLED)
    cv2.addWeighted(overlay, 0.55, debug, 0.45, 0, dst=debug)

    def _put(line_idx, text, colour=(255, 255, 255)):
        y = 22 + 18 * line_idx
        cv2.putText(debug, text, (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)

    _put(0, f"mode = {surface_info.get('mode', 'unknown')}")
    _put(1, f"selected support depth = {surface_z:.4f} m")
    _put(2, f"reason: {surface_info.get('reason', '')[:46]}")
    _put(3, "depth peaks (closest first):")
    for k, p in enumerate(surface_info.get("peaks", []), start=1):
        _put(3 + k,
             f"  {k}: d={p['depth_m']:.4f} m  frac={p['fraction']*100:.1f}%  "
             f"cnt={p['count_px']}",
             colour=(0, 255, 0)
             if k - 1 == surface_info.get("selected_rank", -1)
             else (255, 255, 255))

    cv2.imwrite(str(out_dir / "depth_layers_debug.png"), debug)
    print(f"[save] depth_layers_debug.png written")


def save_piece_roi_debug(out_dir, rgb, roi: tuple, expanded_roi: tuple = None):
    """Render piece_roi_debug.png: RGB with the ROI rectangle (cyan) and,
    if RESTRICT_PIECE_MASK_TO_ROI is on, the expanded segmentation ROI
    (yellow, dashed)."""
    if rgb is None or roi is None:
        return
    import cv2
    import numpy as np

    debug = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    x1, y1, x2, y2 = roi[:4]
    source = roi[4] if len(roi) > 4 else "unknown"
    cv2.rectangle(debug, (x1, y1), (x2 - 1, y2 - 1), (255, 255, 0), 2)   # cyan
    cv2.putText(debug, f"piece ROI ({source})", (x1, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1, cv2.LINE_AA)

    if expanded_roi is not None:
        ex1, ey1, ex2, ey2 = expanded_roi
        cv2.rectangle(debug, (ex1, ey1), (ex2 - 1, ey2 - 1), (0, 255, 255), 1)
        cv2.putText(debug, "segmentation ROI (expanded)",
                    (ex1, min(rgb.shape[0] - 5, ey2 + 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_dir / "piece_roi_debug.png"), debug)
    print(f"[save] piece_roi_debug.png written")


def save_debug_outputs(out_dir, rgb, depth, raw_mask, best_mask,
                       footprint_bgr, points, blob_stats, centroid_world,
                       surface_z):
    """
    Save all required debug artifacts to out_dir.  Every file path is printed
    so you can docker-cp them out of the container immediately.
    """
    import numpy as np
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. RGB
    rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(out_dir / "rgb.png"), rgb_bgr)

    # 2. Depth visualisation (linear colour map; 0 = black)
    valid_d = depth[depth > DEPTH_MIN_VALID]
    if valid_d.size > 0:
        d_min, d_max = float(valid_d.min()), float(valid_d.max())
    else:
        d_min, d_max = 0.0, 1.0
    d_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0.0, 1.0)
    depth_vis = cv2.applyColorMap((d_norm * 255).astype(np.uint8),
                                  cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(str(out_dir / "depth_vis.png"), depth_vis)

    # 3. Raw piece mask (before connected-components selection)
    cv2.imwrite(str(out_dir / "raw_piece_mask.png"),
                (raw_mask.astype(np.uint8) * 255))

    # 4. Final piece mask
    piece_mask_img = (best_mask.astype(np.uint8) * 255) if best_mask is not None \
        else np.zeros_like(raw_mask, dtype=np.uint8)
    cv2.imwrite(str(out_dir / "piece_mask.png"), piece_mask_img)

    # 5. Debug overlay: tint piece mask on RGB, draw centroid
    debug = rgb.copy()
    if best_mask is not None:
        tint_colour = np.array([255, 60, 60], dtype=np.float32)
        debug[best_mask] = (debug[best_mask].astype(np.float32) * 0.3
                            + tint_colour * 0.7).astype(np.uint8)
        if blob_stats:
            cx_px = int(blob_stats[0]["centroid"][0])
            cy_px = int(blob_stats[0]["centroid"][1])
            cv2.circle(debug, (cx_px, cy_px), 6, (255, 255, 0), -1)
            bx, by, bw, bh = blob_stats[0]["bbox"]
            cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
            label = f"({cx_px},{cy_px}) {blob_stats[0]['area_px']}px"
            cv2.putText(debug, label, (bx, max(by - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
    cv2.imwrite(str(out_dir / "piece_debug.png"),
                cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))

    # 6. Footprint
    cv2.imwrite(str(out_dir / "piece_footprint.png"), footprint_bgr)

    # 7. Point cloud
    np.save(str(out_dir / "piece_pointcloud.npy"), points)

    # Print docker-cp helpers
    print("\n[save] All outputs written to:", out_dir)
    for f in sorted(out_dir.iterdir()):
        print(f"  {f}")
    print("\n[save] To copy out of container:")
    for name in ["rgb.png", "depth_vis.png", "raw_piece_mask.png",
                 "piece_mask.png", "piece_debug.png", "piece_footprint.png"]:
        print(f"  docker cp <container>:{out_dir}/{name} ./")


def save_metadata(out_dir, success, best_mask, blob_stats, points,
                  centroid_world, surface_z, n_valid_components=0,
                  selected_rank=-1, error_msg=None, camera_pose: dict = None,
                  piece_roi: tuple = None, piece_roi_expanded: tuple = None,
                  surface_info: dict = None, raw_piece_mask_area: int = 0):
    """
    Write piece_metadata.json conforming to the experiments.md conventions.

    n_valid_components: count of blobs that passed the area filter
                        (always >= 1 when success=True; used to flag ambiguity).
    selected_rank:      index of the selected blob in the deterministic sort order
                        (area DESC, ties broken by centroid_x ASC).
    """
    import numpy as np

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    # ── Selected component block ───────────────────────────────────────────────
    selected_component = {}
    pointcloud_bounds  = {}
    if success and best_mask is not None and blob_stats:
        b = blob_stats[0]
        selected_component = {
            "area_px":          b["area_px"],
            "centroid_px":      list(b["centroid"]),
            "bbox_px":          list(b["bbox"]),   # (x, y, w, h)
            "centroid_world_m": list(centroid_world),
            "surface_depth_m":  float(surface_z),
            "point_count":      int(len(points)),
            "height_range_m":   [float(points[:, 2].min()),
                                 float(points[:, 2].max())],
            "xy_span_m":        [float(points[:, 0].max() - points[:, 0].min()),
                                 float(points[:, 1].max() - points[:, 1].min())],
        }
        pointcloud_bounds = {
            "x_min": float(points[:, 0].min()),
            "x_max": float(points[:, 0].max()),
            "y_min": float(points[:, 1].min()),
            "y_max": float(points[:, 1].max()),
            "z_min": float(points[:, 2].min()),
            "z_max": float(points[:, 2].max()),
        }

    metadata = {
        "script":             "capture_piece_detection.py",
        "timestamp":          ts,
        "project_root":       str(PROJECT_ROOT),
        "capture_name":       CAPTURE_NAME,
        "use_capture_subdir": USE_CAPTURE_SUBDIR,
        "output_dir":         str(out_dir),
        "visible_piece_assumption": "single visible piece",
        "camera_pose": camera_pose if camera_pose is not None else {
            "x": None, "y": None, "z": None, "rot_z_deg": None,
            "source": "unknown",
        },
        "camera_pose_source":     (camera_pose or {}).get("source", "unknown"),
        "camera_pose_overridden": bool(SET_CAMERA_POSE),
        "set_camera_pose":        bool(SET_CAMERA_POSE),
        "piece_roi_enabled":           bool(PIECE_ROI_ENABLED),
        "piece_roi_mode":              PIECE_ROI_MODE,
        "piece_roi_fraction":          float(PIECE_ROI_FRACTION),
        "piece_roi_px": (
            {"x1": piece_roi[0], "y1": piece_roi[1],
             "x2": piece_roi[2], "y2": piece_roi[3],
             "source": piece_roi[4] if len(piece_roi) > 4 else None}
            if piece_roi is not None else None
        ),
        "restrict_piece_mask_to_roi":  bool(RESTRICT_PIECE_MASK_TO_ROI),
        "piece_mask_roi_expand_px":    int(PIECE_MASK_ROI_EXPAND_PX),
        "expanded_piece_roi_px": (
            {"x1": piece_roi_expanded[0], "y1": piece_roi_expanded[1],
             "x2": piece_roi_expanded[2], "y2": piece_roi_expanded[3]}
            if piece_roi_expanded is not None else None
        ),
        "image_resolution": {
            "width":  IMG_W,
            "height": IMG_H,
        },
        "surface_depth_m":           float(surface_z),
        "surface_estimation_mode":   (surface_info or {}).get("mode", SURFACE_ESTIMATION_MODE),
        "depth_peaks":               [
            {"depth_m": p["depth_m"], "count_px": p["count_px"],
             "fraction": p["fraction"]}
            for p in (surface_info or {}).get("peaks", [])
        ],
        "selected_support_peak_rank": (surface_info or {}).get("selected_rank", -1),
        "selected_support_depth_m":   float(surface_z),
        "selected_support_reason":    (surface_info or {}).get("reason", ""),
        "selected_support_is_farthest_major":
            bool((surface_info or {}).get("is_farthest_major", False)),
        "raw_piece_mask_area":       int(raw_piece_mask_area),
        "expanded_roi_area":         int(
            (piece_roi_expanded[2] - piece_roi_expanded[0]) *
            (piece_roi_expanded[3] - piece_roi_expanded[1])
            if piece_roi_expanded is not None else 0),
        "raw_piece_mask_area_ratio": float(
            raw_piece_mask_area /
            ((piece_roi_expanded[2] - piece_roi_expanded[0]) *
             (piece_roi_expanded[3] - piece_roi_expanded[1]))
            if piece_roi_expanded is not None and raw_piece_mask_area > 0 else 0.0),
        "n_valid_components":        n_valid_components,
        "multiple_valid_components": n_valid_components > 1,
        "piece_selection_mode":      PIECE_SELECTION_MODE,
        "manual_component_index":    MANUAL_COMPONENT_INDEX,
        "selected_component_rank_or_index": selected_rank,
        "all_valid_components": [
            {
                "area_px":     b["area_px"],
                "centroid_px": {"x": b["centroid"][0], "y": b["centroid"][1]},
                "bbox_px":     {"x": b["bbox"][0], "y": b["bbox"][1],
                                "w": b["bbox"][2], "h": b["bbox"][3]},
            }
            for b in blob_stats
        ],
        "selected_component":        selected_component,
        "pointcloud_bounds":         pointcloud_bounds,
        "parameters": {
            "camera_prim":         CAMERA_PRIM_PATH,
            "cam_xyz":             [CAM_X, CAM_Y, CAM_Z],
            "cam_rot_z_deg":       CAM_ROT_Z_DEG,
            "resolution":          [IMG_W, IMG_H],
            "focal_mm":            FOCAL_MM,
            "aperture_mm":         APERTURE_MM,
            "surface_depth_range": [SURFACE_DEPTH_MIN, SURFACE_DEPTH_MAX],
            "surface_tolerance_m": SURFACE_TOLERANCE,
            "cc_min_area_px":      CC_MIN_AREA_PX,
            "cc_max_area_px":      CC_MAX_AREA_PX,
            "n_points":            N_POINTS,
        },
        "input_files":  [],
        "output_files": [
            "rgb.png", "depth_vis.png", "raw_piece_mask.png",
            "piece_mask.png", "piece_debug.png", "piece_footprint.png",
            "piece_pointcloud.npy", "piece_metadata.json",
        ],
        "success": success,
        "error":   error_msg,
    }

    meta_path = out_dir / "piece_metadata.json"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(str(meta_path), "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"[metadata] saved → {meta_path}")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

async def main():
    import numpy as np

    print("=" * 60)
    print("capture_piece_detection.py — Phase 1")
    print("=" * 60)
    print(f"[config] SURFACE_ESTIMATION_MODE={SURFACE_ESTIMATION_MODE}")
    print(f"[main] capture_name = {CAPTURE_NAME!r}  "
          f"use_capture_subdir = {USE_CAPTURE_SUBDIR}")
    print(f"[main] output_dir   = {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove stale outputs from the previous run so a failed run never leaves
    # old images sitting around that could be mistaken for current results.
    _stale_files = [
        "rgb.png", "depth_vis.png", "raw_piece_mask.png",
        "piece_mask.png", "piece_debug.png", "piece_footprint.png",
        "piece_pointcloud.npy", "piece_metadata.json",
        "piece_roi_debug.png", "depth_layers_debug.png",
    ]
    for _fname in _stale_files:
        _p = OUT_DIR / _fname
        if _p.exists():
            _p.unlink()
            print(f"[main] removed stale: {_p.name}")

    error_msg      = None
    success        = False
    best_mask      = None
    blob_stats     = []
    selected_rank  = -1
    points         = np.zeros((N_POINTS, 3), dtype=np.float32)
    centroid_w     = (0.0, 0.0)
    surface_z      = 0.0
    rgb            = None
    depth          = None
    raw_mask       = None
    footprint_bgr  = None
    active_camera_pose = None    # populated after Step 1; recorded in metadata
    active_cam_xy      = (0.0, 0.0)
    piece_roi          = None    # (x1, y1, x2, y2, source)
    piece_roi_expanded = None    # ROI used for mask restriction
    surface_info       = None    # info dict from estimate_support_surface_depth

    try:
        # ── Step 1: Camera setup ──────────────────────────────────────────────
        print("\n--- Step 1: Camera setup ---")

        # Defensive warning if the configured override pose looks suspicious
        # (e.g. clearly out of the reachable workspace).  Does not block.
        if SET_CAMERA_POSE and (abs(CAM_X) > 5.0 or abs(CAM_Y) > 5.0 or
                                CAM_Z <= 0.0 or CAM_Z > 5.0):
            print(f"[camera] WARNING: configured pose (CAM_X={CAM_X}, "
                  f"CAM_Y={CAM_Y}, CAM_Z={CAM_Z}) is outside a typical "
                  f"workspace; double-check before relying on this run.")

        if SET_CAMERA_POSE:
            print("[camera] overriding stage camera pose")
            setup_camera(CAM_X, CAM_Y, CAM_Z, CAM_ROT_Z_DEG)
        else:
            print("[camera] using existing stage camera pose")

        cam_x, cam_y, cam_z = get_camera_world_pose()
        print(f"[camera] active world pos = ({cam_x:.4f}, {cam_y:.4f}, {cam_z:.4f}) m")
        active_cam_xy = (cam_x, cam_y)
        active_camera_pose = {
            "x": cam_x, "y": cam_y, "z": cam_z,
            "rot_z_deg": CAM_ROT_Z_DEG if SET_CAMERA_POSE else None,
            "source": "config_override" if SET_CAMERA_POSE else "stage",
        }

        # ── Step 2: Capture ───────────────────────────────────────────────────
        print("\n--- Step 2: Capture RGB + depth ---")
        rgb, depth = await capture_rgb_depth()

        # ── Step 3a: Resolve piece ROI ───────────────────────────────────────
        x1, y1, x2, y2, roi_source = compute_piece_roi(IMG_W, IMG_H)
        piece_roi = (x1, y1, x2, y2, roi_source)
        print(f"[piece_roi] enabled={PIECE_ROI_ENABLED}  source={roi_source}")
        print(f"[piece_roi] x=[{x1}:{x2}] y=[{y1}:{y2}]")

        if RESTRICT_PIECE_MASK_TO_ROI:
            ex1, ey1, ex2, ey2 = expand_roi(piece_roi,
                                            PIECE_MASK_ROI_EXPAND_PX,
                                            IMG_W, IMG_H)
            piece_roi_expanded = (ex1, ey1, ex2, ey2)
            print(f"[piece_roi] segmentation ROI (expanded by "
                  f"{PIECE_MASK_ROI_EXPAND_PX} px) = [{ex1}:{ex2}, {ey1}:{ey2}]")

        # ── Step 3b: Surface estimation ──────────────────────────────────────
        print("\n--- Step 3: Estimate support surface depth ---")
        surface_estimation_roi = (x1, y1, x2, y2) if PIECE_ROI_ENABLED else None
        surface_z, surface_info = estimate_support_surface_depth(
            depth, roi=surface_estimation_roi)

        # ── Step 4: Segmentation ──────────────────────────────────────────────
        print("\n--- Step 4: Segment piece ---")
        seg_roi = piece_roi_expanded if RESTRICT_PIECE_MASK_TO_ROI else None
        raw_mask = segment_piece(depth, surface_z, roi=seg_roi)

        # ── Step 5: Connected components ──────────────────────────────────────
        print("\n--- Step 5: Select best component ---")
        best_mask, blob_stats, selected_rank = select_best_component(raw_mask)

        print(f"[main] n_valid_components = {len(blob_stats)}")
        if len(blob_stats) != 1:
            print(f"[main] WARNING: expected exactly 1 visible piece for "
                  f"capture_name={CAPTURE_NAME!r}, found {len(blob_stats)}. "
                  f"Hide the other pieces in Isaac Sim before capturing.")

        if best_mask is None:
            raise RuntimeError(
                "No valid piece component found. "
                "Inspect raw_piece_mask.png and depth_vis.png. "
                "Typical causes: piece flush with table, wrong CAM_Z, "
                "SURFACE_TOLERANCE too tight, or multiple pieces all filtered out.")

        # ── Step 6: Point cloud ───────────────────────────────────────────────
        print("\n--- Step 6: Build point cloud ---")
        intrinsics = compute_intrinsics(surface_z)
        points, centroid_w = depth_to_pointcloud(
            depth, best_mask, intrinsics, surface_z,
            cam_xy=active_cam_xy, n_samples=N_POINTS)

        # ── Step 7: Footprint ─────────────────────────────────────────────────
        print("\n--- Step 7: Generate 2D footprint ---")
        footprint_bgr = make_footprint_image(points)

        success = True
        print("\n[main] Pipeline completed successfully.")

    except Exception as exc:
        error_msg = str(exc)
        print(f"\n[ERROR] {exc}")
        import traceback
        traceback.print_exc()

    finally:
        # ── Step 8: Save outputs ──────────────────────────────────────────────
        print("\n--- Step 8: Save debug outputs ---")

        # Always write the ROI debug image whenever RGB is available, on both
        # success and failure paths.  Helps diagnosing ROI mis-placement.
        if rgb is not None and piece_roi is not None:
            OUT_DIR.mkdir(parents=True, exist_ok=True)
            save_piece_roi_debug(OUT_DIR, rgb, piece_roi, piece_roi_expanded)

        # Depth-layers debug: written whenever we have a surface_info dict,
        # so failures caused by wrong support estimation are auditable.
        if rgb is not None and depth is not None and surface_info is not None:
            save_depth_layers_debug(OUT_DIR, rgb, depth, piece_roi,
                                    piece_roi_expanded, surface_info,
                                    surface_z)

        if success:
            # All intermediates are guaranteed non-None when success=True
            save_debug_outputs(OUT_DIR, rgb, depth, raw_mask, best_mask,
                               footprint_bgr, points, blob_stats,
                               centroid_w, surface_z)
        else:
            # On failure: only save the files we actually have; never write
            # zero-filled placeholders that look like real captures.
            import numpy as np
            import cv2
            OUT_DIR.mkdir(parents=True, exist_ok=True)

            if rgb is not None and depth is not None:
                rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                cv2.imwrite(str(OUT_DIR / "rgb.png"), rgb_bgr)
                valid_d = depth[depth > DEPTH_MIN_VALID]
                d_min = float(valid_d.min()) if valid_d.size > 0 else 0.0
                d_max = float(valid_d.max()) if valid_d.size > 0 else 1.0
                d_norm = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0.0, 1.0)
                depth_vis = cv2.applyColorMap((d_norm * 255).astype(np.uint8),
                                             cv2.COLORMAP_VIRIDIS)
                cv2.imwrite(str(OUT_DIR / "depth_vis.png"), depth_vis)
                print("[save] rgb.png and depth_vis.png written (capture succeeded)")

            if raw_mask is not None:
                cv2.imwrite(str(OUT_DIR / "raw_piece_mask.png"),
                            (raw_mask.astype(np.uint8) * 255))
                print("[save] raw_piece_mask.png written")

            print(f"[save] Skipping piece_mask, piece_debug, piece_footprint, "
                  f"piece_pointcloud — not produced before failure.")

        # Metadata is always written (success or failure).
        # len(blob_stats) is the count of components that passed the area
        # filter; this is the n_valid_components value.
        save_metadata(OUT_DIR, success, best_mask, blob_stats, points,
                      centroid_w, surface_z,
                      n_valid_components=len(blob_stats),
                      selected_rank=selected_rank,
                      error_msg=error_msg,
                      camera_pose=active_camera_pose,
                      piece_roi=piece_roi,
                      piece_roi_expanded=piece_roi_expanded,
                      surface_info=surface_info,
                      raw_piece_mask_area=int(raw_mask.sum())
                      if raw_mask is not None else 0)

        print("\n" + "=" * 60)
        print(f"  success={success}")
        if not success:
            print(f"  error:  {error_msg}")
        print("=" * 60)


asyncio.ensure_future(main())
