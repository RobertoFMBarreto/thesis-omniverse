"""
capture_cavity_detection.py — Phase 2

Detect geometric cavities in the shape-sorting board from an RGB-D capture.
Cavities are negative geometry: holes in the board top surface.  They appear
as regions where the measured depth is GREATER than the board top surface
depth (the camera is farther from those pixels because the surface dips away).

Outputs per-cavity:
  - binary mask
  - 3-D point cloud (real metric scale, XY centred on cavity centroid)
  - 2-D top-down footprint image
  - per-cavity metadata JSON

Plus global summary outputs:
  - rgb.png, depth_vis.png, raw_cavity_mask.png, cavities_debug.png
  - cavities_summary.json

Run inside Isaac Sim 5.1 Script Editor.

NOTE: __file__ is unreliable when pasted into the Script Editor — it resolves
to a temporary path such as /tmp/carb.../script_*.py.  PROJECT_ROOT is
therefore set explicitly (with an env-var escape hatch for developer machines).
"""

import asyncio
import math
import json
import os
import shutil
import time
import traceback
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

# Project root.  Override with the SHAPE_INSERTION_PROJECT_ROOT environment
# variable if running on a machine with a different layout (e.g. a Mac dev
# workflow that mounts the repo at a different path).
PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        "/workspace/Tese_Roberto/shape_insertion/thesis-omniverse",
    )
)

# All outputs land here.  Layout mirrors the piece script's pattern.
OUT_DIR = PROJECT_ROOT / "data" / "cavities_detected"

# Camera USD prim path in the stage.
# (The piece script calls this CAMERA_PRIM_PATH; here we use CAMERA_PATH as
# requested.  Both refer to the same concept.)
CAMERA_PATH = "/World/Camera"

# Camera pose override.
#
# The cavity board is captured by a different camera placement than the
# piece-table workflow.  By default we DO NOT move the camera: the script
# uses whatever pose is already authored on the stage.  Set
# SET_CAMERA_POSE = True to programmatically override the camera pose
# using the (CAM_X, CAM_Y, CAM_Z, CAM_ROT_Z_DEG) constants below.
#
# The defaults below correspond to the cavity-capture configuration —
# they are NOT the piece-table pose (which was ~(-0.25, 0.45, 0.58)).
SET_CAMERA_POSE = False
CAM_X         =  0.2885
CAM_Y         =  0.0020
CAM_Z         =  1.00    # height above world origin (metres)
CAM_ROT_Z_DEG = -90.0    # rotation around world Z (degrees)

# Render resolution
IMAGE_WIDTH  = 640
IMAGE_HEIGHT = 480

# Replicator subframes (higher = more stable; slower)
RT_SUBFRAMES = 8

# Camera intrinsics — must match the Isaac Sim camera prim settings.
# Horizontal FOV is derived from focal_mm / aperture_mm.
FOCAL_MM    = 24.0
APERTURE_MM = 36.0

# ── Board ROI for surface estimation ─────────────────────────────────────────
# If BOARD_ROI_ENABLED=True, the surface depth histogram uses only a centred
# fraction of the image (BOARD_ROI_FRACTION) instead of the full frame.
# This is OPTIONAL.  Disable when the board fills most of the view.
# Enable when floor/bench at the image edges contaminates the histogram peak.
BOARD_ROI_ENABLED  = False
BOARD_ROI_FRACTION = 0.6   # fraction of each dimension kept (centred)

# ── Surface / cavity depth thresholds ────────────────────────────────────────
# Bracket the board top surface.  Pixels outside this range are ignored when
# computing the dominant surface depth.
SURFACE_DEPTH_MIN = 0.10   # metres
SURFACE_DEPTH_MAX = 0.50   # metres

SURFACE_HIST_BIN = 0.001   # 1 mm histogram bins

# Cavity segmentation: a pixel belongs to a cavity if its depth is deeper than
# the board surface by at least CAVITY_DEPTH_MARGIN and at most MAX_CAVITY_DEPTH.
#
# CAVITY_DEPTH_MARGIN — eliminates board surface noise.  Too small: board
#   texture bleeds in.  Too large: shallow cavities disappear.  Start at 3 mm.
#
# MAX_CAVITY_DEPTH — eliminates holes through the board and floor reflections.
#   30 mm is a safe ceiling for typical shape-sorting toys; raise if cavities
#   are physically deeper.
CAVITY_DEPTH_MARGIN = 0.003   # 3 mm
MAX_CAVITY_DEPTH    = 0.030   # 30 mm

# ── Connected-component filters ───────────────────────────────────────────────
CC_MIN_AREA_PX =   200   # discard blobs smaller than this (noise)
CC_MAX_AREA_PX = 30000   # discard blobs suspiciously large (board bleed)

# ── Point cloud ───────────────────────────────────────────────────────────────
N_POINTS = 2048   # target points per cavity (padded/sampled)

# ── Footprint rendering ───────────────────────────────────────────────────────
FOOTPRINT_RESOLUTION_M_PER_PX = 0.0005   # 0.5 mm / pixel
FOOTPRINT_CANVAS_PX           = 256      # square canvas side length (pixels)

# ── Cavity sort order ─────────────────────────────────────────────────────────
# Cavities are sorted top-to-bottom then left-to-right by centroid pixel
# coordinates so that cavity_00, cavity_01, ... map consistently across runs
# as long as the camera does not move.
#
# Algorithm: bin centroid_y into rows of ROW_BIN_PX pixels (loose, to tolerate
# small Y jitter between runs), then sort by (row_bin, centroid_x).
ROW_BIN_PX = 30

# ── END CONFIG ────────────────────────────────────────────────────────────────


# ── CAMERA SETUP ─────────────────────────────────────────────────────────────

def setup_camera(x: float, y: float, z: float, rot_z_deg: float = 0.0) -> None:
    """
    Set camera world-space translate and Z-axis orientation.

    Supports xformOp:orient (quaternion), xformOp:rotateXYZ, and xformOp:rotateZ
    ops — whichever is present on the camera prim.
    """
    import omni.usd
    from pxr import UsdGeom, Gf

    stage    = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not cam_prim.IsValid():
        raise RuntimeError(f"[setup_camera] Camera prim not found: {CAMERA_PATH}")

    xformable = UsdGeom.Xformable(cam_prim)
    ops_dict  = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}

    if "xformOp:translate" not in ops_dict:
        raise RuntimeError("[setup_camera] Camera prim has no xformOp:translate op")

    ops_dict["xformOp:translate"].Set(Gf.Vec3d(x, y, z))

    half_rad = math.radians(rot_z_deg) / 2.0
    quat     = Gf.Quatd(math.cos(half_rad), 0.0, 0.0, math.sin(half_rad))

    if "xformOp:orient" in ops_dict:
        ops_dict["xformOp:orient"].Set(quat)
    elif "xformOp:rotateXYZ" in ops_dict:
        ops_dict["xformOp:rotateXYZ"].Set(Gf.Vec3f(0.0, 0.0, rot_z_deg))
    elif "xformOp:rotateZ" in ops_dict:
        ops_dict["xformOp:rotateZ"].Set(rot_z_deg)
    else:
        print("[setup_camera] WARNING: no rotation op found on camera prim — "
              "orientation unchanged")

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
    cam_prim = stage.GetPrimAtPath(CAMERA_PATH)
    if not cam_prim.IsValid():
        raise RuntimeError(f"[camera] prim not found: {CAMERA_PATH}")

    xformable = UsdGeom.Xformable(cam_prim)
    world_xf  = xformable.ComputeLocalToWorldTransform(0)
    t         = world_xf.ExtractTranslation()
    return float(t[0]), float(t[1]), float(t[2])


# ── CAPTURE ───────────────────────────────────────────────────────────────────

async def capture_rgb_depth():
    """
    Create a Replicator render product, attach rgb and distance_to_image_plane
    annotators, step the simulation once, and return:
        (rgb_uint8 H×W×3, depth_float32 H×W)

    depth values are in metres (distance to image plane, not ray length).
    NaN / Inf are replaced with 0.

    Both annotators may return an ndarray OR a dict with a "data" key
    (Isaac Sim 5.1 returns dicts in some configurations).  Both branches are
    handled and the returned type is printed once for diagnostics.
    """
    import omni.replicator.core as rep
    import numpy as np

    print(f"[capture] creating render product {IMAGE_WIDTH}x{IMAGE_HEIGHT} "
          f"on {CAMERA_PATH}")
    rp = rep.create.render_product(CAMERA_PATH, (IMAGE_WIDTH, IMAGE_HEIGHT))

    depth_an = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
    rgb_an   = rep.AnnotatorRegistry.get_annotator("rgb")
    depth_an.attach([rp])
    rgb_an.attach([rp])

    print(f"[capture] stepping simulation ({RT_SUBFRAMES} rt_subframes) ...")
    await rep.orchestrator.step_async(rt_subframes=RT_SUBFRAMES)

    raw_depth = depth_an.get_data()
    raw_rgb   = rgb_an.get_data()

    if raw_depth is None or raw_rgb is None:
        raise RuntimeError("[capture] Annotator returned None — check camera "
                           "prim and render product")

    # ── RGB ───────────────────────────────────────────────────────────────────
    print(f"[capture] rgb type = {type(raw_rgb)}")
    if isinstance(raw_rgb, dict):
        raw_rgb = raw_rgb["data"]
    raw_rgb = __import__("numpy").asarray(raw_rgb)
    raw_rgb = raw_rgb.reshape(IMAGE_HEIGHT, IMAGE_WIDTH, -1)
    print(f"[capture] rgb shape = {raw_rgb.shape}")
    rgb = raw_rgb[:, :, :3].astype(__import__("numpy").uint8)

    # ── Depth ─────────────────────────────────────────────────────────────────
    print(f"[capture] depth type = {type(raw_depth)}")
    if isinstance(raw_depth, dict):
        raw_depth = raw_depth["data"]
    raw_depth = __import__("numpy").asarray(raw_depth, dtype=__import__("numpy").float32)
    raw_depth = raw_depth.reshape(IMAGE_HEIGHT, IMAGE_WIDTH)
    print(f"[capture] depth shape = {raw_depth.shape}")
    depth = __import__("numpy").nan_to_num(raw_depth, nan=0.0, posinf=0.0, neginf=0.0)

    valid_d = depth[depth > 0.0]
    if valid_d.size > 0:
        print(f"[capture] depth valid range "
              f"[{valid_d.min():.4f}, {valid_d.max():.4f}] m  "
              f"({valid_d.size} px)")
    else:
        print("[capture] WARNING: no non-zero depth pixels")

    return rgb, depth


# ── SURFACE ESTIMATION ────────────────────────────────────────────────────────

def estimate_board_surface_depth(depth):
    """
    Estimate the depth (distance to camera) of the board top surface by finding
    the dominant histogram peak within [SURFACE_DEPTH_MIN, SURFACE_DEPTH_MAX].

    If BOARD_ROI_ENABLED, restricts the analysis to a centred rectangular sub-
    image of size BOARD_ROI_FRACTION of the full frame, which reduces the
    influence of floor/bench pixels at the image edges.

    Returns the estimated board surface depth in metres.
    """
    import numpy as np

    if BOARD_ROI_ENABLED:
        h, w = depth.shape
        dh   = int(h * (1.0 - BOARD_ROI_FRACTION) / 2.0)
        dw   = int(w * (1.0 - BOARD_ROI_FRACTION) / 2.0)
        roi  = depth[dh : h - dh, dw : w - dw]
        print(f"[surface_est] using ROI [{dh}:{h-dh}, {dw}:{w-dw}]  "
              f"({roi.shape[1]}x{roi.shape[0]} px)")
    else:
        roi = depth
        print("[surface_est] using full image (BOARD_ROI_ENABLED=False)")

    valid = roi[(roi > SURFACE_DEPTH_MIN) & (roi < SURFACE_DEPTH_MAX)]
    if valid.size == 0:
        raise RuntimeError(
            f"[surface_est] No valid depth pixels in "
            f"[{SURFACE_DEPTH_MIN}, {SURFACE_DEPTH_MAX}] m. "
            f"Check CAM_Z and SURFACE_DEPTH_MIN / SURFACE_DEPTH_MAX."
        )

    bins             = np.arange(SURFACE_DEPTH_MIN,
                                  SURFACE_DEPTH_MAX + SURFACE_HIST_BIN,
                                  SURFACE_HIST_BIN)
    hist, edges      = np.histogram(valid, bins=bins)
    peak_bin         = int(np.argmax(hist))
    board_surface_z  = float(edges[peak_bin]) + SURFACE_HIST_BIN / 2.0
    peak_fraction    = float(hist[peak_bin]) / float(valid.size)

    print(f"[surface_est] dominant depth = {board_surface_z:.4f} m  "
          f"({peak_fraction * 100:.1f}% of valid pixels in range)")

    if peak_fraction < 0.05:
        print("[surface_est] WARNING: peak fraction < 5% — histogram is noisy. "
              "Enable BOARD_ROI_ENABLED or adjust SURFACE_DEPTH_MIN/MAX. "
              "Inspect depth_vis.png.")

    return board_surface_z


# ── SEGMENTATION ──────────────────────────────────────────────────────────────

def segment_cavities_from_depth(depth, board_surface_z: float):
    """
    Return a boolean mask of pixels that correspond to cavities.

    Cavity rule: the measured depth is deeper than the board surface by at
    least CAVITY_DEPTH_MARGIN (camera sees farther — the pixel is inside a
    hole) but no more than MAX_CAVITY_DEPTH (rejects floor, holes through the
    board, and far-field noise).

    A morphological open (remove isolated noise pixels) followed by a close
    (fill small gaps inside cavities) is applied with a 3×3 kernel.
    """
    import numpy as np
    import cv2

    lo = board_surface_z + CAVITY_DEPTH_MARGIN
    hi = board_surface_z + MAX_CAVITY_DEPTH

    raw_mask = (depth > lo) & (depth < hi)

    n_raw = int(raw_mask.sum())
    print(f"[segment] board_surface_z={board_surface_z:.4f} m  "
          f"cavity band=[{lo:.4f}, {hi:.4f}] m  "
          f"raw cavity pixels: {n_raw}")

    if n_raw == 0:
        print("[segment] WARNING: zero cavity pixels. "
              "Possible causes: board surface estimate is off, "
              "CAVITY_DEPTH_MARGIN too large, MAX_CAVITY_DEPTH too small, "
              "or depth units mismatch. Inspect depth_vis.png.")

    kernel = np.ones((3, 3), dtype=np.uint8)
    binary = raw_mask.astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN,  kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    cleaned_mask = binary.astype(bool)
    n_cleaned    = int(cleaned_mask.sum())
    print(f"[segment] after morph open+close: {n_cleaned} pixels "
          f"(removed {n_raw - n_cleaned} noise pixels)")

    return cleaned_mask


# ── CONNECTED COMPONENTS ──────────────────────────────────────────────────────

def find_cavity_components(raw_mask):
    """
    Run connected-components analysis on raw_mask, filter blobs by area, and
    return a deterministically sorted list of cavity dicts.

    Sort order (documented so callers can rely on cavity_00, cavity_01, ...):
      1. Bin centroid_y into rows of ROW_BIN_PX pixels.
      2. Sort by (row_bin, centroid_x) — top row, left to right; then next row.

    This order is stable across runs as long as the camera does not move.

    Each dict: {label, area_px, centroid (x,y), bbox (x,y,w,h)}
    """
    import numpy as np
    import cv2

    binary   = (raw_mask.astype(np.uint8)) * 255
    n, labels, stats, centroids = cv2.connectedComponentsWithStats(binary)

    cavities = []
    for i in range(1, n):   # 0 = background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if CC_MIN_AREA_PX <= area <= CC_MAX_AREA_PX:
            cavities.append({
                "label":    i,
                "area_px":  area,
                "centroid": (float(centroids[i][0]), float(centroids[i][1])),
                "bbox":     (int(stats[i, cv2.CC_STAT_LEFT]),
                             int(stats[i, cv2.CC_STAT_TOP]),
                             int(stats[i, cv2.CC_STAT_WIDTH]),
                             int(stats[i, cv2.CC_STAT_HEIGHT])),
            })

    print(f"[cc] total components (excl. background): {n - 1}  "
          f"valid [{CC_MIN_AREA_PX}–{CC_MAX_AREA_PX} px]: {len(cavities)}")

    if not cavities:
        print("[cc] WARNING: no valid cavity components found. "
              "Check CAVITY_DEPTH_MARGIN, MAX_CAVITY_DEPTH, CC_MIN_AREA_PX, "
              "and inspect raw_cavity_mask.png.")
        return cavities

    # Deterministic sort: row_bin (top first) then centroid_x (left first).
    cavities.sort(key=lambda c: (
        int(c["centroid"][1]) // ROW_BIN_PX,
        c["centroid"][0],
    ))

    print("[cc] sorted cavities (top-to-bottom, left-to-right):")
    for k, c in enumerate(cavities):
        cx, cy = c["centroid"]
        bx, by, bw, bh = c["bbox"]
        row_bin = int(cy) // ROW_BIN_PX
        print(f"  cavity_{k:02d}  area={c['area_px']} px  "
              f"centroid=({cx:.1f}, {cy:.1f})  row_bin={row_bin}  "
              f"bbox=({bx}, {by}, {bw}, {bh})")

    return cavities


# ── CAMERA INTRINSICS ─────────────────────────────────────────────────────────

def compute_intrinsics(cam_z: float):
    """
    Compute pixel-space intrinsics from the camera configuration constants.

    cam_z should be the depth at which the XY scale is evaluated — for cavity
    detection this is board_surface_z (the board top surface, not CAM_Z).

    Returns a dict: fx, fy, cx_px, cy_px, mpp_x, mpp_y, cam_z.
    """
    fov_h = 2.0 * math.atan((APERTURE_MM / 2.0) / FOCAL_MM)
    fov_v = fov_h * (IMAGE_HEIGHT / IMAGE_WIDTH)
    mpp_x = (2.0 * cam_z * math.tan(fov_h / 2.0)) / IMAGE_WIDTH
    mpp_y = (2.0 * cam_z * math.tan(fov_v / 2.0)) / IMAGE_HEIGHT
    fx    = cam_z / mpp_x
    fy    = cam_z / mpp_y
    return {
        "fx":    fx,
        "fy":    fy,
        "cx_px": IMAGE_WIDTH  / 2.0,
        "cy_px": IMAGE_HEIGHT / 2.0,
        "mpp_x": mpp_x,
        "mpp_y": mpp_y,
        "cam_z": cam_z,
    }


# ── POINT CLOUD ───────────────────────────────────────────────────────────────

def build_cavity_pointcloud(depth, mask, intrinsics, board_surface_z: float,
                             cam_xy: tuple,
                             n_samples: int = N_POINTS):
    """
    Back-project masked depth pixels into a cavity-local 3-D point cloud.

    Coordinate convention (output):
      X, Y : world-plane position centred on the cavity centroid (metres).
              Real metric scale is preserved; points are only translated, not
              scaled.
      Z    : depth BELOW the board top surface (metres, positive = deeper).
              Z = depth[y,x] - board_surface_z
              Z is NOT centred — its absolute magnitude encodes cavity depth.

    This is geometrically opposite to the piece point cloud, where
    Z = surface_z - depth (height above the table).

    Returns:
      points         — (N_POINTS, 3) float32
      centroid_world — (cx_world, cy_world) in metres, for metadata
    """
    import numpy as np

    ys, xs = np.where(mask)
    if len(xs) == 0:
        raise RuntimeError("[pointcloud] cavity mask is empty — "
                           "cannot build point cloud")

    mpp_x = intrinsics["mpp_x"]
    mpp_y = intrinsics["mpp_y"]
    cx_px = intrinsics["cx_px"]
    cy_px = intrinsics["cy_px"]

    # Back-project to world XY using pinhole model.
    # Camera world position is cam_xy = (cam_x, cam_y); pixel offset scales by mpp.
    cam_x, cam_y = cam_xy
    world_x = cam_x + (xs.astype(np.float64) - cx_px) * mpp_x
    world_y = cam_y - (ys.astype(np.float64) - cy_px) * mpp_y  # image V flips Y

    # Z: how deep the pixel is below the board top surface.
    # depth[y,x] is the camera-to-surface distance; board_surface_z is the
    # camera-to-board-top distance.  Pixels inside a cavity have larger depth.
    world_z = depth[ys, xs].astype(np.float64) - board_surface_z
    world_z = np.clip(world_z, 0.0, None)   # clip numerical noise above board

    # Centre XY on the cavity centroid (world frame).
    cx_world = float(world_x.mean())
    cy_world = float(world_y.mean())
    world_x -= cx_world
    world_y -= cy_world
    # Z is NOT centred — absolute depth below board surface is meaningful.

    points = np.stack([world_x, world_y, world_z], axis=1).astype(np.float32)

    n_raw   = len(points)
    replace = n_raw < n_samples
    rng     = np.random.default_rng(0)
    idx     = rng.choice(n_raw, size=n_samples, replace=replace)
    points  = points[idx]

    print(f"[pointcloud] raw pixels={n_raw}  sampled={n_samples}  "
          f"(replace={replace})")
    print(f"  X=[{points[:, 0].min():.4f}, {points[:, 0].max():.4f}] m")
    print(f"  Y=[{points[:, 1].min():.4f}, {points[:, 1].max():.4f}] m")
    print(f"  Z=[{points[:, 2].min():.4f}, {points[:, 2].max():.4f}] m  "
          f"(depth below board surface)")
    print(f"  centroid_world=({cx_world:.4f}, {cy_world:.4f})")

    return points, (cx_world, cy_world)


# ── FOOTPRINT IMAGE ───────────────────────────────────────────────────────────

def make_cavity_footprint(points):
    """
    Project the cavity point cloud onto the XY plane (top-down view) and
    render as a colour-mapped density image.

    Uses np.add.at for scatter-accumulation — avoids the per-pixel Python loop
    present in the piece script's footprint renderer.

    Points are already centred on the cavity centroid so the image centre
    corresponds to the cavity centre.

    Returns a 3-channel uint8 BGR image suitable for cv2.imwrite.
    """
    import numpy as np
    import cv2

    half_world = (FOOTPRINT_CANVAS_PX / 2.0) * FOOTPRINT_RESOLUTION_M_PER_PX

    u = ((points[:, 0] + half_world) / FOOTPRINT_RESOLUTION_M_PER_PX).astype(np.int32)
    v = ((half_world - points[:, 1]) / FOOTPRINT_RESOLUTION_M_PER_PX).astype(np.int32)

    valid = (u >= 0) & (u < FOOTPRINT_CANVAS_PX) & \
            (v >= 0) & (v < FOOTPRINT_CANVAS_PX)
    u = u[valid]
    v = v[valid]

    canvas = np.zeros((FOOTPRINT_CANVAS_PX, FOOTPRINT_CANVAS_PX), dtype=np.int32)
    np.add.at(canvas, (v, u), 1)   # vectorised scatter — no Python loop

    # Clip to uint8 range and normalise for display
    canvas_u8 = np.clip(canvas, 0, 255).astype(np.uint8)
    if canvas_u8.max() > 0:
        vis = cv2.normalize(canvas_u8, None, 0, 255, cv2.NORM_MINMAX)
    else:
        vis = canvas_u8

    footprint_bgr = cv2.applyColorMap(vis.astype(np.uint8), cv2.COLORMAP_HOT)

    # Crosshair at the cavity centroid (= image centre)
    mid = FOOTPRINT_CANVAS_PX // 2
    cv2.line(footprint_bgr, (mid - 10, mid), (mid + 10, mid), (0, 255, 0), 1)
    cv2.line(footprint_bgr, (mid, mid - 10), (mid, mid + 10), (0, 255, 0), 1)

    n_in  = int(valid.sum())
    n_tot = len(points)
    print(f"[footprint] canvas={FOOTPRINT_CANVAS_PX}px  "
          f"res={FOOTPRINT_RESOLUTION_M_PER_PX * 1000:.1f} mm/px  "
          f"world span={half_world * 2 * 100:.1f} cm  "
          f"projected {n_in}/{n_tot} points")

    if n_in < int(n_tot * 0.5):
        print("[footprint] WARNING: fewer than 50% of points fit in the canvas. "
              "Consider increasing FOOTPRINT_CANVAS_PX or "
              "FOOTPRINT_RESOLUTION_M_PER_PX.")

    return footprint_bgr


# ── GLOBAL DEBUG OUTPUTS ──────────────────────────────────────────────────────

def save_global_outputs(out_dir: Path, rgb, depth, raw_mask, cavities):
    """
    Write the four global output images.  Skips any that are None (so a partial
    run only writes the files that were actually produced).

    cavities — list of cavity dicts as returned by find_cavity_components.
    """
    import numpy as np
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. RGB
    if rgb is not None:
        rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "rgb.png"), rgb_bgr)
        print(f"[save_global] rgb.png")

    # 2. Depth visualisation
    if depth is not None:
        valid_d = depth[depth > 0.0]
        d_min   = float(valid_d.min()) if valid_d.size > 0 else 0.0
        d_max   = float(valid_d.max()) if valid_d.size > 0 else 1.0
        d_norm  = np.clip((depth - d_min) / (d_max - d_min + 1e-8), 0.0, 1.0)
        depth_vis = cv2.applyColorMap((d_norm * 255).astype(np.uint8),
                                      cv2.COLORMAP_VIRIDIS)
        cv2.imwrite(str(out_dir / "depth_vis.png"), depth_vis)
        print(f"[save_global] depth_vis.png")

    # 3. Raw cavity mask (before connected-component filtering)
    if raw_mask is not None:
        cv2.imwrite(str(out_dir / "raw_cavity_mask.png"),
                    (raw_mask.astype(np.uint8) * 255))
        print(f"[save_global] raw_cavity_mask.png")

    # 4. Cavities debug overlay (all cavities tinted and numbered on RGB)
    if rgb is not None and cavities:
        _colours = [
            (255,  60,  60),   # red
            ( 60, 180, 255),   # cyan-blue
            ( 60, 255,  60),   # green
            (255, 200,  60),   # amber
            (200,  60, 255),   # purple
            (255, 255,  60),   # yellow
            ( 60, 255, 200),   # teal
            (255, 120, 200),   # pink
        ]
        debug = rgb.copy()
        # We need the per-cavity masks; reconstruct them from labeled image.
        # Build the full labeled image so we can tint each cavity.
        binary   = (raw_mask.astype(np.uint8) * 255) if raw_mask is not None \
            else np.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=np.uint8)
        n_cc, labels = cv2.connectedComponents(binary)

        for k, cav in enumerate(cavities):
            colour  = _colours[k % len(_colours)]
            cav_lbl = cav["label"]
            cav_msk = (labels == cav_lbl)
            tint    = np.array(colour, dtype=np.float32)
            debug[cav_msk] = (debug[cav_msk].astype(np.float32) * 0.35
                               + tint * 0.65).astype(np.uint8)
            cx, cy  = int(cav["centroid"][0]), int(cav["centroid"][1])
            bx, by, bw, bh = cav["bbox"]
            cv2.circle(debug, (cx, cy), 5, colour[::-1], -1)   # BGR in cv2
            cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), colour[::-1], 2)
            label_str = f"C{k:02d} {cav['area_px']}px"
            cv2.putText(debug, label_str, (bx, max(by - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, colour[::-1], 1,
                        cv2.LINE_AA)

        debug_bgr = cv2.cvtColor(debug, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / "cavities_debug.png"), debug_bgr)
        print(f"[save_global] cavities_debug.png  ({len(cavities)} cavities)")

    elif rgb is not None:
        # No cavities — save a plain copy so the debug image still exists
        cv2.imwrite(str(out_dir / "cavities_debug.png"),
                    cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        print("[save_global] cavities_debug.png  (no cavities detected)")


# ── PER-CAVITY OUTPUTS ────────────────────────────────────────────────────────

def save_cavity_outputs(out_dir: Path, cavity_id: int, cavity_dict: dict,
                        cavity_mask, points, footprint_bgr,
                        rgb, board_surface_z: float, centroid_world):
    """
    Write all per-cavity files into out_dir / f"cavity_{cavity_id:02d}".

    cavity_mask : H×W bool array for this specific cavity
    points      : (N, 3) float32 point cloud
    footprint_bgr : colourmap footprint image
    rgb         : H×W×3 uint8 RGB image (used for debug overlay)
    """
    import numpy as np
    import cv2

    cav_dir = out_dir / f"cavity_{cavity_id:02d}"
    cav_dir.mkdir(parents=True, exist_ok=True)

    # ── Cavity mask ───────────────────────────────────────────────────────────
    cv2.imwrite(str(cav_dir / "cavity_mask.png"),
                (cavity_mask.astype(np.uint8) * 255))

    # ── Per-cavity debug overlay ──────────────────────────────────────────────
    if rgb is not None:
        debug = rgb.copy()
        debug[cavity_mask] = (debug[cavity_mask].astype(np.float32) * 0.3
                               + np.array([60, 180, 255], np.float32) * 0.7
                               ).astype(np.uint8)
        cx, cy = int(cavity_dict["centroid"][0]), int(cavity_dict["centroid"][1])
        bx, by, bw, bh = cavity_dict["bbox"]
        cv2.circle(debug, (cx, cy), 6, (255, 255, 0), -1)
        cv2.rectangle(debug, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
        label_str = (f"cavity_{cavity_id:02d}  "
                     f"({cx},{cy})  {cavity_dict['area_px']} px")
        cv2.putText(debug, label_str, (bx, max(by - 6, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 0), 1,
                    cv2.LINE_AA)
        cv2.imwrite(str(cav_dir / "cavity_debug.png"),
                    cv2.cvtColor(debug, cv2.COLOR_RGB2BGR))

    # ── Footprint ─────────────────────────────────────────────────────────────
    cv2.imwrite(str(cav_dir / "cavity_footprint.png"), footprint_bgr)

    # ── Point cloud ───────────────────────────────────────────────────────────
    np.save(str(cav_dir / "cavity_pointcloud.npy"), points)

    # ── Per-cavity metadata ───────────────────────────────────────────────────
    cx_w, cy_w = centroid_world
    xy_span_x  = float(points[:, 0].max() - points[:, 0].min())
    xy_span_y  = float(points[:, 1].max() - points[:, 1].min())
    z_min      = float(points[:, 2].min())
    z_max      = float(points[:, 2].max())

    cavity_meta = {
        "cavity_id":         cavity_id,
        "area_px":           cavity_dict["area_px"],
        "centroid_px": {
            "x": cavity_dict["centroid"][0],
            "y": cavity_dict["centroid"][1],
        },
        "bbox_px": {
            "x": cavity_dict["bbox"][0],
            "y": cavity_dict["bbox"][1],
            "w": cavity_dict["bbox"][2],
            "h": cavity_dict["bbox"][3],
        },
        "centroid_world_m": {
            "x": cx_w,
            "y": cy_w,
        },
        "board_surface_depth_m": float(board_surface_z),
        "point_count":       N_POINTS,
        "xy_span_m": {
            "x": xy_span_x,
            "y": xy_span_y,
        },
        "z_depth_range_m": {
            "min": z_min,
            "max": z_max,
        },
        "pointcloud_bounds_m": {
            "x_min": float(points[:, 0].min()),
            "x_max": float(points[:, 0].max()),
            "y_min": float(points[:, 1].min()),
            "y_max": float(points[:, 1].max()),
            "z_min": z_min,
            "z_max": z_max,
        },
        "files": {
            "footprint":   "cavity_footprint.png",
            "mask":        "cavity_mask.png",
            "pointcloud":  "cavity_pointcloud.npy",
            "debug":       "cavity_debug.png",
        },
    }

    meta_path = cav_dir / "cavity_metadata.json"
    with open(str(meta_path), "w") as f:
        json.dump(cavity_meta, f, indent=2)

    print(f"[save_cavity] cavity_{cavity_id:02d}  "
          f"z_range=[{z_min:.4f}, {z_max:.4f}] m  "
          f"xy_span=({xy_span_x:.4f}, {xy_span_y:.4f}) m  → {cav_dir}")

    return cavity_meta


# ── SUMMARY METADATA ──────────────────────────────────────────────────────────

def save_summary_metadata(out_dir: Path, success: bool, board_surface_z: float,
                           raw_pixels: int, cavities_meta: list,
                           camera_pose: dict = None,
                           error_msg=None):
    """
    Write cavities_summary.json.  Always called, even on failure.
    """
    ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    summary = {
        "script":           "capture_cavity_detection.py",
        "timestamp":        ts,
        "project_root":     str(PROJECT_ROOT),
        "output_dir":       str(out_dir),
        "success":          success,
        "camera_pose": camera_pose if camera_pose is not None else {
            "x": None, "y": None, "z": None, "rot_z_deg": None,
        },
        "camera_pose_overridden": bool(SET_CAMERA_POSE),
        "image_resolution": {
            "width":  IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
        },
        "parameters": {
            "focal_mm":             FOCAL_MM,
            "aperture_mm":          APERTURE_MM,
            "board_roi_enabled":    BOARD_ROI_ENABLED,
            "board_roi_fraction":   BOARD_ROI_FRACTION,
            "surface_depth_min":    SURFACE_DEPTH_MIN,
            "surface_depth_max":    SURFACE_DEPTH_MAX,
            "cavity_depth_margin":  CAVITY_DEPTH_MARGIN,
            "max_cavity_depth":     MAX_CAVITY_DEPTH,
            "cc_min_area_px":       CC_MIN_AREA_PX,
            "cc_max_area_px":       CC_MAX_AREA_PX,
            "n_points":             N_POINTS,
            "footprint_res_m_per_px": FOOTPRINT_RESOLUTION_M_PER_PX,
            "footprint_canvas_px":  FOOTPRINT_CANVAS_PX,
            "row_bin_px":           ROW_BIN_PX,
        },
        "board_surface_depth_m": board_surface_z,
        "raw_cavity_pixels":     raw_pixels,
        "n_detected_cavities":   len(cavities_meta),
        "cavities":              cavities_meta,
        "error":                 error_msg,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    meta_path = out_dir / "cavities_summary.json"
    with open(str(meta_path), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[summary] saved → {meta_path}")


# ── STALE FILE CLEANUP ────────────────────────────────────────────────────────

def _cleanup_stale(out_dir: Path) -> None:
    """
    Delete the four global output files and all existing cavity_XX subdirectories
    before the pipeline runs.  Ensures a failed run cannot leave old artefacts
    that could be mistaken for current results.
    """
    _global_files = [
        "rgb.png", "depth_vis.png", "raw_cavity_mask.png",
        "cavities_debug.png", "cavities_summary.json",
    ]
    for fname in _global_files:
        p = out_dir / fname
        if p.exists():
            p.unlink()
            print(f"[cleanup] removed stale: {fname}")

    for subdir in sorted(out_dir.glob("cavity_*")):
        if subdir.is_dir():
            shutil.rmtree(subdir)
            print(f"[cleanup] removed stale: {subdir.name}/")


# ── MAIN PIPELINE ─────────────────────────────────────────────────────────────

async def main():
    import numpy as np

    print("=" * 60)
    print("capture_cavity_detection.py — Phase 2")
    print("=" * 60)
    print(f"[main] output_dir = {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    _cleanup_stale(OUT_DIR)

    # State variables — kept at module scope of main() so the finally block
    # always has something to write, even on early failure.
    error_msg       = None
    success         = False
    rgb             = None
    depth           = None
    raw_mask        = None
    board_surface_z = 0.0
    raw_pixels      = 0
    cavities        = []        # list of component dicts
    cavities_meta   = []        # list of per-cavity metadata dicts (for summary)
    active_camera_pose = None   # populated after Step 1; recorded in metadata

    try:
        # ── Step 1: Camera ────────────────────────────────────────────────────
        print("\n--- Step 1: Camera setup ---")

        # Warn if the configured override pose still looks like the old
        # piece-table pose (CAM_X<0 and CAM_Y>0.2 was the working piece pose).
        if CAM_X < 0.0 and CAM_Y > 0.2:
            print(f"[camera] WARNING: configured CAM_X={CAM_X}, CAM_Y={CAM_Y} "
                  f"look like the piece-table pose, not a cavity-board pose. "
                  f"Update the constants if SET_CAMERA_POSE=True.")

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

        # ── Step 3: Board surface estimation ─────────────────────────────────
        print("\n--- Step 3: Estimate board surface depth ---")
        board_surface_z = estimate_board_surface_depth(depth)

        # ── Step 4: Cavity segmentation ───────────────────────────────────────
        print("\n--- Step 4: Segment cavities ---")
        raw_mask   = segment_cavities_from_depth(depth, board_surface_z)
        raw_pixels = int(raw_mask.sum())

        # ── Step 5: Connected components ──────────────────────────────────────
        print("\n--- Step 5: Find cavity components ---")
        cavities = find_cavity_components(raw_mask)

        if not cavities:
            raise RuntimeError(
                "No cavity components found after connected-component analysis. "
                "Inspect raw_cavity_mask.png and depth_vis.png. "
                "Typical causes: CAVITY_DEPTH_MARGIN too large, "
                "MAX_CAVITY_DEPTH too small, board surface estimate wrong, "
                "CC_MIN_AREA_PX too large."
            )

        # ── Step 6: Per-cavity processing ────────────────────────────────────
        print(f"\n--- Step 6: Process {len(cavities)} cavity/cavities ---")

        # Build the labeled image once so we can extract individual masks
        import cv2 as _cv2
        binary_u8  = (raw_mask.astype(np.uint8)) * 255
        _n_cc, labels_img = _cv2.connectedComponents(binary_u8)
        del _cv2

        intrinsics = compute_intrinsics(board_surface_z)
        print(f"[intrinsics] mpp_x={intrinsics['mpp_x']*1000:.3f} mm/px  "
              f"mpp_y={intrinsics['mpp_y']*1000:.3f} mm/px  "
              f"(at board surface depth {board_surface_z:.4f} m)")

        for k, cav in enumerate(cavities):
            print(f"\n  [cavity_{k:02d}] label={cav['label']}  "
                  f"area={cav['area_px']} px  "
                  f"centroid=({cav['centroid'][0]:.1f}, {cav['centroid'][1]:.1f})")

            cav_mask = (labels_img == cav["label"])

            # Point cloud
            points, centroid_world = build_cavity_pointcloud(
                depth, cav_mask, intrinsics, board_surface_z,
                cam_xy=active_cam_xy, n_samples=N_POINTS)

            # Footprint
            footprint_bgr = make_cavity_footprint(points)

            # Save per-cavity files and collect metadata
            cav_meta = save_cavity_outputs(
                OUT_DIR, k, cav, cav_mask, points, footprint_bgr,
                rgb, board_surface_z, centroid_world)
            cavities_meta.append(cav_meta)

        success = True
        print(f"\n[main] Pipeline completed successfully.  "
              f"Detected {len(cavities)} cavities.")

    except Exception as exc:
        error_msg = str(exc)
        print(f"\n[ERROR] {exc}")
        traceback.print_exc()

    finally:
        # ── Step 7: Save global outputs ───────────────────────────────────────
        print("\n--- Step 7: Save global outputs ---")

        save_global_outputs(OUT_DIR, rgb, depth, raw_mask, cavities)

        save_summary_metadata(
            OUT_DIR, success, board_surface_z, raw_pixels,
            cavities_meta, camera_pose=active_camera_pose,
            error_msg=error_msg)

        print("\n[main] Files written to:", OUT_DIR)
        if OUT_DIR.exists():
            for p in sorted(OUT_DIR.rglob("*")):
                rel = p.relative_to(OUT_DIR)
                print(f"  {rel}")

        print("\n" + "=" * 60)
        print(f"  success={success}  cavities={len(cavities_meta)}")
        if not success:
            print(f"  error:  {error_msg}")
        print("=" * 60)


asyncio.ensure_future(main())
