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
import time
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Tune these constants for your scene BEFORE running.

# Camera USD path in the stage
CAMERA_PRIM_PATH = "/World/Camera"

# Camera pose: X, Y, Z in metres (world frame, Z = up)
# Position the camera above the piece table looking straight down.
CAM_X = -0.25
CAM_Y =  0.45
CAM_Z =  0.58   # height above world origin

# Camera rotation around Z-axis in degrees (0 = looking in -Y, no rotation).
# If the piece appears rotated or upside-down, adjust this.
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

# Surface estimation: look for the dominant depth peak in this range [m].
# Should bracket the table/board surface.
SURFACE_DEPTH_MIN = 0.10
SURFACE_DEPTH_MAX = 0.50
SURFACE_HIST_BIN  = 0.001   # 1 mm bins

# Segmentation: a pixel belongs to the piece if its depth is MORE than this
# margin below the surface estimate.  Too small → table noise bleeds in.
# Too large → thin/flat pieces disappear.
SURFACE_TOLERANCE = 0.004   # 4 mm

# Minimum depth for any valid measurement (clips near-field noise)
DEPTH_MIN_VALID = 0.02

# Connected-component filters
CC_MIN_AREA_PX =  300   # discard blobs smaller than this
CC_MAX_AREA_PX = 50000  # discard blobs suspiciously large (probably table leak)

# Point cloud sampling target
N_POINTS = 2048

# Output directory
OUT_DIR = Path("/workspace/Tese_Roberto/shape_insertion/thesis-omniverse/data/pieces_detected")

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

def estimate_support_surface_depth(depth):
    """
    Estimate the depth (distance to camera) of the support surface by finding
    the dominant peak in a histogram of valid depth values within the configured
    range.

    Returns the estimated surface depth in metres, or raises if not found.

    Risk: if the piece covers most of the frame (very close camera) the piece
    itself may dominate the histogram and be mis-identified as the surface.
    Mitigation: keep the camera far enough that the surrounding table is visible.
    """
    import numpy as np

    valid = depth[(depth > SURFACE_DEPTH_MIN) & (depth < SURFACE_DEPTH_MAX)]
    if valid.size == 0:
        raise RuntimeError(
            f"[surface_est] No valid depth pixels in range "
            f"[{SURFACE_DEPTH_MIN}, {SURFACE_DEPTH_MAX}] m. "
            f"Check CAM_Z and SURFACE_DEPTH_MIN/MAX.")

    bins = np.arange(SURFACE_DEPTH_MIN, SURFACE_DEPTH_MAX + SURFACE_HIST_BIN,
                     SURFACE_HIST_BIN)
    hist, edges = np.histogram(valid, bins=bins)

    peak_bin = int(np.argmax(hist))
    surface_d = float(edges[peak_bin]) + SURFACE_HIST_BIN / 2.0

    # Sanity: peak bin should contain a meaningful fraction of valid pixels
    peak_fraction = float(hist[peak_bin]) / float(valid.size)
    print(f"[surface_est] dominant depth = {surface_d:.4f} m  "
          f"({peak_fraction*100:.1f}% of valid pixels)")

    if peak_fraction < 0.05:
        print("[surface_est] WARNING: peak fraction < 5% — depth histogram is "
              "noisy; surface estimate may be unreliable. Inspect depth_vis.png.")

    return surface_d


# ── SEGMENTATION ──────────────────────────────────────────────────────────────

def segment_piece(depth, surface_z):
    """
    Return a raw boolean mask of pixels that are above the support surface by
    more than SURFACE_TOLERANCE metres.

    Pixels closer to the camera than DEPTH_MIN_VALID are ignored (near-field
    sensor noise in Isaac Sim).

    depth:     H×W float32, metres
    surface_z: estimated surface distance in metres
    """
    import numpy as np

    # Piece pixels are CLOSER to the camera than the surface
    # (smaller distance_to_image_plane value)
    threshold = surface_z - SURFACE_TOLERANCE
    mask = (depth > DEPTH_MIN_VALID) & (depth < threshold)

    n_pixels = int(mask.sum())
    print(f"[segment] surface_z={surface_z:.4f}m  threshold={threshold:.4f}m  "
          f"pixels above surface: {n_pixels}")

    if n_pixels == 0:
        print("[segment] WARNING: zero pixels above surface. "
              "Possible causes: piece is flush with table, camera too high, "
              "SURFACE_TOLERANCE too small, depth units mismatch.")

    return mask


# ── CONNECTED COMPONENTS ──────────────────────────────────────────────────────

def select_best_component(raw_mask):
    """
    Run connected-components analysis on raw_mask, filter blobs by area, and
    return (best_mask, stats_list).

    best_mask:  H×W bool, only the largest valid blob set to True
    stats_list: list of dicts with per-blob info, sorted by area descending

    Returns (None, stats_list) if no valid blob is found.
    """
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

    blobs.sort(key=lambda b: b["area_px"], reverse=True)

    print(f"[cc] total components (excl. background): {n - 1}  "
          f"valid [{CC_MIN_AREA_PX}–{CC_MAX_AREA_PX} px]: {len(blobs)}")
    for idx, b in enumerate(blobs):
        print(f"  blob {idx}: area={b['area_px']}px  "
              f"centroid=({b['centroid'][0]:.1f},{b['centroid'][1]:.1f})  "
              f"bbox={b['bbox']}")

    if not blobs:
        print("[cc] WARNING: no valid blob found. "
              "Check CC_MIN_AREA_PX, SURFACE_TOLERANCE, and inspect "
              "raw_piece_mask.png.")
        return None, blobs

    best = blobs[0]
    best_mask = (labels == best["label"])
    print(f"[cc] selected blob 0: area={best['area_px']}px  "
          f"centroid={best['centroid']}")

    return best_mask, blobs


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

def depth_to_pointcloud(depth, mask, intrinsics, surface_z, n_samples=N_POINTS):
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
    # Camera X/Y world position is the CONFIG value; pixel offset scales by mpp.
    world_x = CAM_X + (xs.astype(np.float64) - cx_px) * mpp_x
    world_y = CAM_Y - (ys.astype(np.float64) - cy_px) * mpp_y  # image V flips Y

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
                  centroid_world, surface_z, error_msg=None):
    """
    Write piece_metadata.json conforming to the experiments.md conventions.
    """
    import numpy as np

    ts = time.strftime("%Y-%m-%dT%H:%M:%S")

    piece_metrics = {}
    if success and best_mask is not None and blob_stats:
        b = blob_stats[0]
        ys, xs = __import__('numpy').where(best_mask)
        depth_values_placeholder = []   # actual heights stored in point cloud Z
        piece_metrics = {
            "area_px":           b["area_px"],
            "centroid_px":       b["centroid"],
            "bbox_px":           b["bbox"],
            "centroid_world_m":  list(centroid_world),
            "surface_depth_m":   float(surface_z),
            "point_count":       int(len(points)),
            "height_range_m":    [float(points[:, 2].min()),
                                  float(points[:, 2].max())],
            "xy_span_m":         [float((points[:, 0].max() - points[:, 0].min())),
                                  float((points[:, 1].max() - points[:, 1].min()))],
        }

    metadata = {
        "script":      "capture_piece_detection.py",
        "timestamp":   ts,
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
        "success":       success,
        "error":         error_msg,
        "piece_metrics": piece_metrics,
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

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Remove stale outputs from the previous run so a failed run never leaves
    # old images sitting around that could be mistaken for current results.
    _stale_files = [
        "rgb.png", "depth_vis.png", "raw_piece_mask.png",
        "piece_mask.png", "piece_debug.png", "piece_footprint.png",
        "piece_pointcloud.npy", "piece_metadata.json",
    ]
    for _fname in _stale_files:
        _p = OUT_DIR / _fname
        if _p.exists():
            _p.unlink()
            print(f"[main] removed stale: {_p.name}")

    error_msg    = None
    success      = False
    best_mask    = None
    blob_stats   = []
    points       = np.zeros((N_POINTS, 3), dtype=np.float32)
    centroid_w   = (0.0, 0.0)
    surface_z    = 0.0
    rgb          = None
    depth        = None
    raw_mask     = None
    footprint_bgr = None

    try:
        # ── Step 1: Camera setup ──────────────────────────────────────────────
        print("\n--- Step 1: Camera setup ---")
        setup_camera(CAM_X, CAM_Y, CAM_Z, CAM_ROT_Z_DEG)

        # ── Step 2: Capture ───────────────────────────────────────────────────
        print("\n--- Step 2: Capture RGB + depth ---")
        rgb, depth = await capture_rgb_depth()

        # ── Step 3: Surface estimation ────────────────────────────────────────
        print("\n--- Step 3: Estimate support surface depth ---")
        surface_z = estimate_support_surface_depth(depth)

        # ── Step 4: Segmentation ──────────────────────────────────────────────
        print("\n--- Step 4: Segment piece ---")
        raw_mask = segment_piece(depth, surface_z)

        # ── Step 5: Connected components ──────────────────────────────────────
        print("\n--- Step 5: Select best component ---")
        best_mask, blob_stats = select_best_component(raw_mask)

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
            depth, best_mask, intrinsics, surface_z, n_samples=N_POINTS)

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

        # Metadata is always written (success or failure)
        save_metadata(OUT_DIR, success, best_mask, blob_stats, points,
                      centroid_w, surface_z, error_msg=error_msg)

        print("\n" + "=" * 60)
        print(f"  success={success}")
        if not success:
            print(f"  error:  {error_msg}")
        print("=" * 60)


asyncio.ensure_future(main())
