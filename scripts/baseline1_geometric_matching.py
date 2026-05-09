"""
baseline1_geometric_matching.py — Baseline 1: Deterministic Geometric Footprint Matching

Runs OUTSIDE Isaac Sim. Standard Python (NumPy + OpenCV as hard deps;
matplotlib optional for plots).

For every (piece, cavity) pair:
  - Rasterise XY point clouds to binary masks at a fixed real-world resolution.
  - Test 180 rotations (0..358 deg in 2-deg steps).
  - Score by IoU + inside ratio - outside ratio.
  - Select best cavity per piece; report compatibility.

No hardcoded shape→cavity mappings. Folder names are organisational labels only.

Outputs: data/baseline1_geometric_matching/
"""

import csv
import json
import math
import os
import shutil
import sys
import traceback
from datetime import datetime
from glob import glob
from pathlib import Path

import cv2
import numpy as np

# ── PROJECT ROOT ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        str(Path(__file__).resolve().parent.parent),
    )
)

# ── CONFIG ────────────────────────────────────────────────────────────────────

ROTATION_STEP_DEG             = 2           # 180 angles, 0..358 inclusive
CLEARANCE_DILATION_M          = 0.001       # 1 mm tolerance
FOOTPRINT_RESOLUTION_M_PER_PX = 0.00025     # 0.25 mm/px
WORLD_CANVAS_M                = 0.080       # 80 mm world span

CANVAS_PX = int(round(WORLD_CANVAS_M / FOOTPRINT_RESOLUTION_M_PER_PX))   # = 320
assert CANVAS_PX == 320, f"Expected CANVAS_PX=320, got {CANVAS_PX}"

COMPATIBLE_INSIDE_MIN  = 0.80
COMPATIBLE_OUTSIDE_MAX = 0.20
COMPATIBLE_IOU_MIN     = 0.55

SUSPICIOUS_AREA_RATIO_MAX  = 0.50
LOW_RAW_SUPPORT_AREA_PX    = 200
TIE_MARGIN                 = 0.01

# Score weights
W_IOU     = 0.55
W_INSIDE  = 0.35
W_OUTSIDE = 0.10

OVERLAY_SCALE = 4   # 320 → 1280 px upscale for legibility

# Main experimental shape set. The star was replaced by triangle in the MVP
# (see docs/thesis_notes/03_geometric_baseline.md, section 11). If a `star/`
# folder still exists under PIECES_DIR it is intentionally NOT matched here:
# star is reserved as an optional concave-shape stress test.
PIECE_ORDER  = ["rectangle", "square", "circle", "triangle"]
PIECES_DIR   = PROJECT_ROOT / "data" / "pieces_detected"
CAVITIES_DIR = PROJECT_ROOT / "data" / "cavities_detected"
OUT_ROOT     = PROJECT_ROOT / "data" / "baseline1_geometric_matching"

# Dilation radius in pixels for clearance
_DIL_RADIUS_PX = int(round(CLEARANCE_DILATION_M / FOOTPRINT_RESOLUTION_M_PER_PX))  # = 4
_DIL_KERNEL    = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (2 * _DIL_RADIUS_PX + 1, 2 * _DIL_RADIUS_PX + 1)
)

# Rasteriser convex-hull fallback trigger thresholds
MIN_FILL_RATIO_VS_BBOX            = 0.20   # filled_px / bbox_area must exceed this
MAX_CONTOUR_COUNT_FOR_FILLED_MASK = 20     # too many fragments → hull fallback
MIN_LARGEST_CONTOUR_FRACTION      = 0.50   # largest contour / filled_px must exceed this

# ── LOGGING / TEE ─────────────────────────────────────────────────────────────

_log_fh   = None
_orig_stdout = sys.stdout
_orig_stderr = sys.stderr


class _TeeStream:
    """Write to both a file and the original stream."""

    def __init__(self, original, fh):
        self._orig = original
        self._fh   = fh

    def write(self, text):
        self._orig.write(text)
        self._fh.write(text)

    def flush(self):
        self._orig.flush()
        self._fh.flush()

    def fileno(self):
        return self._orig.fileno()


def setup_run_logging(log_path: Path) -> None:
    global _log_fh
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log_fh = open(str(log_path), "w", encoding="utf-8")
    sys.stdout = _TeeStream(_orig_stdout, _log_fh)
    sys.stderr = _TeeStream(_orig_stderr, _log_fh)
    print(f"[log] run_log.txt -> {log_path}")


def teardown_run_logging() -> None:
    global _log_fh
    sys.stdout = _orig_stdout
    sys.stderr = _orig_stderr
    if _log_fh is not None:
        _log_fh.close()
        _log_fh = None


# ── INPUT LOADING ─────────────────────────────────────────────────────────────

def load_inputs() -> tuple[list[dict], list[dict]]:
    """
    Load all piece and cavity data.
    Returns (pieces, cavities).
    Each piece dict: {name, xy, meta}
    Each cavity dict: {cid, xy, meta, area_px_raw, low_raw_support}
    """
    pieces   = []
    cavities = []

    # ── Pieces ────────────────────────────────────────────────────────────────
    for name in PIECE_ORDER:
        piece_dir = PIECES_DIR / name
        pc_path   = piece_dir / "piece_pointcloud.npy"
        meta_path = piece_dir / "piece_metadata.json"

        if not pc_path.exists():
            print(f"[load] WARNING: piece pointcloud missing: {pc_path} — skipping {name}")
            continue
        if not meta_path.exists():
            print(f"[load] WARNING: piece metadata missing: {meta_path} — skipping {name}")
            continue

        pc   = np.load(str(pc_path)).astype(np.float32)
        meta = json.loads(meta_path.read_text())

        if pc.ndim != 2 or pc.shape[1] < 2:
            print(f"[load] WARNING: bad pointcloud shape {pc.shape} for {name} — skipping")
            continue

        xy = pc[:, :2]   # use only XY columns; Z ignored for footprint
        print(f"[load] piece '{name}': {len(xy)} points, "
              f"X=[{xy[:,0].min():.4f},{xy[:,0].max():.4f}] m, "
              f"Y=[{xy[:,1].min():.4f},{xy[:,1].max():.4f}] m")
        pieces.append({"name": name, "xy": xy, "meta": meta})

    # ── Cavities ──────────────────────────────────────────────────────────────
    cav_dirs = sorted(glob(str(CAVITIES_DIR / "cavity_*")))
    for cav_path_str in cav_dirs:
        cav_dir   = Path(cav_path_str)
        cid       = cav_dir.name          # e.g. "cavity_00"
        pc_path   = cav_dir / "cavity_pointcloud.npy"
        meta_path = cav_dir / "cavity_metadata.json"

        if not pc_path.exists():
            print(f"[load] WARNING: cavity pointcloud missing: {pc_path} — skipping {cid}")
            continue
        if not meta_path.exists():
            print(f"[load] WARNING: cavity metadata missing: {meta_path} — skipping {cid}")
            continue

        pc   = np.load(str(pc_path)).astype(np.float32)
        meta = json.loads(meta_path.read_text())

        if pc.ndim != 2 or pc.shape[1] < 2:
            print(f"[load] WARNING: bad cavity pointcloud shape {pc.shape} for {cid} — skipping")
            continue

        xy      = pc[:, :2]
        raw_apx = int(meta.get("area_px", 0))
        low_sup = raw_apx < LOW_RAW_SUPPORT_AREA_PX

        print(f"[load] cavity '{cid}': {len(xy)} points, "
              f"X=[{xy[:,0].min():.4f},{xy[:,0].max():.4f}] m, "
              f"Y=[{xy[:,1].min():.4f},{xy[:,1].max():.4f}] m, "
              f"area_px={raw_apx}, low_raw_support={low_sup}")
        cavities.append({
            "cid":             cid,
            "xy":              xy,
            "meta":            meta,
            "area_px_raw":     raw_apx,
            "low_raw_support": low_sup,
        })

    print(f"[load] loaded {len(pieces)} pieces and {len(cavities)} cavities")
    return pieces, cavities


# ── RASTERISATION ─────────────────────────────────────────────────────────────

_HALF_PX     = FOOTPRINT_RESOLUTION_M_PER_PX / 2.0   # 0.125 mm dedup resolution
_HALF_CANVAS = WORLD_CANVAS_M / 2.0                   # 40 mm


def rasterise_xy_to_mask(xy_m: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Project XY points (centred, metres) onto a CANVAS_PX x CANVAS_PX binary mask.

    Steps:
      1. Dedup at half-pixel resolution (0.125 mm).
      2. Project to pixel grid with Y-axis flip matching make_footprint convention.
      3. Splat points (set pixels to 255).
      4. Morphological close (3x3) to fill 1-px splatting gaps.
      5. Fill external contours (preserves concavities of outermost boundary).
      6. Convex-hull fallback if filled mask has < 50 pixels.

    Returns (mask_uint8, info_dict).
    """
    info = {
        "n_input_points":           int(len(xy_m)),
        "n_unique_points":          0,
        "n_pixels_after_splat":     0,
        "n_pixels_after_close":     0,
        "convex_hull_fallback":     False,
        "fallback_reason":          "none",
        "pre_fallback_filled_px":   0,
        "post_fallback_filled_px":  0,
        "bbox_area_px":             0,
        "n_external_contours":      0,
        "largest_contour_area_px":  0,
    }

    # 1. Dedup: round to half-pixel resolution, keep unique rows
    xy_rounded = np.round(xy_m / _HALF_PX) * _HALF_PX
    xy_unique  = np.unique(xy_rounded, axis=0)
    info["n_unique_points"] = int(len(xy_unique))

    # 2. Project to pixel grid
    #    u = (x + half_canvas) / res   (column = rightward)
    #    v = (half_canvas - y) / res   (row = downward, Y flips)
    u = np.round((xy_unique[:, 0] + _HALF_CANVAS) / FOOTPRINT_RESOLUTION_M_PER_PX).astype(np.int32)
    v = np.round((_HALF_CANVAS - xy_unique[:, 1]) / FOOTPRINT_RESOLUTION_M_PER_PX).astype(np.int32)

    # Drop out-of-canvas
    in_canvas = (u >= 0) & (u < CANVAS_PX) & (v >= 0) & (v < CANVAS_PX)
    u = u[in_canvas]
    v = v[in_canvas]

    # 3. Splat
    mask = np.zeros((CANVAS_PX, CANVAS_PX), dtype=np.uint8)
    if len(u) > 0:
        mask[v, u] = 255
    info["n_pixels_after_splat"] = int((mask > 0).sum())

    # 4. Morphological close (3x3 SE) to fill single-pixel gaps from sparse splatting
    close_kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)
    info["n_pixels_after_close"] = int((mask > 0).sum())

    # Compute bbox_area from surviving (in_canvas) splat coordinates
    if len(u) > 0:
        bbox_area = int((int(u.max()) - int(u.min()) + 1) * (int(v.max()) - int(v.min()) + 1))
    else:
        bbox_area = 0
    info["bbox_area_px"] = bbox_area

    # 5. Fill external contours (RETR_EXTERNAL + CHAIN_APPROX_NONE preserves concavities)
    filled = np.zeros_like(mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if contours:
        cv2.drawContours(filled, contours, -1, 255, thickness=cv2.FILLED)

    n_external_contours     = len(contours)
    largest_contour_area_px = int(max((cv2.contourArea(c) for c in contours), default=0))
    info["n_external_contours"]     = n_external_contours
    info["largest_contour_area_px"] = largest_contour_area_px

    filled_px = int((filled > 0).sum())
    info["pre_fallback_filled_px"] = filled_px

    # 6. Extended convex-hull fallback — triggers if ANY condition is met:
    #    (a) fewer than 50 filled pixels (original gate)
    #    (b) fill ratio vs bounding-box area is too low
    #    (c) too many disconnected contour fragments
    #    (d) largest contour is not dominant among filled pixels
    fallback_reason = "none"
    if filled_px < 50:
        fallback_reason = "too_few_pixels"
    elif bbox_area > 0 and (filled_px / bbox_area) < MIN_FILL_RATIO_VS_BBOX:
        fallback_reason = "low_fill_vs_bbox"
    elif n_external_contours > MAX_CONTOUR_COUNT_FOR_FILLED_MASK:
        fallback_reason = "too_many_contours"
    elif filled_px > 0 and (largest_contour_area_px / filled_px) < MIN_LARGEST_CONTOUR_FRACTION:
        fallback_reason = "largest_contour_too_small"

    info["fallback_reason"] = fallback_reason

    if fallback_reason != "none":
        info["convex_hull_fallback"] = True
        if len(u) >= 3:
            pts = np.stack([u, v], axis=1).reshape(-1, 1, 2).astype(np.int32)
            hull = cv2.convexHull(pts)
            filled = np.zeros((CANVAS_PX, CANVAS_PX), dtype=np.uint8)
            cv2.fillPoly(filled, [hull], 255)
        elif len(u) > 0:
            filled = mask.copy()
        # else: remains all-zero

    info["post_fallback_filled_px"] = int((filled > 0).sum())

    return filled, info


# ── SCORING ───────────────────────────────────────────────────────────────────

def score_pair(
    xy_piece:    np.ndarray,
    mask_c:      np.ndarray,   # dilated cavity mask, uint8
    mask_c_undil: np.ndarray,  # undilated cavity mask, uint8
) -> tuple[list[dict], dict]:
    """
    Test all rotations for one (piece, cavity) pair.

    Returns:
      rotation_records  — list of 180 dicts (one per angle)
      best_record       — the record with highest score
    """
    # Centroid-centre the piece (should already be centred but enforce)
    xy_c = xy_piece - xy_piece.mean(axis=0)

    c_area    = float((mask_c_undil > 0).sum())    # undilated area for area_ratio
    c_dil_area = float((mask_c > 0).sum())

    rotation_records = []
    best_score   = -1e9
    best_record  = None

    n_angles = 360 // ROTATION_STEP_DEG
    for i in range(n_angles):
        theta = i * ROTATION_STEP_DEG
        rad   = math.radians(theta)
        cos_t = math.cos(rad)
        sin_t = math.sin(rad)

        # Rotate piece XY around origin
        rot_mat   = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float32)
        xy_rot    = xy_c @ rot_mat

        mask_p, p_info = rasterise_xy_to_mask(xy_rot)

        p_area = float((mask_p > 0).sum())

        # Containment metrics use the DILATED cavity mask (tolerance/clearance intent)
        inter_dil = float(((mask_p > 0) & (mask_c > 0)).sum())
        inside    = inter_dil / max(p_area, 1.0)
        outside   = 1.0 - inside

        # IoU uses the NON-DILATED cavity opening mask (true geometric overlap)
        p_bin   = mask_p > 0
        c_bin   = mask_c_undil > 0
        iou_inter = float((p_bin & c_bin).sum())
        iou_union = float((p_bin | c_bin).sum())
        iou       = iou_inter / max(iou_union, 1.0)

        score   = W_IOU * iou + W_INSIDE * inside - W_OUTSIDE * outside

        record = {
            "rotation_deg":       theta,
            "inside_ratio":       round(inside,    6),
            "outside_ratio":      round(outside,   6),
            "iou":                round(iou,       6),
            "score":              round(score,     6),
            "p_area_px":          int(p_area),
            "iou_intersection_px": int(iou_inter),
            "iou_union_px":        int(iou_union),
            "convex_hull_fallback_piece": p_info["convex_hull_fallback"],
        }
        rotation_records.append(record)

        if score > best_score:
            best_score  = score
            best_record = record

    # Best rotation mask (re-rasterise at best theta)
    best_theta = best_record["rotation_deg"]
    rad        = math.radians(best_theta)
    cos_t, sin_t = math.cos(rad), math.sin(rad)
    rot_mat   = np.array([[cos_t, sin_t], [-sin_t, cos_t]], dtype=np.float32)
    xy_rot    = xy_c @ rot_mat
    mask_p_best, _ = rasterise_xy_to_mask(xy_rot)

    p_area_best = float((mask_p_best > 0).sum())
    area_ratio  = (min(p_area_best, c_area) / max(p_area_best, c_area, 1.0))
    suspicious  = area_ratio < SUSPICIOUS_AREA_RATIO_MAX

    best_record["p_area_at_best_px"]     = int(p_area_best)
    best_record["c_undilated_area_px"]   = int(c_area)
    best_record["c_dilated_area_px"]     = int(c_dil_area)
    best_record["area_ratio"]            = round(area_ratio, 6)
    best_record["suspicious_scale"]      = suspicious
    best_record["mask_p_best"]           = mask_p_best   # retained for overlay; not serialised
    # New diagnostic fields (four required by experiment spec)
    best_record["opening_mask_area_px"]  = int(c_area)       # non-dilated opening area
    best_record["dilated_mask_area_px"]  = int(c_dil_area)   # dilated mask area
    # iou_intersection_px / iou_union_px already stored per-rotation; expose best-rotation values
    best_record["iou_intersection_px"]   = int(best_record.get("iou_intersection_px", 0))
    best_record["iou_union_px"]          = int(best_record.get("iou_union_px", 0))

    return rotation_records, best_record


# ── OVERLAY ───────────────────────────────────────────────────────────────────

def make_overlay(
    piece_name:   str,
    cavity_id:    str,
    mask_p:       np.ndarray,   # uint8 piece mask at best rotation
    mask_c:       np.ndarray,   # uint8 dilated cavity mask
    mask_c_undil: np.ndarray,   # uint8 undilated cavity mask
    best_record:  dict,
    low_raw:      bool,
    chf_cavity:   bool,
    scale:        int = OVERLAY_SCALE,
) -> np.ndarray:
    """
    Render coloured overlay (BGR, uint8) at CANVAS_PX*scale resolution.

    Colours:
      background   — white
      cavity mask  — blue  (255,   0,   0)
      piece mask   — red   (  0,   0, 255)
      overlap      — green (  0, 255,   0)
      piece outside cavity — yellow (0, 255, 255)
      undilated cavity contour — dark blue (180, 50, 20)
    """
    canvas_big = CANVAS_PX * scale

    p = mask_p > 0
    c = mask_c > 0
    overlap  = p & c
    only_p   = p & ~c
    only_c   = c & ~p

    img = np.full((CANVAS_PX, CANVAS_PX, 3), 255, dtype=np.uint8)   # white bg

    img[only_c]  = (255,   0,   0)   # blue  — cavity only
    img[only_p]  = (  0,   0, 255)   # red   — piece only (outside)
    img[overlap] = (  0, 255,   0)   # green — overlap

    # Upscale
    img = cv2.resize(img, (canvas_big, canvas_big), interpolation=cv2.INTER_NEAREST)

    # Undilated cavity contour
    c_undil_big = cv2.resize(mask_c_undil, (canvas_big, canvas_big),
                             interpolation=cv2.INTER_NEAREST)
    contours_u, _ = cv2.findContours(c_undil_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours_u, -1, (180, 50, 20), 1)

    # Text annotations
    rot   = best_record["rotation_deg"]
    score = best_record["score"]
    ins   = best_record["inside_ratio"]
    outs  = best_record["outside_ratio"]
    iou_v = best_record["iou"]
    ar    = best_record.get("area_ratio", 0.0)
    chf_p = best_record.get("convex_hull_fallback_piece", False)
    susp  = best_record.get("suspicious_scale", False)

    font  = cv2.FONT_HERSHEY_SIMPLEX
    fscl  = 0.45
    thick = 1
    col   = (0, 0, 0)

    lines = [
        f"piece={piece_name}  cavity={cavity_id}  rot={rot}deg  score={score:.3f}",
        f"inside={ins:.3f}  outside={outs:.3f}  iou={iou_v:.3f}  area_ratio={ar:.3f}",
        f"flags: suspicious_scale={susp}  low_raw_support={low_raw}  "
        f"chf_piece={chf_p}  chf_cavity={chf_cavity}",
    ]
    y0 = 16
    for k, line in enumerate(lines):
        cv2.putText(img, line, (6, y0 + k * 16), font, fscl, col, thick, cv2.LINE_AA)

    return img


# ── SCORE CURVE ───────────────────────────────────────────────────────────────

def make_score_curve_matplotlib(
    records: list[dict],
    best_theta: int,
    out_path: Path,
    title: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thetas = [r["rotation_deg"] for r in records]
    scores = [r["score"]        for r in records]
    ious   = [r["iou"]          for r in records]

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.plot(thetas, scores, label="score",  color="blue")
    ax.plot(thetas, ious,   label="iou",    color="green", linestyle="--")
    ax.axvline(best_theta, color="red", linestyle=":", linewidth=1.5,
               label=f"best={best_theta}deg")
    ax.set_xlabel("rotation (deg)")
    ax.set_ylabel("value")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.set_xlim(0, 358)
    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100)
    plt.close(fig)


def make_score_curve_opencv(
    records: list[dict],
    best_theta: int,
    out_path: Path,
    title: str,
) -> None:
    """Minimal OpenCV fallback: draw score and iou curves as a line plot."""
    H, W = 200, 600
    img = np.full((H, W, 3), 245, dtype=np.uint8)

    thetas = np.array([r["rotation_deg"] for r in records], dtype=np.float32)
    scores = np.array([r["score"]        for r in records], dtype=np.float32)
    ious   = np.array([r["iou"]          for r in records], dtype=np.float32)

    margin_l, margin_r, margin_t, margin_b = 40, 10, 20, 30
    plot_w = W - margin_l - margin_r
    plot_h = H - margin_t - margin_b

    def _to_pixel(x_vals, y_vals, x_range=(0, 358), y_range=(0.0, 1.0)):
        xs = ((x_vals - x_range[0]) / (x_range[1] - x_range[0]) * plot_w + margin_l).astype(np.int32)
        ys = (H - margin_b - (y_vals - y_range[0]) / (y_range[1] - y_range[0]) * plot_h).astype(np.int32)
        return xs, ys

    y_min = float(min(scores.min(), ious.min()))
    y_max = float(max(scores.max(), ious.max()))
    y_range = (y_min - 0.05, y_max + 0.05)

    xs_s, ys_s = _to_pixel(thetas, scores, y_range=y_range)
    xs_i, ys_i = _to_pixel(thetas, ious,   y_range=y_range)

    # Draw axes
    cv2.rectangle(img, (margin_l, margin_t), (W - margin_r, H - margin_b), (0, 0, 0), 1)

    # Draw curves
    for k in range(len(xs_s) - 1):
        cv2.line(img, (xs_s[k], ys_s[k]), (xs_s[k + 1], ys_s[k + 1]), (200, 0, 0), 1)
        cv2.line(img, (xs_i[k], ys_i[k]), (xs_i[k + 1], ys_i[k + 1]), (0, 160, 0), 1)

    # Best-rotation vertical line
    bx = int((best_theta / 358.0) * plot_w + margin_l)
    cv2.line(img, (bx, margin_t), (bx, H - margin_b), (0, 0, 200), 1)

    # Title
    cv2.putText(img, title, (margin_l, margin_t - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.4, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)


def make_score_curve(
    records: list[dict],
    best_theta: int,
    out_path: Path,
    title: str,
) -> None:
    try:
        import matplotlib  # noqa: F401
        make_score_curve_matplotlib(records, best_theta, out_path, title)
    except ImportError:
        make_score_curve_opencv(records, best_theta, out_path, title)


# ── HEATMAP ───────────────────────────────────────────────────────────────────

def make_heatmap_matplotlib(
    piece_names:  list[str],
    cavity_ids:   list[str],
    score_matrix: np.ndarray,   # shape (n_pieces, n_cavities)
    out_path:     Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(score_matrix, vmin=0.0, vmax=1.0, cmap="YlOrRd", aspect="auto")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(cavity_ids)))
    ax.set_yticks(range(len(piece_names)))
    ax.set_xticklabels(cavity_ids, fontsize=9)
    ax.set_yticklabels(piece_names, fontsize=9)
    ax.set_xlabel("cavity")
    ax.set_ylabel("piece")
    ax.set_title("Best score per (piece, cavity)")

    for i in range(len(piece_names)):
        for j in range(len(cavity_ids)):
            ax.text(j, i, f"{score_matrix[i, j]:.3f}",
                    ha="center", va="center", fontsize=8,
                    color="black" if score_matrix[i, j] < 0.7 else "white")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100)
    plt.close(fig)


def make_heatmap_opencv(
    piece_names:  list[str],
    cavity_ids:   list[str],
    score_matrix: np.ndarray,
    out_path:     Path,
) -> None:
    cell_h, cell_w = 60, 110
    label_col_w    = 90
    header_row_h   = 30
    n_p = len(piece_names)
    n_c = len(cavity_ids)

    H = header_row_h + n_p * cell_h
    W = label_col_w  + n_c * cell_w
    img = np.full((H, W, 3), 240, dtype=np.uint8)

    # Header: cavity ids
    for j, cid in enumerate(cavity_ids):
        x = label_col_w + j * cell_w + 5
        cv2.putText(img, cid, (x, header_row_h - 8), cv2.FONT_HERSHEY_SIMPLEX,
                    0.38, (0, 0, 0), 1, cv2.LINE_AA)

    for i, pname in enumerate(piece_names):
        # Row label
        y_top = header_row_h + i * cell_h
        cv2.putText(img, pname, (4, y_top + cell_h // 2 + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 1, cv2.LINE_AA)

        for j in range(n_c):
            s   = float(score_matrix[i, j])
            col = (
                int(255 * (1 - s)),
                int(200 * s),
                int(50  * s),
            )
            x0 = label_col_w + j * cell_w
            y0 = y_top
            cv2.rectangle(img, (x0, y0), (x0 + cell_w, y0 + cell_h), col, -1)
            cv2.rectangle(img, (x0, y0), (x0 + cell_w, y0 + cell_h), (80, 80, 80), 1)
            cv2.putText(img, f"{s:.3f}", (x0 + 28, y0 + cell_h // 2 + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(str(out_path), img)


def make_heatmap(
    piece_names:  list[str],
    cavity_ids:   list[str],
    score_matrix: np.ndarray,
    out_path:     Path,
) -> None:
    try:
        import matplotlib  # noqa: F401
        make_heatmap_matplotlib(piece_names, cavity_ids, score_matrix, out_path)
    except ImportError:
        make_heatmap_opencv(piece_names, cavity_ids, score_matrix, out_path)


# ── BEST MATCH GRID ───────────────────────────────────────────────────────────

def make_best_grid(
    piece_results:   list[dict],   # one dict per piece with keys from process_piece
    all_pair_masks:  dict,         # {(piece_name, cid): (mask_p_best, mask_c_dil, mask_c_undil)}
    out_path:        Path,
) -> None:
    """
    4 rows (one per piece). Each row: [piece raster | cavity raster | overlay].
    """
    cell = CANVAS_PX * 2   # 640 px cells
    col_labels = ["piece footprint", "cavity footprint", "overlay"]
    n_cols = 3
    header_h = 24
    row_label_w = 90

    rows_img = []
    for pr in piece_results:
        pname    = pr["piece"]
        best_cid = pr.get("best_cavity_id", "")
        score    = pr.get("best_score", 0.0)
        best_rot = pr.get("best_rotation_deg", 0)

        key = (pname, best_cid)
        if key not in all_pair_masks:
            # fallback: blank row
            blank = np.full((header_h + cell, row_label_w + n_cols * cell, 3), 200, dtype=np.uint8)
            cv2.putText(blank, f"{pname} — no data", (10, header_h // 2 + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            rows_img.append(blank)
            continue

        mask_p, mask_c_dil, mask_c_undil = all_pair_masks[key]

        def _to_bgr(mask_uint8: np.ndarray) -> np.ndarray:
            m = cv2.resize(mask_uint8, (cell, cell), interpolation=cv2.INTER_NEAREST)
            return cv2.cvtColor(m, cv2.COLOR_GRAY2BGR)

        p_bgr    = _to_bgr(mask_p)
        c_bgr    = _to_bgr(mask_c_dil)

        # Tiny overlay (no annotation text for compactness)
        p = mask_p > 0
        c = mask_c_dil > 0
        ov = np.full((CANVAS_PX, CANVAS_PX, 3), 255, dtype=np.uint8)
        ov[c & ~p]  = (255,   0,   0)
        ov[p & ~c]  = (  0,   0, 255)
        ov[p & c]   = (  0, 255,   0)
        ov_bgr = cv2.resize(ov, (cell, cell), interpolation=cv2.INTER_NEAREST)

        row_cells = np.hstack([p_bgr, c_bgr, ov_bgr])   # (cell, 3*cell, 3)

        # Row label bar
        label_bar = np.full((header_h, row_cells.shape[1], 3), 40, dtype=np.uint8)
        label_text = (f"{pname} -> {best_cid}  rot={best_rot}deg  "
                      f"score={score:.3f}")
        cv2.putText(label_bar, label_text, (6, header_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        row_img = np.vstack([label_bar, row_cells])
        rows_img.append(row_img)

    if not rows_img:
        print("[grid] WARNING: no rows to assemble for best_match_grid.png")
        return

    # Pad rows to same width
    max_w = max(r.shape[1] for r in rows_img)
    padded = []
    for r in rows_img:
        if r.shape[1] < max_w:
            pad = np.full((r.shape[0], max_w - r.shape[1], 3), 200, dtype=np.uint8)
            r   = np.hstack([r, pad])
        padded.append(r)

    grid = np.vstack(padded)
    cv2.imwrite(str(out_path), grid)
    print(f"[grid] best_match_grid.png -> {out_path}")


# ── ALL CAVITIES COMPARISON ───────────────────────────────────────────────────

def make_all_cavities_comparison(
    piece_name: str,
    cavity_ids: list[str],
    pair_records: dict,   # {cid: best_record}
    pair_masks:   dict,   # {cid: (mask_p_best, mask_c_dil, mask_c_undil)}
    out_path: Path,
) -> None:
    """
    Single row of 4 best-rotation overlays, one per cavity, with scores annotated.
    """
    cell = CANVAS_PX * 2   # 640 px per cell
    header_h = 40

    cells = []
    for cid in cavity_ids:
        if cid not in pair_masks or cid not in pair_records:
            blank = np.full((cell + header_h, cell, 3), 200, dtype=np.uint8)
            cv2.putText(blank, f"{cid} MISSING", (10, cell // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            cells.append(blank)
            continue

        mask_p, mask_c_dil, mask_c_undil = pair_masks[cid]
        br = pair_records[cid]

        p = mask_p > 0
        c = mask_c_dil > 0
        ov = np.full((CANVAS_PX, CANVAS_PX, 3), 255, dtype=np.uint8)
        ov[c & ~p]  = (255,   0,   0)
        ov[p & ~c]  = (  0,   0, 255)
        ov[p & c]   = (  0, 255,   0)
        ov_big = cv2.resize(ov, (cell, cell), interpolation=cv2.INTER_NEAREST)

        # Header bar per cell
        bar = np.full((header_h, cell, 3), 40, dtype=np.uint8)
        line1 = f"{cid}  rot={br['rotation_deg']}  sc={br['score']:.3f}"
        line2 = f"in={br['inside_ratio']:.3f} iou={br['iou']:.3f}"
        cv2.putText(bar, line1, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(bar, line2, (4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 230, 255), 1, cv2.LINE_AA)

        cell_img = np.vstack([bar, ov_big])
        cells.append(cell_img)

    if not cells:
        return

    # Pad to same height
    max_h = max(c.shape[0] for c in cells)
    padded = []
    for c in cells:
        if c.shape[0] < max_h:
            pad = np.full((max_h - c.shape[0], c.shape[1], 3), 200, dtype=np.uint8)
            c   = np.vstack([c, pad])
        padded.append(c)

    row_img = np.hstack(padded)
    cv2.imwrite(str(out_path), row_img)


# ── PROCESS PAIR ──────────────────────────────────────────────────────────────

def process_pair(
    piece: dict,
    cavity: dict,
    pair_out_dir: Path,
    mask_c: np.ndarray,       # dilated cavity mask (precomputed)
    mask_c_undil: np.ndarray, # undilated cavity mask (precomputed)
    c_info: dict,
) -> dict:
    """
    Score all rotations for one (piece, cavity) pair, write per-pair outputs.
    Returns pair_summary dict.
    """
    pname = piece["name"]
    cid   = cavity["cid"]

    print(f"  [pair] {pname} vs {cid} ...", end="", flush=True)

    pair_out_dir.mkdir(parents=True, exist_ok=True)

    rotation_records, best_record = score_pair(
        piece["xy"], mask_c, mask_c_undil
    )

    # ── rotation_scores.csv ───────────────────────────────────────────────────
    csv_path = pair_out_dir / "rotation_scores.csv"
    with open(str(csv_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["rotation_deg", "inside_ratio", "outside_ratio", "iou", "score"]
        )
        writer.writeheader()
        for r in rotation_records:
            writer.writerow({k: r[k] for k in writer.fieldnames})

    # ── pair_summary.json ─────────────────────────────────────────────────────
    chf_piece   = best_record.get("convex_hull_fallback_piece", False)
    chf_cavity  = c_info.get("convex_hull_fallback", False)
    low_raw_sup = cavity["low_raw_support"]

    pair_summary = {
        "piece":                   pname,
        "cavity":                  cid,
        "best_rotation_deg":       int(best_record["rotation_deg"]),
        "best_score":              float(best_record["score"]),
        "best_inside":             float(best_record["inside_ratio"]),
        "best_outside":            float(best_record["outside_ratio"]),
        "best_iou":                float(best_record["iou"]),
        "area_ratio":              float(best_record.get("area_ratio", 0.0)),
        "suspicious_scale":        bool(best_record.get("suspicious_scale", False)),
        "low_raw_support":         bool(low_raw_sup),
        "convex_hull_fallback_piece":  bool(chf_piece),
        "convex_hull_fallback_cavity": bool(chf_cavity),
        "p_area_at_best_px":       int(best_record.get("p_area_at_best_px", 0)),
        "c_undilated_area_px":     int(best_record.get("c_undilated_area_px", 0)),
        "c_dilated_area_px":       int(best_record.get("c_dilated_area_px", 0)),
        "opening_mask_area_px":    int(best_record.get("opening_mask_area_px", 0)),
        "dilated_mask_area_px":    int(best_record.get("dilated_mask_area_px", 0)),
        "iou_intersection_px":     int(best_record.get("iou_intersection_px", 0)),
        "iou_union_px":            int(best_record.get("iou_union_px", 0)),
        "failed":                  False,
        "failure_reason":          "",
        # Cavity rasteriser fallback diagnostics (new fields — run C)
        "cavity_fallback_reason":          str(c_info.get("fallback_reason", "none")),
        "cavity_pre_fallback_filled_px":   int(c_info.get("pre_fallback_filled_px", 0)),
        "cavity_post_fallback_filled_px":  int(c_info.get("post_fallback_filled_px", 0)),
        "cavity_bbox_area_px":             int(c_info.get("bbox_area_px", 0)),
        "cavity_n_external_contours":      int(c_info.get("n_external_contours", 0)),
        "cavity_largest_contour_area_px":  int(c_info.get("largest_contour_area_px", 0)),
    }

    with open(str(pair_out_dir / "pair_summary.json"), "w", encoding="utf-8") as f:
        json.dump(pair_summary, f, indent=2)

    # ── overlay_best.png ──────────────────────────────────────────────────────
    mask_p_best = best_record["mask_p_best"]
    overlay = make_overlay(
        pname, cid, mask_p_best, mask_c, mask_c_undil,
        best_record, low_raw_sup, chf_cavity
    )
    cv2.imwrite(str(pair_out_dir / "overlay_best.png"), overlay)

    # ── score_curve.png ───────────────────────────────────────────────────────
    make_score_curve(
        rotation_records,
        int(best_record["rotation_deg"]),
        pair_out_dir / "score_curve.png",
        title=f"{pname} vs {cid}",
    )

    print(f" score={best_record['score']:.3f} rot={best_record['rotation_deg']}deg "
          f"inside={best_record['inside_ratio']:.3f}")

    # Attach mask for later use by grid generators (not serialised)
    pair_summary["_mask_p_best"] = mask_p_best

    return pair_summary


# ── PROCESS PIECE ─────────────────────────────────────────────────────────────

def process_piece(
    piece:   dict,
    cavities: list[dict],
    piece_out_dir: Path,
    all_results: dict,        # mutated in-place: {piece_name: {cid: summary}}
    all_pair_masks: dict,     # mutated in-place: {(pname, cid): masks}
) -> dict:
    """
    Process all cavities for one piece. Write per-piece outputs.
    Returns piece_best dict.
    """
    pname = piece["name"]
    print(f"\n[piece] Processing '{pname}' ...")

    piece_out_dir.mkdir(parents=True, exist_ok=True)
    all_results[pname] = {}

    cavity_summaries = []   # list of pair_summary dicts (one per cavity)
    cavity_masks     = {}   # {cid: (mask_p_best, mask_c_dil, mask_c_undil)}
    cavity_records   = {}   # {cid: best_record for comparison png}

    for cavity in cavities:
        cid = cavity["cid"]
        pair_out_dir = piece_out_dir / f"vs_{cid}"

        # ── Precompute cavity masks (once per cavity per piece, but really once
        #    per cavity overall — keep logic here for simplicity)
        mask_c_undil, c_info = rasterise_xy_to_mask(cavity["xy"])
        mask_c = cv2.dilate(mask_c_undil, _DIL_KERNEL)

        try:
            summary = process_pair(
                piece, cavity, pair_out_dir, mask_c, mask_c_undil, c_info
            )
        except Exception as exc:
            tb_str = traceback.format_exc()
            print(f"\n  [pair] ERROR in {pname} vs {cid}: {exc}")
            print(tb_str)
            summary = {
                "piece":          pname,
                "cavity":         cid,
                "best_rotation_deg": 0,
                "best_score":     0.0,
                "best_inside":    0.0,
                "best_outside":   1.0,
                "best_iou":       0.0,
                "area_ratio":     0.0,
                "suspicious_scale": False,
                "low_raw_support": cavity["low_raw_support"],
                "convex_hull_fallback_piece":  False,
                "convex_hull_fallback_cavity": False,
                "p_area_at_best_px": 0,
                "c_undilated_area_px": 0,
                "c_dilated_area_px":   0,
                "failed":         True,
                "failure_reason": str(exc),
                "_mask_p_best":   np.zeros((CANVAS_PX, CANVAS_PX), dtype=np.uint8),
            }

        all_results[pname][cid] = {k: v for k, v in summary.items()
                                   if not k.startswith("_")}
        cavity_summaries.append(summary)

        mask_p_best = summary.get("_mask_p_best",
                                  np.zeros((CANVAS_PX, CANVAS_PX), dtype=np.uint8))
        cavity_masks[cid]   = (mask_p_best, mask_c, mask_c_undil)
        cavity_records[cid] = {
            "rotation_deg": summary["best_rotation_deg"],
            "score":        summary["best_score"],
            "inside_ratio": summary["best_inside"],
            "iou":          summary["best_iou"],
        }

    # Sort cavities by best score descending for ranking
    cavity_summaries.sort(key=lambda s: s["best_score"], reverse=True)

    # ── Ranking ───────────────────────────────────────────────────────────────
    ranking = []
    for s in cavity_summaries:
        compatible = (
            s["best_inside"]  >= COMPATIBLE_INSIDE_MIN
            and s["best_outside"] <= COMPATIBLE_OUTSIDE_MAX
            and s["best_iou"]     >= COMPATIBLE_IOU_MIN
        )
        ranking.append({
            "cavity":          s["cavity"],
            "best_score":      s["best_score"],
            "best_rotation_deg": s["best_rotation_deg"],
            "best_inside":     s["best_inside"],
            "best_outside":    s["best_outside"],
            "best_iou":        s["best_iou"],
            "area_ratio":      s["area_ratio"],
            "suspicious_scale": s["suspicious_scale"],
            "low_raw_support": s["low_raw_support"],
            "compatible":      compatible,
            "failed":          s["failed"],
            "failure_reason":  s.get("failure_reason", ""),
        })

    with open(str(piece_out_dir / "ranking.json"), "w", encoding="utf-8") as f:
        json.dump(ranking, f, indent=2)

    # ── best_match.json ───────────────────────────────────────────────────────
    top = cavity_summaries[0] if cavity_summaries else {}
    second = cavity_summaries[1] if len(cavity_summaries) > 1 else {}
    tie = (
        len(cavity_summaries) > 1
        and abs(top.get("best_score", 0.0) - second.get("best_score", 0.0)) < TIE_MARGIN
    )
    tie_candidates = (
        [s["cavity"] for s in cavity_summaries
         if abs(s["best_score"] - top.get("best_score", 0.0)) < TIE_MARGIN]
        if tie else []
    )

    best_inside  = top.get("best_inside",  0.0)
    best_outside = top.get("best_outside", 1.0)
    best_iou     = top.get("best_iou",     0.0)
    compatible   = (
        best_inside  >= COMPATIBLE_INSIDE_MIN
        and best_outside <= COMPATIBLE_OUTSIDE_MAX
        and best_iou     >= COMPATIBLE_IOU_MIN
    )

    failure_reason = ""
    if not compatible:
        parts = []
        if best_inside  < COMPATIBLE_INSIDE_MIN:
            parts.append(f"inside={best_inside:.3f}<{COMPATIBLE_INSIDE_MIN}")
        if best_outside > COMPATIBLE_OUTSIDE_MAX:
            parts.append(f"outside={best_outside:.3f}>{COMPATIBLE_OUTSIDE_MAX}")
        if best_iou     < COMPATIBLE_IOU_MIN:
            parts.append(f"iou={best_iou:.3f}<{COMPATIBLE_IOU_MIN}")
        failure_reason = "; ".join(parts)

    piece_best = {
        "piece":             pname,
        "best_cavity_id":    top.get("cavity",             ""),
        "best_rotation_deg": top.get("best_rotation_deg",  0),
        "best_score":        top.get("best_score",         0.0),
        "best_inside":       best_inside,
        "best_outside":      best_outside,
        "best_iou":          best_iou,
        "area_ratio":        top.get("area_ratio",         0.0),
        "suspicious_scale":  top.get("suspicious_scale",   False),
        "low_raw_support":   top.get("low_raw_support",    False),
        "compatible":        compatible,
        "tie":               tie,
        "tie_candidates":    tie_candidates,
        "failure_reason":    failure_reason,
    }

    with open(str(piece_out_dir / "best_match.json"), "w", encoding="utf-8") as f:
        json.dump(piece_best, f, indent=2)

    # ── all_cavities_comparison.png ───────────────────────────────────────────
    cavity_ids_sorted = [c["cid"] for c in cavities]
    make_all_cavities_comparison(
        pname,
        cavity_ids_sorted,
        cavity_records,
        cavity_masks,
        piece_out_dir / "all_cavities_comparison.png",
    )

    # Store masks for global grid
    best_cid = piece_best["best_cavity_id"]
    if best_cid and best_cid in cavity_masks:
        all_pair_masks[(pname, best_cid)] = cavity_masks[best_cid]

    return piece_best


# ── GLOBAL OUTPUTS ────────────────────────────────────────────────────────────

def write_global_outputs(
    pieces:        list[dict],
    cavities:      list[dict],
    all_results:   dict,
    piece_bests:   list[dict],
    all_pair_masks: dict,
    run_start:     datetime,
    run_success:   bool,
) -> None:
    piece_names = [p["name"] for p in pieces]
    cavity_ids  = [c["cid"]  for c in cavities]

    # ── results_all.json ──────────────────────────────────────────────────────
    with open(str(OUT_ROOT / "results_all.json"), "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)

    # ── results_matrix.csv ────────────────────────────────────────────────────
    score_matrix = np.zeros((len(piece_names), len(cavity_ids)), dtype=np.float32)
    csv_matrix_path = OUT_ROOT / "results_matrix.csv"
    with open(str(csv_matrix_path), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["piece"] + cavity_ids)
        for i, pname in enumerate(piece_names):
            row = [pname]
            for j, cid in enumerate(cavity_ids):
                s = all_results.get(pname, {}).get(cid, {}).get("best_score", 0.0)
                score_matrix[i, j] = s
                row.append(f"{s:.6f}")
            writer.writerow(row)

    # ── summary.txt ───────────────────────────────────────────────────────────
    summary_lines = [
        "Baseline 1 — Geometric Matching Summary",
        f"Run: {run_start.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"{'piece':<12} {'best_cavity':<12} {'rot':>5} {'score':>7} "
        f"{'inside':>7} {'outside':>8} {'iou':>7} {'ar':>6} "
        f"{'susp':>5} {'low_raw':>7} {'compat':>7}",
        "-" * 95,
    ]
    for pb in piece_bests:
        summary_lines.append(
            f"{pb['piece']:<12} {pb['best_cavity_id']:<12} "
            f"{pb['best_rotation_deg']:>5} {pb['best_score']:>7.3f} "
            f"{pb['best_inside']:>7.3f} {pb['best_outside']:>8.3f} "
            f"{pb['best_iou']:>7.3f} {pb['area_ratio']:>6.3f} "
            f"{str(pb['suspicious_scale']):<5} {str(pb['low_raw_support']):<7} "
            f"{str(pb['compatible']):<7}"
        )
    summary_text = "\n".join(summary_lines) + "\n"
    (OUT_ROOT / "summary.txt").write_text(summary_text, encoding="utf-8")
    print("\n" + summary_text)

    # ── run_metadata.json ─────────────────────────────────────────────────────
    run_end = datetime.now()
    run_meta = {
        "script":    "baseline1_geometric_matching.py",
        "timestamp": run_start.isoformat(),
        "duration_s": (run_end - run_start).total_seconds(),
        "inputs": {
            "pieces":   piece_names,
            "cavities": cavity_ids,
        },
        "parameters": {
            "ROTATION_STEP_DEG":             ROTATION_STEP_DEG,
            "CLEARANCE_DILATION_M":          CLEARANCE_DILATION_M,
            "FOOTPRINT_RESOLUTION_M_PER_PX": FOOTPRINT_RESOLUTION_M_PER_PX,
            "WORLD_CANVAS_M":                WORLD_CANVAS_M,
            "CANVAS_PX":                     CANVAS_PX,
            "COMPATIBLE_INSIDE_MIN":         COMPATIBLE_INSIDE_MIN,
            "COMPATIBLE_OUTSIDE_MAX":        COMPATIBLE_OUTSIDE_MAX,
            "COMPATIBLE_IOU_MIN":            COMPATIBLE_IOU_MIN,
            "SUSPICIOUS_AREA_RATIO_MAX":     SUSPICIOUS_AREA_RATIO_MAX,
            "LOW_RAW_SUPPORT_AREA_PX":       LOW_RAW_SUPPORT_AREA_PX,
            "TIE_MARGIN":                    TIE_MARGIN,
            "W_IOU":                         W_IOU,
            "W_INSIDE":                      W_INSIDE,
            "W_OUTSIDE":                     W_OUTSIDE,
        },
        "output_root": str(OUT_ROOT),
        "success":     run_success,
    }
    with open(str(OUT_ROOT / "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(run_meta, f, indent=2)

    # ── score_matrix_heatmap.png ──────────────────────────────────────────────
    make_heatmap(piece_names, cavity_ids, score_matrix, OUT_ROOT / "score_matrix_heatmap.png")

    # ── best_match_grid.png ───────────────────────────────────────────────────
    make_best_grid(piece_bests, all_pair_masks, OUT_ROOT / "best_match_grid.png")

    print(f"\n[outputs] All global outputs written to: {OUT_ROOT}")
    for f in sorted(OUT_ROOT.iterdir()):
        if f.is_file():
            print(f"  {f.name}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    run_start = datetime.now()

    # ── Clean and recreate output directory ───────────────────────────────────
    if OUT_ROOT.exists():
        shutil.rmtree(str(OUT_ROOT))
        print(f"[main] removed previous output: {OUT_ROOT}")
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    log_path = OUT_ROOT / "run_log.txt"
    setup_run_logging(log_path)

    print("=" * 70)
    print("baseline1_geometric_matching.py — Baseline 1: Geometric Matching")
    print("=" * 70)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"OUT_ROOT     : {OUT_ROOT}")
    print(f"CANVAS_PX    : {CANVAS_PX}  ({WORLD_CANVAS_M*1000:.0f}mm @ "
          f"{FOOTPRINT_RESOLUTION_M_PER_PX*1000:.3f}mm/px)")
    print(f"ROTATIONS    : {360 // ROTATION_STEP_DEG} @ {ROTATION_STEP_DEG}deg step")
    print(f"DIL_RADIUS   : {_DIL_RADIUS_PX} px ({CLEARANCE_DILATION_M*1000:.1f}mm)")

    run_success = False
    all_results  = {}
    piece_bests  = []
    all_pair_masks = {}

    try:
        pieces, cavities = load_inputs()

        if not pieces:
            raise RuntimeError("No pieces loaded — check PIECES_DIR and folder names")
        if not cavities:
            raise RuntimeError("No cavities loaded — check CAVITIES_DIR")

        for piece in pieces:
            pname         = piece["name"]
            piece_out_dir = OUT_ROOT / pname

            try:
                pb = process_piece(piece, cavities, piece_out_dir,
                                   all_results, all_pair_masks)
                piece_bests.append(pb)
            except Exception as exc:
                tb_str = traceback.format_exc()
                print(f"\n[main] ERROR processing piece '{pname}': {exc}")
                print(tb_str)
                piece_bests.append({
                    "piece":             pname,
                    "best_cavity_id":    "",
                    "best_rotation_deg": 0,
                    "best_score":        0.0,
                    "best_inside":       0.0,
                    "best_outside":      1.0,
                    "best_iou":          0.0,
                    "area_ratio":        0.0,
                    "suspicious_scale":  False,
                    "low_raw_support":   False,
                    "compatible":        False,
                    "tie":               False,
                    "tie_candidates":    [],
                    "failure_reason":    str(exc),
                    "failed":            True,
                })

        write_global_outputs(
            pieces, cavities, all_results, piece_bests,
            all_pair_masks, run_start, run_success=True
        )
        run_success = True

    except Exception as exc:
        tb_str = traceback.format_exc()
        print(f"\n[main] FATAL ERROR: {exc}")
        print(tb_str)
    finally:
        status = "SUCCESS" if run_success else "FAILURE"
        print(f"\n{'='*70}")
        print(f"  Run {status}  ({(datetime.now()-run_start).total_seconds():.1f}s)")
        print(f"{'='*70}\n")
        teardown_run_logging()


if __name__ == "__main__":
    main()
