"""
sweep_cavity_opening_params.py

Offline parameter sweep for the cavity-opening derivation in
scripts/capture_cavity_detection.py.

Goal: find values of the board-surface depth tolerance, cavity-opening
expansion, and morphological cleanup that produce stable, full cavity
openings — without re-running Isaac Sim.

Inputs (read from data/cavities_detected/):
  depth.npy                   — REQUIRED, raw depth (float32, metres)
  board_region_mask.png       — REQUIRED, filled board footprint
  board_surface_mask.png      — optional, current board top mask (baseline)
  cavities_summary.json       — REQUIRED, for board_surface_depth_m, fx_px,
                                  fy_px, image_resolution
  rgb.png                     — optional, for the overlay underlay
  data/expected_cad_dimensions.json — optional, REFERENCE-ONLY CAD legend.
                                       NOT used to drive the score.

Outputs (written to data/cavities_detected/param_sweep/):
  sweep_summary.csv
  sweep_summary.json
  sweep_grid.png
  best_candidate_debug.png

Hard deps: numpy, opencv.  matplotlib optional (used for the grid layout).

Read-only with respect to capture/baseline/validate scripts.  Does not modify
any masks, just recomputes them from the raw depth array.
"""

from __future__ import annotations

import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import cv2

try:
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    _HAVE_MPL = True
except Exception:
    _HAVE_MPL = False


# ── PATH RESOLUTION ──────────────────────────────────────────────────────────
PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        str(Path(__file__).resolve().parent.parent),
    )
)
DATA_DIR = PROJECT_ROOT / "data" / "cavities_detected"
OUT_DIR  = DATA_DIR / "param_sweep"
CAD_PATH = PROJECT_ROOT / "data" / "expected_cad_dimensions.json"

# ── SWEEP CONFIG ─────────────────────────────────────────────────────────────
TOLERANCE_M_VALUES   = [0.001, 0.002, 0.003, 0.005, 0.008]
EXPAND_PX_VALUES     = [0, 1, 2, 3, 4]
MORPHOLOGY_VALUES    = ["none", "close_3", "close_5"]

# Component filters — match what capture_cavity_detection.py uses.
CC_MIN_AREA_PX = 200
CC_MAX_AREA_PX = 30000

# Plausibility tagging thresholds (mm and px).
SPAN_TOO_SMALL_MM    = 20.0
SPAN_TOO_LARGE_MM    = 100.0
AREA_TOO_LARGE_PX    = 20000
NOISY_AREA_PX        = 400
NOISY_SPAN_MM        = 25.0
EXPECTED_N_CAVITIES  = 4  # mild prior, NOT shape-driven


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _die(msg: str, code: int = 1):
    print(f"[error] {msg}")
    sys.exit(code)


def _load_mask_png(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        _die(f"could not read {path}")
    return img > 127


def _load_summary() -> dict:
    p = DATA_DIR / "cavities_summary.json"
    if not p.exists():
        _die(f"missing {p}")
    with open(p) as f:
        return json.load(f)


def _load_cad() -> dict:
    if not CAD_PATH.exists():
        return {}
    try:
        with open(CAD_PATH) as f:
            return json.load(f)
    except Exception:
        return {}


def _morph(mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return mask
    if mode == "close_3":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    if mode == "close_5":
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        return cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, k) > 0
    return mask


def _dilate(mask: np.ndarray, expand_px: int) -> np.ndarray:
    if expand_px <= 0:
        return mask
    r = expand_px
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*r + 1, 2*r + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def _classify(area_px: int, span_x_mm: float, span_y_mm: float) -> str:
    if (span_x_mm > SPAN_TOO_LARGE_MM or span_y_mm > SPAN_TOO_LARGE_MM
            or area_px > AREA_TOO_LARGE_PX):
        return "too_large_or_merged"
    if span_x_mm < SPAN_TOO_SMALL_MM or span_y_mm < SPAN_TOO_SMALL_MM:
        return "too_small"
    if (area_px < NOISY_AREA_PX
            and (span_x_mm < NOISY_SPAN_MM or span_y_mm < NOISY_SPAN_MM)):
        return "noisy"
    return "plausible"


def _eval_combo(opening_mask: np.ndarray, fx_px: float, fy_px: float,
                board_z: float) -> list:
    """Connected components + per-component metrics. Returns list of dicts."""
    binary = (opening_mask.astype(np.uint8)) * 255
    n, _labels, stats, centroids = cv2.connectedComponentsWithStats(binary)
    comps = []
    for i in range(1, n):  # 0 = background
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (CC_MIN_AREA_PX <= area <= CC_MAX_AREA_PX):
            continue
        bx = int(stats[i, cv2.CC_STAT_LEFT])
        by = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        # Pinhole at board surface depth.
        span_x_m = bw / fx_px * board_z
        span_y_m = bh / fy_px * board_z
        note = _classify(area, span_x_m * 1000.0, span_y_m * 1000.0)
        comps.append({
            "label":    int(i),
            "area_px":  area,
            "centroid_px": (float(centroids[i][0]), float(centroids[i][1])),
            "bbox_px":  (bx, by, bw, bh),
            "xy_span_m": {"x": span_x_m, "y": span_y_m},
            "xy_span_mm": {"x": span_x_m * 1000.0, "y": span_y_m * 1000.0},
            "note":     note,
        })
    return comps


def _score_combo(comps: list) -> dict:
    """Plausibility score — does NOT use CAD."""
    n_total     = len(comps)
    n_plausible = sum(1 for c in comps if c["note"] == "plausible")
    n_noisy     = sum(1 for c in comps if c["note"] == "noisy")
    n_too_large = sum(1 for c in comps if c["note"] == "too_large_or_merged")
    score = (n_plausible
             - 2.0 * n_too_large
             - 0.5 * n_noisy
             - 0.5 * abs(n_total - EXPECTED_N_CAVITIES))
    return {
        "n_components": n_total,
        "n_plausible":  n_plausible,
        "n_noisy":      n_noisy,
        "n_too_large":  n_too_large,
        "score":        score,
    }


def _draw_overview(opening_mask: np.ndarray, comps: list, rgb: np.ndarray,
                   title: str, height: int = 240) -> np.ndarray:
    """Render an overlay panel for the grid: opening mask tinted on RGB,
    bboxes + per-component span text, with the title on top."""
    h, w = opening_mask.shape
    if rgb is not None:
        base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    else:
        base = np.zeros((h, w, 3), dtype=np.uint8)
        base[opening_mask] = (60, 60, 60)

    # Tint opening pixels in cyan
    base[opening_mask] = (
        base[opening_mask].astype(np.float32) * 0.35
        + np.array([255, 220, 60], np.float32) * 0.65
    ).astype(np.uint8)

    palette = [(255, 80, 80), (60, 200, 60), (60, 160, 255), (255, 200, 60),
               (220, 60, 220), (60, 220, 220), (255, 140, 60)]
    for i, c in enumerate(comps):
        col = palette[i % len(palette)]
        bx, by, bw, bh = c["bbox_px"]
        cv2.rectangle(base, (bx, by), (bx + bw, by + bh), col, 2)
        cx_p, cy_p = int(c["centroid_px"][0]), int(c["centroid_px"][1])
        cv2.circle(base, (cx_p, cy_p), 4, col, -1)
        label = (f"{c['xy_span_mm']['x']:.1f}×{c['xy_span_mm']['y']:.1f}mm "
                 f"a={c['area_px']} {c['note']}")
        cv2.putText(base, label, (bx, max(by - 4, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, col, 1, cv2.LINE_AA)

    # Title bar
    title_h = 22
    tile = np.zeros((h + title_h, w, 3), dtype=np.uint8)
    tile[title_h:] = base
    cv2.putText(tile, title, (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    # Rescale to a uniform height for grid composition
    scale = height / float(tile.shape[0])
    new_w = int(round(tile.shape[1] * scale))
    return cv2.resize(tile, (new_w, height), interpolation=cv2.INTER_AREA)


def _build_grid_image(panels_per_cell: dict, out_path: Path,
                      tol_values: list, expand_values: list):
    """Build a grid where rows = tolerance, cols = expand_px.  Each cell has
    the best-morph panel (chosen by score)."""
    cells = []
    for t in tol_values:
        row = []
        for e in expand_values:
            key = (t, e)
            best = panels_per_cell.get(key)
            if best is None:
                row.append(np.zeros((240, 320, 3), dtype=np.uint8))
            else:
                row.append(best)
        # Pad row to matching widths
        max_w = max(c.shape[1] for c in row)
        row = [
            c if c.shape[1] == max_w
            else np.hstack([c, np.zeros((c.shape[0], max_w - c.shape[1], 3),
                                          dtype=np.uint8)])
            for c in row
        ]
        cells.append(np.hstack(row))
    max_w = max(r.shape[1] for r in cells)
    cells = [
        r if r.shape[1] == max_w
        else np.hstack([r, np.zeros((r.shape[0], max_w - r.shape[1], 3),
                                      dtype=np.uint8)])
        for r in cells
    ]
    big = np.vstack(cells)
    cv2.imwrite(str(out_path), big)


def _build_best_debug(opening_mask: np.ndarray, comps: list, rgb: np.ndarray,
                       params: dict, score_info: dict, cad: dict,
                       out_path: Path):
    """Full-size annotated overview of the best parameter set."""
    h, w = opening_mask.shape
    if rgb is not None:
        base = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    else:
        base = np.zeros((h, w, 3), dtype=np.uint8)
    base[opening_mask] = (
        base[opening_mask].astype(np.float32) * 0.35
        + np.array([255, 220, 60], np.float32) * 0.65
    ).astype(np.uint8)

    palette = [(255, 80, 80), (60, 200, 60), (60, 160, 255), (255, 200, 60)]
    for i, c in enumerate(comps):
        col = palette[i % len(palette)]
        bx, by, bw, bh = c["bbox_px"]
        cv2.rectangle(base, (bx, by), (bx + bw, by + bh), col, 2)
        cx_p, cy_p = int(c["centroid_px"][0]), int(c["centroid_px"][1])
        cv2.circle(base, (cx_p, cy_p), 5, col, -1)
        label = (f"#{i}  {c['xy_span_mm']['x']:.1f}×{c['xy_span_mm']['y']:.1f}mm  "
                 f"area={c['area_px']}  {c['note']}")
        cv2.putText(base, label, (bx, max(by - 6, 14)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    # Header band with parameters + CAD reference (printed only)
    band_h = 110
    band   = np.zeros((band_h, w, 3), dtype=np.uint8)
    cv2.putText(band, f"BEST candidate (NOT chosen via CAD)", (10, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(band,
                f"tol={params['tolerance_m']*1000:.1f}mm  "
                f"expand={params['expand_px']}px  morph={params['morph']}  "
                f"score={score_info['score']:.2f}  "
                f"(plaus={score_info['n_plausible']}/"
                f"{score_info['n_components']}, noisy={score_info['n_noisy']}, "
                f"too_large={score_info['n_too_large']})",
                (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 255), 1, cv2.LINE_AA)
    # CAD reference (printed only)
    cad_pieces  = (cad.get("cavities") or {})
    if cad_pieces:
        cad_lines = []
        for name in ("rectangle", "square", "circle", "triangle"):
            entry = cad_pieces.get(name)
            if not entry:
                continue
            xs = entry.get("x_span_m", 0) * 1000.0
            ys = entry.get("y_span_m", 0) * 1000.0
            cad_lines.append(f"{name}={xs:.0f}×{ys:.0f}mm")
        line = "CAD ref (NOT used for scoring): " + "  ".join(cad_lines)
        cv2.putText(band, line, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 200, 160), 1,
                    cv2.LINE_AA)
        clr = (cad.get("nominal_clearance_total_m") or 0) * 1000.0
        cv2.putText(band,
                    f"CAD clearance (total / per side): "
                    f"{clr:.1f}mm / "
                    f"{(cad.get('nominal_clearance_per_side_m') or 0)*1000:.1f}mm",
                    (10, 92),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 200, 160), 1,
                    cv2.LINE_AA)

    final = np.vstack([band, base])
    cv2.imwrite(str(out_path), final)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("sweep_cavity_opening_params.py")
    print("=" * 60)
    print(f"DATA_DIR : {DATA_DIR}")
    print(f"OUT_DIR  : {OUT_DIR}")

    if not DATA_DIR.exists():
        _die(f"DATA_DIR does not exist: {DATA_DIR}")

    depth_path = DATA_DIR / "depth.npy"
    if not depth_path.exists():
        _die(
            f"depth.npy not found at {depth_path}. "
            f"Run scripts/capture_cavity_detection.py first (the capture "
            f"script now writes it on every run)."
        )

    region_mask_path = DATA_DIR / "board_region_mask.png"
    if not region_mask_path.exists():
        _die(f"board_region_mask.png not found at {region_mask_path}")

    summary = _load_summary()
    cad     = _load_cad()

    # Pull intrinsics + board surface depth from summary
    fx_px = summary.get("fx_px")
    fy_px = summary.get("fy_px")
    if fx_px is None or fy_px is None:
        _die("cavities_summary.json missing fx_px / fy_px (intrinsics fix not "
             "applied? Re-capture cavities after the intrinsics correction).")

    board_z = summary.get("board_surface_depth_m")
    if board_z is None:
        _die("cavities_summary.json missing board_surface_depth_m")

    print(f"[load] fx_px={fx_px:.2f}  fy_px={fy_px:.2f}  "
          f"board_surface_depth={board_z:.4f} m")

    depth          = np.load(str(depth_path)).astype(np.float32)
    region_mask    = _load_mask_png(region_mask_path)
    rgb_path       = DATA_DIR / "rgb.png"
    rgb            = None
    if rgb_path.exists():
        rgb_bgr = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
        rgb     = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)

    print(f"[load] depth={depth.shape}  region_mask={region_mask.shape}  "
          f"rgb={'yes' if rgb is not None else 'no'}")

    # Output directory cleanup
    if OUT_DIR.exists():
        for p in OUT_DIR.iterdir():
            if p.is_file():
                p.unlink()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Sweep ────────────────────────────────────────────────────────────────
    all_records = []
    panels_by_combo = {}      # (t, e, m) → overview panel
    best_per_cell   = {}      # (t, e) → best (score, panel)
    best_global     = None    # (score, params, comps, opening_mask)

    for t in TOLERANCE_M_VALUES:
        # Recomputed board surface mask = pixels close to board_z
        bsm = np.abs(depth - board_z) <= t
        for m in MORPHOLOGY_VALUES:
            for e in EXPAND_PX_VALUES:
                opening = region_mask & ~bsm
                opening = _morph(opening, m)
                opening = _dilate(opening, e)

                comps  = _eval_combo(opening, fx_px, fy_px, board_z)
                sinfo  = _score_combo(comps)

                rec = {
                    "tolerance_m": t,
                    "expand_px":   e,
                    "morph":       m,
                    "n_components": sinfo["n_components"],
                    "n_plausible":  sinfo["n_plausible"],
                    "n_noisy":      sinfo["n_noisy"],
                    "n_too_large":  sinfo["n_too_large"],
                    "score":        sinfo["score"],
                    "components":   comps,
                }
                if comps:
                    spans_x = [c["xy_span_mm"]["x"] for c in comps]
                    spans_y = [c["xy_span_mm"]["y"] for c in comps]
                    areas   = [c["area_px"] for c in comps]
                    rec["mean_xy_span_x_mm"] = float(np.mean(spans_x))
                    rec["mean_xy_span_y_mm"] = float(np.mean(spans_y))
                    rec["max_area_px"]       = max(areas)
                    rec["min_area_px"]       = min(areas)
                else:
                    rec["mean_xy_span_x_mm"] = 0.0
                    rec["mean_xy_span_y_mm"] = 0.0
                    rec["max_area_px"]       = 0
                    rec["min_area_px"]       = 0

                all_records.append(rec)

                # Build panel for the grid (one per combo)
                title = (f"t={t*1000:.1f}mm exp={e}px morph={m} "
                         f"sc={sinfo['score']:.1f}")
                panel = _draw_overview(opening, comps, rgb, title, height=240)
                panels_by_combo[(t, e, m)] = panel

                cur = best_per_cell.get((t, e))
                if cur is None or sinfo["score"] > cur[0]:
                    best_per_cell[(t, e)] = (sinfo["score"], panel)

                # Tie-breakers for best global: higher score, smaller expand,
                # smaller tolerance, prefer "none" morph.
                tiebreak = (
                    -sinfo["score"],
                    e,
                    t,
                    {"none": 0, "close_3": 1, "close_5": 2}[m],
                )
                if best_global is None or tiebreak < best_global[0]:
                    best_global = (
                        tiebreak,
                        {"tolerance_m": t, "expand_px": e, "morph": m},
                        comps, opening, sinfo,
                    )

    # ── Outputs ──────────────────────────────────────────────────────────────
    # 1. CSV
    csv_path = OUT_DIR / "sweep_summary.csv"
    cols = ["tolerance_m", "expand_px", "morph",
            "n_components", "n_plausible", "n_noisy", "n_too_large", "score",
            "mean_xy_span_x_mm", "mean_xy_span_y_mm",
            "max_area_px", "min_area_px"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_records:
            w.writerow({k: r.get(k) for k in cols})
    print(f"[save] {csv_path}")

    # 2. JSON
    json_path = OUT_DIR / "sweep_summary.json"
    with open(json_path, "w") as jf:
        json.dump(all_records, jf, indent=2, default=float)
    print(f"[save] {json_path}")

    # 3. Grid image
    grid_path = OUT_DIR / "sweep_grid.png"
    panels_for_grid = {k: v[1] for k, v in best_per_cell.items()}
    _build_grid_image(panels_for_grid, grid_path,
                      TOLERANCE_M_VALUES, EXPAND_PX_VALUES)
    print(f"[save] {grid_path}")

    # 4. Best candidate debug
    _, best_params, best_comps, best_opening, best_sinfo = best_global
    best_path = OUT_DIR / "best_candidate_debug.png"
    _build_best_debug(best_opening, best_comps, rgb, best_params,
                      best_sinfo, cad, best_path)
    print(f"[save] {best_path}")

    # ── Console: top-5 table ─────────────────────────────────────────────────
    sorted_recs = sorted(all_records,
                         key=lambda r: (-r["score"], r["expand_px"],
                                         r["tolerance_m"]))
    print("\nTop 5 parameter combinations by score:")
    print(f"{'rank':>4} {'tol(mm)':>8} {'exp':>4} {'morph':>8} "
          f"{'#cmp':>5} {'plaus':>5} {'noisy':>5} {'too_lg':>6} "
          f"{'score':>6} {'mean_span_mm':>20}")
    for i, r in enumerate(sorted_recs[:5], start=1):
        spans = (f"{r['mean_xy_span_x_mm']:5.1f}×{r['mean_xy_span_y_mm']:5.1f}")
        print(f"{i:>4} {r['tolerance_m']*1000:8.1f} {r['expand_px']:>4} "
              f"{r['morph']:>8} {r['n_components']:>5} "
              f"{r['n_plausible']:>5} {r['n_noisy']:>5} "
              f"{r['n_too_large']:>6} {r['score']:>6.2f} {spans:>20}")

    print(f"\nBest candidate: {best_params} score={best_sinfo['score']:.2f}")
    print(f"Inspect first: {best_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
