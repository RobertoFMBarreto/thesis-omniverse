"""
make_cavity_diagnostic_report.py

Read-only diagnostic for the cavity-detection outputs.  Builds:

  data/cavities_detected/cavity_diagnostic_report.png   — labelled grid of all
      relevant masks / debug images, both global and per-cavity, with on-image
      explanations of what each panel means.
  data/cavities_detected/cavity_diagnostic_report.md    — Markdown summary
      stating which files exist, which mask is the primary one, whether each
      cavity looks usable for Baseline 1, and what to inspect first.

Runs outside Isaac Sim with plain Python.  Hard deps: numpy + opencv.
matplotlib is preferred for the grid render (cleaner labels); falls back to
OpenCV if matplotlib is unavailable.

Does NOT modify any capture script, baseline script, or mask file.
"""

from __future__ import annotations

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

# ── EXPECTED FILES ───────────────────────────────────────────────────────────
GLOBAL_PANELS = [
    # (filename, title, explanation shown beside/under)
    ("rgb.png",                    "RGB",                 "scene as captured"),
    ("depth_vis.png",              "Depth",               "viridis depth map"),
    ("board_surface_mask.png",     "Board surface",       "board top (with cavity holes)"),
    ("board_region_mask.png",      "Board region",        "filled board footprint"),
    ("cavity_opening_mask.png",    "Cavity opening",      "PRIMARY — top apertures"),
    ("cavity_depth_mask.png",      "Cavity depth",        "AUX — visible deep pixels"),
    ("depth_band_cavity_mask.png", "Depth-band (legacy)", "diagnostic only"),
    ("raw_cavity_mask.png",        "Raw cavity (active)", "= mask used by CC"),
    ("cavities_debug.png",         "Cavities debug",      "labelled overlay on RGB"),
]

# Per-cavity panel layout — order matters for left-to-right reading.
PER_CAVITY_PANELS = [
    ("cavity_debug.png",             "Debug overlay"),
    ("cavity_mask.png",              "Primary mask"),
    ("cavity_opening_mask.png",      "Opening mask"),
    ("cavity_depth_mask.png",        "Depth mask (aux)"),
    ("cavity_footprint.png",         "Primary footprint"),
    ("cavity_opening_footprint.png", "Opening footprint"),
]

EXPLANATIONS = [
    "Opening mask = top aperture used for matching.",
    "Depth mask = visible bottom/deeper pixels, auxiliary only.",
    "Baseline 1 should use the opening footprint, not the depth footprint.",
]


# ── HELPERS ──────────────────────────────────────────────────────────────────

def _read_image(path: Path):
    """Read an image as BGR or grayscale.  Returns (img_or_None, exists_bool)."""
    if not path.exists():
        return None, False
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    return img, img is not None


def _to_rgb(img):
    """Coerce any image to a 3-channel RGB uint8 array for plotting."""
    if img is None:
        return None
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    if img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def _placeholder_rgb(text: str, h: int = 240, w: int = 320):
    """Black tile with white text saying which file is missing."""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (10, h // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def _list_cavity_dirs(data_dir: Path):
    return sorted([p for p in data_dir.glob("cavity_*") if p.is_dir()])


def _interpret_cavity_meta(meta: dict) -> dict:
    """Pull a small subset of fields used by the markdown analysis."""
    return {
        "primary_matching_representation":
            meta.get("primary_matching_representation"),
        "footprint_source":
            meta.get("footprint_source"),
        "xy_projection_depth_mode":
            meta.get("xy_projection_depth_mode"),
        "opening_area_px":
            meta.get("opening_area_px"),
        "depth_area_px":
            meta.get("depth_area_px"),
        "opening_xy_span_m":
            meta.get("opening_xy_span_m"),
        "depth_xy_span_m":
            meta.get("depth_xy_span_m"),
        "z_depth_median_m":
            meta.get("z_depth_median_m"),
        "z_depth_max_m":
            meta.get("z_depth_max_m"),
        "board_surface_depth_m":
            meta.get("board_surface_depth_m"),
        "cavity_id":
            meta.get("cavity_id"),
    }


def _classify_primary_footprint(meta: dict) -> str:
    """
    One of:
      "full_opening"        — opening representation is wired correctly
      "depth_only"          — primary is depth-band; risk of partial silhouette
      "partial_opening"     — opening exists but is much smaller than depth area
                              (suggests opening derivation broken)
      "unclear"             — metadata insufficient
    """
    pmr = meta.get("primary_matching_representation")
    fs  = meta.get("footprint_source")
    if pmr == "cavity_opening_pointcloud" and fs == "opening_from_board_region":
        op = meta.get("opening_area_px") or 0
        dp = meta.get("depth_area_px")  or 0
        if op == 0 and dp > 0:
            return "depth_only"
        if op > 0 and dp > 0 and op < 0.5 * dp:
            return "partial_opening"
        if op > 0:
            return "full_opening"
        return "unclear"
    if fs == "depth_band":
        return "depth_only"
    return "unclear"


def _usability_for_baseline1(meta: dict, classification: str) -> tuple:
    """Return (verdict_str, short_reason)."""
    if classification == "full_opening":
        op = meta.get("opening_area_px") or 0
        if op < 200:
            return ("MARGINAL",
                    f"opening_area_px={op} is small; check raw mask for tightness")
        return ("USABLE", "opening representation present and non-trivial")
    if classification == "partial_opening":
        return ("NOT USABLE",
                "opening mask much smaller than depth area — opening derivation broken")
    if classification == "depth_only":
        return ("NOT USABLE",
                "primary is the depth/bottom region, not the top aperture")
    return ("UNKNOWN", "insufficient metadata to judge")


# ── PNG REPORT BUILDER (matplotlib path) ─────────────────────────────────────

def _build_png_matplotlib(data_dir: Path, cavities: list, out_path: Path):
    """Render the diagnostic grid using matplotlib."""
    n_global   = len(GLOBAL_PANELS)
    n_cavities = len(cavities)
    n_per_cav  = len(PER_CAVITY_PANELS)

    # Layout: explanations row + global rows + per-cavity rows
    ncols      = 6   # tile width of the grid
    global_rows = (n_global + ncols - 1) // ncols
    cav_rows    = n_cavities * 1            # one row per cavity, n_per_cav cols
    nrows       = 1 + global_rows + cav_rows

    fig_w = max(18, n_per_cav * 3.2)
    fig_h = max(6, 3.2 * nrows)
    fig = plt.figure(figsize=(fig_w, fig_h))
    gs  = GridSpec(nrows, max(ncols, n_per_cav), figure=fig,
                   hspace=0.35, wspace=0.18)

    # Title bar at the top
    title_lines = [
        f"Cavity diagnostic report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Source: {data_dir}",
    ] + EXPLANATIONS
    fig.suptitle("\n".join(title_lines), fontsize=12, ha="center",
                 y=0.995, wrap=True)

    # Reserve the first row for nothing (empty axis used as spacer for the
    # title block; matplotlib doesn't size the title; this gives breathing room).
    spacer = fig.add_subplot(gs[0, :])
    spacer.axis("off")

    # ── GLOBAL PANELS ────────────────────────────────────────────────────────
    for i, (fname, title, explain) in enumerate(GLOBAL_PANELS):
        r, c = i // ncols, i % ncols
        ax = fig.add_subplot(gs[1 + r, c])
        img, exists = _read_image(data_dir / fname)
        if exists:
            ax.imshow(_to_rgb(img))
            ax.set_title(f"{title}\n{fname}", fontsize=8)
        else:
            ax.imshow(_placeholder_rgb(f"MISSING\n{fname}"))
            ax.set_title(f"{title} — MISSING", fontsize=8, color="red")
        ax.set_xticks([])
        ax.set_yticks([])
        # Caption-like explanation
        ax.set_xlabel(explain, fontsize=7)

    # ── PER-CAVITY PANELS ────────────────────────────────────────────────────
    base_row = 1 + global_rows
    for ci, cav_dir in enumerate(cavities):
        for pi, (fname, title) in enumerate(PER_CAVITY_PANELS):
            ax = fig.add_subplot(gs[base_row + ci, pi])
            img, exists = _read_image(cav_dir / fname)
            if exists:
                ax.imshow(_to_rgb(img))
                ax.set_title(f"{cav_dir.name} / {title}\n{fname}", fontsize=8)
            else:
                ax.imshow(_placeholder_rgb(f"MISSING\n{fname}"))
                ax.set_title(f"{cav_dir.name} / {title} — MISSING",
                             fontsize=8, color="red")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.savefig(str(out_path), dpi=110, bbox_inches="tight")
    plt.close(fig)


# ── PNG REPORT BUILDER (OpenCV fallback) ─────────────────────────────────────

def _resize_tile(img, target_w: int = 320, target_h: int = 240):
    if img is None:
        return _placeholder_rgb("MISSING", h=target_h, w=target_w)
    img = _to_rgb(img)
    return cv2.resize(img, (target_w, target_h),
                      interpolation=cv2.INTER_AREA)


def _add_caption(tile, text: str, footer: str = ""):
    h, w = tile.shape[:2]
    bar_h = 36
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[:h] = tile
    cv2.putText(out, text, (6, h + 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1, cv2.LINE_AA)
    if footer:
        cv2.putText(out, footer, (6, h + 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (180, 180, 180), 1, cv2.LINE_AA)
    return out


def _build_png_opencv(data_dir: Path, cavities: list, out_path: Path):
    """Fallback grid using OpenCV concatenation only."""
    GLOBAL_NCOLS = 6
    rows = []

    # Header bar
    header_text = "Cavity diagnostic report — " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    header_w    = GLOBAL_NCOLS * 320
    header      = np.zeros((90, header_w, 3), dtype=np.uint8)
    cv2.putText(header, header_text, (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(header, f"Source: {data_dir}", (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)
    for i, line in enumerate(EXPLANATIONS):
        cv2.putText(header, line, (10, 60 + i * 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 220, 255), 1, cv2.LINE_AA)
    rows.append(header)

    # Global panels
    tiles_g = []
    for fname, title, explain in GLOBAL_PANELS:
        img, exists = _read_image(data_dir / fname)
        tile = _resize_tile(img if exists else None)
        tiles_g.append(_add_caption(tile,
                                    f"{title}  [{fname}]",
                                    explain if exists else "MISSING"))
    while len(tiles_g) % GLOBAL_NCOLS != 0:
        tiles_g.append(_resize_tile(None))
    for r in range(0, len(tiles_g), GLOBAL_NCOLS):
        rows.append(np.hstack(tiles_g[r:r + GLOBAL_NCOLS]))

    # Per-cavity rows
    PER_NCOLS = len(PER_CAVITY_PANELS)
    pad_w = GLOBAL_NCOLS * 320
    for cav_dir in cavities:
        tiles_c = []
        for fname, title in PER_CAVITY_PANELS:
            img, exists = _read_image(cav_dir / fname)
            tile = _resize_tile(img if exists else None)
            tiles_c.append(_add_caption(tile,
                                        f"{cav_dir.name}  {title}",
                                        fname if exists else "MISSING"))
        row = np.hstack(tiles_c)
        if row.shape[1] < pad_w:
            pad = np.zeros((row.shape[0], pad_w - row.shape[1], 3),
                           dtype=np.uint8)
            row = np.hstack([row, pad])
        rows.append(row)

    # Make all row widths equal (pad shorter ones)
    max_w = max(r.shape[1] for r in rows)
    rows  = [
        r if r.shape[1] == max_w
        else np.hstack([r, np.zeros((r.shape[0], max_w - r.shape[1], 3),
                                     dtype=np.uint8)])
        for r in rows
    ]
    big = np.vstack(rows)
    cv2.imwrite(str(out_path), cv2.cvtColor(big, cv2.COLOR_RGB2BGR))


def build_png_report(data_dir: Path, cavities: list, out_path: Path):
    if _HAVE_MPL:
        try:
            _build_png_matplotlib(data_dir, cavities, out_path)
            return "matplotlib"
        except Exception as exc:
            print(f"[png] matplotlib failed: {exc}; falling back to OpenCV")
    _build_png_opencv(data_dir, cavities, out_path)
    return "opencv"


# ── MARKDOWN REPORT BUILDER ──────────────────────────────────────────────────

def build_markdown_report(data_dir: Path, cavities: list,
                           out_path: Path, png_path: Path,
                           summary_meta: dict):
    """Produce the textual companion report."""
    lines = []
    lines.append(f"# Cavity diagnostic report")
    lines.append("")
    lines.append(f"- Generated: `{datetime.now().isoformat(timespec='seconds')}`")
    lines.append(f"- Source: `{data_dir}`")
    lines.append(f"- PNG: `{png_path.name}`")
    lines.append("")

    # 1. File presence
    lines.append("## 1. File presence")
    lines.append("")
    lines.append("**Global files:**")
    for fname, title, _ in GLOBAL_PANELS:
        present = (data_dir / fname).exists()
        lines.append(f"- {'✅' if present else '❌'}  `{fname}`  ({title})")
    for extra in ("cavities_summary.json", "run_log.txt", "board_mask.png"):
        present = (data_dir / extra).exists()
        lines.append(f"- {'✅' if present else '❌'}  `{extra}`")
    lines.append("")
    lines.append(f"**Cavity folders found:** {len(cavities)}")
    for cav_dir in cavities:
        lines.append(f"- `{cav_dir.name}/`:")
        for fname, title in PER_CAVITY_PANELS:
            present = (cav_dir / fname).exists()
            lines.append(f"  - {'✅' if present else '❌'} `{fname}`  ({title})")
        for extra in ("cavity_metadata.json",
                      "cavity_pointcloud.npy",
                      "cavity_opening_pointcloud.npy",
                      "cavity_depth_pointcloud.npy"):
            present = (cav_dir / extra).exists()
            lines.append(f"  - {'✅' if present else '❌'} `{extra}`")
    lines.append("")

    # 2. Primary cavity mask
    lines.append("## 2. Primary cavity mask in use")
    lines.append("")
    cdm = summary_meta.get("cavity_detection_mode", "unknown")
    lines.append(f"`cavities_summary.json` reports "
                 f"`cavity_detection_mode = \"{cdm}\"`.")
    if cdm == "opening_from_board_region":
        lines.append("")
        lines.append("Primary mask is **`cavity_opening_mask.png`** "
                     "(the negative space inside the detected board region: "
                     "`board_region_mask AND NOT board_surface_mask`). "
                     "Each `cavity_NN/cavity_mask.png` is an alias of "
                     "`cavity_NN/cavity_opening_mask.png`.")
        lines.append("")
        lines.append("Auxiliary mask is **`cavity_depth_mask.png`** "
                     "(visible deeper pixels inside each opening — used to "
                     "estimate cavity depth, NOT used as the primary "
                     "footprint).")
    elif cdm == "depth_band":
        lines.append("")
        lines.append("⚠️ Primary mask is the **depth-band mask** "
                     "(legacy mode). The footprint may capture only side "
                     "walls / deeper visible pixels rather than the full top "
                     "aperture. Consider switching `CAVITY_DETECTION_MODE` "
                     "back to `opening_from_board_region` in "
                     "`scripts/capture_cavity_detection.py`.")
    else:
        lines.append("")
        lines.append("⚠️ Could not determine primary cavity mask "
                     "from the summary metadata.")
    lines.append("")

    # 3. Per-cavity classification
    lines.append("## 3. Per-cavity primary footprint classification")
    lines.append("")
    lines.append("| Cavity | classification | usability for Baseline 1 | "
                 "opening_xy (mm) | opening_area px | depth_area px | "
                 "z_depth median (mm) |")
    lines.append("|---|---|---|---|---|---|---|")

    classifications = []
    for cav_dir in cavities:
        meta_path = cav_dir / "cavity_metadata.json"
        if not meta_path.exists():
            lines.append(f"| `{cav_dir.name}` | metadata missing | "
                         f"UNKNOWN | — | — | — | — |")
            classifications.append((cav_dir.name, "unclear", "no metadata"))
            continue
        with open(meta_path) as f:
            meta = json.load(f)
        m   = _interpret_cavity_meta(meta)
        cls = _classify_primary_footprint(m)
        verdict, _reason = _usability_for_baseline1(m, cls)

        op_xy = m.get("opening_xy_span_m") or {}
        op_x  = (op_xy.get("x") or 0) * 1000
        op_y  = (op_xy.get("y") or 0) * 1000
        opa   = m.get("opening_area_px") or 0
        dpa   = m.get("depth_area_px")   or 0
        zmed  = (m.get("z_depth_median_m") or 0) * 1000

        lines.append(f"| `{cav_dir.name}` | {cls} | {verdict} | "
                     f"{op_x:.1f} × {op_y:.1f} | {opa} | {dpa} | "
                     f"{zmed:.1f} |")
        classifications.append((cav_dir.name, cls, verdict))
    lines.append("")

    # 4. Baseline 1 readiness
    lines.append("## 4. Baseline 1 readiness")
    lines.append("")
    if all(v == "USABLE" for _, _, v in classifications) and classifications:
        lines.append("✅ All cavities present a usable opening representation. "
                     "`cavity_pointcloud.npy` (the alias) feeds Baseline 1 "
                     "directly without changes.")
    else:
        lines.append("⚠️ Not all cavities pass the heuristic for usability. "
                     "Inspect the PNG report and verify the opening mask of "
                     "the flagged ones.")
    lines.append("")

    # 5. Inspect first
    lines.append("## 5. What to inspect first")
    lines.append("")
    lines.append("1. Open `cavity_diagnostic_report.png` and read the global "
                 "row first.")
    lines.append("2. Confirm `cavity_opening_mask.png` shows **filled, "
                 "shape-correct apertures** (full circle, full triangle, "
                 "full square, full rectangle). If any look like crescents "
                 "or partial outlines, the opening derivation is broken "
                 "upstream (`board_surface_mask` or `board_region_mask`).")
    lines.append("3. Compare `cavity_opening_mask.png` vs "
                 "`cavity_depth_mask.png` — the opening should be "
                 "substantially larger than the depth (the depth captures "
                 "only what the camera can see down the hole).")
    lines.append("4. For each `cavity_NN/`, check that "
                 "`cavity_opening_footprint.png` looks like the expected "
                 "shape silhouette and that `cavity_metadata.json` has "
                 "`primary_matching_representation = "
                 "\"cavity_opening_pointcloud\"`.")
    lines.append("")

    out_path.write_text("\n".join(lines))


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> int:
    print("=" * 60)
    print("make_cavity_diagnostic_report.py")
    print("=" * 60)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATA_DIR     : {DATA_DIR}")

    if not DATA_DIR.exists():
        print(f"[ERROR] DATA_DIR does not exist: {DATA_DIR}")
        return 1

    cavities = _list_cavity_dirs(DATA_DIR)
    print(f"Cavity folders: {[c.name for c in cavities]}")

    # Load summary metadata for the markdown report
    summary_meta = {}
    summary_path = DATA_DIR / "cavities_summary.json"
    if summary_path.exists():
        try:
            with open(summary_path) as f:
                summary_meta = json.load(f)
        except Exception as exc:
            print(f"[warn] could not parse cavities_summary.json: {exc}")

    png_path = DATA_DIR / "cavity_diagnostic_report.png"
    md_path  = DATA_DIR / "cavity_diagnostic_report.md"

    print("\n[png] building grid ...")
    backend = build_png_report(DATA_DIR, cavities, png_path)
    print(f"[png] backend={backend}  written: {png_path}")

    print("\n[md] building markdown ...")
    build_markdown_report(DATA_DIR, cavities, md_path, png_path, summary_meta)
    print(f"[md] written: {md_path}")

    print("\nDone.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
