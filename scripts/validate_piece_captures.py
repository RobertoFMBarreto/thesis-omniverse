"""
validate_piece_captures.py

Offline validation of per-piece capture artefacts produced by
capture_piece_detection.py.  Runs OUTSIDE Isaac Sim — plain Python
(NumPy + OpenCV, matplotlib optional).

For each expected piece subfolder the script checks:
  1. piece_metadata.json exists.
  2. piece_pointcloud.npy exists.
  3. piece_footprint.png exists.
  4. piece_debug.png exists.
  5. n_valid_components in metadata equals 1 (if present).
  6. multiple_valid_components in metadata is false (if present).
  7. Point cloud shape: ndim==2, shape[1]==3, at least 100 points.
  8. Point cloud bounds: X/Y span > 0, Z span >= 0, no NaN, no Inf.
  9. Footprint image is not empty (readable and has non-zero pixels).

Outputs written to DATA_DIR:
  - validation_summary.json
  - validation_summary.csv
  - footprints_grid.png  (2x2 grid of piece_footprint.png images)

Usage:
  python3 scripts/validate_piece_captures.py
  SHAPE_INSERTION_PROJECT_ROOT=/some/path python3 scripts/validate_piece_captures.py
"""

import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import cv2

# ── CONFIG ────────────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        str(Path(__file__).resolve().parent.parent),   # <repo>/scripts/ → <repo>/
    )
)
DATA_DIR = PROJECT_ROOT / "data" / "pieces_detected"
EXPECTED = ["rectangle", "square", "circle", "star"]

PC_MIN_POINTS = 100       # minimum acceptable point count
FOOTPRINT_GRID_CELL = 256 # pixels per cell in the 2x2 grid

# ── CHECK HELPERS ─────────────────────────────────────────────────────────────

def check_metadata(piece_dir: Path) -> dict:
    """
    Check 1: metadata JSON exists and is valid.
    Check 5/6: n_valid_components == 1, multiple_valid_components is false.
    Returns a dict with keys: ok, path, content, n_components_ok, reason.
    """
    result = {
        "exists": False,
        "n_components_ok": False,
        "reason_exists": "",
        "reason_n_components": "",
        "content": None,
    }
    meta_path = piece_dir / "piece_metadata.json"
    if not meta_path.exists():
        result["reason_exists"] = "piece_metadata.json not found"
        return result

    result["exists"] = True
    try:
        with open(meta_path, "r") as f:
            meta = json.load(f)
        result["content"] = meta
    except Exception as exc:
        result["reason_exists"] = f"JSON parse error: {exc}"
        result["exists"] = False
        return result

    # Checks 5 + 6
    n_comp = meta.get("n_valid_components", None)
    multi  = meta.get("multiple_valid_components", None)

    if n_comp is None and multi is None:
        result["n_components_ok"] = True
        result["reason_n_components"] = "fields absent (skipped)"
    else:
        n_ok    = (n_comp == 1) if n_comp is not None else True
        multi_ok = (multi is False) if multi is not None else True
        if n_ok and multi_ok:
            result["n_components_ok"] = True
        else:
            parts = []
            if not n_ok:
                parts.append(f"n_valid_components={n_comp} (want 1)")
            if not multi_ok:
                parts.append(f"multiple_valid_components={multi} (want false)")
            result["reason_n_components"] = "; ".join(parts)

    return result


def check_pointcloud(piece_dir: Path) -> dict:
    """
    Checks 2, 7, 8: .npy exists, shape correct, bounds sane.
    Returns a dict describing each sub-check.
    """
    result = {
        "exists": False,
        "shape_ok": False,
        "no_nan": False,
        "no_inf": False,
        "n_points": 0,
        "x_span_m": 0.0,
        "y_span_m": 0.0,
        "z_span_m": 0.0,
        "x_min": 0.0, "x_max": 0.0,
        "y_min": 0.0, "y_max": 0.0,
        "z_min": 0.0, "z_max": 0.0,
        "reason_exists": "",
        "reason_shape": "",
        "reason_bounds": "",
    }
    pc_path = piece_dir / "piece_pointcloud.npy"
    if not pc_path.exists():
        result["reason_exists"] = "piece_pointcloud.npy not found"
        return result

    result["exists"] = True
    try:
        pc = np.load(str(pc_path))
    except Exception as exc:
        result["reason_exists"] = f"load error: {exc}"
        result["exists"] = False
        return result

    # Shape check
    if pc.ndim == 2 and pc.shape[1] == 3 and pc.shape[0] >= PC_MIN_POINTS:
        result["shape_ok"] = True
        result["n_points"] = int(pc.shape[0])
    else:
        issues = []
        if pc.ndim != 2:
            issues.append(f"ndim={pc.ndim} (want 2)")
        elif pc.shape[1] != 3:
            issues.append(f"shape[1]={pc.shape[1]} (want 3)")
        if pc.ndim == 2 and pc.shape[0] < PC_MIN_POINTS:
            issues.append(f"n_points={pc.shape[0]} < {PC_MIN_POINTS}")
        result["reason_shape"] = "; ".join(issues)
        result["n_points"] = int(pc.shape[0]) if pc.ndim >= 1 else 0
        return result

    # NaN / Inf checks
    has_nan = bool(np.any(np.isnan(pc)))
    has_inf = bool(np.any(np.isinf(pc)))
    result["no_nan"] = not has_nan
    result["no_inf"] = not has_inf

    # Span checks (only meaningful when no NaN/Inf)
    if not has_nan and not has_inf:
        x_span = float(pc[:, 0].max() - pc[:, 0].min())
        y_span = float(pc[:, 1].max() - pc[:, 1].min())
        z_span = float(pc[:, 2].max() - pc[:, 2].min())
        result["x_span_m"] = x_span
        result["y_span_m"] = y_span
        result["z_span_m"] = z_span
        result["x_min"]    = float(pc[:, 0].min())
        result["x_max"]    = float(pc[:, 0].max())
        result["y_min"]    = float(pc[:, 1].min())
        result["y_max"]    = float(pc[:, 1].max())
        result["z_min"]    = float(pc[:, 2].min())
        result["z_max"]    = float(pc[:, 2].max())

        issues = []
        if x_span <= 0.0:
            issues.append("X span == 0")
        if y_span <= 0.0:
            issues.append("Y span == 0")
        if z_span < 0.0:
            issues.append("Z span < 0")
        if has_nan:
            issues.append("NaN values present")
        if has_inf:
            issues.append("Inf values present")
        if issues:
            result["reason_bounds"] = "; ".join(issues)
    else:
        result["reason_bounds"] = ("NaN present" if has_nan else "") + \
                                  (" Inf present" if has_inf else "")

    return result


def _bounds_ok(pc_result: dict) -> bool:
    """Return True if all point cloud bounds checks pass."""
    return (
        pc_result["shape_ok"]
        and pc_result["no_nan"]
        and pc_result["no_inf"]
        and pc_result["x_span_m"] > 0.0
        and pc_result["y_span_m"] > 0.0
        and pc_result["z_span_m"] >= 0.0
    )


def check_image_exists(piece_dir: Path, filename: str) -> tuple:
    """Return (ok: bool, reason: str)."""
    p = piece_dir / filename
    if not p.exists():
        return False, f"{filename} not found"
    return True, ""


def check_footprint_nonempty(piece_dir: Path) -> tuple:
    """
    Check 9: piece_footprint.png is readable and has at least some non-zero
    pixels.  Returns (ok: bool, reason: str).
    """
    p = piece_dir / "piece_footprint.png"
    if not p.exists():
        return False, "piece_footprint.png not found"
    img = cv2.imread(str(p))
    if img is None:
        return False, "cv2.imread returned None"
    if np.count_nonzero(img) == 0:
        return False, "image is all-zero (empty)"
    return True, ""


# ── VALIDATE ONE PIECE ────────────────────────────────────────────────────────

def validate_piece(piece_name: str) -> dict:
    """
    Run all checks for one piece folder.
    Returns a flat result dict suitable for JSON / CSV export.
    """
    piece_dir = DATA_DIR / piece_name
    folder_exists = piece_dir.is_dir()

    result = {
        "piece": piece_name,
        "folder_exists": folder_exists,
        "metadata_exists": False,
        "metadata_n_components_ok": False,
        "pc_exists": False,
        "pc_shape_ok": False,
        "pc_no_nan": False,
        "pc_no_inf": False,
        "pc_bounds_ok": False,
        "pc_n_points": 0,
        "pc_x_span_m": 0.0,
        "pc_y_span_m": 0.0,
        "pc_z_span_m": 0.0,
        "pc_x_min": 0.0, "pc_x_max": 0.0,
        "pc_y_min": 0.0, "pc_y_max": 0.0,
        "pc_z_min": 0.0, "pc_z_max": 0.0,
        "footprint_exists": False,
        "footprint_nonempty": False,
        "debug_exists": False,
        "reasons": [],
        "overall": False,
    }

    if not folder_exists:
        result["reasons"].append(f"folder {piece_dir} does not exist")
        return result

    # Check 1: metadata
    meta_check = check_metadata(piece_dir)
    result["metadata_exists"]        = meta_check["exists"]
    result["metadata_n_components_ok"] = meta_check["n_components_ok"]
    if not meta_check["exists"]:
        result["reasons"].append(meta_check["reason_exists"])
    if not meta_check["n_components_ok"]:
        result["reasons"].append(meta_check["reason_n_components"])

    # Checks 2, 7, 8: point cloud
    pc_check = check_pointcloud(piece_dir)
    result["pc_exists"]    = pc_check["exists"]
    result["pc_shape_ok"]  = pc_check["shape_ok"]
    result["pc_no_nan"]    = pc_check["no_nan"]
    result["pc_no_inf"]    = pc_check["no_inf"]
    result["pc_n_points"]  = pc_check["n_points"]
    result["pc_x_span_m"]  = pc_check["x_span_m"]
    result["pc_y_span_m"]  = pc_check["y_span_m"]
    result["pc_z_span_m"]  = pc_check["z_span_m"]
    result["pc_x_min"]     = pc_check["x_min"]
    result["pc_x_max"]     = pc_check["x_max"]
    result["pc_y_min"]     = pc_check["y_min"]
    result["pc_y_max"]     = pc_check["y_max"]
    result["pc_z_min"]     = pc_check["z_min"]
    result["pc_z_max"]     = pc_check["z_max"]
    result["pc_bounds_ok"] = _bounds_ok(pc_check)

    if not pc_check["exists"]:
        result["reasons"].append(pc_check["reason_exists"])
    if pc_check["exists"] and not pc_check["shape_ok"]:
        result["reasons"].append("pc shape: " + pc_check["reason_shape"])
    if pc_check["exists"] and pc_check["shape_ok"] and not result["pc_bounds_ok"]:
        result["reasons"].append("pc bounds: " + pc_check["reason_bounds"])

    # Check 3: footprint exists
    fp_exists, fp_reason = check_image_exists(piece_dir, "piece_footprint.png")
    result["footprint_exists"] = fp_exists
    if not fp_exists:
        result["reasons"].append(fp_reason)

    # Check 9: footprint not empty
    fp_nonempty, fp_ne_reason = check_footprint_nonempty(piece_dir)
    result["footprint_nonempty"] = fp_nonempty
    if not fp_nonempty and fp_exists:
        result["reasons"].append("footprint: " + fp_ne_reason)

    # Check 4: debug image exists
    dbg_exists, dbg_reason = check_image_exists(piece_dir, "piece_debug.png")
    result["debug_exists"] = dbg_exists
    if not dbg_exists:
        result["reasons"].append(dbg_reason)

    # Overall pass: all mandatory checks must pass
    result["overall"] = (
        result["folder_exists"]
        and result["metadata_exists"]
        and result["metadata_n_components_ok"]
        and result["pc_exists"]
        and result["pc_shape_ok"]
        and result["pc_bounds_ok"]
        and result["footprint_exists"]
        and result["footprint_nonempty"]
        and result["debug_exists"]
    )

    return result


# ── CONSOLE TABLE ─────────────────────────────────────────────────────────────

def _tick(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def print_summary_table(results: list) -> None:
    """Print a plain-ASCII aligned summary table to stdout."""
    header = (
        f"{'piece':<12} "
        f"{'exists':<8} "
        f"{'n_comp':<8} "
        f"{'pc_shape':<10} "
        f"{'pc_bounds':<10} "
        f"{'footprint':<10} "
        f"{'OVERALL':<8}"
    )
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)
    for r in results:
        print(
            f"{r['piece']:<12} "
            f"{_tick(r['folder_exists']):<8} "
            f"{_tick(r['metadata_n_components_ok']):<8} "
            f"{_tick(r['pc_shape_ok']):<10} "
            f"{_tick(r['pc_bounds_ok']):<10} "
            f"{_tick(r['footprint_nonempty']):<10} "
            f"{_tick(r['overall']):<8}"
        )
    print(sep)

    print("\nPer-piece point cloud spans (metres) and point counts:")
    span_header = f"  {'piece':<12} {'X span':>10} {'Y span':>10} {'Z span':>10} {'n_points':>10}"
    print(span_header)
    print("  " + "-" * (len(span_header) - 2))
    for r in results:
        print(
            f"  {r['piece']:<12} "
            f"{r['pc_x_span_m']:>10.5f} "
            f"{r['pc_y_span_m']:>10.5f} "
            f"{r['pc_z_span_m']:>10.5f} "
            f"{r['pc_n_points']:>10}"
        )

    print("\nFailure reasons:")
    any_fail = False
    for r in results:
        if r["reasons"]:
            any_fail = True
            print(f"  {r['piece']}: {'; '.join(r['reasons'])}")
    if not any_fail:
        print("  (none — all checks passed)")


# ── SAVE JSON ─────────────────────────────────────────────────────────────────

def save_json(results: list, out_path: Path) -> None:
    payload = []
    for r in results:
        payload.append({
            "piece":                     r["piece"],
            "folder_exists":             r["folder_exists"],
            "metadata_exists":           r["metadata_exists"],
            "metadata_n_components_ok":  r["metadata_n_components_ok"],
            "pc_exists":                 r["pc_exists"],
            "pc_shape_ok":               r["pc_shape_ok"],
            "pc_no_nan":                 r["pc_no_nan"],
            "pc_no_inf":                 r["pc_no_inf"],
            "pc_bounds_ok":              r["pc_bounds_ok"],
            "pc_n_points":               r["pc_n_points"],
            "pc_x_span_m":               r["pc_x_span_m"],
            "pc_y_span_m":               r["pc_y_span_m"],
            "pc_z_span_m":               r["pc_z_span_m"],
            "pc_bounds": {
                "x_min": r["pc_x_min"], "x_max": r["pc_x_max"],
                "y_min": r["pc_y_min"], "y_max": r["pc_y_max"],
                "z_min": r["pc_z_min"], "z_max": r["pc_z_max"],
            },
            "footprint_exists":          r["footprint_exists"],
            "footprint_nonempty":        r["footprint_nonempty"],
            "debug_exists":              r["debug_exists"],
            "failure_reasons":           r["reasons"],
            "overall_pass":              r["overall"],
        })
    with open(str(out_path), "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[output] validation_summary.json -> {out_path}")


# ── SAVE CSV ──────────────────────────────────────────────────────────────────

def save_csv(results: list, out_path: Path) -> None:
    fieldnames = [
        "piece", "exists", "files_ok", "metadata_n_components_ok",
        "pc_shape_ok", "pc_no_nan", "pc_no_inf",
        "pc_n_points", "pc_x_span_m", "pc_y_span_m", "pc_z_span_m",
        "footprint_ok", "overall",
    ]
    with open(str(out_path), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            files_ok = (
                r["metadata_exists"]
                and r["pc_exists"]
                and r["footprint_exists"]
                and r["debug_exists"]
            )
            writer.writerow({
                "piece":                    r["piece"],
                "exists":                   r["folder_exists"],
                "files_ok":                 files_ok,
                "metadata_n_components_ok": r["metadata_n_components_ok"],
                "pc_shape_ok":              r["pc_shape_ok"],
                "pc_no_nan":                r["pc_no_nan"],
                "pc_no_inf":                r["pc_no_inf"],
                "pc_n_points":              r["pc_n_points"],
                "pc_x_span_m":              round(r["pc_x_span_m"], 6),
                "pc_y_span_m":              round(r["pc_y_span_m"], 6),
                "pc_z_span_m":              round(r["pc_z_span_m"], 6),
                "footprint_ok":             r["footprint_nonempty"],
                "overall":                  r["overall"],
            })
    print(f"[output] validation_summary.csv  -> {out_path}")


# ── FOOTPRINTS GRID ───────────────────────────────────────────────────────────

def _make_tile_opencv(piece_name: str, img_path: Path, cell_px: int) -> np.ndarray:
    """
    Load a footprint image, resize to cell_px x cell_px, and add a label via
    cv2.putText.  If missing or unreadable, return a black tile with a MISSING
    label.
    """
    cell = np.zeros((cell_px, cell_px, 3), dtype=np.uint8)

    if img_path.exists():
        img = cv2.imread(str(img_path))
        if img is not None:
            cell = cv2.resize(img, (cell_px, cell_px))
        else:
            cv2.putText(cell, f"READ ERROR", (10, cell_px // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
            cv2.putText(cell, piece_name, (10, cell_px // 2 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)
    else:
        cv2.putText(cell, f"MISSING", (10, cell_px // 2 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 200), 1)
        cv2.putText(cell, piece_name, (10, cell_px // 2 + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 200), 1)

    # Label bar at top
    label_h = 24
    label_bar = np.zeros((label_h, cell_px, 3), dtype=np.uint8)
    cv2.putText(label_bar, piece_name, (6, label_h - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return np.vstack([label_bar, cell])


def save_footprints_grid_opencv(results: list, out_path: Path, cell_px: int) -> None:
    """Build a 2x2 grid using OpenCV only."""
    tiles = []
    for r in results:
        img_path = DATA_DIR / r["piece"] / "piece_footprint.png"
        tile = _make_tile_opencv(r["piece"], img_path, cell_px)
        tiles.append(tile)

    # Ensure we have exactly 4 tiles
    while len(tiles) < 4:
        blank = np.zeros((cell_px + 24, cell_px, 3), dtype=np.uint8)
        tiles.append(blank)

    row0 = np.hstack([tiles[0], tiles[1]])
    row1 = np.hstack([tiles[2], tiles[3]])
    grid = np.vstack([row0, row1])
    cv2.imwrite(str(out_path), grid)
    print(f"[output] footprints_grid.png     -> {out_path}  (OpenCV)")


def save_footprints_grid_matplotlib(results: list, out_path: Path, cell_px: int) -> None:
    """Build a 2x2 grid using matplotlib (preferred when available)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(2, 2, figsize=(8, 8))
    axes = axes.flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        img_path = DATA_DIR / r["piece"] / "piece_footprint.png"
        ax.set_title(r["piece"], fontsize=12)
        ax.axis("off")

        if img_path.exists():
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is not None:
                img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                ax.imshow(img_rgb)
            else:
                ax.text(0.5, 0.5, f"READ ERROR\n{r['piece']}",
                        ha="center", va="center", transform=ax.transAxes,
                        color="red", fontsize=10)
        else:
            ax.text(0.5, 0.5, f"MISSING\n{r['piece']}",
                    ha="center", va="center", transform=ax.transAxes,
                    color="red", fontsize=10)

    # Fill any unused axes (in case fewer than 4 results)
    for idx in range(len(results), 4):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(str(out_path), dpi=100)
    plt.close(fig)
    print(f"[output] footprints_grid.png     -> {out_path}  (matplotlib)")


def save_footprints_grid(results: list, out_path: Path, cell_px: int) -> None:
    """Try matplotlib first, fall back to OpenCV."""
    try:
        import matplotlib  # noqa: F401
        save_footprints_grid_matplotlib(results, out_path, cell_px)
    except ImportError:
        print("[output] matplotlib not available — using OpenCV for grid")
        save_footprints_grid_opencv(results, out_path, cell_px)


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 60)
    print("validate_piece_captures.py")
    print("=" * 60)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"DATA_DIR     : {DATA_DIR}")
    print(f"Expected     : {EXPECTED}")

    if not DATA_DIR.exists():
        print(f"\n[ERROR] DATA_DIR does not exist: {DATA_DIR}")
        print("The capture data has not been pulled to this machine yet.")
        sys.exit(1)

    # Run checks
    results = []
    for piece_name in EXPECTED:
        print(f"\n[validate] checking '{piece_name}' ...")
        r = validate_piece(piece_name)
        results.append(r)
        status = "PASS" if r["overall"] else "FAIL"
        print(f"  -> {status}  reasons: {r['reasons'] if r['reasons'] else '(none)'}")

    # Console table
    print_summary_table(results)

    # Save outputs
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    save_json(results,  DATA_DIR / "validation_summary.json")
    save_csv(results,   DATA_DIR / "validation_summary.csv")
    save_footprints_grid(results, DATA_DIR / "footprints_grid.png", FOOTPRINT_GRID_CELL)

    # Final verdict
    n_pass = sum(1 for r in results if r["overall"])
    n_fail = len(results) - n_pass
    print(f"\n{'='*60}")
    print(f"Overall: {n_pass}/{len(results)} pieces passed all checks.")
    if n_fail > 0:
        print(f"WARNING: {n_fail} piece(s) failed — inspect failure reasons above.")
    else:
        print("All captures are structurally valid.")
    print(f"{'='*60}\n")

    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
