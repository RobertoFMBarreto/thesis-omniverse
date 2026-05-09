"""
generate_phaseE_sdf_embedding_dataset.py

Phase E.1 — generate a 2D-footprint SDF dataset for learning geometric
embeddings of insertion-affordance compatibility.

Reuses Phase D.7 procedural shape generation and label rule (partial
insertion through opening). Perception remains deterministic and frozen.

Outputs (under data/phaseE_learned_embeddings/):
  phaseE_pairs_metadata.csv
  phaseE_dataset_summary.json
  phaseE_dataset_report.md
  phaseE_sdf_pairs.npz       (compressed; piece_sdf, cavity_sdf, piece_mask,
                              cavity_mask, labels, sample_ids — int8/uint8)
  debug/contact_sheet_positive_examples.png
  debug/contact_sheet_negative_examples.png
  debug/contact_sheet_family_examples.png

NOT robot control. NOT 3D reconstruction. NOT learned perception.
Dataset only.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        "/Users/robertofmbarreto/Documents/Mestrado/tese/code/thesis-omniverse",
    )
)

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from generate_phaseD_3d_affordance_dataset import (   # noqa: E402
    FAMILY_GENERATORS,
    PROCEDURAL_FAMILIES,
    INSTANCES_PER_FAMILY,
    MVP_SHAPES,
    PIECE_HEIGHT_RANGE_MM,
    CAVITY_DEPTH_RANGE_MM,
    ROTATIONS_DEG,
    LATERAL_OUTSIDE_MAX,
    LATERAL_INSIDE_MIN,
    DEPTH_TOLERANCE_MM,
    MIN_REQUIRED_INSERTION_MM,
    INSERTION_FRACTION,
    MIN_INSERTION_GUIDANCE_MM,
    RNG_SEED,
    CAVITIES_PER_PIECE,
    SHAPE_BBOX_LIMIT_MM,
    CLEARANCE_PER_SIDE_MM,
    rotate_xy,
    shoelace_area_mm2,
    make_matching_cavity,
    make_distorted_cavity,
    compute_pair_metrics,
    make_piece,
    make_cavity_pool,
)

OUT_DIR = PROJECT_ROOT / "data" / "phaseE_learned_embeddings"
DEBUG_DIR = OUT_DIR / "debug"

CSV_PATH       = OUT_DIR / "phaseE_pairs_metadata.csv"
SUMMARY_PATH   = OUT_DIR / "phaseE_dataset_summary.json"
REPORT_PATH    = OUT_DIR / "phaseE_dataset_report.md"
NPZ_PATH       = OUT_DIR / "phaseE_sdf_pairs.npz"

# ── Phase E representation config ─────────────────────────────────────────────

EMBED_CANVAS_PX        = 128
EMBED_CANVAS_WORLD_MM  = 80.0    # match Phase D rasteriser world span
EMBED_RES_MM_PER_PX    = EMBED_CANVAS_WORLD_MM / EMBED_CANVAS_PX   # 0.625 mm/px
EMBED_HALF_CANVAS_MM   = EMBED_CANVAS_WORLD_MM / 2.0

# SDF normalisation: clip to [-SDF_CLIP_MM, +SDF_CLIP_MM] then scale to [-1, 1].
SDF_CLIP_MM = 20.0      # signed-distance values beyond ±20 mm are saturated

# Quantisation:
#   piece_sdf, cavity_sdf as int8 in [-127, 127] = SDF * 127 / SDF_CLIP_MM
#   piece_mask, cavity_mask as uint8 {0, 255}
SDF_INT8_SCALE = 127.0 / SDF_CLIP_MM   # ~6.35 per mm

# Train/val/test split (deterministic; same fractions Phase D used)
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)


# ── Rasterisation helpers (Phase E 128×128) ───────────────────────────────────

def rasterise_polygon_filled_phaseE(xy_mm: np.ndarray) -> np.ndarray:
    """Filled polygon raster onto the Phase E 128×128 canvas (centroid-centred input)."""
    u = (xy_mm[:, 0] + EMBED_HALF_CANVAS_MM) / EMBED_RES_MM_PER_PX
    v = (EMBED_HALF_CANVAS_MM - xy_mm[:, 1]) / EMBED_RES_MM_PER_PX
    pts = np.stack([u, v], axis=1).astype(np.int32)
    mask = np.zeros((EMBED_CANVAS_PX, EMBED_CANVAS_PX), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def signed_distance_field(mask: np.ndarray) -> np.ndarray:
    """
    Compute signed distance field in mm.
    Convention: positive INSIDE the shape; negative OUTSIDE; ~0 on boundary.
    """
    inside  = (mask > 0).astype(np.uint8)
    outside = 1 - inside
    # cv2.distanceTransform returns positive distance to nearest zero pixel.
    # Distance inside the shape = distance from interior pixels to the boundary
    # (where boundary = nearest "outside" pixel).
    dist_in  = cv2.distanceTransform(inside,  cv2.DIST_L2, 5)
    dist_out = cv2.distanceTransform(outside, cv2.DIST_L2, 5)
    sdf_px = dist_in - dist_out
    return sdf_px * EMBED_RES_MM_PER_PX   # convert pixels to millimetres


def quantise_sdf_to_int8(sdf_mm: np.ndarray) -> np.ndarray:
    """Clip SDF to ±SDF_CLIP_MM and scale to int8 range [-127, 127]."""
    clipped = np.clip(sdf_mm, -SDF_CLIP_MM, SDF_CLIP_MM)
    return np.round(clipped * SDF_INT8_SCALE).astype(np.int8)


def dequantise_sdf(sdf_int8: np.ndarray) -> np.ndarray:
    """Inverse of quantise_sdf_to_int8 — return mm."""
    return sdf_int8.astype(np.float32) / SDF_INT8_SCALE


# ── Dataset generation ────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("generate_phaseE_sdf_embedding_dataset.py")
    print("=" * 70)
    print(f"PROJECT_ROOT       : {PROJECT_ROOT}")
    print(f"OUT_DIR            : {OUT_DIR}")
    print(f"canvas             : {EMBED_CANVAS_PX}x{EMBED_CANVAS_PX} px @ "
          f"{EMBED_RES_MM_PER_PX:.3f} mm/px ({EMBED_CANVAS_WORLD_MM:.1f} mm world)")
    print(f"SDF clip           : ±{SDF_CLIP_MM:.1f} mm  -> int8 in [-127, 127]")
    print(f"families           : {PROCEDURAL_FAMILIES}")
    print(f"instances/family   : {INSTANCES_PER_FAMILY}")
    print(f"cavities/piece     : {CAVITIES_PER_PIECE}")
    print(f"rotations          : {len(ROTATIONS_DEG)} (every 10 degrees)")
    print(f"RNG_SEED           : {RNG_SEED}")
    print()

    rng = np.random.default_rng(RNG_SEED)

    # 1. Build piece list (same procedure as Phase D so labels match exactly)
    pieces = []
    next_uid = 0
    for family in PROCEDURAL_FAMILIES:
        for inst in range(INSTANCES_PER_FAMILY):
            piece = make_piece(family, inst, rng)
            piece["piece_id"] = f"{family}_{inst:03d}"
            piece["uid"] = next_uid
            piece["is_mvp"] = False
            pieces.append(piece)
            next_uid += 1

    for mvp_name, (family, params, height_mm, _) in MVP_SHAPES.items():
        gen = FAMILY_GENERATORS[family]
        xy_mm, params_out = gen(rng, params=params)
        pieces.append({
            "family":    family,
            "instance":  -1,
            "xy_mm":     xy_mm,
            "height_mm": height_mm,
            "params":    params_out,
            "piece_id":  mvp_name,
            "uid":       next_uid,
            "is_mvp":    True,
        })
        next_uid += 1

    print(f"[gen] pieces total = {len(pieces)} "
          f"({len(PROCEDURAL_FAMILIES) * INSTANCES_PER_FAMILY} procedural + "
          f"{len(MVP_SHAPES)} MVP)")

    # 2. For each piece + cavity_pool + rotation, build masks/SDFs and labels
    #    Pre-allocate flat arrays for the npz
    n_total = len(pieces) * CAVITIES_PER_PIECE * len(ROTATIONS_DEG)
    print(f"[gen] expected total configurations: {n_total}")

    piece_sdf_arr   = np.empty((n_total, EMBED_CANVAS_PX, EMBED_CANVAS_PX),
                                dtype=np.int8)
    cavity_sdf_arr  = np.empty((n_total, EMBED_CANVAS_PX, EMBED_CANVAS_PX),
                                dtype=np.int8)
    piece_mask_arr  = np.empty((n_total, EMBED_CANVAS_PX, EMBED_CANVAS_PX),
                                dtype=np.uint8)
    cavity_mask_arr = np.empty((n_total, EMBED_CANVAS_PX, EMBED_CANVAS_PX),
                                dtype=np.uint8)
    labels_arr      = np.empty(n_total, dtype=np.uint8)
    sample_ids_arr  = np.arange(n_total, dtype=np.int32)

    csv_rows: list[dict] = []
    n_pos = 0
    per_family_totals = {}
    per_family_pos    = {}

    family_to_fold = {f: i for i, f in enumerate(PROCEDURAL_FAMILIES)}

    write_idx = 0
    for piece in pieces:
        cavity_pool = make_cavity_pool(piece, pieces, rng)

        # Centroid-centre piece XY ONCE
        piece_xy_centred = piece["xy_mm"] - piece["xy_mm"].mean(axis=0)

        per_family_totals.setdefault(piece["family"], 0)
        per_family_pos.setdefault(piece["family"], 0)

        # Pre-rasterise each cavity once (cavity is not rotated)
        cavity_cache = []
        for cav_idx, cavity in enumerate(cavity_pool):
            cav_xy_centred = cavity["xy_mm"] - cavity["xy_mm"].mean(axis=0)
            cmask = rasterise_polygon_filled_phaseE(cav_xy_centred)
            csdf  = signed_distance_field(cmask)
            cavity_cache.append({
                "cavity_id":   f"{piece['piece_id']}_cav{cav_idx:02d}",
                "source":      cavity["source"],
                "depth_mm":    float(cavity["depth_mm"]),
                "xy_centred":  cav_xy_centred,
                "mask":        cmask,
                "sdf_int8":    quantise_sdf_to_int8(csdf),
            })

        for cav_idx, cav_entry in enumerate(cavity_cache):
            for rot_deg in ROTATIONS_DEG:
                xy_rot = rotate_xy(piece_xy_centred, rot_deg)
                pmask  = rasterise_polygon_filled_phaseE(xy_rot)
                psdf   = signed_distance_field(pmask)

                # Phase D.7 labels (using Phase D 320-px rasteriser to match
                # Phase D dataset row-for-row)
                metrics = compute_pair_metrics(xy_rot, cav_entry["xy_centred"])
                lateral_ok = (metrics["outside_ratio"] <= LATERAL_OUTSIDE_MAX
                              and metrics["inside_ratio"] >= LATERAL_INSIDE_MIN)
                insertion_required = max(MIN_REQUIRED_INSERTION_MM,
                                          INSERTION_FRACTION * piece["height_mm"])
                depth_ok = (cav_entry["depth_mm"] >= insertion_required - DEPTH_TOLERANCE_MM
                            and cav_entry["depth_mm"] >= MIN_INSERTION_GUIDANCE_MM)
                label = int(lateral_ok and depth_ok)

                # Sanity guard: skip empty masks
                p_area_px = int((pmask > 0).sum())
                c_area_px = int((cav_entry["mask"] > 0).sum())
                if p_area_px == 0 or c_area_px == 0:
                    print(f"[skip] piece={piece['piece_id']} cav={cav_entry['cavity_id']} "
                          f"rot={rot_deg}: empty mask (p={p_area_px}, c={c_area_px})")
                    continue

                # Deterministic split assignment per row using a rng draw
                split_r = rng.random()
                if split_r < SPLIT_FRACTIONS[0]:
                    split = "train"
                elif split_r < SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1]:
                    split = "val"
                else:
                    split = "test"

                piece_sdf_arr[write_idx]   = quantise_sdf_to_int8(psdf)
                cavity_sdf_arr[write_idx]  = cav_entry["sdf_int8"]
                piece_mask_arr[write_idx]  = pmask
                cavity_mask_arr[write_idx] = cav_entry["mask"]
                labels_arr[write_idx]      = label

                csv_rows.append({
                    "sample_id":            write_idx,
                    "piece_id":             piece["piece_id"],
                    "cavity_id":            cav_entry["cavity_id"],
                    "shape_family":         piece["family"],
                    "is_mvp":               bool(piece.get("is_mvp", False)),
                    "rotation_deg":         int(rot_deg),
                    "label":                int(label),
                    "split":                split,
                    "heldout_family_fold":  family_to_fold.get(piece["family"], -1),
                    "cavity_source":        cav_entry["source"],
                    "piece_height_mm":      round(float(piece["height_mm"]), 3),
                    "cavity_depth_mm":      round(float(cav_entry["depth_mm"]), 3),
                    "p_mask_area_px":       p_area_px,
                    "c_mask_area_px":       c_area_px,
                    "iou":                  round(float(metrics["iou"]), 6),
                })

                per_family_totals[piece["family"]] += 1
                per_family_pos[piece["family"]]    += label
                if label == 1:
                    n_pos += 1
                write_idx += 1

    # Trim arrays to actual write count (in case of skips)
    piece_sdf_arr   = piece_sdf_arr[:write_idx]
    cavity_sdf_arr  = cavity_sdf_arr[:write_idx]
    piece_mask_arr  = piece_mask_arr[:write_idx]
    cavity_mask_arr = cavity_mask_arr[:write_idx]
    labels_arr      = labels_arr[:write_idx]
    sample_ids_arr  = sample_ids_arr[:write_idx]

    n_total_actual = write_idx
    pos_rate = n_pos / max(n_total_actual, 1)
    print()
    print(f"[summary] n_samples         = {n_total_actual}")
    print(f"[summary] n_positive        = {n_pos} ({pos_rate*100:.2f}%)")
    print(f"[summary] piece_sdf shape   = {piece_sdf_arr.shape}, dtype={piece_sdf_arr.dtype}")
    print(f"[summary] cavity_sdf shape  = {cavity_sdf_arr.shape}, dtype={cavity_sdf_arr.dtype}")

    # SDF range diagnostics
    sample_for_sdf_diag = piece_sdf_arr[: min(2000, n_total_actual)]
    sdf_mm_min = float(dequantise_sdf(sample_for_sdf_diag).min())
    sdf_mm_max = float(dequantise_sdf(sample_for_sdf_diag).max())
    print(f"[summary] SDF range (mm) on first 2000 piece SDFs: "
          f"[{sdf_mm_min:.2f}, {sdf_mm_max:.2f}]")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    DEBUG_DIR.mkdir(parents=True, exist_ok=True)

    # Write CSV
    print(f"[write] {CSV_PATH}")
    with CSV_PATH.open("w") as f:
        if csv_rows:
            keys = list(csv_rows[0].keys())
            f.write(",".join(keys) + "\n")
            for r in csv_rows:
                f.write(",".join(str(r[k]) for k in keys) + "\n")

    # Write npz (compressed)
    print(f"[write] {NPZ_PATH} (compressed) ...")
    np.savez_compressed(
        NPZ_PATH,
        piece_sdf_int8=piece_sdf_arr,
        cavity_sdf_int8=cavity_sdf_arr,
        piece_mask_uint8=piece_mask_arr,
        cavity_mask_uint8=cavity_mask_arr,
        labels=labels_arr,
        sample_ids=sample_ids_arr,
    )
    npz_size_mb = NPZ_PATH.stat().st_size / (1024 * 1024)
    print(f"[write] {NPZ_PATH}  size={npz_size_mb:.1f} MB")

    # Per-split stats
    split_stats = {"train": {"n": 0, "pos": 0},
                   "val":   {"n": 0, "pos": 0},
                   "test":  {"n": 0, "pos": 0}}
    for r in csv_rows:
        s = r["split"]
        split_stats[s]["n"]   += 1
        split_stats[s]["pos"] += r["label"]
    for s in split_stats:
        n = split_stats[s]["n"]
        split_stats[s]["positive_rate"] = round(split_stats[s]["pos"] / max(n, 1), 6)

    summary = {
        "schema_version": 1,
        "script_name":    "generate_phaseE_sdf_embedding_dataset.py",
        "phase":          "Phase E.1 — SDF embedding dataset",
        "phase_note": (
            "Footprint masks + signed distance fields at 128x128, derived from "
            "the deterministic Phase D.7 procedural shapes and partial-insertion "
            "labels. Supports learning of geometric embeddings for affordance "
            "ranking; does NOT train a robot controller, NOT learn perception, "
            "NOT reconstruct full 3D, NOT model contact dynamics."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rng_seed":      RNG_SEED,
        "canvas": {
            "px":           EMBED_CANVAS_PX,
            "world_mm":     EMBED_CANVAS_WORLD_MM,
            "mm_per_px":    EMBED_RES_MM_PER_PX,
        },
        "sdf_convention": {
            "positive_inside_shape": True,
            "clip_mm":               SDF_CLIP_MM,
            "int8_scale":            SDF_INT8_SCALE,
            "dequantise":            "sdf_mm = sdf_int8 / int8_scale",
        },
        "thresholds_inherited_from_phaseD7": {
            "lateral_outside_max":         LATERAL_OUTSIDE_MAX,
            "lateral_inside_min":          LATERAL_INSIDE_MIN,
            "depth_tolerance_mm":          DEPTH_TOLERANCE_MM,
            "min_required_insertion_mm":   MIN_REQUIRED_INSERTION_MM,
            "insertion_fraction":          INSERTION_FRACTION,
            "min_insertion_guidance_mm":   MIN_INSERTION_GUIDANCE_MM,
            "clearance_per_side_mm":       CLEARANCE_PER_SIDE_MM,
        },
        "summary": {
            "n_pieces":          len(pieces),
            "n_procedural":      len(PROCEDURAL_FAMILIES) * INSTANCES_PER_FAMILY,
            "n_mvp":             len(MVP_SHAPES),
            "cavities_per_piece": CAVITIES_PER_PIECE,
            "rotations_per_pair": len(ROTATIONS_DEG),
            "n_samples":         int(n_total_actual),
            "n_positive":        int(n_pos),
            "n_negative":        int(n_total_actual - n_pos),
            "positive_rate":     round(pos_rate, 6),
            "per_family":        {
                fam: {
                    "n_samples":     per_family_totals[fam],
                    "n_positive":    int(per_family_pos[fam]),
                    "positive_rate": round(per_family_pos[fam] / max(per_family_totals[fam], 1), 6),
                    "fold_id":       family_to_fold[fam],
                }
                for fam in PROCEDURAL_FAMILIES
            },
            "per_split":        split_stats,
            "any_family_zero_positive": any(per_family_pos[f] == 0 for f in PROCEDURAL_FAMILIES),
            "sdf_range_mm_first2000": {"min": sdf_mm_min, "max": sdf_mm_max},
            "npz_file_size_mb":        round(npz_size_mb, 2),
        },
        "leakage_diagnostic_columns": [
            "iou", "p_mask_area_px", "c_mask_area_px",
        ],
        "identifier_columns_excluded_from_features": [
            "sample_id", "piece_id", "cavity_id", "shape_family", "is_mvp",
            "cavity_source", "split", "heldout_family_fold",
        ],
    }
    print(f"[write] {SUMMARY_PATH}")
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))

    # Markdown report
    print(f"[write] {REPORT_PATH}")
    write_report(summary, REPORT_PATH)

    # Debug contact sheets
    write_contact_sheets(csv_rows, piece_sdf_arr, cavity_sdf_arr,
                          piece_mask_arr, cavity_mask_arr, labels_arr,
                          DEBUG_DIR)

    print("[done].")


# ── Report writer ─────────────────────────────────────────────────────────────

def write_report(summary: dict, path: Path) -> None:
    s = summary["summary"]
    th = summary["thresholds_inherited_from_phaseD7"]
    lines = []
    lines.append("# Phase E.1 — Footprint SDF Dataset for Embedding Learning")
    lines.append("")
    lines.append("> **This dataset supports learning geometric embeddings for "
                 "affordance ranking; it does not train a robot controller.**")
    lines.append("")
    lines.append("> Status: Phase E.1 (dataset generation only). No model has "
                 "been trained from this data.")
    lines.append("")

    lines.append("## Objective")
    lines.append("")
    lines.append("Provide a procedurally-generated dataset of (piece footprint, "
                 "cavity opening footprint, partial-insertion label) triples, "
                 "rasterised at 128×128 with mask + signed distance field "
                 "channels, suitable for training a learned geometric "
                 "embedding (e.g. shallow CNN siamese encoder, see Doc 09).")
    lines.append("")

    lines.append("## Representation choice")
    lines.append("")
    lines.append("Each pair stores BOTH the binary mask and the signed distance "
                 "field of piece and cavity at 128×128. The SDF is the primary "
                 "candidate input for the embedding model; the mask is "
                 "available for ablation and debug visualisation.")
    lines.append("")

    lines.append("## SDF convention")
    lines.append("")
    lines.append(f"- Positive **inside** the shape; negative **outside**; ~0 on the boundary.")
    lines.append(f"- Computed in millimetres via `cv2.distanceTransform` on each side, then differenced.")
    lines.append(f"- Clipped to ±{summary['sdf_convention']['clip_mm']:.1f} mm.")
    lines.append(f"- Quantised to `int8` via `sdf_int8 = round(sdf_mm × "
                 f"{summary['sdf_convention']['int8_scale']:.4f})`.")
    lines.append(f"- Dequantise: `sdf_mm = sdf_int8 / {summary['sdf_convention']['int8_scale']:.4f}`.")
    lines.append("")

    lines.append("## Dataset source")
    lines.append("")
    lines.append("Procedural shape generation reused from Phase D.7 "
                 "(`scripts/generate_phaseD_3d_affordance_dataset.py`) via direct "
                 "Python import. Same RNG seed, same families, same instances, "
                 "same matching/mismatched cavity recipes. Labels generated by "
                 "the same Phase D.7 partial-insertion rule; perception is "
                 "deterministic and frozen.")
    lines.append("")

    lines.append("## Label rule (inherited from Phase D.7)")
    lines.append("")
    lines.append("```")
    lines.append("lateral_ok = (outside_ratio_raw <= "
                 f"{th['lateral_outside_max']}) AND")
    lines.append("              (inside_ratio_raw  >= "
                 f"{th['lateral_inside_min']})")
    lines.append("")
    lines.append("insertion_required_mm = max(")
    lines.append(f"    {th['min_required_insertion_mm']},")
    lines.append(f"    {th['insertion_fraction']} * piece_height_mm")
    lines.append(")")
    lines.append("")
    lines.append("depth_ok = (cavity_depth_mm >= insertion_required_mm "
                 f"- {th['depth_tolerance_mm']}) AND")
    lines.append("            (cavity_depth_mm >= "
                 f"{th['min_insertion_guidance_mm']})")
    lines.append("")
    lines.append("label = lateral_ok AND depth_ok")
    lines.append("```")
    lines.append("")
    lines.append("Constants are **fixed operating points**, not free tuning parameters.")
    lines.append("")

    lines.append("## Dataset statistics")
    lines.append("")
    lines.append(f"- **Pieces**: {s['n_pieces']} ({s['n_procedural']} procedural + {s['n_mvp']} MVP hold-in)")
    lines.append(f"- **Cavities per piece**: {s['cavities_per_piece']}")
    lines.append(f"- **Rotations per pair**: {s['rotations_per_pair']}")
    lines.append(f"- **Total samples**: {s['n_samples']}")
    lines.append(f"- **Positive samples**: {s['n_positive']} "
                 f"({s['positive_rate']*100:.2f}% positive rate)")
    lines.append(f"- **Negative samples**: {s['n_negative']}")
    lines.append(f"- **Any family with zero positives**: {s['any_family_zero_positive']}")
    lines.append(f"- **NPZ file size**: {s['npz_file_size_mb']:.1f} MB")
    lines.append(f"- **SDF range (mm) on first 2000 piece SDFs**: "
                 f"[{s['sdf_range_mm_first2000']['min']:.2f}, "
                 f"{s['sdf_range_mm_first2000']['max']:.2f}]")
    lines.append("")

    lines.append("### Per-family")
    lines.append("")
    lines.append("| family | fold_id | n_samples | n_positive | positive_rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for fam, d in s["per_family"].items():
        lines.append(f"| `{fam}` | {d['fold_id']} | {d['n_samples']} | "
                     f"{d['n_positive']} | {d['positive_rate']:.4f} |")
    lines.append("")

    lines.append("### Per-split")
    lines.append("")
    lines.append("| split | n | n_positive | positive_rate |")
    lines.append("|---|---:|---:|---:|")
    for split_name, d in s["per_split"].items():
        lines.append(f"| {split_name} | {d['n']} | {d['pos']} | "
                     f"{d['positive_rate']:.4f} |")
    lines.append("")

    lines.append("## Leakage / identifier columns (NOT model inputs)")
    lines.append("")
    lines.append("**Diagnostics stored in CSV but should NOT be used as model inputs**:")
    for c in summary["leakage_diagnostic_columns"]:
        lines.append(f"- `{c}`")
    lines.append("")
    lines.append("**Identifiers / metadata stored in CSV (tracing only)**:")
    for c in summary["identifier_columns_excluded_from_features"]:
        lines.append(f"- `{c}`")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- **Synthetic only**: labels generated by raster geometry; not physical insertion.")
    lines.append("- **Convex prismatic only**: extrusion assumption inherited from Phase D.")
    lines.append("- **No XY offset**: rotations only.")
    lines.append("- **Perception frozen**: this dataset feeds learned embeddings, not learned perception.")
    lines.append("- **No model trained from this data yet** (Phase E.1 is dataset generation).")
    lines.append("- **MVP hold-in is a small minority** (4 pieces vs 100 procedural).")
    lines.append("")

    lines.append("## Closing note")
    lines.append("")
    lines.append("This dataset supports learning geometric embeddings for "
                 "affordance ranking. It does **not** train a robot controller, "
                 "**not** learn perception, **not** reconstruct full 3D, **not** "
                 "model contact dynamics.")
    path.write_text("\n".join(lines) + "\n")


# ── Debug contact sheets ──────────────────────────────────────────────────────

def write_contact_sheets(csv_rows: list[dict],
                          piece_sdf: np.ndarray, cavity_sdf: np.ndarray,
                          piece_mask: np.ndarray, cavity_mask: np.ndarray,
                          labels: np.ndarray, debug_dir: Path) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[debug] matplotlib not available; skipping contact sheets")
        return

    debug_dir.mkdir(parents=True, exist_ok=True)

    def _plot_grid(indices: list[int], title: str, out_path: Path) -> None:
        n = min(len(indices), 8)
        if n == 0:
            return
        fig, axs = plt.subplots(4, n, figsize=(2.0 * n, 8.0))
        if n == 1:
            axs = np.array(axs).reshape(4, 1)
        for col, idx in enumerate(indices[:n]):
            r = csv_rows[idx]
            axs[0, col].imshow(piece_mask[idx], cmap="gray")
            axs[0, col].set_title(f"piece mask\n{r['piece_id']}\nrot {r['rotation_deg']}°",
                                    fontsize=7)
            axs[1, col].imshow(piece_sdf[idx], cmap="seismic", vmin=-127, vmax=127)
            axs[1, col].set_title("piece SDF (int8)", fontsize=7)
            axs[2, col].imshow(cavity_mask[idx], cmap="gray")
            axs[2, col].set_title(f"cavity mask\n{r['cavity_id']}", fontsize=7)
            axs[3, col].imshow(cavity_sdf[idx], cmap="seismic", vmin=-127, vmax=127)
            axs[3, col].set_title(
                f"cavity SDF\nlabel={r['label']} fam={r['shape_family'][:8]}",
                fontsize=7)
            for row in range(4):
                axs[row, col].axis("off")
        fig.suptitle(title, fontsize=10)
        fig.tight_layout()
        try:
            fig.savefig(out_path, dpi=110, bbox_inches="tight")
            print(f"[debug] {out_path}")
        finally:
            plt.close(fig)

    rng = np.random.default_rng(0)
    pos_idx = [i for i, r in enumerate(csv_rows) if r["label"] == 1]
    neg_idx = [i for i, r in enumerate(csv_rows) if r["label"] == 0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    _plot_grid(pos_idx[:8], "Positive examples (label=1)",
                debug_dir / "contact_sheet_positive_examples.png")
    _plot_grid(neg_idx[:8], "Negative examples (label=0)",
                debug_dir / "contact_sheet_negative_examples.png")

    # Family-stratified contact sheet: one column per family
    by_family = {}
    for i, r in enumerate(csv_rows):
        by_family.setdefault(r["shape_family"], []).append(i)
    fam_indices = []
    for fam in PROCEDURAL_FAMILIES:
        if by_family.get(fam):
            fam_indices.append(by_family[fam][0])
    _plot_grid(fam_indices, "Per-family example (one per family)",
                debug_dir / "contact_sheet_family_examples.png")


if __name__ == "__main__":
    main()
