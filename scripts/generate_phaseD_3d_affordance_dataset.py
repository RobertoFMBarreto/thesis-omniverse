"""
generate_phaseD_3d_affordance_dataset.py

Phase D.1/D.2 — procedural 3D-extrusion geometric affordance dataset.

Pipeline:
    1. Generate procedural convex prismatic pieces from 4-5 families.
    2. For each piece, generate a matching cavity + N mismatched cavities.
    3. For each (piece, cavity, rotation) configuration, compute:
       - identity-free entity and pair descriptors,
       - deterministic affordance label (lateral feasibility AND depth feasibility),
       - diagnostics (inside_ratio_raw, outside_ratio_raw — label-only).
    4. Write CSV + JSON summary + Markdown report.

This script generates DATA only. It does not train any model. It does not
involve Isaac Sim, robot control, force feedback, or any learning.

Geometric assumption (controlled simplification):
    piece  ~ 2D footprint extruded vertically by piece_height_mm
    cavity ~ 2D opening extruded downward by cavity_depth_mm
Valid only for convex prismatic shapes and a fixed vertical insertion direction.
"""

from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

# ── Project root ──────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(
    os.environ.get(
        "SHAPE_INSERTION_PROJECT_ROOT",
        "/Users/robertofmbarreto/Documents/Mestrado/tese/code/thesis-omniverse",
    )
)

OUT_DIR     = PROJECT_ROOT / "data" / "phaseD_3d_affordance"
CSV_PATH    = OUT_DIR / "configurations_labelled.csv"
PARQUET_PATH = OUT_DIR / "configurations_labelled.parquet"
SUMMARY_PATH = OUT_DIR / "dataset_summary.json"
REPORT_PATH  = OUT_DIR / "dataset_report.md"

# ── CONFIG ────────────────────────────────────────────────────────────────────

CANVAS_PX           = 320
RES_M_PER_PX        = 0.00025          # 0.25 mm/px
WORLD_CANVAS_M      = CANVAS_PX * RES_M_PER_PX   # 80 mm
WORLD_HALF_CANVAS_M = WORLD_CANVAS_M / 2.0       # 40 mm

INSTANCES_PER_FAMILY = 20
SHAPE_BBOX_LIMIT_MM  = 60.0

# Phase D.7 — partial-insertion-through-opening regime.
# Pieces may be taller than cavities are deep; this is the normal case for
# the MVP shape-sorter. Ranges widened accordingly.
PIECE_HEIGHT_RANGE_MM = (20.0, 150.0)
CAVITY_DEPTH_RANGE_MM = (10.0, 100.0)

# Clearance: Baseline 1 spec is 0.5 mm per side / 1 mm total.
CLEARANCE_PER_SIDE_MM = 0.5
CLEARANCE_PX          = int(round((CLEARANCE_PER_SIDE_MM / 1000.0) / RES_M_PER_PX))   # = 2 px
DIL_KERNEL            = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE,
    (2 * CLEARANCE_PX + 1, 2 * CLEARANCE_PX + 1),
)

ROTATIONS_DEG = list(range(0, 360, 10))   # 36 angles

# Cavity pool composition per piece
N_MATCHING_CAVITIES_PER_PIECE   = 1
N_MISMATCHED_CAVITIES_PER_PIECE = 6
CAVITIES_PER_PIECE              = N_MATCHING_CAVITIES_PER_PIECE + N_MISMATCHED_CAVITIES_PER_PIECE

# Phase D.7 affordance label thresholds — FROZEN operating points.
# Partial insertion through opening (NOT full containment).
LATERAL_OUTSIDE_MAX        = 0.05
LATERAL_INSIDE_MIN         = 0.80
DEPTH_TOLERANCE_MM         = 0.5
MIN_REQUIRED_INSERTION_MM  = 5.0
INSERTION_FRACTION         = 0.25
MIN_INSERTION_GUIDANCE_MM  = 5.0

# Sampling / split RNG
RNG_SEED = 12345

# Shape families (for procedural generation)
PROCEDURAL_FAMILIES = [
    "rectangle",
    "ellipse",
    "regular_polygon",
    "convex_irregular_polygon",
    "rounded_rectangle",
]

# MVP shapes (real-data hold-in). Names start with "mvp_" so they are clearly
# tagged as MVP instances; the model never sees these names — only the
# family + numeric features.
MVP_SHAPES = {
    "mvp_rectangle": ("rectangle",       {"w_mm": 75.0, "h_mm": 50.0}, 105.0, 75.0),
    "mvp_square":    ("rectangle",       {"w_mm": 50.0, "h_mm": 50.0}, 105.0, 75.0),
    "mvp_circle":    ("ellipse",         {"semi_a_mm": 25.0, "semi_b_mm": 25.0}, 105.0, 75.0),
    "mvp_triangle":  ("regular_polygon", {"n_sides": 3, "circumradius_mm": 28.87}, 105.0, 75.0),
}

# Held-out family fold mapping (LOFO): each family is one fold ID
FAMILY_FOLD_ID = {fam: idx for idx, fam in enumerate(PROCEDURAL_FAMILIES)}

# ── Shape generators ──────────────────────────────────────────────────────────

def gen_rectangle(rng: np.random.Generator, params: dict | None = None):
    if params is None:
        ar = rng.uniform(1.0, 2.0)
        long_side = rng.uniform(20.0, SHAPE_BBOX_LIMIT_MM)
        short_side = long_side / ar
        w, h = (long_side, short_side) if rng.random() < 0.5 else (short_side, long_side)
    else:
        w, h = float(params["w_mm"]), float(params["h_mm"])
    hw, hh = w / 2.0, h / 2.0
    pts = np.array([[-hw, -hh], [hw, -hh], [hw, hh], [-hw, hh]], dtype=np.float64)
    return pts, {"w_mm": w, "h_mm": h}


def gen_ellipse(rng: np.random.Generator, params: dict | None = None, n_pts: int = 64):
    if params is None:
        a = rng.uniform(15.0, SHAPE_BBOX_LIMIT_MM / 2.0)
        b_ratio = rng.uniform(0.5, 1.0)
        b = a * b_ratio
    else:
        a, b = float(params["semi_a_mm"]), float(params["semi_b_mm"])
    theta = np.linspace(0.0, 2.0 * np.pi, n_pts, endpoint=False)
    pts = np.stack([a * np.cos(theta), b * np.sin(theta)], axis=1)
    return pts, {"semi_a_mm": a, "semi_b_mm": b}


def gen_regular_polygon(rng: np.random.Generator, params: dict | None = None):
    if params is None:
        n = int(rng.choice([3, 5, 6, 8]))
        r = rng.uniform(15.0, SHAPE_BBOX_LIMIT_MM / 2.0)
    else:
        n, r = int(params["n_sides"]), float(params["circumradius_mm"])
    theta = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False) + np.pi / 2.0
    pts = np.stack([r * np.cos(theta), r * np.sin(theta)], axis=1)
    return pts, {"n_sides": n, "circumradius_mm": r}


def gen_convex_irregular_polygon(rng: np.random.Generator, params: dict | None = None):
    n_seed = int(rng.integers(4, 9))
    seed_pts = rng.uniform(
        -SHAPE_BBOX_LIMIT_MM / 2.0, SHAPE_BBOX_LIMIT_MM / 2.0, size=(n_seed, 2)
    )
    hull = cv2.convexHull(seed_pts.astype(np.float32)).reshape(-1, 2)
    return hull.astype(np.float64), {"n_seed": n_seed, "n_hull": int(len(hull))}


def gen_rounded_rectangle(rng: np.random.Generator, params: dict | None = None,
                           n_corner_pts: int = 8):
    ar = rng.uniform(1.0, 2.0)
    long_side = rng.uniform(20.0, SHAPE_BBOX_LIMIT_MM)
    short_side = long_side / ar
    w, h = long_side, short_side
    r = rng.uniform(0.05, 0.20) * min(w, h)
    pts = []
    # Corner centre offsets and starting angles for each quadrant
    corner_data = [
        ( w / 2 - r,  h / 2 - r,           0.0),  # top-right
        ( w / 2 - r, -h / 2 + r,    1.5 * np.pi),  # bottom-right
        (-w / 2 + r, -h / 2 + r,           np.pi),  # bottom-left
        (-w / 2 + r,  h / 2 - r,    0.5 * np.pi),  # top-left
    ]
    for cx, cy, a0 in corner_data:
        for k in range(n_corner_pts):
            t = a0 + (k / max(n_corner_pts - 1, 1)) * (np.pi / 2.0)
            pts.append([cx + r * np.cos(t), cy + r * np.sin(t)])
    return np.array(pts, dtype=np.float64), {"w_mm": w, "h_mm": h, "corner_r_mm": r}


FAMILY_GENERATORS = {
    "rectangle":               gen_rectangle,
    "ellipse":                 gen_ellipse,
    "regular_polygon":         gen_regular_polygon,
    "convex_irregular_polygon": gen_convex_irregular_polygon,
    "rounded_rectangle":       gen_rounded_rectangle,
}


# ── Geometry helpers ──────────────────────────────────────────────────────────

def rotate_xy(xy_mm: np.ndarray, theta_deg: float) -> np.ndarray:
    th = math.radians(theta_deg)
    c, s = math.cos(th), math.sin(th)
    R = np.array([[c, s], [-s, c]], dtype=np.float64)
    return xy_mm @ R


def rasterise_polygon_filled(xy_mm: np.ndarray) -> np.ndarray:
    """Filled polygon raster on 320x320 px canvas, centroid-centred at canvas centre."""
    u = (xy_mm[:, 0] / 1000.0 + WORLD_HALF_CANVAS_M) / RES_M_PER_PX
    v = (WORLD_HALF_CANVAS_M - xy_mm[:, 1] / 1000.0) / RES_M_PER_PX
    pts = np.stack([u, v], axis=1).astype(np.int32)
    mask = np.zeros((CANVAS_PX, CANVAS_PX), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    return mask


def shoelace_area_mm2(xy_mm: np.ndarray) -> float:
    x = xy_mm[:, 0]
    y = xy_mm[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def perimeter_mm(xy_mm: np.ndarray) -> float:
    pts = np.vstack([xy_mm, xy_mm[:1]])
    diffs = np.diff(pts, axis=0)
    return float(np.sum(np.linalg.norm(diffs, axis=1)))


def bbox_aspect_ratio(xy_mm: np.ndarray) -> float:
    w = float(xy_mm[:, 0].max() - xy_mm[:, 0].min())
    h = float(xy_mm[:, 1].max() - xy_mm[:, 1].min())
    long_side  = max(w, h)
    short_side = max(min(w, h), 1e-9)
    return long_side / short_side


def compactness(area_mm2: float, perim_mm: float) -> float:
    if perim_mm <= 0.0:
        return 0.0
    return float(4.0 * math.pi * area_mm2 / (perim_mm ** 2))


def make_matching_cavity(piece_xy_mm: np.ndarray) -> np.ndarray:
    """
    Matching cavity is the piece footprint scaled outward by clearance.
    For a convex polygon, scale around the centroid by the factor that
    increases its inscribed-equivalent radius by CLEARANCE_PER_SIDE_MM.
    Approximation: scale by (1 + CLEARANCE / characteristic_radius).
    """
    centroid = piece_xy_mm.mean(axis=0)
    centred = piece_xy_mm - centroid
    # characteristic radius = mean distance from centroid to vertices
    r_char = float(np.mean(np.linalg.norm(centred, axis=1)))
    if r_char <= 1e-6:
        scale = 1.05
    else:
        scale = 1.0 + CLEARANCE_PER_SIDE_MM / r_char
    return centred * scale + centroid


def make_distorted_cavity(piece_xy_mm: np.ndarray, scale: float) -> np.ndarray:
    """Scale piece footprint by a custom factor (mismatched cavity)."""
    centroid = piece_xy_mm.mean(axis=0)
    centred = piece_xy_mm - centroid
    return centred * scale + centroid


# ── Per-pair scoring (raster-based) ───────────────────────────────────────────

def compute_pair_metrics(piece_xy_mm: np.ndarray, cavity_xy_mm: np.ndarray) -> dict:
    """
    Rasterise both, dilate cavity by clearance, compute IoU + inside/outside ratios.

    Both shapes are centroid-centred and rasterised onto the canvas.
    Returns p_area_px, c_area_px, c_dil_area_px, inter_dil_px, inter_undil_px, union_undil_px.
    """
    piece_centred  = piece_xy_mm  - piece_xy_mm.mean(axis=0)
    cavity_centred = cavity_xy_mm - cavity_xy_mm.mean(axis=0)

    mask_p = rasterise_polygon_filled(piece_centred)
    mask_c_undil = rasterise_polygon_filled(cavity_centred)
    mask_c_dil   = cv2.dilate(mask_c_undil, DIL_KERNEL)

    p_bool      = mask_p      > 0
    c_undil_bool = mask_c_undil > 0
    c_dil_bool  = mask_c_dil  > 0

    p_area      = int(p_bool.sum())
    c_area      = int(c_undil_bool.sum())
    c_dil_area  = int(c_dil_bool.sum())

    inter_dil   = int((p_bool & c_dil_bool).sum())
    inter_undil = int((p_bool & c_undil_bool).sum())
    union_undil = int((p_bool | c_undil_bool).sum())

    inside_ratio  = inter_dil / max(p_area, 1)
    outside_ratio = 1.0 - inside_ratio
    iou           = inter_undil / max(union_undil, 1)

    return {
        "p_area_px":      p_area,
        "c_area_px":      c_area,
        "c_dil_area_px":  c_dil_area,
        "inter_dil_px":   inter_dil,
        "inter_undil_px": inter_undil,
        "union_undil_px": union_undil,
        "inside_ratio":   float(inside_ratio),
        "outside_ratio":  float(outside_ratio),
        "iou":            float(iou),
    }


# ── Configuration assembly ────────────────────────────────────────────────────

def make_piece(family: str, instance_idx: int, rng: np.random.Generator,
               override: dict | None = None) -> dict:
    """Generate one piece instance: footprint XY (mm), height, family, params."""
    if override is not None:
        family_used = override.get("family", family)
        params      = override.get("params", None)
        height_mm   = float(override.get("height_mm",
                                         rng.uniform(*PIECE_HEIGHT_RANGE_MM)))
    else:
        family_used = family
        params      = None
        height_mm   = float(rng.uniform(*PIECE_HEIGHT_RANGE_MM))

    gen = FAMILY_GENERATORS[family_used]
    xy_mm, params_out = gen(rng, params=params)
    return {
        "family":     family_used,
        "instance":   instance_idx,
        "xy_mm":      xy_mm,
        "height_mm":  height_mm,
        "params":     params_out,
    }


def make_cavity_pool(piece: dict, all_pieces: list[dict], rng: np.random.Generator) -> list[dict]:
    """
    Build cavity pool for a piece: 1 matching + 6 mismatched.
    Each cavity is a dict with xy_mm, depth_mm, source_label.
    """
    cavities = []

    # 1 matching cavity (piece footprint + clearance, depth satisfies the
    # partial-insertion rule: cavity_depth >= INSERTION_FRACTION * piece_height).
    matching_xy = make_matching_cavity(piece["xy_mm"])
    insertion_required = max(MIN_REQUIRED_INSERTION_MM,
                              INSERTION_FRACTION * piece["height_mm"])
    matching_depth_lo = max(insertion_required, MIN_INSERTION_GUIDANCE_MM)
    matching_depth_hi = CAVITY_DEPTH_RANGE_MM[1]
    if matching_depth_lo >= matching_depth_hi:
        matching_depth = float(matching_depth_lo)
    else:
        matching_depth = float(rng.uniform(matching_depth_lo, matching_depth_hi))
    cavities.append({
        "xy_mm":         matching_xy,
        "depth_mm":      matching_depth,
        "source":        "matching_clearance",
    })

    # Mismatched cavities — sample from other pieces / scale variants
    other_indices = [i for i, p in enumerate(all_pieces)
                     if not (p["family"] == piece["family"] and p["instance"] == piece["instance"])]
    rng.shuffle(other_indices)

    mismatch_recipes = [
        ("other_piece_matching",   None,  None),
        ("scale_minus_20_percent", 0.80,  None),
        ("scale_plus_10_percent",  1.10,  None),
        ("zero_clearance",         "zero", None),
        ("shallow_depth",          None,  "shallow"),
        ("other_piece_no_clearance", None, "other_no_clearance"),
    ]

    other_used = 0
    for recipe_name, scale_factor, depth_mode in mismatch_recipes:
        if recipe_name in ("other_piece_matching", "other_piece_no_clearance"):
            if other_used >= len(other_indices):
                # fall back to scale variant if no other pieces
                xy = make_distorted_cavity(piece["xy_mm"], 1.20)
            else:
                other = all_pieces[other_indices[other_used]]
                other_used += 1
                if recipe_name == "other_piece_matching":
                    xy = make_matching_cavity(other["xy_mm"])
                else:
                    xy = other["xy_mm"].copy()
        elif scale_factor == "zero":
            xy = piece["xy_mm"].copy()  # exact piece, zero clearance → infeasible
        elif isinstance(scale_factor, (int, float)):
            xy = make_distorted_cavity(piece["xy_mm"], float(scale_factor))
        else:
            # No scale specified (e.g. shallow-depth recipe): use the piece
            # footprint with nominal clearance so feasibility is decided by
            # depth alone.
            xy = make_matching_cavity(piece["xy_mm"])

        if depth_mode == "shallow":
            # Genuine shallow-failure case: cavity_depth < MIN_INSERTION_GUIDANCE_MM
            depth = float(rng.uniform(1.0, MIN_INSERTION_GUIDANCE_MM - 0.5))
        else:
            depth = float(rng.uniform(*CAVITY_DEPTH_RANGE_MM))

        cavities.append({
            "xy_mm":    xy,
            "depth_mm": depth,
            "source":   recipe_name,
        })

    return cavities[:CAVITIES_PER_PIECE]


# ── Dataset generation ────────────────────────────────────────────────────────

def generate_dataset() -> tuple[list[dict], dict]:
    rng = np.random.default_rng(RNG_SEED)

    # 1. Build piece list
    pieces = []
    next_piece_id = 0
    for family in PROCEDURAL_FAMILIES:
        for inst in range(INSTANCES_PER_FAMILY):
            piece = make_piece(family, inst, rng)
            piece["piece_id"] = f"{family}_{inst:03d}"
            piece["uid"] = next_piece_id
            pieces.append(piece)
            next_piece_id += 1

    # MVP pieces (real-data hold-in)
    for mvp_name, (family, params, height_mm, _) in MVP_SHAPES.items():
        gen = FAMILY_GENERATORS[family]
        xy_mm, params_out = gen(rng, params=params)
        piece = {
            "family":    family,
            "instance":  -1,
            "xy_mm":     xy_mm,
            "height_mm": height_mm,
            "params":    params_out,
            "piece_id":  mvp_name,
            "uid":       next_piece_id,
            "is_mvp":    True,
        }
        pieces.append(piece)
        next_piece_id += 1

    print(f"[gen] pieces total = {len(pieces)} "
          f"({len(PROCEDURAL_FAMILIES) * INSTANCES_PER_FAMILY} procedural + "
          f"{len(MVP_SHAPES)} MVP)")

    # 2. For each piece, generate cavity pool + per-rotation configs
    rows = []
    next_config_id = 0

    for piece in pieces:
        is_mvp = bool(piece.get("is_mvp", False))

        cavity_pool = make_cavity_pool(piece, pieces, rng)

        # piece descriptors (rasterise once)
        piece_centred  = piece["xy_mm"] - piece["xy_mm"].mean(axis=0)
        piece_area     = shoelace_area_mm2(piece_centred)
        piece_perim    = perimeter_mm(piece_centred)
        piece_compact  = compactness(piece_area, piece_perim)
        piece_bbox_ar  = bbox_aspect_ratio(piece_centred)
        piece_height   = float(piece["height_mm"])
        piece_volume   = piece_area * piece_height

        for cav_idx, cavity in enumerate(cavity_pool):
            cav_centred = cavity["xy_mm"] - cavity["xy_mm"].mean(axis=0)
            cav_area    = shoelace_area_mm2(cav_centred)
            cav_perim   = perimeter_mm(cav_centred)
            cav_compact = compactness(cav_area, cav_perim)
            cav_bbox_ar = bbox_aspect_ratio(cav_centred)
            cav_depth   = float(cavity["depth_mm"])
            cav_volume  = cav_area * cav_depth

            cavity_id   = f"{piece['piece_id']}_cav{cav_idx:02d}"
            depth_offset = cav_depth - piece_height
            insertion_required_mm = max(MIN_REQUIRED_INSERTION_MM,
                                          INSERTION_FRACTION * piece_height)

            for rot_deg in ROTATIONS_DEG:
                piece_rot = rotate_xy(piece_centred, rot_deg)

                metrics = compute_pair_metrics(piece_rot, cav_centred)

                area_ratio = piece_area / max(cav_area, 1e-6)
                lateral_clearance_proxy = cav_area - piece_area  # mm^2
                bbox_aspect_diff = abs(piece_bbox_ar - cav_bbox_ar)
                compactness_diff = abs(piece_compact - cav_compact)

                # Phase D.7 partial-insertion affordance label
                lateral_ok = (metrics["outside_ratio"] <= LATERAL_OUTSIDE_MAX
                              and metrics["inside_ratio"] >= LATERAL_INSIDE_MIN)
                depth_ok   = (cav_depth >= insertion_required_mm - DEPTH_TOLERANCE_MM
                              and cav_depth >= MIN_INSERTION_GUIDANCE_MM)
                label = int(lateral_ok and depth_ok)

                if not lateral_ok and not depth_ok:
                    label_reason = "lateral_fail+depth_fail"
                elif not lateral_ok:
                    label_reason = "lateral_fail"
                elif not depth_ok:
                    label_reason = "depth_fail"
                else:
                    label_reason = "feasible"

                # Random split assignment (deterministic with seed)
                # 70 / 15 / 15 train / val / test, stratified later if needed.
                split_r = rng.random()
                if split_r < 0.70:
                    split = "train"
                elif split_r < 0.85:
                    split = "val"
                else:
                    split = "test"

                row = {
                    # ── Identifiers (NOT model features; tracing/debug only) ──
                    "config_id":     next_config_id,
                    "piece_id":      piece["piece_id"],
                    "cavity_id":     cavity_id,
                    "shape_family":  piece["family"],
                    "is_mvp":        is_mvp,
                    "cavity_source": cavity["source"],
                    "split":         split,
                    "heldout_family_fold": FAMILY_FOLD_ID.get(piece["family"], -1),

                    # ── Action / candidate ──
                    "candidate_rotation_deg": rot_deg,

                    # ── Piece descriptors (3D extrusion, identity-free) ──
                    "piece_area_mm2":        round(piece_area, 6),
                    "piece_perimeter_mm":    round(piece_perim, 6),
                    "piece_compactness":     round(piece_compact, 6),
                    "piece_height_mm":       round(piece_height, 6),
                    "piece_volume_mm3":      round(piece_volume, 6),
                    "piece_bbox_aspect_ratio": round(piece_bbox_ar, 6),

                    # ── Cavity descriptors (3D extrusion, identity-free) ──
                    "cavity_area_mm2":        round(cav_area, 6),
                    "cavity_perimeter_mm":    round(cav_perim, 6),
                    "cavity_compactness":     round(cav_compact, 6),
                    "cavity_depth_mm":        round(cav_depth, 6),
                    "cavity_volume_mm3":      round(cav_volume, 6),
                    "cavity_bbox_aspect_ratio": round(cav_bbox_ar, 6),

                    # ── Pair / action descriptors ──
                    "area_ratio":              round(area_ratio, 6),
                    "depth_offset_mm":             round(depth_offset, 6),
                    "diag_insertion_required_mm":  round(insertion_required_mm, 6),
                    "bbox_aspect_diff":        round(bbox_aspect_diff, 6),
                    "compactness_diff":        round(compactness_diff, 6),
                    "iou":                     round(metrics["iou"], 6),
                    "lateral_clearance_proxy_mm2": round(lateral_clearance_proxy, 6),

                    # ── Diagnostics (LABEL-ONLY; NOT model features) ──
                    "diag_inside_ratio_raw":  round(metrics["inside_ratio"], 6),
                    "diag_outside_ratio_raw": round(metrics["outside_ratio"], 6),
                    "diag_p_area_px":         metrics["p_area_px"],
                    "diag_c_area_px":         metrics["c_area_px"],
                    "diag_label_reason":      label_reason,

                    # ── Label ──
                    "label": label,
                }
                rows.append(row)
                next_config_id += 1

    # 3. Summary
    n_total   = len(rows)
    n_pos     = sum(r["label"] for r in rows)
    n_neg     = n_total - n_pos
    pos_rate  = n_pos / max(n_total, 1)

    per_family = {}
    for r in rows:
        fam = r["shape_family"]
        if fam not in per_family:
            per_family[fam] = {"n": 0, "pos": 0}
        per_family[fam]["n"]   += 1
        per_family[fam]["pos"] += r["label"]

    summary = {
        "n_pieces":          len(pieces),
        "n_procedural":      len(PROCEDURAL_FAMILIES) * INSTANCES_PER_FAMILY,
        "n_mvp":             len(MVP_SHAPES),
        "cavities_per_piece": CAVITIES_PER_PIECE,
        "rotations_per_pair": len(ROTATIONS_DEG),
        "n_configurations":  n_total,
        "n_positive":        int(n_pos),
        "n_negative":        int(n_neg),
        "positive_rate":     round(pos_rate, 6),
        "per_family":        {
            fam: {"n_configs": d["n"], "n_positive": int(d["pos"]),
                  "positive_rate": round(d["pos"] / max(d["n"], 1), 6),
                  "fold_id": FAMILY_FOLD_ID.get(fam, -1)}
            for fam, d in per_family.items()
        },
        "any_family_zero_positive": any(d["pos"] == 0 for d in per_family.values()),
        "leakage_diagnostic_columns": [
            "diag_inside_ratio_raw", "diag_outside_ratio_raw",
            "diag_p_area_px", "diag_c_area_px", "diag_label_reason",
        ],
        "identifier_columns_excluded_from_features": [
            "config_id", "piece_id", "cavity_id", "shape_family", "is_mvp",
            "cavity_source", "split", "heldout_family_fold",
        ],
    }

    return rows, summary


# ── Output writers ────────────────────────────────────────────────────────────

def write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fields = list(rows[0].keys())
    with path.open("w") as f:
        f.write(",".join(fields) + "\n")
        for r in rows:
            line = ",".join(str(r[k]) for k in fields)
            f.write(line + "\n")
    print(f"[write] {path} ({len(rows)} rows)")


def write_parquet(rows: list[dict], path: Path) -> None:
    try:
        import pandas as pd
    except ImportError:
        print("[write] pandas not available, skipping parquet output")
        return
    try:
        df = pd.DataFrame(rows)
        df.to_parquet(str(path), engine="pyarrow", index=False)
        print(f"[write] {path}")
    except Exception as exc:
        print(f"[write] parquet write failed ({exc}), skipping")


def write_summary_json(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "script_name":    "generate_phaseD_3d_affordance_dataset.py",
        "phase":          "Phase D.7 — 3D-extrusion partial-insertion affordance dataset",
        "phase_note": (
            "Procedural convex-prismatic shapes + extruded MVP pieces. "
            "Deterministic geometric labels under PARTIAL-INSERTION-THROUGH-OPENING "
            "rule (NOT full containment). NOT robot control. NOT contact "
            "dynamics. Dataset only."
        ),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "rng_seed":      RNG_SEED,
        "thresholds":    {
            "lateral_outside_max":         LATERAL_OUTSIDE_MAX,
            "lateral_inside_min":          LATERAL_INSIDE_MIN,
            "depth_tolerance_mm":          DEPTH_TOLERANCE_MM,
            "min_required_insertion_mm":   MIN_REQUIRED_INSERTION_MM,
            "insertion_fraction":          INSERTION_FRACTION,
            "min_insertion_guidance_mm":   MIN_INSERTION_GUIDANCE_MM,
            "clearance_per_side_mm":       CLEARANCE_PER_SIDE_MM,
        },
        "summary":       summary,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"[write] {path}")


def write_report_md(summary: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# Phase D — 3D-extrusion Affordance Dataset")
    lines.append("")
    lines.append("> **This dataset supports learning a geometric affordance "
                 "score, not robot control.**")
    lines.append("")
    lines.append("> **Status**: Phase D.1/D.2 (dataset generation only). "
                 "No model has been trained from this data.")
    lines.append("")

    lines.append("## Objective")
    lines.append("")
    lines.append("Generate a procedural dataset of (piece, cavity, rotation) "
                 "configurations with deterministic affordance labels, to be "
                 "consumed in Phase D.3+ by an interpretable classifier "
                 "(logistic regression / shallow tree). Pieces and cavities "
                 "are convex prismatic shapes; the affordance label is "
                 "computed from a deterministic geometric fit rule (lateral "
                 "feasibility AND depth feasibility).")
    lines.append("")

    lines.append("## Controlled 3D-extrusion assumption")
    lines.append("")
    lines.append("Piece 3D shape ≈ vertical extrusion of the 2D footprint by "
                 "`piece_height_mm`. Cavity 3D shape ≈ vertical depression of "
                 "the 2D opening footprint by `cavity_depth_mm`. Valid only "
                 "for **convex prismatic shapes** and a **fixed vertical "
                 "insertion direction**. Side-face geometry, undercuts, "
                 "internal voids, non-prismatic cross-sections, and concave "
                 "outlines are out of scope.")
    lines.append("")

    lines.append("## Procedural shape families")
    lines.append("")
    for fam in PROCEDURAL_FAMILIES:
        lines.append(f"- `{fam}` — {INSTANCES_PER_FAMILY} instances")
    lines.append(f"- MVP real-data hold-in: {len(MVP_SHAPES)} pieces "
                 "(rectangle, square, circle, triangle) with their CAD-nominal "
                 "heights and cavity depths.")
    lines.append("")

    lines.append("## Affordance label rule (deterministic, partial-insertion)")
    lines.append("")
    lines.append("**Phase D.7 task definition: partial insertion through an "
                 "opening, NOT full containment.** A piece may be taller than "
                 "the cavity is deep; the relevant question is whether the "
                 "piece cross-section fits through the opening AND the cavity "
                 "is deep enough to engage the piece by a mechanically "
                 "meaningful depth.")
    lines.append("")
    lines.append("For each (piece, cavity, rotation) configuration:")
    lines.append("")
    lines.append(f"- **Lateral feasibility**: `outside_ratio_raw ≤ "
                 f"{LATERAL_OUTSIDE_MAX}` AND "
                 f"`inside_ratio_raw ≥ {LATERAL_INSIDE_MIN}`.")
    lines.append("")
    lines.append(f"- **Required insertion depth**:")
    lines.append("")
    lines.append(f"  `insertion_required_mm = max("
                 f"MIN_REQUIRED_INSERTION_MM={MIN_REQUIRED_INSERTION_MM} mm, "
                 f"INSERTION_FRACTION={INSERTION_FRACTION} * piece_height_mm)`")
    lines.append("")
    lines.append(f"- **Depth feasibility**:")
    lines.append("")
    lines.append(f"  `cavity_depth_mm ≥ insertion_required_mm − "
                 f"DEPTH_TOLERANCE_MM={DEPTH_TOLERANCE_MM} mm`")
    lines.append("")
    lines.append(f"  AND `cavity_depth_mm ≥ "
                 f"MIN_INSERTION_GUIDANCE_MM={MIN_INSERTION_GUIDANCE_MM} mm`.")
    lines.append("")
    lines.append("- **Affordance label** = 1 iff BOTH; 0 otherwise.")
    lines.append("")
    lines.append("Thresholds are **fixed operating points**, not free tuning "
                 "parameters.")
    lines.append("")

    lines.append("## Feature list (identity-free)")
    lines.append("")
    lines.append("**Piece descriptors**: `piece_area_mm2`, "
                 "`piece_perimeter_mm`, `piece_compactness`, "
                 "`piece_height_mm`, `piece_volume_mm3`, "
                 "`piece_bbox_aspect_ratio`.")
    lines.append("")
    lines.append("**Cavity descriptors**: `cavity_area_mm2`, "
                 "`cavity_perimeter_mm`, `cavity_compactness`, "
                 "`cavity_depth_mm`, `cavity_volume_mm3`, "
                 "`cavity_bbox_aspect_ratio`.")
    lines.append("")
    lines.append("**Pair / action descriptors**: `area_ratio`, "
                 "`depth_offset_mm`, `insertion_required_mm`, "
                 "`bbox_aspect_diff`, `compactness_diff`, "
                 "`candidate_rotation_deg`, `iou`, "
                 "`lateral_clearance_proxy_mm2`.")
    lines.append("")

    lines.append("## Excluded / leakage-prone columns")
    lines.append("")
    lines.append("**Diagnostics ONLY (NOT model features — used for label "
                 "generation)**: `diag_inside_ratio_raw`, "
                 "`diag_outside_ratio_raw`, `diag_p_area_px`, "
                 "`diag_c_area_px`, `diag_label_reason`.")
    lines.append("")
    lines.append("**Identifiers ONLY (NOT model features — tracing / "
                 "debugging)**: `config_id`, `piece_id`, `cavity_id`, "
                 "`shape_family`, `is_mvp`, `cavity_source`, `split`, "
                 "`heldout_family_fold`.")
    lines.append("")
    lines.append("Phase D training MUST exclude these columns from the "
                 "classifier input.")
    lines.append("")

    lines.append("## Dataset statistics")
    lines.append("")
    lines.append(f"- Pieces total: **{summary['n_pieces']}** "
                 f"({summary['n_procedural']} procedural + {summary['n_mvp']} MVP)")
    lines.append(f"- Cavities per piece: {summary['cavities_per_piece']}")
    lines.append(f"- Rotations per (piece, cavity): {summary['rotations_per_pair']}")
    lines.append(f"- Total configurations: **{summary['n_configurations']}**")
    lines.append(f"- Positive labels: **{summary['n_positive']}** "
                 f"({summary['positive_rate']*100:.2f}% positive rate)")
    lines.append(f"- Negative labels: {summary['n_negative']}")
    lines.append(f"- Any family with zero positives: **{summary['any_family_zero_positive']}**")
    lines.append("")

    lines.append("### Per-family breakdown")
    lines.append("")
    lines.append("| family | fold_id | n_configs | n_positive | positive_rate |")
    lines.append("|---|---:|---:|---:|---:|")
    for fam, d in summary["per_family"].items():
        lines.append(f"| `{fam}` | {d['fold_id']} | {d['n_configs']} | "
                     f"{d['n_positive']} | {d['positive_rate']:.4f} |")
    lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append("- **Synthetic only**: labels are generated from raster "
                 "geometry, not physical insertion trials. Any sim-to-real "
                 "claim is out of scope.")
    lines.append("- **Convex prismatic only**: the extrusion assumption "
                 "breaks for non-prismatic / concave shapes. Star and other "
                 "stress shapes are deferred.")
    lines.append("- **No XY offset** in this first dataset: only rotations are "
                 "swept. Adding offsets is a future extension; the current "
                 "design intentionally keeps configurations interpretable.")
    lines.append("- **No height/depth measurement noise**: a noise ablation "
                 "is reserved for a future run, kept separate from the "
                 "primary dataset.")
    lines.append("- **No model trained from this data yet**.")
    lines.append("")

    lines.append("## Closing note")
    lines.append("")
    lines.append("This dataset supports learning a **geometric affordance "
                 "score, not robot control**. The downstream classifier (Phase "
                 "D.3) will rank candidate cavities by predicted affordance "
                 "and output a top-1 cavity per piece, plus a rank margin. "
                 "Insertion execution, grasp planning, force feedback, and "
                 "any robotic action remain out of scope.")

    path.write_text("\n".join(lines) + "\n")
    print(f"[write] {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("=" * 70)
    print("generate_phaseD_3d_affordance_dataset.py")
    print("=" * 70)
    print(f"PROJECT_ROOT : {PROJECT_ROOT}")
    print(f"OUT_DIR      : {OUT_DIR}")
    print(f"RNG_SEED     : {RNG_SEED}")
    print(f"families     : {PROCEDURAL_FAMILIES}")
    print(f"instances/family: {INSTANCES_PER_FAMILY}")
    print(f"cavities/piece  : {CAVITIES_PER_PIECE}")
    print(f"rotations       : {len(ROTATIONS_DEG)} (every 10 degrees)")
    print()

    rows, summary = generate_dataset()

    print()
    print(f"[summary] n_configurations = {summary['n_configurations']}")
    print(f"[summary] n_positive       = {summary['n_positive']} "
          f"({summary['positive_rate']*100:.2f}%)")
    print(f"[summary] any_family_zero_positive = "
          f"{summary['any_family_zero_positive']}")
    print()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(rows, CSV_PATH)
    write_parquet(rows, PARQUET_PATH)
    write_summary_json(summary, SUMMARY_PATH)
    write_report_md(summary, REPORT_PATH)

    print("[done].")


if __name__ == "__main__":
    main()
