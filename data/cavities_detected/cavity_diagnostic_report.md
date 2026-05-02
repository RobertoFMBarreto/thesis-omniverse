# Cavity diagnostic report

- Generated: `2026-05-02T15:14:15`
- Source: `/Users/robertofmbarreto/Documents/Mestrado/tese/code/thesis-omniverse/data/cavities_detected`
- PNG: `cavity_diagnostic_report.png`

## 1. File presence

**Global files:**
- ✅  `rgb.png`  (RGB)
- ✅  `depth_vis.png`  (Depth)
- ✅  `board_surface_mask.png`  (Board surface)
- ✅  `board_region_mask.png`  (Board region)
- ✅  `cavity_opening_mask.png`  (Cavity opening)
- ✅  `cavity_depth_mask.png`  (Cavity depth)
- ✅  `depth_band_cavity_mask.png`  (Depth-band (legacy))
- ✅  `raw_cavity_mask.png`  (Raw cavity (active))
- ✅  `cavities_debug.png`  (Cavities debug)
- ✅  `cavities_summary.json`
- ✅  `run_log.txt`
- ✅  `board_mask.png`

**Cavity folders found:** 4
- `cavity_00/`:
  - ✅ `cavity_debug.png`  (Debug overlay)
  - ✅ `cavity_mask.png`  (Primary mask)
  - ✅ `cavity_opening_mask.png`  (Opening mask)
  - ✅ `cavity_depth_mask.png`  (Depth mask (aux))
  - ✅ `cavity_footprint.png`  (Primary footprint)
  - ✅ `cavity_opening_footprint.png`  (Opening footprint)
  - ✅ `cavity_metadata.json`
  - ✅ `cavity_pointcloud.npy`
  - ✅ `cavity_opening_pointcloud.npy`
  - ✅ `cavity_depth_pointcloud.npy`
- `cavity_01/`:
  - ✅ `cavity_debug.png`  (Debug overlay)
  - ✅ `cavity_mask.png`  (Primary mask)
  - ✅ `cavity_opening_mask.png`  (Opening mask)
  - ✅ `cavity_depth_mask.png`  (Depth mask (aux))
  - ✅ `cavity_footprint.png`  (Primary footprint)
  - ✅ `cavity_opening_footprint.png`  (Opening footprint)
  - ✅ `cavity_metadata.json`
  - ✅ `cavity_pointcloud.npy`
  - ✅ `cavity_opening_pointcloud.npy`
  - ✅ `cavity_depth_pointcloud.npy`
- `cavity_02/`:
  - ✅ `cavity_debug.png`  (Debug overlay)
  - ✅ `cavity_mask.png`  (Primary mask)
  - ✅ `cavity_opening_mask.png`  (Opening mask)
  - ✅ `cavity_depth_mask.png`  (Depth mask (aux))
  - ✅ `cavity_footprint.png`  (Primary footprint)
  - ✅ `cavity_opening_footprint.png`  (Opening footprint)
  - ✅ `cavity_metadata.json`
  - ✅ `cavity_pointcloud.npy`
  - ✅ `cavity_opening_pointcloud.npy`
  - ✅ `cavity_depth_pointcloud.npy`
- `cavity_03/`:
  - ✅ `cavity_debug.png`  (Debug overlay)
  - ✅ `cavity_mask.png`  (Primary mask)
  - ✅ `cavity_opening_mask.png`  (Opening mask)
  - ✅ `cavity_depth_mask.png`  (Depth mask (aux))
  - ✅ `cavity_footprint.png`  (Primary footprint)
  - ✅ `cavity_opening_footprint.png`  (Opening footprint)
  - ✅ `cavity_metadata.json`
  - ✅ `cavity_pointcloud.npy`
  - ✅ `cavity_opening_pointcloud.npy`
  - ✅ `cavity_depth_pointcloud.npy`

## 2. Primary cavity mask in use

`cavities_summary.json` reports `cavity_detection_mode = "opening_from_board_region"`.

Primary mask is **`cavity_opening_mask.png`** (the negative space inside the detected board region: `board_region_mask AND NOT board_surface_mask`). Each `cavity_NN/cavity_mask.png` is an alias of `cavity_NN/cavity_opening_mask.png`.

Auxiliary mask is **`cavity_depth_mask.png`** (visible deeper pixels inside each opening — used to estimate cavity depth, NOT used as the primary footprint).

## 3. Per-cavity primary footprint classification

| Cavity | classification | usability for Baseline 1 | opening_xy (mm) | opening_area px | depth_area px | z_depth median (mm) |
|---|---|---|---|---|---|---|
| `cavity_00` | full_opening | USABLE | 38.8 × 61.6 | 4260 | 0 | 0.0 |
| `cavity_01` | full_opening | USABLE | 37.3 × 37.3 | 1299 | 0 | 0.0 |
| `cavity_02` | full_opening | USABLE | 38.8 × 36.5 | 2544 | 0 | 0.0 |
| `cavity_03` | full_opening | USABLE | 48.7 × 46.4 | 3121 | 0 | 0.0 |

## 4. Baseline 1 readiness

✅ All cavities present a usable opening representation. `cavity_pointcloud.npy` (the alias) feeds Baseline 1 directly without changes.

## 5. What to inspect first

1. Open `cavity_diagnostic_report.png` and read the global row first.
2. Confirm `cavity_opening_mask.png` shows **filled, shape-correct apertures** (full circle, full triangle, full square, full rectangle). If any look like crescents or partial outlines, the opening derivation is broken upstream (`board_surface_mask` or `board_region_mask`).
3. Compare `cavity_opening_mask.png` vs `cavity_depth_mask.png` — the opening should be substantially larger than the depth (the depth captures only what the camera can see down the hole).
4. For each `cavity_NN/`, check that `cavity_opening_footprint.png` looks like the expected shape silhouette and that `cavity_metadata.json` has `primary_matching_representation = "cavity_opening_pointcloud"`.
