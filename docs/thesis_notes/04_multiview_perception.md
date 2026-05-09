# 04 — Multi-view perception (Baseline 2)

> Implementation note for future conversion into a LaTeX section.
> Status: Baseline 2 — Phase A (multi-view capture proof-of-life).
> Capture-only; **no** fusion, **no** multi-view matching yet.
> Date: 2026-05-09.

---

## 1. Phase A — Multi-view capture proof-of-life

### 1.1 Objective of Phase A

Phase A validates the **multi-view capture infrastructure** that
Baseline 2 will rely on, in isolation from matching, fusion or
scoring. The scope is the question "can the *pipeline* place a
virtual camera at several known poses around a target piece,
capture coherent RGB-D data from each pose, and persist per-view
artifacts with full provenance?".

Phase A **does not** validate fusion of the per-view observations,
**does not** validate multi-view matching or compatibility scoring,
and **does not** replace Baseline 1. It produces the per-view
inputs that Phase B (deterministic fusion) will consume; it does
not perform that fusion. Artifacts are infrastructure evidence,
not scientific results about insertion compatibility.

The script is `scripts/capture_multiview_piece.py`. The target
piece is the `rectangle` only — same piece as in the validated
single-view of doc 01 — section 18.8.

### 1.2 Why multi-view follows Baseline 1

Baseline 1 (doc 03 — section 17) closes the deterministic
*top-down* matching on the final main set under the single-view,
single-piece, single-top-face premise carried since Phase 1.
Several limitations of that premise are documented but not
addressed by Baseline 1: lateral and bottom faces are not
observable (doc 01 — section 14, item 6); the piece auxiliary
depth point cloud may degenerate to a *2.5D heightmap* on
prismatic pieces (doc 01 — section 18.9, item 1); the cavity
auxiliary depth representation can also degenerate when the depth
annotator does not robustly observe the cavity interior (doc 02 —
section 20.11).

Baseline 2 targets these limitations by combining observations
from several camera poses. Phase A is the prerequisite step:
before fusion or matching across views can be attempted, the
per-view capture chain has to exist, has to be reproducible, and
has to record enough provenance for downstream phases. The
depth-edge bias and partial cavity-wall observation issues
identified in the Baseline 2 design proposal (cross-referenced by
doc 03 — section 17) are not measured here; they belong to Phase E.

### 1.3 Target selection and auto-centring

The first run of `capture_multiview_piece.py` placed the cameras
relative to the world origin instead of the rectangle, framing the
wrong region. The fix introduced an automatic target resolution
step that is now the default:

- `TARGET_MODE = "auto_prim_bbox"`;
- `TARGET_PRIM_NAME_HINTS = ["rectangle", "Rectangle", "Rect", "rect"]`.

Resolution is a read-only USD stage traversal. The bounding box of
each candidate prim is computed via
`UsdGeom.BBoxCache(Usd.TimeCode.Default(),
includedPurposes=[UsdGeom.Tokens.default_, UsdGeom.Tokens.render],
useExtentsHint=True)`, with a mesh-points fallback when the cache
returns an empty range. The implementation mirrors the read-only
pattern of `scripts/inspect_cavity_scene_scale.py` (cf. doc 02 —
section 20.5) and does not modify the stage.

Selection rule, in order: exact-name match against the hints takes
precedence; otherwise the candidate with the largest bounding-box
volume wins; ties are broken by shallowest path depth. Prims whose
type contains `Material`, `Shader`, `Light`, `Camera` or `Scope`
are excluded. If auto-resolution fails, the script raises a
`RuntimeError` with explicit instructions to either adjust the hint
list or switch to `TARGET_MODE = "manual"`. There is no silent
fallback to the world origin.

Run-time evidence: the resolver selected `/World/Rectangle`,
bounding-box size **50 × 75 × 105 mm** (consistent with the CAD
nominal of doc 01 — section 18.2), bounding-box centre in world
coordinates **(−0.2500, 0.4500, 0.4525) m**. Stored in the
top-level `target_resolution` block of
`multiview_capture_summary.json` and propagated into each per-view
`metadata.json`.

### 1.4 Camera layout

The Phase A layout is a **sequential-camera proof-of-life**
configuration, not the final architecture: cameras are
programmatically repositioned between captures within one script
execution. The architecture intended for Phase B onwards is
multiple **static** cameras authored in USD with no programmatic
pose changes at capture time (recorded as a limitation in 1.7).

Three views, derived from the resolved target centre
`(−0.2500, 0.4500, 0.4525) m`:

- `top_down`: offset `(0.00, 0.00, +0.50)` m;
- `front_oblique`: offset `(0.00, −0.30, +0.40)` m
  (≈ 36.9° from vertical);
- `side_oblique`: offset `(+0.30, 0.00, +0.40)` m
  (≈ 36.9° from vertical).

Constants: `TOP_DOWN_HEIGHT = 0.50 m`, `OBLIQUE_HEIGHT = 0.40 m`,
`OBLIQUE_OFFSET = 0.30 m`. Angle from vertical for the obliques is
`atan(0.30 / 0.40) ≈ 36.9°`, inside the 35°–45° range targeted by
the Phase A design.

A previous layout used `oblique_left` + `oblique_right` on
opposing X-axis offsets. It was replaced by `front_oblique`
(offset on −Y) plus `side_oblique` (offset on +X), so each
principal world axis (X, Y, Z) is represented by exactly one view.
A Phase A scoping decision, not a measured optimum.

### 1.5 Outputs generated

Per-view intrinsics are identical across all three views:
640 × 480, `focal_mm = 24.0`, `aperture_mm = 36.0`,
`(cx_px, cy_px) = (320.0, 240.0)`,
`fx_px = fy_px = 426.667`,
`intrinsics_model = "pinhole_square_pixels"` — same square-pixel
correction validated in Baseline 1 (doc 01 — section 18.12).

Outputs under `data/multiview_captures/pieces/rectangle/`:

- `run_log.txt`;
- `views_contact_sheet.png`;
- `multiview_capture_summary.json` (with a top-level
  `target_resolution` block);
- `view_00_top_down/{rgb.png, depth.npy, depth_vis.png, metadata.json}`;
- `view_01_front_oblique/{rgb.png, depth.npy, depth_vis.png, metadata.json}`;
- `view_02_side_oblique/{rgb.png, depth.npy, depth_vis.png, metadata.json}`.

Per-view `metadata.json` records: `view_name`,
`camera_prim_path`, `requested_pose`,
`measured_pose_read_back_from_stage`, `image_width/height`,
`focal_mm/aperture_mm/fx_px/fy_px/cx_px/cy_px`,
`intrinsics_model`, `depth_valid_min_m`, `depth_valid_max_m`,
`n_valid_depth_pixels`, `timestamp_utc`, `run_id`, `phase_note`,
plus the auto-target fields `target_mode`,
`selected_target_prim_path`, `target_bbox_center_world_m`,
`target_bbox_size_mm`, `target_bbox_method`, `requested_look_at`,
`camera_offset_from_target`. The pose-read-back field is kept
distinct from the requested pose so any discrepancy is auditable.

### 1.6 Results and per-view diagnosis

The validated run reports `success = True` with
`views_succeeded = 3/3`. Qualifications below are preserved
verbatim because each view exposes a different aspect of the
capture chain.

**top_down — PASS.** Piece visible, roughly centred, not clipped.
Depth sensible: three distinct levels — floor at ≈ 0.95 m, board
at ≈ 0.70–0.75 m, top of the piece at ≈ 0.45–0.50 m. Pose match
exact: requested `(−0.25, 0.45, 0.9525)` = measured; quaternion
`(1, 0, 0, 0)` (identity, straight down). Depth range
`[0.4475, 0.9525]` m matches the geometry exactly. **⚠ Scene
intruder** in the bottom-right corner: a grey box with depth
≈ 0.50–0.60 m overlapping the depth band of the piece. Not the
rectangle. Will need handling before Phase B segmentation.

**front_oblique — PASS.** Piece visible, not clipped, near the
horizontal centre. Genuinely complementary to `top_down` (front
face of the board exposed). Depth a smooth gradient,
non-degenerate. Pose match exact: `(−0.25, 0.15, 0.8525)`;
quaternion `(0.9487, 0.3162, 0, 0)` — rotation about X, downward
tilt from −Y. Depth range `[0.4355, 1.8405]` m — plausible (close
board edge to distant background).

**side_oblique — PASS with caution.** Piece visible, not clipped,
but **off-centre to the left**: the piece sits at world X = −0.25 m
and the camera is offset +0.30 m on X relative to the target,
which puts the piece far from the camera's principal axis.
Genuinely complementary (board seen near edge-on; perpendicular
face of the piece exposed). Depth smooth, non-degenerate. Pose
match exact: `(0.05, 0.45, 0.8525)`; quaternion
`(0.9487, 0, 0.3162, 0)` — rotation about Y, looking left. Depth
range `[0.3622, 2.4308]` m — wider than the other two views; this
compresses the colour map of `depth_vis.png` so the piece appears
only subtly distinct from the board, but the raw `.npy` retains
the signal.

Cross-cutting: intrinsics identical across all views (no
inconsistency); pose discrepancy zero within float precision; the
up-vector / orientation is correct (the long edge of the rectangle
remains consistent across views).

### 1.7 Limitations and risks before fusion

1. **Sequential camera movement.** A single camera moved between
   three programmatic poses within one script execution.
   Acceptable as proof-of-life only. The architecture intended for
   Phase B onwards is multiple **static** cameras authored in USD,
   with no programmatic pose changes at capture time — a
   reproducibility requirement: a static authored layout makes the
   pose set recoverable from the USD file alone.
2. **Scene intruder in the top-down view.** The grey box in the
   bottom-right corner overlaps the depth band of the piece. If
   propagated unchanged, an additive occupancy fusion in Phase B
   would treat it as part of the piece. Must be handled before
   Phase B segmentation (scene fix, per-view ROI, or per-view
   masking).
3. **`side_oblique` offset may need tuning.** With the rectangle
   at world X = −0.25 m, an `OBLIQUE_OFFSET = 0.30 m` on +X puts
   the piece off-centre. Reducing it to ≈ 0.20 m would re-centre
   the piece without changing the angular character. Optional
   adjustment, not a Phase A blocker.
4. **No fused representation produced.** Phase A artifacts are
   per-view only; no fused point cloud, no fused footprint, no
   multi-view consistency check. By design.
5. **Auto-target resolver depends on prim naming hints.** If the
   USD scene is restructured or the rectangle prim is renamed,
   the hints list must be updated. The script raises a clear
   `RuntimeError` rather than aiming silently elsewhere — a
   deliberate design choice, not a robustness gap.

### 1.8 Current Phase A status

- `scripts/capture_multiview_piece.py` runs successfully on the
  current scene with `TARGET_MODE = "auto_prim_bbox"`, resolving
  `/World/Rectangle` deterministically;
- three views captured with identical square-pixel intrinsics and
  per-view pose round-trip exact within float precision;
- per-view artifacts (`rgb.png`, `depth.npy`, `depth_vis.png`,
  `metadata.json`) and global artifacts (`run_log.txt`,
  `views_contact_sheet.png`, `multiview_capture_summary.json`)
  written under `data/multiview_captures/pieces/rectangle/`;
- cavity-side multi-view capture, fusion and multi-view matching
  are out of scope for this phase and are not implemented;
- Baseline 1 remains the current reference for piece-cavity
  matching (doc 03 — section 17.8).

### 1.9 Notes for the author

- Frame Phase A as **infrastructure validation** — capture
  pipeline, target resolution, intrinsics consistency, pose
  round-trip. The matching outcome, which is the scientific
  question of Baseline 2, belongs to Phase D. Phase A must not be
  presented as evidence about matching.
- Do not claim the auto-target resolution generalises beyond the
  current scene. It is a deterministic substring-match heuristic;
  correct here because the rectangle prim is unambiguously named.
- The choice of `top + front + side` instead of `top + two
  opposing X obliques` is a Phase A scoping decision, not a
  measured optimum — one of several reasonable minimal sets,
  chosen for one-axis-per-view coverage.
- The depth-edge bias and partial cavity-wall observation
  limitations identified in the Baseline 2 design proposal
  (cross-referenced by doc 03 — section 17) are **not** evaluated
  by Phase A. Phase E will.
- Archive a copy of the validated `multiview_capture_summary.json`
  and the three per-view `metadata.json` files alongside the
  capture artifacts, so the resolved target pose and the
  requested/measured pose pairs are recoverable independently of
  any later script re-execution.
- Save a screenshot of the USD *stage* inspector at the moment of
  the validated capture, showing the `/World/Rectangle` prim and
  the camera prim path.
- Record the exact version of Isaac Sim and of the container in
  which the validated capture was produced.
