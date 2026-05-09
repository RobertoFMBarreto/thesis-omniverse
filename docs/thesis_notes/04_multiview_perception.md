# 04 — Multi-view perception (Baseline 2)

> Implementation note for future conversion into a LaTeX section.
> Status: Baseline 2 — Phase A (multi-view capture proof-of-life).
> Capture-only; **no** fusion, **no** multi-view matching yet.
> Date: 2026-05-09.

---

## 1. Phase A — Multi-view capture proof-of-life

> Cross-reference: section 2 below is the formal completion record of Phase A once the capture pipeline was extended from the rectangle alone to the four MVP pieces (rectangle, square, circle, triangle); this section 1 is preserved verbatim as the chronological proof-of-life record from the rectangle-only stage.

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

---

## 2. Phase A — Completion across the MVP set

### 2.1 Introduction / Motivation

Section 1 documents the rectangle-only proof-of-life. That stage
demonstrated the chain (target auto-resolution, three-view
layout, intrinsics consistency, pose round-trip), but it was not
sufficient to close Phase A: Baseline 2 commits the capture
pipeline to the full MVP set (`rectangle, square, circle,
triangle`; doc 03 — section 11), on which Baseline 1 already
operates (doc 03 — section 17). Reporting Phase A as complete on
one piece would have been an over-claim.

Section 2 records the formal completion: the same infrastructure
was extended to iterate over the four MVP pieces in one
execution, producing per-piece three-view captures with the same
intrinsics, provenance and artefact layout. The exercise remains
**capture-only proof-of-life**: no multi-view fusion, no
multi-view matching, no 3D reconstruction, no pose estimation,
no learned component.

### 2.2 Multi-view strategy

The strategy is the same as section 1.4, inherited unchanged:
three viewpoints — `top_down`, `front_oblique`, `side_oblique` —
one per principal world axis (Z, Y, X). Justification:

- **Simplicity.** Smallest set exposing one observation per
  principal axis; minimum required to discuss the depth-edge
  bias and partial cavity-wall observation identified in the
  Baseline 2 design proposal (cross-referenced from doc 03 —
  section 17).
- **Low computational cost.** Three captures per piece at the
  intrinsics of section 1.5; no per-view optimisation at capture
  time.
- **Geometric appropriateness for the MVP.** The four MVP pieces
  are simple convex prismatic geometries; top + front + side
  exposes their principal faces. The non-convex star is out of
  scope (see section 2.6).
- **Minimum experimental complexity.** A larger view set would
  not address any question Phase A is meant to answer.

Sequential single-camera relocation is also inherited from
section 1.4. It is a Phase A scoping choice, not the final
architecture: static authored cameras in USD are intended from
Phase B onwards (section 2.6 and section 1.7, item 1).

### 2.3 Pipeline architecture

The script is the same `scripts/capture_multiview_piece.py` of
section 1, extended with per-piece iteration and visibility
control. Configurable constants: `CAPTURE_ALL_PIECES` (boolean
switch between the rectangle-only legacy mode and the full-MVP
mode), `PIECE_CAPTURE_ORDER = ["rectangle", "square", "circle",
"triangle"]`, and `PIECE_NAME_HINTS` (per-piece prim-name hints,
generalising the rectangle hints of section 1.3).

Helpers introduced:

- `collect_mvp_piece_prims` — read-only stage traversal resolving
  each MVP piece root using the section 1.3 selection rule
  (exact-name match first, then largest bbox volume, ties broken
  by shallowest path depth).
- `set_piece_visibility` — operates on `UsdGeom.Imageable`,
  flipping the `visibility` token between `inherited` and
  `invisible`.
- `snapshot_visibility` / `restore_visibility` — capture and
  restore the per-prim original `visibility` token in an outer
  `try/finally`, so visibility is always restored even if a
  per-piece capture fails.

Per-piece capture loop (ASCII, no Unicode box characters):

```
   for piece_name in PIECE_CAPTURE_ORDER:
       try:
           snapshot_visibility(all MVP piece roots)
           hide all MVP piece roots except `piece_name`
           show `piece_name`
           target_centre = resolve_target_look_at(piece_name)
           sanity_warn = check_against_cad(target_centre, piece_name)
           for view in (top_down, front_oblique, side_oblique):
               set camera pose from target_centre + view offset
               capture rgb, depth
               save rgb.png, depth.npy, depth_vis.png, metadata.json
           build per-piece contact sheet
           build per-piece summary json
       except Exception as exc:
           record failure for this piece, continue
       finally:
           restore_visibility(all MVP piece roots)
```

Visibility control is allowed **only as scene-setup automation**.
Piece identity is **not** consumed by any perception, scoring or
matching step; it is used solely to decide which prim to show or
hide and where the camera is aimed. Per-view metadata, per-piece
summary and global aggregate all record the disclaimer
`"Piece identity used only for experimental scene setup, not for
perception/matching."` together with
`visibility_control_enabled = True`, so this constraint is
recoverable from the artefacts independently of the script.

CAD sanity warning: after `resolve_target_look_at`, the largest
dimension of the resolved bbox is compared against
`data/expected_cad_dimensions.json`; if outside
`±SANITY_WARN_TOLERANCE = 0.30` (±30 % of the CAD largest
dimension), a `[sanity_warn]` line is emitted and recorded in
the per-view metadata, the per-piece summary and the global
aggregate. The warning never aborts the run; it flags scene
drift (swapped piece, unit mismatch, CAD desynchronised from
USD) without enforcing policy at capture time.

The per-piece `try/except` prevents one failure from stopping
the run; the outer `try/finally` guarantees the scene's original
visibility state is restored.

### 2.4 Experimental configuration

- Isaac Sim 5.1 in container, via WebRTC (same as section 1).
- Execution from the Script Editor with the standard
  `asyncio.ensure_future(main())` pattern.
- Pieces (subset of the project's MVP set): `rectangle`,
  `square`, `circle`, `triangle`. The star is out of scope for
  the MVP (section 2.6).
- Intrinsics are unchanged from the rectangle proof-of-life
  (see section 1.4 / 1.5); the same square-pixel intrinsics are
  reused across the four pieces and the three views per piece,
  by construction.

Output organisation per piece (4 × 3 = 12 captures total):

- `data/multiview_captures/pieces/<piece_name>/run_log.txt`
- `data/multiview_captures/pieces/<piece_name>/views_contact_sheet.png`
- `data/multiview_captures/pieces/<piece_name>/multiview_capture_summary.json`
- `data/multiview_captures/pieces/<piece_name>/view_00_top_down/{rgb.png, depth.npy, depth_vis.png, metadata.json}`
- `data/multiview_captures/pieces/<piece_name>/view_01_front_oblique/{...}`
- `data/multiview_captures/pieces/<piece_name>/view_02_side_oblique/{...}`

Plus a global aggregate at
`data/multiview_captures/pieces/multiview_phaseA_all_pieces_summary.json`
collecting per-piece success status, target-resolution blocks,
sanity-warning records and pointers to the per-piece summaries.

### 2.5 Results and validation

- Four pieces, three views per piece, **12 captures total**.
- Per-piece contact sheets generated and visually validated for
  the four pieces.
- **Object isolation correct.** The visibility-control mechanism
  shows only the target piece per capture; verified visually
  from the four contact sheets.
- **No observable clipping** in any of the 12 frames.
- **Geometric consistency across views** maintained: the long
  edge of the rectangle remains visually consistent across its
  three views (as in section 1.6), and analogous visual
  consistency holds for the square, the circle and the triangle.
  This is visual confirmation, not a measured metric: Phase A
  does not produce cross-view consistency numbers, by design.

No fused representation, no IoU, no Chamfer distance and no
cross-view consistency number are reported here; none is
produced by the Phase A pipeline. They belong to Phase B onwards.

### 2.6 Limitations

1. **No multi-view fusion** implemented (Phase B).
2. **No 3D reconstruction** implemented.
3. **No pose estimation** implemented.
4. **Viewpoints are manually defined.** The three viewpoints
   (top + front + side) and the offset constants of section 1.4
   are reused unchanged across pieces; chosen for simplicity and
   one-axis-per-view coverage, **not** optimised for downstream
   matching or fusion quality.
5. **Pipeline is deterministic and geometry-only by design.** No
   learned component is used or proposed in this phase. Piece
   identity is used only for scene setup (section 2.3), not for
   perception or matching.
6. **Not validated for complex (non-convex) geometries.** The
   star piece remains a future stress test, out of scope for the
   MVP, consistent with doc 03 — section 11.
7. **Sequential camera relocation is not the final
   architecture.** It remains a proof-of-life convenience; the
   intended Phase B-onwards architecture authors multiple static
   cameras in USD (see section 1.7, item 1, and section 2.7).

### 2.7 Future work

Phase A closes the capture-only stage. Directions recorded as
next steps; none implemented in this section.

- **Phase B — deterministic fusion** of the per-view captures
  produced here, in the spirit of the Baseline 2 design proposal
  cross-referenced from doc 03 — section 17 (e.g. world-frame
  point-cloud accumulation across the three views per piece, no
  learned component).
- **Geometric descriptors derivable from fused
  representations**, without committing to any specific
  descriptor family.
- **More robust matching approaches** consuming fused multi-view
  inputs while keeping the Baseline 1 scoring head frozen as a
  reference, per the design principle in doc 03 — section 17.
- **Static-camera authoring in USD** as a reproducibility
  upgrade: replace sequential single-camera relocation by
  multiple static authored cameras, so the pose set is
  recoverable from the USD file alone.
