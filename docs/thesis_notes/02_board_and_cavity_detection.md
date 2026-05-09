# 02 — Automatic detection of the board and the cavities

> Implementation note for future conversion into a LaTeX section.
> Status: Phase 2 — deterministic perception, no learned component.
> Date: 2026-05-01.

---

## 1. Objective of the phase

The objective of this phase is to automatically detect the board on
the bench and, within the board, segment each geometric cavity as a
region of negative depth relative to the upper face of the board. For
each detected cavity, the geometric artifacts that will later feed
the deterministic baseline of piece-cavity matching
(*footprint matching*) are exported.

This phase **does not** classify cavity shapes, **does not**
establish any piece-cavity correspondence, **does not** involve
learned models, and **does not** control the robot. It is exclusively
a step of geometric perception based on depth.

---

## 2. Experimental context

The virtual bench in NVIDIA Isaac Sim 5.1 contains:

- a table/background plane;
- a rectangular board on the table, with through cavities;
- the camera view is approximately *top-down* over the board;
- the camera is the same as in Phase 1, but here repositioned
  directly in the USD *stage* to frame the entire board.

Capture is orchestrated via the Isaac Sim *Script Editor*, with the
same asynchronous pattern and the same `rgb` and
`distance_to_image_plane` annotators as in Phase 1.

---

## 3. Geometric model of the scene

The depth interpretation adopted distinguishes three levels, ordered
by distance to the camera (from closest to farthest):

1. **Top of the board** — smallest distance to the camera, because
   the board is elevated relative to the table by its thickness.
2. **Table/background** — greater distance, around the board.
3. **Pixels in the interior of the cavities** — distance approximately
   equal to that of the table, because the camera observes the table
   plane through the hole of the cavity.

This ordering justifies all the segmentation decisions that follow:
the board is what is above the table; the cavities are holes in the
board that re-expose the depth of the table.

---

## 4. RGB-D acquisition in Isaac Sim

The capture module is implemented in
`scripts/capture_cavity_detection.py` and follows the pattern
validated in Phase 1:

- Creation of a *render product* over the configured camera.
- Attachment of the `rgb` and `distance_to_image_plane` annotators.
- Execution of an asynchronous simulation step with
  `await rep.orchestrator.step_async(rt_subframes=RT_SUBFRAMES)`.
- Defensive reading via `get_data()`, with normalization of the
  *ndarray*-or-dictionary format across Replicator versions.

By default, the camera **is not moved** by the *script* (variable
`SET_CAMERA_POSE = False`): the authoritative pose in the USD *stage*
is treated as canonical. The *script* limits itself to reading the
world position of the camera via `get_camera_world_pose()` and using
it in the inverse projection to world coordinates. This decision
avoids inadvertent overlay of the Phase 1 camera pose onto the
visually verified configuration for Phase 2.

---

## 5. Automatic estimation of the table/background depth

Before any segmentation of the board, the dominant depth of the
background plane (table) is estimated through the histogram of the
complete depth image in the configurable interval
`[SURFACE_DEPTH_MIN, SURFACE_DEPTH_MAX]`. The dominant mode is
assumed to be the depth of the table, given that in a *top-down*
scene with a board small relative to the field of view, the table
occupies the largest fraction of valid *pixels*.

The depth of the table serves as a **negative reference**: the board
will be systematically closer to the camera than this value.

---

## 6. Automatic detection of the board

Board detection is executed in four deterministic steps:

1. **Candidate mask**: *pixels* whose depth is less than
   `table_depth - BOARD_ABOVE_TABLE_MARGIN` (the default margin is
   5 mm, and must be lower than the physical thickness of the
   board).
2. **Connected components** over that mask, with area filter
   `[BOARD_MIN_AREA_PX, BOARD_MAX_AREA_PX]`.
3. **Rectangularity filter**: ratio
   `area / bounding_box_area >= BOARD_RECTANGULARITY_MIN`. A fully
   rectangular board approaches 1; the defensive threshold is 0.70,
   sufficient to tolerate edges with some noise without accepting
   manifestly irregular components.
4. **Selection of the dominant candidate** among those that pass the
   filters (largest area).

The resulting mask (`board_mask`) represents only the upper face of
the board; it has **holes with the shape of the cavities**. For the
subsequent detection of the cavities, a **filled** version of that
mask is constructed — `board_region_mask` — through `BOARD_FILL_MODE`:

- `"contour"` (preferred): the largest external contour is drawn
  filled with `cv2.drawContours(..., thickness=cv2.FILLED)`;
- `"bbox"` (robust alternative): filling of the bounding box of the
  board.

`board_region_mask` is the **search domain** for the cavities: it is
geometrically impossible for a cavity to exist outside this region.

---

## 7. Estimation of the depth of the top of the board

The depth of the upper face of the board is estimated as the mode of
the depth histogram restricted to the *pixels* of `board_mask` (i.e.,
only the visible upper face, excluding the cavities). This
restriction is decisive: feeding the histogram with the full image
would lock onto the table plane, not the top of the board.

The obtained value is `board_surface_z`. When the fraction of the
dominant *bin* is below a warning threshold, the *script* emits a
message indicating a potentially noisy estimate.

---

## 8. Cavity segmentation

A cavity is, by construction, a region whose depth lies **below** the
face of the board but within a plausible window. The rule is:

```
board_surface_z + CAVITY_DEPTH_MARGIN  <  depth  <  board_surface_z + MAX_CAVITY_DEPTH
```

and is applied **only inside `board_region_mask`** (boolean AND
operation). The window simultaneously eliminates noise from the board
surface (lower bound) and excessively deep holes or noise originating
from other surfaces (upper bound).

After thresholding, a soft morphological operation (*open + close*)
is applied to remove small *speckles* without destroying thin
contours.

---

## 9. Identification and ordering of cavities

On the resulting binary mask, connected-components analysis is
executed, with area filter `[CC_MIN_AREA_PX, CC_MAX_AREA_PX]`.

The final ordering of the cavities is deterministic and documented,
so that `cavity_00`, `cavity_01`, ... maintain correspondence between
executions provided that the camera does not move:

1. *Bin* of the centroid y-coordinate in rows of `ROW_BIN_PX`
   *pixels* (tolerance for nominally aligned cavities);
2. Ordering by `(row_bin, centroid_x)` — line by line, from top to
   bottom, and within each line from left to right.

The identifiers `cavity_NN` are therefore **deterministic spatial
identifiers**, **never** semantic labels.

---

## 10. Generation of point clouds and footprints per cavity

For each accepted cavity, a local geometric representation is
generated with the same convention as in Phase 1, with the only
difference being the sign in Z:

- **X and Y**: centred on the world centroid of the cavity, in
  metres, with real-world scale preserved.
- **Z**: **depth below the top of the board**, computed as
  `depth[pixel] - board_surface_z`; it is, by construction, positive
  (deeper = greater value).
- **Fixed sampling**: each cloud contains exactly `N_POINTS = 2048`
  points. When the mask contains fewer *pixels* than `N_POINTS`,
  sampling with replacement is performed. This replication is
  recorded in the metadata.
- **2D footprint** *top-down*: 256 × 256 *pixels* at 0.5 mm/*pixel*.

The camera intrinsics are evaluated at the depth of the top of the
board, ensuring metric coherence for the metres-per-pixel computation
at the depth effectively observed.

---

## 11. Saved outputs

Each execution produces, in `data/cavities_detected/`, a set of
global outputs and one subfolder per detected cavity.

**Global outputs:**

| File | Content |
|---|---|
| `rgb.png` | Captured colour image. |
| `depth_vis.png` | Coloured visualization of the depth image. |
| `board_mask.png` | Upper face of the board (with the cavity holes). |
| `board_region_mask.png` | Filled board (cavities included — search domain). |
| `board_debug.png` | RGB overlay with board tinted, filled contour, bounding box and centroid. |
| `board_roi_auto_debug.png` | Diagnostic of the board detection (also written in case of failure). |
| `raw_cavity_mask.png` | Binary mask after depth threshold restricted to `board_region_mask`. |
| `cavities_debug.png` | RGB overlay with each detected cavity tinted and numbered. |
| `cavities_summary.json` | Global metadata (board, parameters, list of cavities, rejected components). |
| `run_log.txt` | Copy of the console output, overwritten on each execution. |

**Outputs per cavity**, in `cavity_NN/`:

| File | Content |
|---|---|
| `cavity_mask.png` | Binary mask of the cavity. |
| `cavity_debug.png` | Overlay of the mask over the RGB image. |
| `cavity_footprint.png` | 2D *top-down* footprint. |
| `cavity_pointcloud.npy` | 3D point cloud in metres, *shape* `(2048, 3)`. |
| `cavity_metadata.json` | Metadata of the individual cavity. |

The write policy preserves the rule adopted in Phase 1: files from
previous executions are removed at the beginning; in case of failure,
**no** *placeholders* are produced; the `cavities_summary.json` is
always written, with `success=False` and an error message when
applicable.

---

## 12. Validation procedure

`scripts/validate_cavity_captures.py` runs outside Isaac Sim in
conventional Python. For each subfolder `cavity_NN/`, it verifies:

1. presence of files (`cavity_metadata.json`,
   `cavity_pointcloud.npy`, `cavity_footprint.png`,
   `cavity_debug.png`, `cavity_mask.png`);
2. cloud structure (dimension 2, second dimension 3, ≥ 100 points);
3. numerical validity (no NaN, no infinities);
4. geometric bounds: positive X and Y spans, `Z maximum > 0`,
   non-negative Z span;
5. *footprint* readable and non-empty;
6. coherence with the metadata fields (`cavity_id`,
   `centroid_world_m`, `xy_span_m`, `z_depth_range_m`).

It also verifies, at the global level, the presence of the global
outputs listed above (including `cavities_summary.json` and
`run_log.txt`).

The following are produced:

- `data/cavities_detected/validation_summary.json`
- `data/cavities_detected/validation_summary.csv`
- `data/cavities_detected/footprints_grid.png` — grid with the
  footprints of all the cavities, labelled by `cavity_NN`.

---

## 13. Summary of validation results

**4 cavities** were detected and successfully validated, all passing
all validation criteria. Each point cloud contains 2048 points; no
NaN or infinities; non-empty *footprints*. The following table
summarises the geometric metrics extracted from
`data/cavities_detected/validation_summary.csv`.

| Cavity     | Area (px) | X span (mm) | Y span (mm) | Z span (mm) | Points |
|------------|-----------|-------------|-------------|-------------|--------|
| cavity_00  | 114       | 10.67       | 10.77       | 20.0        | 2048   |
| cavity_01  | 897       | 19.57       | 31.48       | 16.1        | 2048   |
| cavity_02  | 383       | 18.68       | 17.40       | 17.3        | 2048   |
| cavity_03  | 506       | 19.57       | 17.40       | 14.0        | 2048   |

Manual visual inspection of the footprints grid confirmed that
`cavity_00` corresponds to the star cavity and not to noise. The
remaining cavities have areas and spans coherent with nominal
rectangular and square/circular pieces.

---

## 14. Problems encountered and corrections

This section documents the technical problems effectively observed
during the development of Phase 2 and the corrections applied.

1. **Improper override of the camera pose**.
   The first version of the *script* moved the camera to the pose
   used in Phase 1 (pieces), even though the camera was correctly
   positioned over the board in the USD *stage*.
   *Correction*: introduction of `SET_CAMERA_POSE = False` by
   default and introduction of `get_camera_world_pose()` so that the
   inverse projection uses the effective pose of the *prim*.
   Addition of a warning when the pose constants are compatible with
   the old Phase 1 pose.

2. **Wrong estimation of the board surface**.
   The first version estimated the dominant depth from the complete
   image. As the table occupies most of the field of view, the
   histogram peak corresponded to the table and not to the top of
   the board, invalidating the depth window used in cavity
   segmentation.
   *Correction*: prior automatic detection of the board and
   rewriting of `estimate_board_surface_depth` to accept
   `board_mask` as a histogram restriction, ensuring that only
   *pixels* of the top of the board contribute to the estimate.

3. **Dependence on a manually configured ROI**.
   The previous version depended on a ROI adjusted by `BOARD_ROI_*`
   constants. This approach is fragile under camera or scene
   changes.
   *Correction*: introduction of the automatic board detection
   *pipeline* described in section 6. The `BOARD_ROI_*` constants
   are preserved as an alternative path only when
   `AUTO_DETECT_BOARD = False`.

4. **Cavities missed by area filter**.
   In one execution, 4 connected components were identified but
   only 3 passed the area filter. The rejected component had an
   area of 114 *pixels* (the star cavity) and was rejected by the
   threshold `CC_MIN_AREA_PX = 200`.
   *Correction*: reduction to `CC_MIN_AREA_PX = 80`, with an
   explicit comment justifying the value — small enough to preserve
   small cavities like the star, and still above the typical depth
   *speckle* threshold. Detailed diagnostics (full list of
   components with reason for rejection) were added to the console
   and to `cavities_summary.json` to make this kind of future
   adjustment simpler and more auditable.

5. **Need for an execution log for reproducibility**.
   Reproducing problems depended on manually copying the console
   output of the *Script Editor*.
   *Correction*: addition of `setup_run_logging()` with a class
   `_TeeStream` that duplicates `sys.stdout`/`stderr` to
   `data/cavities_detected/run_log.txt`, overwritten on each
   execution. The solution is idempotent across consecutive
   executions in the same *Script Editor* process (it does not
   stack *wrappers*).

6. **Ambiguity in the outputs in case of failure**.
   As in Phase 1, situations were detected in which zero
   *placeholders* were written after failures, misleading the
   subsequent analysis.
   *Correction*: application of the same policy as in Phase 1 —
   only artifacts effectively produced are written, prior *cleanup*
   removing files and `cavity_*` subfolders from the previous
   execution, and `cavities_summary.json` always written with
   `success=False` and an error message if applicable.

---

## 15. Limitations of the current approach

1. **Geometric coverage of the cavities is partial**.
   The point cloud is constructed from a single *top-down* view;
   lateral walls and the bottom of the cavities are only partially
   observable. The resulting representation is essentially a
   *2.5D heightmap* of the opening.

2. **Very small cavities have low effective density**.
   `cavity_00` was reconstructed from 114 *pixels* resampled with
   replacement up to 2048 points. The cloud is usable for
   methodologies based on footprint, IoU or *Chamfer*, but **must
   not** be used in a criterion based on local point density.

3. **Sensitivity to depth parameters**.
   `BOARD_ABOVE_TABLE_MARGIN`, `CAVITY_DEPTH_MARGIN`,
   `MAX_CAVITY_DEPTH`, `CC_MIN/MAX_AREA_PX` and
   `BOARD_RECTANGULARITY_MIN` continue to depend on the geometry
   and on the chosen resolution; substantially different scenes
   will require retuning.

4. **Geometric model assumes elevated board**.
   The premise "board closer to the camera than the table" is
   structural. Scenarios with embedded board, at the level of the
   table, or observed under oblique angles will require an
   alternative segmentation scheme.

5. **`board_region_mask` by external contour**.
   The `"contour"` version uses the largest external contour of the
   board. For boards with non-simply-connected geometry (e.g. large
   internal opening), it may not correspond to intuition. The
   `"bbox"` mode is a permissive alternative but may include noise
   around the edges.

6. **Identifiers `cavity_NN` are positional, not semantic**. The
   ordering is deterministic under a fixed camera, but reorders if
   the camera or the board are moved. For any future work that
   depends on stable identity of the cavities, this limitation has
   to be explicitly addressed (e.g., geometric correspondence
   instead of index).

---

## 16. Relevance for the thesis objective

As in Phase 1, this phase **does not** correspond to the learned
approach intended by the thesis. Its role is to prepare the ground:

- Provide geometric representatives of the cavities (footprint,
  point cloud with real-world scale preserved, mask), in parity
  with the representatives of the pieces produced in Phase 1.
- Allow, in the next phase, the construction of a **deterministic
  baseline** of piece-cavity matching — for example, comparison of
  footprints via IoU or Chamfer distance under candidate rotations.
- Establish a *de facto* geometrically annotated data source (and
  not by human labels) for later confrontation with learned
  methods.

Important points to keep explicit in the report:

- the phase **does not** classify cavities;
- the identifiers `cavity_NN` are **not** semantic labels;
- real-world scale is preserved, a necessary condition for any
  subsequent reasoning about insertion;
- the possibility of *multi-view* remains as a future extension to
  enrich the 3D representation of the cavities, in particular of
  the lateral walls.

---

## 17. Figures to include later in LaTeX

See `docs/thesis_notes/figures_index.md` for the consolidated table
of candidate figures (Phases 1 and 2) with identifiers, paths,
proposed names for LaTeX and captions in European Portuguese. The
most relevant figures of this phase are:

- `data/cavities_detected/board_debug.png` — illustration of the
  automatic detection of the board;
- `data/cavities_detected/board_mask.png` and
  `board_region_mask.png` — pair illustrating the difference between
  detected surface and filled domain;
- `data/cavities_detected/raw_cavity_mask.png` — result of the
  segmentation by restricted depth;
- `data/cavities_detected/cavities_debug.png` — global view of the
  detected cavities with identifiers;
- `data/cavities_detected/footprints_grid.png` — footprints of the
  cavities in a labelled grid.

---

## 18. Nominal CAD dimensions of the board and the cavities

This section records the final CAD dimensions of the board and the
cavities used from this version onwards. The values are also stored
in `data/expected_cad_dimensions.json`, which serves as the single
reference for scale auditing. **These values are for
validation/reporting only — they are not consumed by the detection
algorithm.**

| Cavity      | Nominal XY (mm)            | Depth (mm) |
|-------------|----------------------------|------------|
| square      | 51 × 51                    | 75         |
| rectangular | 51 × 76                    | 75         |
| triangular  | base 51, geom. height 51   | 75         |
| circular    | diameter 51                | 75         |

**Board**:
- thickness/height: **75 mm** (= nominal depth of the cavities,
  assuming through cavities);
- external X / Y dimensions: still **not recorded** (`null` fields
  in `expected_cad_dimensions.json`); they should be obtained in
  Fusion and added to that file.

**Nominal clearance** between piece and cavity:
- total clearance: **1 mm**;
- clearance per side: **0.5 mm**.

Implications for the detection *script* parameters
(`capture_cavity_detection.py`):

- `BOARD_ABOVE_TABLE_MARGIN = 5 mm` is compatible with a 75 mm
  board, but the effective lower limit is the margin; values above
  ~70 mm would be incompatible with the thickness.
- `MAX_CAVITY_DEPTH = 30 mm` is smaller than the nominal cavity
  depth (75 mm). If the cavities are through and the sensor
  observes them down to the bottom, this limit will truncate the
  observation. **Reassess this parameter** as soon as the next
  execution is done; raising it to 80 mm is defensible.
- `CAVITY_DEPTH_MARGIN = 3 mm` is geometrically safe relative to
  the 1 mm clearance.

The star remains as a reserved *stress* case, recorded in
`data/expected_cad_dimensions.json` under
`optional_stress_test_shapes` — when it returns, it will likely
need a dedicated cavity not present in the current bench.

---

## 19. Pending verification — vertical intrinsics bias

The piece-capture *script*
(`scripts/capture_piece_detection.py`) revealed a systematic bias
of approximately −7 to −8 % in the dimensions measured in Y,
attributed to an incorrect formula for computing the vertical
*focal* in *pixels*:

```
fov_v          = fov_h × (IMG_H / IMG_W)        # linear scaling, wrong for large FOVs
tan_half_fov_y = tan(fov_v / 2)
fy_px          = (IMG_H / 2) / tan_half_fov_y
```

The appropriate correction for square *pixels* is:

```
tan_half_fov_y = tan_half_fov_x × (IMG_H / IMG_W)
fy_px          = (IMG_H / 2) / tan_half_fov_y      # algebraically equal to fx_px
```

(Full detail in doc 01 — section 18.10.)

`scripts/capture_cavity_detection.py` shares the constants
`FOCAL_MM`, `APERTURE_MM`, `IMAGE_WIDTH`, `IMAGE_HEIGHT` and the
function `compute_intrinsics(...)`. **Verify** whether the formula
of the vertical *focal* is the same and, if so, correct and
recapture the cavities before any final scale audit. The cavity XY
spans reported in section 13 (previous validation, with the camera
in a different scenario) are therefore subject to the same
reanalysis — they should be treated as intermediate diagnostic, not
as an absolute reference.

---

## 20. Cavity-scale audit and CAD correction episode

### 20.1 Objective of the phase

The objective of this phase is to audit the absolute scale of the
cavity representations produced by the *pipeline*, immediately
before running Baseline 1, and to reconcile any deviation observed
between CAD and measured dimensions. The phase covers the
detection of a dimensional anomaly on `cavity_03`, the systematic
elimination of perception-side hypotheses, the discovery of an
upstream CAD modelling error, the correction of that error in the
source CAD, and the revalidation of the resulting cavities.

This phase **does not** alter the geometric model of the
*pipeline*, **does not** introduce new representations, and
**does not** establish any piece-cavity correspondence. It is a
scale-auditing phase whose output is a corrected set of cavity
representations and an explicit numerical CAD-vs-measured
comparison.

### 20.2 Experimental context

The audit was carried out with the cavity perception stage of the
*pipeline* immediately preceding Baseline 1. The execution
environment is NVIDIA Isaac Sim 5.1, container + WebRTC client.

The camera pose is the manual *stage* pose: `SET_CAMERA_POSE =
False`; the *script* reads the pose via `get_camera_world_pose()`
and uses it directly in the inverse projection. This is consistent
with the convention adopted in section 4.

The cavity detection mode used in this phase is
`CAVITY_DETECTION_MODE = "opening_from_board_region"`. In this mode,
the cavity opening is derived as the boolean difference between
`board_region_mask` and `board_surface_mask`, placing the opening
on the board top plane (Z = 0) by construction. The opening
footprint is therefore the primary geometric representation
consumed by Baseline 1; the auxiliary depth point cloud is
secondary in this mode.

### 20.3 Discovery of the dimensional anomaly

After a routine execution of `capture_cavity_detection.py`,
`cavity_03` was measured at approximately 60.84 × 60.84 mm in the
opening footprint plane. The footprint was perfectly isotropic in
X and Y at this measurement.

The CAD nominal value for the circular cavity in the experimental
set is Ø51 mm. The measured value therefore deviates from the
nominal by ~+19.3 %, an excursion that is large enough to be
incompatible with a benign quantization or rasterization artifact.
The remaining three cavities (`cavity_00`, `cavity_01`,
`cavity_02`) had measured spans within a few percent of their
nominal values, which made the anomaly even more puzzling: the
deviation was not a global scale error.

The size of the deviation (~+19.3 %) and the perfect isotropy of
the inflation across X and Y were the two facts that triggered the
audit. A scale error of this magnitude on a single cavity, with
the others looking nominally correct, ruled out any naive global
explanation and demanded a full investigation.

### 20.4 Initially considered hypotheses

Six families of hypotheses were considered at the outset, all of
them locating the defect on the perception side:

1. Camera intrinsics error (incorrect focal length, principal
   point, or aspect-ratio bug similar to the one previously fixed
   in section 19).
2. Per-pixel back-projection scaling error (incorrect application
   of the *pinhole* formula at the pixel level for some cavities
   only).
3. Metric reconstruction error (e.g. depth unit confusion,
   millimetres vs metres in part of the chain).
4. Opening segmentation spill / morphology artifact /
   connected-component merge of `cavity_03` with neighbouring
   pixels.
5. Cavity geometry extraction error (e.g. `board_region_mask` or
   `board_surface_mask` leaking pixels into the opening
   computation).
6. Anisotropic projection distortion that would inflate one cavity
   more than the others.

All six were considered plausible *a priori* because the previous
section (19) had already identified one intrinsics-side bug, and
because the perception *pipeline* is the part of the system most
exposed to silent geometric errors.

### 20.5 Root-cause investigation

The hypotheses listed in 20.4 were ruled out one by one through
the following evidence chain:

- **Anisotropic projection distortion ruled out.** The inflation
  on `cavity_03` was perfectly isotropic across X and Y
  (60.84 × 60.84 mm). An anisotropic projection bug would have
  produced unequal inflation in the two axes.
- **Global scaling and intrinsics error ruled out.** The
  metres-per-pixel scale was consistent across all four cavities
  (0.749 vs 0.751 mm/px). A global intrinsics bug would have
  produced the same fractional inflation in every cavity, not in
  one cavity alone.
- **Opening segmentation spill ruled out.** Direct visual
  inspection of the opening masks for all four cavities showed
  clean masks, no speckles, no contamination, no merges with
  neighbouring regions. A morphology artifact or
  connected-component merge would have left visible traces in the
  mask.
- **USD scene-graph scaling ruled out.** A read-only USD scene
  inspector script, `scripts/inspect_cavity_scene_scale.py`, was
  written specifically for this audit. It traverses every prim
  whose name matches `cavity`, `circle`, `cylinder`, `board` or
  `hole`, reports the local `xformOp:scale`, the parent-chain
  scales, and the world-space bounding box. The traversal
  reported zero non-unity scales anywhere in the stage. A USD
  scaling factor on the cavity geometry would have produced the
  observed inflation, but no such factor exists in the scene.

At the end of this chain, all perception-side, projection-side and
scene-graph-side hypotheses had been eliminated. The only
remaining possibility was that the geometry being captured was
itself not what it was assumed to be: that the CAD model of the
board did not actually have a Ø51 mm circular cavity.

### 20.6 Fusion 360 verification

The board model was opened in Fusion 360 and the circular cavity
sketch was inspected directly. The sketch was modelled with
diameter 62.00 mm (radius 31.00 mm), instead of the intended
51 mm. The inflation observed in the perception pipeline
(60.84 mm vs nominal 51 mm) is therefore explained by the CAD
model itself: the *pipeline* was correctly measuring a cavity that
was, in the source CAD, oversized.

The error is a CAD modelling mistake on the board, predating any
export to USD or capture in Isaac Sim. It is not a bug in the
perception pipeline, not a bug in the USD scene, and not a bug in
the export. The discrepancy of ~10 mm between the diameter modelled
in CAD (62 mm) and the measured value (~60.84 mm) is consistent
with the residual contraction caused by the depth-threshold
segmentation; this residual is discussed in section 20.13.

### 20.7 CAD model correction

The board model was corrected in Fusion 360 by setting the
circular cavity diameter to 51 mm, matching the intended value of
the experimental set. The corrected board was re-exported to USD
and the resulting USD file was updated in the Isaac Sim scene.

This is a single-source-of-truth correction: the CAD is the source
of geometric truth for the board, and the *pipeline* downstream
consumes the USD export of that CAD. Correcting the CAD propagates
the fix to every subsequent capture without any change in the
*pipeline* itself.

### 20.8 Recapture and revalidation

After the CAD fix and re-export, a recapture was performed with
`capture_cavity_detection.py`. A small parallel issue surfaced
during this step and is documented here because it affected the
recapture protocol, even though it is unrelated to the CAD error.

After the re-export, the relative distance between the board and
the camera shifted slightly. The previously hardcoded depth
window `[0.10, 0.50]` m, which was used by both
`estimate_table_or_background_depth` and
`estimate_board_surface_depth`, no longer matched the observed
valid depth range, which had moved to approximately
`[0.525, 1.000]` m. With the configured window now empty of valid
pixels, the depth estimators returned no usable estimate.

The estimators were extended with an adaptive valid-depth
fallback. When the configured window is empty, a percentile window
(p05..p70 of the source pool) is used instead, with a minimum band
width of 4 mm (`MIN_ADAPTIVE_DEPTH_BAND_M = 0.004`) to avoid
degeneracy when the source pool collapses to a single depth. The
configured window remains the default; the fallback only triggers
when the configured window is empty. This is a small robustness
improvement to the depth estimators, not a change to the geometric
model or to the produced representations.

### 20.9 Final validation results

The recapture, after the CAD fix and the depth-estimator
robustness improvement, yields:

- `capture_cavity_detection.py`: `success=True`, `cavities=4`.
- Final XY spans (millimetres):
  - cavity_00 = 50.50 × 73.90
  - cavity_01 = 48.03 × 49.27
  - cavity_02 = 50.50 × 49.27
  - cavity_03 = 49.27 × 50.50
- Deviations vs CAD (signed, percent of nominal); shape attribution
  inferred from the spans only and explicitly noted as such, not
  read from any stored mapping (cf. section 9 — `cavity_NN`
  identifiers are positional, not semantic):
  - cavity_00 vs rectangle 51 × 76 → −1.0 % / −2.8 %
  - cavity_01 vs square 51 × 51 → −5.8 % / −3.4 %
  - cavity_02 vs square 51 × 51 → −1.0 % / −3.4 %
  - cavity_03 vs circle Ø51 → −3.4 % / −1.0 %
- All deviations are within ~6 % of the nominal value; the
  previous +19.3 % anomaly on `cavity_03` is resolved.

`scripts/validate_cavity_captures.py` reports 4/4 cavities with
`overall_pass = True`. Structural checks all pass: `files_ok`,
`pc_shape`, `pc_no_nan`, `pc_no_inf`, `pc_xy_ok`, `pc_z_ok`,
`footprint`, `metadata`. The artifacts produced by the validation
are `data/cavities_detected/validation_summary.json`,
`data/cavities_detected/validation_summary.csv`,
`data/cavities_detected/footprints_grid.png`,
`data/cavities_detected/cavity_diagnostic_report.png` and
`data/cavities_detected/cavity_diagnostic_report.md`.

### 20.10 Technical interpretation of the results

The audit and correction episode produces several technical
observations that are relevant for the report and for any future
reader of the *pipeline*.

- **Why scale auditing is necessary.** Baseline 1 operates on
  real-world units; a silent scale error on either the piece side
  or the cavity side corrupts all downstream matching scores and
  any future comparison with learned methods. Without an explicit
  CAD-vs-measured comparison performed before Baseline 1 is run,
  the matching results carry an undocumented systematic offset
  that is impossible to deconvolve from the algorithmic
  performance. Scale auditing is therefore a necessary
  precondition for any quantitative claim about Baseline 1.
- **Why visual inspection alone was insufficient.** The inflated
  circular opening looked plausible at first glance. The visual
  appearance of an isotropic 60.84 mm circular footprint is not
  meaningfully distinguishable from a 51 mm circular footprint
  without a numerical reference at hand. Only the deterministic
  geometric baseline, combined with a numerical CAD-vs-measured
  comparison, surfaced the anomaly. This is an argument for
  treating the deterministic baseline not just as a comparison
  target for future learned methods, but as a sanity-check tool
  for the perception *pipeline* itself.
- **How CAD errors can masquerade as perception errors.** When
  the perception *pipeline* preserves real-world scale faithfully
  (as is the case here by design), any upstream geometric error
  in the CAD model propagates linearly into the measured
  representations and is indistinguishable from a perception bug
  without a scene-level scale audit. In this episode, six
  perception-side hypotheses were considered and ruled out before
  the actual upstream cause was identified. This is the expected
  behaviour and is, in itself, a positive property of the
  *pipeline*: the *pipeline* did not absorb the CAD error
  silently; it propagated it faithfully.
- **Why cross-validation between CAD and perception outputs is
  critical.** The CAD-vs-measured comparison turns a single point
  of failure (the trust in either the CAD or the *pipeline*) into
  a redundancy check. Numerical agreement across all four
  cavities, after the fix, is the only safe basis for trusting
  the geometric baseline. Any single-cavity check would have been
  insufficient: the original anomaly was localized to
  `cavity_03`, and a check that did not include all four cavities
  (or that was performed only on the most "obvious" pieces)
  would have missed the inconsistency entirely.
- **How the deterministic geometric baseline helped expose the
  issue.** The matching scores produced by Baseline 1 with the
  inflated `cavity_03` were poor for the circular piece, which is
  what triggered the scale audit. Without the baseline acting as
  a numerical sanity check, the CAD error might have remained
  hidden until robot deployment, where it would have manifested
  as a physical insertion failure with a much higher cost of
  diagnosis.

### 20.11 Notes on opening-footprint representation

The `CAVITY_DETECTION_MODE = "opening_from_board_region"` mode
produces the opening footprint as a boolean difference between
`board_region_mask` and `board_surface_mask`. By construction, the
opening lies on the board top plane (Z = 0). This is the primary
representation consumed by Baseline 1.

A consequence of this construction is that, for cavities whose
interior is not robustly observed by the depth annotator, the
auxiliary depth point cloud may degenerate. In the current
recapture, `cavity_02` produces a degenerate auxiliary depth point
cloud (≤ 1 unique point, Z ≈ 0). This happens because no pixels in
that region are deeper than `board_surface_z +
BOARD_SURFACE_DEPTH_TOLERANCE_M` (1 mm tolerance), so the "below
the surface" set used to build the auxiliary cloud is empty or
near-empty.

This degeneracy **does not** invalidate `cavity_02` for matching:
the primary representation used by Baseline 1 is the opening
footprint (Z = 0 by construction); the depth point cloud is
auxiliary and is not consumed by Baseline 1. The opening footprint
of `cavity_02` is well-formed and structurally validated.

This observation is a concrete reason to keep the auxiliary depth
point cloud and the opening footprint as separate representations
in the artifact set: a failure mode of the auxiliary representation
must not block the use of the primary one.

### 20.12 Problems encountered and corrections

Specific to this audit episode:

1. **Hardcoded depth window invalidated by board re-export.**
   After the CAD correction and re-export, the board/camera
   distance shifted and the previously valid window
   `[0.10, 0.50]` m no longer enclosed the depth distribution
   (which moved to ~`[0.525, 1.000]` m). The depth estimators
   returned no usable estimate, blocking the recapture.
   *Correction*: addition of an adaptive valid-depth fallback in
   `estimate_table_or_background_depth` and
   `estimate_board_surface_depth`. When the configured window is
   empty, a percentile window (p05..p70 of the source pool) is
   used, with a minimum band width of 4 mm
   (`MIN_ADAPTIVE_DEPTH_BAND_M = 0.004`). The configured window
   remains the default; the fallback only triggers when the
   configured window is empty. This is a small robustness
   improvement to the depth estimators, not a change to the
   geometric model.

2. **Six perception-side hypotheses considered before the actual
   upstream cause was identified.** The investigation cost was
   non-trivial: a dedicated scene inspector script
   (`scripts/inspect_cavity_scene_scale.py`) had to be written to
   rule out USD scene-graph scaling. This investigation cost is
   itself a finding — it is the price of having a *pipeline* that
   propagates upstream errors faithfully rather than absorbing
   them. The *script* remains in the codebase as a future audit
   tool.

3. **Auxiliary depth point cloud degenerate on `cavity_02`.** As
   discussed in 20.11, no pixels in the `cavity_02` region are
   deeper than `board_surface_z +
   BOARD_SURFACE_DEPTH_TOLERANCE_M`. The auxiliary cloud
   collapses to ≤ 1 unique point.
   *Correction*: none required at the *pipeline* level for
   Baseline 1, since Baseline 1 consumes the opening footprint and
   not the auxiliary cloud. Documented here as a known property
   of the current capture for `cavity_02`.

### 20.13 Current limitations

1. **Residual deviations trend negative (~6 % at most).** The
   measured spans are systematically smaller than the CAD
   nominals. This trend is consistent with under-segmentation of
   the opening contours by the depth threshold. A −5.8 %
   deviation on `cavity_01` corresponds to approximately 3 mm at
   a 51 mm nominal; this is non-negligible for tight matching
   thresholds.
2. **`BOARD_SURFACE_DEPTH_TOLERANCE_M = 1 mm` and absence of
   morphology.** The current parameters are sweep-validated and
   keep the masks faithful to the depth signal, but accept the
   slight contraction described above. Tightening the tolerance
   or applying morphological dilation would reduce the contraction
   but would also risk reintroducing speckle noise; the trade-off
   has not been re-tuned in this episode.
3. **Auxiliary depth point cloud may degenerate.** As described
   in 20.11 for `cavity_02`. Does not affect Baseline 1 but must
   be kept in mind for any future use of the auxiliary cloud
   (e.g. depth-based plausibility checks).
4. **Single-view geometry.** The audit was performed under the
   same single-view top-down assumption as the rest of Phase 2.
   It does not address the geometric coverage limitation
   discussed in section 15.

### 20.14 Relevance for Baseline 1

The audit and correction episode is a precondition for any
quantitative claim about Baseline 1 on the current experimental
set. After this episode:

- the cavity-side representations are within ~6 % of the CAD
  nominals, with the previous +19.3 % anomaly resolved;
- the residual deviations are documented and trend negative,
  which means Baseline 1 will systematically see slightly smaller
  cavity footprints than the CAD nominals (this should be kept
  explicit in any reading of the matching scores);
- the auxiliary depth representation of `cavity_02` is degenerate
  but does not block Baseline 1, which consumes the opening
  footprint;
- the CAD-vs-measured comparison is now part of the audit chain
  and can be re-run after any future scene change.

The episode also establishes, in practice, that the deterministic
geometric baseline functions as a numerical sanity check on the
perception *pipeline*, not only as a reference for future learned
methods. This dual role of Baseline 1 is consistent with the
positioning given in doc 03 — section 14.

### 20.15 Current pipeline status

After this episode, the cavity perception stage of the *pipeline*
is in the following state:

- the board CAD is corrected and re-exported to USD;
- `capture_cavity_detection.py` runs successfully with the
  corrected scene under the manual *stage* camera pose;
- the depth estimators
  (`estimate_table_or_background_depth`,
  `estimate_board_surface_depth`) include the adaptive
  valid-depth fallback;
- `CAVITY_DETECTION_MODE = "opening_from_board_region"` is the
  current detection mode;
- `validate_cavity_captures.py` reports 4/4 with
  `overall_pass = True`;
- the scale audit produced `cavity_diagnostic_report.md` and
  `cavity_diagnostic_report.png` as part of the validation
  outputs;
- `scripts/inspect_cavity_scene_scale.py` is available as a
  read-only audit tool for any future investigation of suspected
  USD-side scaling.

### 20.16 Notes for the author

Items that should be recorded manually, outside this document, in
addition to those in the closing notes of this file:

- A copy of the corrected Fusion 360 board file should be archived
  with a date stamp, so that the CAD state at the moment of the
  validated capture is recoverable.
- The pre-correction CAD (with the 62 mm circular cavity) should
  also be archived, so that the audit episode itself can be
  reproduced and shown in the report if needed.
- The output of `scripts/inspect_cavity_scene_scale.py` at the
  moment of the validated capture should be saved alongside the
  capture artifacts, so that the absence of USD-side scaling can
  be cited verbatim in the report.
- The numerical CAD-vs-measured comparison shown in 20.9 should
  be cross-checked against `expected_cad_dimensions.json` before
  any final report run, to ensure that the two sources of
  nominals are consistent.
- The investigation cost of the audit (six hypotheses ruled out
  before the upstream cause was identified) should be discussed
  in the report as evidence that the *pipeline* propagates
  upstream errors faithfully and does not absorb them silently.

---

## Notes for the author

Items that should be recorded manually, outside this document, and
that are not captured in the output files:

- External X / Y dimensions of the board in Fusion (currently
  `null` in `data/expected_cad_dimensions.json`).
- Empirically confirm whether the cavities are through; if not,
  record the actual depth in `expected_cad_dimensions.json` in the
  `depth_m` field per cavity (currently 75 mm = thickness of the
  board is assumed).
- Physical pose of the virtual camera in USD (translation and
  orientation) at the moment of the validated capture.
- Exact version of Isaac Sim and of the container.
- Justification of the number and arrangement of the cavities on
  the board.
- Decision about the strategy of cavity identity to use in later
  phases (positional index vs. geometric correspondence vs.
  other).
