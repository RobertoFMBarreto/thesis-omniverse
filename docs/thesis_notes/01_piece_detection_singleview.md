# 01 — Piece detection (single view)

> Implementation note for future conversion into a LaTeX section.
> Status: Phase 1 — deterministic perception, no learned component.
> Date: 2026-05-01.

---

## 1. Objective of the phase

The objective of this phase is to obtain a reliable geometric
representation of a piece visible in the scene, from a single RGB-D
capture in a simulated environment. The intent is to construct the
input artifacts that will later feed the deterministic baseline for
piece-cavity matching (*footprint matching*).

This phase **does not** classify shapes, **does not** infer insertion
affinities, and **does not** involve any learned model. It is
exclusively a step of geometric perception.

---

## 2. Experimental context

The complete system is inspired by children's shape-sorting toys
(*shape sorter*). The experimental bench contains:

- a board with geometric cavities;
- geometric pieces modelled in Fusion (rectangle, square, circle and
  star), used here as the initial set;
- a virtual RGB-D camera in NVIDIA Isaac Sim 5.1, executed inside a
  container and accessed through the WebRTC client.

Capture is orchestrated via the Isaac Sim *Script Editor*, using the
compatible asynchronous pattern and the `rgb` and
`distance_to_image_plane` annotators of the `omni.replicator.core`
module.

---

## 3. Input data and scene assumptions

Current assumptions for Phase 1:

- **Single view**: the camera is placed above the board, with
  approximately vertical orientation (*top-down*).
- **Single visible piece per capture**: the remaining pieces are
  manually hidden in Isaac Sim. This restriction is an experimental
  decision, not an intrinsic limitation of the detector — the
  algorithm detects multiple connected components, but the current
  pipeline selects only one.
- **Approximately planar support surface** within the field of view.
- **Real-world scale preserved**: depth in metres, without any global
  normalization.

The names of the capture folders (`rectangle/`, `square/`, `circle/`,
`star/`) are **experimental organization labels** and do not influence
any geometric decision of the *script*. The detector is unaware of the
nominal shape of the piece.

---

## 4. RGB-D acquisition in Isaac Sim

The capture module is implemented in `scripts/capture_piece_detection.py`
and follows the recommended pattern for the *Script Editor*:

- Creation of a *render product* over the configured camera
  (`/World/Camera`).
- Attachment of the `rgb` (colour image) and
  `distance_to_image_plane` (depth in metres, distance to image
  plane) annotators.
- Execution of an asynchronous simulation step via
  `rep.orchestrator.step_async(rt_subframes=RT_SUBFRAMES)`, with
  `RT_SUBFRAMES = 8` to stabilise the *render*.
- Reading of the data via `get_data()`, with defensive normalization
  between the *ndarray* and dictionary-with-`"data"`-key formats that
  different Replicator versions may return.

The camera pose is defined programmatically through the constants
`CAM_X`, `CAM_Y`, `CAM_Z`, `CAM_ROT_Z_DEG`. The function `setup_camera`
supports the USD transformation operations `xformOp:orient`,
`xformOp:rotateXYZ` and `xformOp:rotateZ`.

**Default resolution**: 640 × 480 *pixels*.
**Intrinsics**: focal length of 24 mm and horizontal aperture of
36 mm, defined as constants and matching the `UsdGeom.Camera`
attributes of the camera *prim*.

---

## 5. Estimation of the support surface

The depth of the support surface (table/board) is estimated
automatically from the histogram of the depth image, restricted to
the configurable interval `[SURFACE_DEPTH_MIN, SURFACE_DEPTH_MAX]`
(by default `[0.10, 0.50]` m).

The estimated value is the centre of the dominant *bin*, with a *bin*
width of 1 mm. The algorithm emits an explicit warning when the
fraction of *pixels* in the dominant *bin* is below 5 %, signalling a
noisy histogram or a poorly framed support surface.

This approach assumes that the support surface occupies a
non-negligible fraction of the field of view. If the piece fills
almost the entire field, the histogram peak may correspond to the
piece itself and not to the table.

---

## 6. Piece segmentation

Segmentation operates on the depth image, **not** on the RGB image.
The piece is treated as positive geometry above the support surface:
a *pixel* belongs to the piece if its depth is lower than the surface
depth subtracted by a configurable tolerance `SURFACE_TOLERANCE` (by
default 4 mm).

This convention respects world geometry: a smaller distance to the
camera means closer to the camera, that is, higher relative to the
table in a *top-down* view.

The choice of tolerance is a trade-off:

- values that are too small let through noise from the support
  surface itself;
- values that are too large eliminate thin pieces or low prisms.

---

## 7. Connected-component selection

After binary segmentation, connected-component analysis
(`cv2.connectedComponentsWithStats`) is executed with area filtering
(`CC_MIN_AREA_PX`, `CC_MAX_AREA_PX`).

Valid components are ordered deterministically by decreasing area,
with ties broken by ascending centroid x-coordinate. The selection of
the component of interest is controlled by three configurable modes:

- `largest` — selects the component with the largest area (rank 0).
  This is the default mode and was used in all validated captures.
- `closest_to_center` — selects the component whose centroid is
  closest to the centre of the image. Useful for capturing a specific
  piece by repositioning the camera, without resorting to shape
  classification.
- `manual_index` — selects the component at position
  `MANUAL_COMPONENT_INDEX` of the ordered list.

In any mode, the total number of valid detected components is
recorded in the metadata (`n_valid_components` and
`multiple_valid_components` fields), allowing later diagnosis of
scene ambiguity.

---

## 8. Point cloud generation

The point cloud is constructed by *backprojection* of the *pixels* of
the selected piece mask, using *pinhole* camera intrinsics (`mpp_x`,
`mpp_y` derived from the focal length, aperture and the distance to
the estimated surface `surface_z`).

Conventions:

- **X and Y axes**: centred on the world centroid of the piece,
  expressed in metres.
- **Z axis**: represents the height above the support surface,
  computed as `surface_z - depth[pixel]`. It is always non-negative.
- **Real-world scale preserved**: coordinates are in metres and are
  **not** normalized to a unit scale. This decision is fundamental
  for the future piece-cavity matching, in which absolute size is
  part of the relevant geometric information.
- **Fixed sampling**: each point cloud contains exactly
  `N_POINTS = 2048` points. When the mask contains fewer *pixels*
  than `N_POINTS`, sampling with replacement is performed (recorded
  in the metadata).

The intrinsics are evaluated at `surface_z` (and not at a nominal
camera height) so that the metres-per-pixel relation is correct at
the effective depth of the piece.

---

## 9. Footprint generation

The 2D footprint is the *top-down* projection of the point cloud onto
the XY plane, rendered on a square canvas of 256 *pixels* with a
resolution of 0.5 mm per *pixel*. The image is saved with a hot
colour map to facilitate visual inspection.

The footprint is the main artifact to be consumed by the
deterministic geometric baseline of the next phase: comparison by
overlap (IoU) or Chamfer between the piece footprint and the cavity
footprints, under different rotations.

---

## 10. Saved outputs

By default, each capture produces a subfolder inside
`data/pieces_detected/<CAPTURE_NAME>/` with the following files:

| File | Content |
|---|---|
| `rgb.png` | Captured colour image. |
| `depth_vis.png` | Coloured visualization of the depth image. |
| `raw_piece_mask.png` | Binary mask after the depth threshold. |
| `piece_mask.png` | Mask of the selected connected component. |
| `piece_debug.png` | Overlay of the mask, bounding box and centroid over the RGB image. |
| `piece_footprint.png` | 2D *top-down* footprint. |
| `piece_pointcloud.npy` | 3D point cloud in metres, *shape* `(N_POINTS, 3)`. |
| `piece_metadata.json` | Full capture metadata (parameters, intrinsics, metrics, list of valid components). |

The write policy guarantees that, in case of failure at an
intermediate stage, **no** placebo files are produced: invalid
artifacts or those from previous executions are removed at the
beginning and only artifacts effectively produced in the current
execution are written. The `piece_metadata.json` is always written,
with `success=False` and the error message when applicable.

---

## 11. Validation procedure

An independent *script* was implemented,
`scripts/validate_piece_captures.py`, which runs outside Isaac Sim in
conventional Python. For each expected capture subfolder
(`rectangle`, `square`, `circle`, `star`), the validation checks:

1. presence of essential files (`piece_metadata.json`,
   `piece_pointcloud.npy`, `piece_footprint.png`,
   `piece_debug.png`);
2. coherence of the metadata — in particular, `n_valid_components == 1`
   and `multiple_valid_components == false`;
3. point cloud structure: dimension 2, second dimension equal to 3,
   at least 100 points;
4. geometric bounds of the cloud: positive X and Y spans, non-negative
   Z span, absence of NaN and infinities;
5. *footprint* readable and non-empty.

The validation produces three artifacts:

- `data/pieces_detected/validation_summary.json`
- `data/pieces_detected/validation_summary.csv`
- `data/pieces_detected/footprints_grid.png` (2 × 2 grid with the
  *footprints* of the four pieces, labelled by folder).

---

## 12. Summary of validation results

All criteria passed for the four captured pieces. The table below
summarises the point cloud spans and the point counts, extracted
directly from `data/pieces_detected/validation_summary.json`.

| Piece     | X span (mm) | Y span (mm) | Z span (mm) | Points |
|-----------|-------------|-------------|-------------|--------|
| rectangle | 37.7        | 19.7        | 0.0         | 2048   |
| square    | 21.2        | 20.1        | 20.7        | 2048   |
| circle    | 21.2        | 19.7        | 0.0         | 2048   |
| star      | 19.9        | 17.7        | 26.3        | 2048   |

Z bounds (in metres), also extracted from validation:

| Piece     | Z minimum | Z maximum |
|-----------|-----------|-----------|
| rectangle | 0.03050   | 0.03050   |
| square    | 0.00983   | 0.03050   |
| circle    | 0.03050   | 0.03050   |
| star      | 0.00418   | 0.03050   |

The null Z span on the `rectangle` and `circle` pieces is consistent
with the hypothesis that the upper face of those pieces is strictly
planar and the only one visible in a *top-down* view. All visible
*pixels* project to the same depth value at the level of the
`float32` quantization of the annotator, resulting in constant Z. In
light of the inspection performed, this is not a defect of the
*pipeline*: it is a property of the observed geometry combined with
the precision of the *render*.

---

## 13. Problems encountered and corrections

This section documents the technical problems effectively observed
during the development of Phase 1 and the corrections applied, so
that the final report can present the development trajectory and not
only the final method.

1. **Non-existent initialization of the Replicator orchestrator**.
   The first version of the *script* invoked
   `rep.orchestrator.initialize_async()` before the
   `step_async(...)`. In the Isaac Sim 5.1 environment used, that
   method does not exist and the capture failed with `AttributeError`.
   *Correction*: removal of the call and adoption of the pattern
   already validated in the project's earlier *scripts*: create a
   *render product*, attach annotators, execute
   `await rep.orchestrator.step_async(rt_subframes=...)` and read with
   `get_data()`.

2. **Path resolution via `__file__` in the *Script Editor***.
   Initially, the output directory was derived from
   `Path(__file__).resolve().parent.parent`. When the *script* is
   pasted/executed in the Isaac Sim *Script Editor*, `__file__` may
   resolve to a temporary path of the form
   `/tmp/carb.../script_*.py`, causing the outputs to be written
   outside the repository.
   *Correction*: explicit definition of `PROJECT_ROOT`, with optional
   environment variable `SHAPE_INSERTION_PROJECT_ROOT` for override
   in other environments (e.g. development machine). The path became
   stable independently of the execution context.

3. **Multiple pieces visible simultaneously**.
   In initial captures, several pieces remained visible in the scene.
   The *pipeline* selected the largest-area connected component,
   without any guarantee about which piece was chosen.
   *Methodological decision for this phase*: manually hide the
   non-intended pieces in Isaac Sim and capture one piece at a time.
   Configurable selection modes were added (`largest`,
   `closest_to_center`, `manual_index`) to make the choice
   deterministic without resorting to shape classification.

4. **Annotator format compatibility**.
   In different Replicator versions, `get_data()` may return either
   directly an *ndarray* or a dictionary with key `"data"`. In the
   first executions, `TypeError` *crashes* were obtained from
   applying direct slicing to a dictionary.
   *Correction*: defensive normalization — if the return is a
   dictionary, convert to *ndarray* via
   `np.asarray(d["data"]).reshape(IMG_H, IMG_W, -1)` before use;
   single print of the type and *shape* for diagnosis.

5. **Spurious outputs after failure**.
   In an intermediate version, the `finally` block produced
   *placeholders* with zero matrices to avoid write errors. This left
   files with the appearance of a valid capture when, in reality, the
   capture had failed.
   *Correction*: removal of *placeholders*; only artifacts
   effectively produced by the current execution are written; the
   `piece_metadata.json` is always written, with `success=False` and
   error message when applicable; files from previous executions are
   removed at the beginning so that they cannot be confused with the
   current result.

6. **Null Z span on prismatic pieces**.
   The `rectangle` and `circle` pieces showed Z span exactly equal
   to zero. It was found to be a joint property of the observed
   geometry (strictly planar upper face) and the `float32`
   quantization of the depth annotator, and not a defect of the
   *pipeline*. The Z span is useful information but, at this phase,
   is not strictly necessary for the footprint-based geometric
   baseline.

7. **Real-scale verification**.
   In the first executions, the camera intrinsics function was fed
   with the nominal camera height instead of the effective surface
   depth. This introduced a systematic XY scale error proportional
   to the difference between the two depths.
   *Correction*: the estimated surface depth (`surface_z`) is now
   used to compute metres-per-pixel, ensuring metric coherence
   between the *pixel grid* and the world.

---

## 14. Limitations of the current approach

1. **Single view and dominant planar top**: from a single *top-down*
   capture, prismatic pieces with planar upper face produce point
   clouds with little or no variation in Z. The resulting
   representation essentially encodes the footprint and the height,
   but not the lateral shape of the piece.

2. **Selection restricted to one component**: the pipeline assumes
   one piece visible per capture. If multiple pieces are visible,
   only one component is selected according to the configured
   criterion, without any reasoning about piece identity.

3. **Sensitivity to thresholds**: surface estimation and segmentation
   depend on constants that must be tuned to the scene
   (`SURFACE_DEPTH_MIN/MAX`, `SURFACE_TOLERANCE`, `CC_MIN_AREA_PX`,
   `CC_MAX_AREA_PX`).

4. **Dependence on camera pose**: an approximately vertical camera
   over the surface is assumed. Significant deviations invalidate
   the interpretation `world_z = surface_z - depth` as height above
   the table.

5. **Hardcoded intrinsics**: the focal length and aperture are
   constants in the *script*; they must coincide with the attributes
   of the camera *prim* in USD. Discrepancies produce a systematic
   XY scale error.

6. **Partial geometric coverage**: lateral and bottom faces of the
   piece are not observable. The point cloud is, in practice, a
   *2.5D heightmap* of the visible upper face.

---

## 15. Relevance for the thesis objective

The central objective of the thesis is the learning of
perception-action relations based on geometry, with the case study of
piece insertion into cavities. The deterministic perception described
here is the initial step: it provides the geometric representatives
of the pieces that will subsequently be confronted with cavities to
infer compatibility, insertion rotation and approximate pose.

Positioning of this phase in the global plan:

- **Does not replace** the intended learned approach — it provides
  the input artifacts and establishes the geometric reference
  baseline.
- **Does not classify** shapes. The *pipeline* output is not a
  label such as "this is a square", but a reusable geometric
  representation (mask, footprint, point cloud, metadata).
- **Preserves real-world scale**, a necessary condition for any
  subsequent reasoning about insertion, in which the absolute size
  of the piece and the cavity is meaningful information.
- **The 2D footprint representation is adequate for the
  deterministic geometric baseline** of piece-cavity matching, for
  example via IoU or Chamfer distance under candidate rotations.

For richer 3D representations — necessary if one wishes to model the
piece across all faces — a future extension with *multi-view* capture
is foreseen, outside the scope of this phase.

---

## 16. Figures to include later in LaTeX

| Identifier | Current path | Suggested caption |
|---|---|---|
| `fig:rgb` | `data/pieces_detected/<piece>/rgb.png` | RGB image captured by the virtual camera. |
| `fig:depth_vis` | `data/pieces_detected/<piece>/depth_vis.png` | Coloured visualization of the depth image. |
| `fig:raw_mask` | `data/pieces_detected/<piece>/raw_piece_mask.png` | Binary mask resulting from the threshold over depth. |
| `fig:piece_debug` | `data/pieces_detected/<piece>/piece_debug.png` | Overlay of the selected mask, bounding box and centroid. |
| `fig:footprint` | `data/pieces_detected/<piece>/piece_footprint.png` | 2D *top-down* footprint of the piece. |
| `fig:footprints_grid` | `data/pieces_detected/footprints_grid.png` | Grid of the footprints of the four captured pieces. |

A composite figure of four panels (RGB, depth, *debug* mask and
*footprint*) per representative piece is suggested, plus the summary
grid of the four footprints. The metrics of Tables 12.1 and 12.2 may
enter as tables.

---

## 17. Nominal CAD dimensions of the pieces

This section records the final CAD dimensions of the experimental
set used from this version onwards. The values are canonical and are
also stored in `data/expected_cad_dimensions.json`, a file that
serves as the single reference for scale auditing. **These values
are for validation/reporting only — they are not consumed by the
matching algorithm.**

Main set (after replacement of the star by triangle, see doc 03 —
section 11):

| Piece       | Nominal XY (mm)            | Height/extrusion (mm) |
|-------------|----------------------------|-----------------------|
| square      | 50 × 50                    | 105                   |
| rectangle   | 50 × 75                    | 105                   |
| triangle    | base 50, geom. height 50   | 105                   |
| circle      | diameter 50                | 105                   |

The piece extrusion dimension (105 mm) is not used by Baseline 1,
which is purely based on the XY footprint (see doc 03 — section 12).
It is recorded here because it will be necessary for future phases
of 3D perception, *multi-view* and execution of robotic insertion
(in particular: 105 mm of piece vs. 75 mm of cavity depth implies a
protrusion of 30 mm above the top of the board).

The star remains as a concave *stress* case reserved for future
work, recorded in `data/expected_cad_dimensions.json` under
`optional_stress_test_shapes`.

---

## 18. Update — current state of the capture *pipeline*

This section replaces, for the purpose of reading the **current**
state, sections 4–14 above (which remain as a historical record of
the development). It documents the *pipeline* as it exists after the
set of changes described in section 13 plus the additional changes
introduced subsequently to solve problems of camera control, surface
estimation and metric projection.

### 18.1 Final shape set

The main set of pieces used from this version onwards is:

- `rectangle`
- `square`
- `circle`
- `triangle` (replaces the `star`)

The `star` was removed from the main set for being excessively
sensitive to segmentation and to absolute scale, as documented in
doc 03 — section 11. It remains recorded in
`data/expected_cad_dimensions.json`, under
`optional_stress_test_shapes`, as a concave *stress* case reserved
for future work.

### 18.2 Final CAD dimensions (recap)

| Piece       | Nominal XY (mm)            | Extrusion (mm) |
|-------------|----------------------------|----------------|
| rectangle   | 75 × 50                    | 105            |
| square      | 50 × 50                    | 105            |
| circle      | diameter 50                | 105            |
| triangle    | base 50, geom. height 50   | 105            |

For the board and cavities, see doc 02 — section 18; for nominal
clearance (1 mm total, 0.5 mm per side), see
`data/expected_cad_dimensions.json`.

### 18.3 Camera control via the *stage*

`scripts/capture_piece_detection.py` now adopts the same convention
as `scripts/capture_cavity_detection.py`:

- `SET_CAMERA_POSE = False` by default. The *script* **does not**
  move the camera: it uses the authoritative pose in the USD
  *stage*, which the user positions manually in Isaac Sim.
- The effective world pose of the camera is read via
  `get_camera_world_pose()`, printed at the start of the execution,
  and recorded in `piece_metadata.json` under `camera_pose` with
  `source = "stage"`.
- If `SET_CAMERA_POSE = True`, the constants `CAM_X/Y/Z` and
  `CAM_ROT_Z_DEG` are applied via `setup_camera()` and
  `source = "config_override"`. This path is kept only for
  deterministic reproduction of previous captures.

The effective pose is then passed as `cam_xy` to the
*back-projection* function, ensuring that the world XY coordinates
correspond to the pose actually in force (and not to configuration
constants that might not coincide with the *stage*).

### 18.4 Automatic estimation of the support surface by depth layers

Estimation by dominant depth mode failed in scenarios with several
planes in the field of view (for example, piece + local board + a
second table/wall in the background). The adopted solution is
estimation by **depth layers**:

1. Restriction of the analysis to a **ROI** centred on the piece
   capture field (`PIECE_ROI_ENABLED = True`,
   `PIECE_ROI_MODE = "center_fraction"`,
   `PIECE_ROI_FRACTION = 0.60`).
2. Collection of valid depth *pixels* (positive, finite) within that
   ROI.
3. Computation of **adaptive bounds** from the distribution itself
   (`AUTO_SURFACE_DEPTH_BOUNDS = True`):
   `lower = p01 − margin`, `upper = p99 + margin` with
   `SURFACE_DEPTH_MARGIN_M = 0.005` m. The static constants
   `SURFACE_DEPTH_MIN/MAX` no longer act as a hard filter in this
   mode (they remain in use only in the legacy `dominant_depth`
   mode).
4. Construction of a histogram with 1 mm *bin*
   (`SURFACE_HIST_BIN_M`).
5. Extraction of local maxima and merging of nearby peaks
   (`SURFACE_PEAK_MERGE_DISTANCE_M = 0.004` m).
6. Ordering of the peaks from the closest to the farthest.
7. Selection of the support peak:
   - small near peaks (fraction ≤
     `PIECE_MAX_PEAK_FRACTION = 0.08`) are skipped as probable
     "piece top";
   - the first peak with fraction ≥
     `SUPPORT_MIN_PEAK_FRACTION = 0.10` is accepted;
   - if no peak reaches that threshold, the peak with the largest
     fraction is used as fallback and this is recorded in
     `selected_support_reason`.
8. Safeguard: if the resulting raw mask covers more than 50 % of
   the segmentation ROI, a **reselection** is attempted with the
   immediately closer peak — recorded in
   `selected_support_reason` as `"... | RESELECTED (initial
   mask >50% of ROI)"`.

Diagnostics available in the console and in metadata:

- complete listing of the detected peaks (depth, count, fraction);
- selected peak, identified by `rank` and `reason`;
- explicit warning `[WARNING] selected support appears to be the
  farthest layer ...` when the chosen peak is the last one (a
  typical sign of background winning);
- warning `[WARNING] raw piece mask covers too much of ROI ...`;
- specific warning for the current configuration (`> 0.68 m`)
  indicating proximity to the background plane;
- image `depth_layers_debug.png` with the surface ROI (cyan),
  expanded segmentation ROI (yellow) and a text panel with the
  list of peaks, with the selected peak highlighted in green.

### 18.5 Piece segmentation

The rule remains geometrically correct:

```
piece_mask = (depth > DEPTH_MIN_VALID) AND (depth < surface_z − SURFACE_TOLERANCE)
```

with `SURFACE_TOLERANCE = 0.004` m. Pixels closer to the camera than
the surface (in a *top-down* observation) are classified as "above
the support".

If `RESTRICT_PIECE_MASK_TO_ROI = True`, the raw mask is also
restricted to the ROI **expanded** by `PIECE_MASK_ROI_EXPAND_PX =
20` *pixels*, ensuring that objects outside the capture field can
never enter the connected-components analysis.

### 18.6 *Point cloud* and *footprint* generation — per-pixel projection

The computation of world XY coordinates was changed to
**depth-dependent per-pixel projection**. The previous version used
`mpp` computed at `surface_z` for all *pixels*, which was acceptable
for thin pieces but systematically inflated the dimensions for tall
pieces (105 mm).

Current convention (canonical for the *pinhole* model):

```
world_x = cam_x + (u − cx_px) / fx_px × depth_px
world_y = cam_y − (v − cy_px) / fy_px × depth_px
world_z = surface_z − depth_px            (height above support)
```

where `fx_px` and `fy_px` are focal lengths expressed in *pixels*
(independent of depth), `cam_x`/`cam_y` are the actual world
coordinates of the camera obtained from the *stage*, and `depth_px`
is the observed depth of each *pixel* (distance to the image plane).
XY is then centred on the centroid of the piece; Z is kept in
absolute values (height above the support).

Real-world scale is preserved. Z = 0 is typically reported for
planar upper faces, due to the `float32` quantization of the
annotator — it does not constitute a *pipeline* defect.

New per-capture diagnostics:

- `projection_depth_mode = "per_pixel_depth"`
- `support_surface_depth_m`
- `piece_depth_median_m`
- `piece_height_median_m`
- `piece_height_min_m`
- `piece_height_max_m`
- `xy_projection_note` (literal formula, for citation in the
  report).

The 2D footprint continues to be constructed by the *top-down*
projection of the point cloud onto a 256 × 256 *pixel* canvas at
0.5 mm/px.

### 18.7 Problems encountered in this iteration

Summary of the problems observed and resolved during this iteration
of the *pipeline*. More operational detail in section 13.

1. **Camera moved by the *script* without need.** Decision:
   `SET_CAMERA_POSE = False` by default; use the *stage* pose.
2. **Dominant-mode estimator "grabbing" the background.** The
   dominant depth in a scene with several tables/walls could be the
   farthest plane. Decision: layered estimator
   (`auto_depth_layers`) which prefers the large near peak that is
   not the top of the piece.
3. **`SURFACE_DEPTH_MIN/MAX` window too narrow.** The static
   bounds excluded useful layers (piece, local support) and left
   only the background in the histogram. Decision: adaptive bounds
   by p01/p99 of the distribution of the ROI itself when in
   `auto_depth_layers` mode.
4. **Systematic XY inflation (~1.5×) for 105 mm tall pieces.**
   Caused by the use of `mpp(surface_z)` in uniform
   *back-projection*. Decision: per-pixel projection with
   `fx_px`/`fy_px` independent of depth.

The previous history (initial Phase 1) is recorded in section 13.

### 18.8 Current validation (after corrections)

Result of `scripts/validate_piece_captures.py` over the four folders
`data/pieces_detected/{rectangle, square, circle, triangle}/`:

- 4/4 pieces pass all structural criteria;
- all required files present;
- point clouds with shape `(2048, 3)`, without NaN or infinities;
- *footprints* readable and non-empty;
- exactly one valid component per capture
  (`n_valid_components = 1`).

Geometric metrics measured (extracted from
`data/pieces_detected/validation_summary.csv` and from the
individual metadata):

| Piece     | X measured (mm) | Y measured (mm) | Z span (mm) | `piece_height_median` (mm) |
|-----------|-----------------|-----------------|-------------|----------------------------|
| rectangle | 49.8            | 69.4            | 0           | 104.5                      |
| square    | 49.8            | 46.4            | 0           | 104.5                      |
| circle    | 49.4            | 46.0            | 0           | 104.5                      |
| triangle  | 49.4            | 46.0            | 0           | 104.5                      |

Comparison with the CAD:

- **Piece height**: measured 104.5 mm vs CAD 105 mm ⇒ ≈ 0.5 %
  deviation. Confirms that the surface estimation (≈ 0.2995 m) and
  the per-pixel projection are aligned.
- **X dimension**: error ≤ ≈ 1.2 % in all cases (50 mm ⇒
  49.4–49.8 mm; 75 mm ⇒ not applicable to X here because the long
  piece is oriented with 75 mm in Y).
- **Y dimension**: systematic error of approximately −7 to −8 %
  (50 mm ⇒ 46.0–46.4 mm; 75 mm ⇒ 69.4 mm). See section 18.10
  below.
- Z `span` = 0 is maintained. Acceptable for footprint matching in
  Baseline 1.

### 18.9 Current limitations

1. **Z span = 0 on planar upper faces** — joint property of the
   observed geometry and of the `float32` quantization of the
   annotator. Without consequence for Baseline 1 (matching by 2D
   footprint); requires a complementary approach (multi-view or
   sub-pixel depth) for vertical insertion verification.
2. **Systematic Y bias (~7–8 %)** — described in detail in
   section 18.10.
3. **Restriction to one piece visible per capture** —
   experimental premise; the *script* detects multiple components
   but selects one, controlled by `PIECE_SELECTION_MODE`.
4. **Sensitivity to the parameters of the layered estimator**
   (`SUPPORT_MIN_PEAK_FRACTION`, `PIECE_MAX_PEAK_FRACTION`,
   `SURFACE_PEAK_MERGE_DISTANCE_M`). The current values work for
   the validated scene; new scenes may require readjustment.
5. **Partial geometric coverage** — single *top-down* observation;
   lateral and bottom faces are not observable.

### 18.10 Residual Y bias (vertical intrinsics) — *RESOLVED*

> **Note:** this section describes the problem as it was observed
> and diagnosed **before** the correction. For the record of the
> applied correction and of the new validated results, see section
> **18.12** below.

The function `compute_intrinsics()` computes the vertical *focal*
in *pixels* through:

```
fov_v          = fov_h × (IMG_H / IMG_W)
tan_half_fov_y = tan(fov_v / 2)
fy_px          = (IMG_H / 2) / tan_half_fov_y
```

This **linear scaling in degrees/radians** between `fov_h` and
`fov_v` is only geometrically exact for very small FOVs. For the
current sensor (FOCAL = 24 mm, APERTURE = 36 mm, horizontal FOV
≈ 73.7°), the accumulated error is non-negligible: it produces
`fy_px ≈ 459` instead of the correct value for square *pixels*,
`fy_px = fx_px ≈ 426.7`. The ratio 459/426.7 ≈ 1.077 explains
exactly the reduction of ≈ 7.7 % observed in the dimensions
measured in Y.

Appropriate correction (consistent with square *pixels*):

```
tan_half_fov_y = tan_half_fov_x × (IMG_H / IMG_W)
fy_px          = (IMG_H / 2) / tan_half_fov_y
                  → algebraically equal to fx_px
```

This correction is **not** applied in the code at the time of this
note, by explicit instruction not to modify the *script* while the
structural validation passes. It must be applied before any
absolute-scale assertion in the final report. For Baseline 1
(relative piece-cavity matching), the bias partially cancels if the
cavity *script* uses the same formula — see doc 02 — section 18.

### 18.11 Next actions

Updated state of the originally recommended sequence:

1. ~~Visually inspect
   `data/pieces_detected/footprints_grid.png`.~~ — **done**.
2. ~~Correct the computation of `fy_px` in
   `compute_intrinsics()`.~~ — **done** (see 18.12).
3. ~~Recapture the four pieces and re-validate with
   `scripts/validate_piece_captures.py`.~~ — **done** (see 18.12).
4. ~~Verify whether `scripts/capture_cavity_detection.py` shares
   the same incorrect formula.~~ — **done**: it did; it has
   already been corrected by the same change (see doc 02 —
   section 19). **Pending**: recapture the cavities with the
   corrected formula and re-validate.
5. **Scale audit** of the XY spans and measured height of the
   cavities against `data/expected_cad_dimensions.json` (after
   the recapture).
6. **Re-execute Baseline 1** with the `triangle` set
   (rectangle, square, circle, triangle), after item 5.
7. **Document** the updated results in doc 03.

This sequence is treated as a precondition. The current Baseline 1
results (with the set that included the `star`) remain recorded as
intermediate diagnostic but **do not constitute the final result**.

### 18.12 Intrinsics correction and scale validation

**Problem encountered.** After replacing the `star` with
`triangle`, recalibrating the CAD dimensions and structurally
revalidating the four pieces, it was observed that the point clouds
were dimensionally inconsistent with the CAD. The X dimensions
approached the expected value, but the Y dimensions were
**systematically underestimated by 7–8 %**.

**Evidence.** Measurements made on
`data/pieces_detected/validation_summary.csv` before the correction:

| Piece     | Y measured (mm) | Y CAD (mm) | error    |
|-----------|-----------------|------------|----------|
| square    | 46.4            | 50         | −7.2 %   |
| circle    | 46.0            | 50         | −8.0 %   |
| triangle  | 46.0            | 50         | −8.0 %   |
| rectangle | 69.4            | 75         | −7.5 %   |

In parallel:
- the median height (`piece_height_median`) was correct
  (104.5 mm vs CAD 105 mm), which ruled out an error in the global
  depth scale or in the support-plane estimation;
- the X dimension was correct (≤ 1.2 % error), which localised
  the problem **exclusively to the vertical direction of the
  image**.

**Cause.** The function `compute_intrinsics()` in
`scripts/capture_piece_detection.py` (and the homonymous function
in `scripts/capture_cavity_detection.py`) computed the vertical FOV
through a linear scaling in radians:

```
fov_v = fov_h × (IMG_H / IMG_W)
```

This approximation is only valid for very small FOVs. For the
sensor used (FOCAL = 24 mm, APERTURE = 36 mm, horizontal FOV
≈ 73.7°), it produced `fy_px ≈ 459` instead of the geometrically
correct value for square *pixels*, `fy_px = fx_px ≈ 426.67`. The
ratio 459 / 426.67 ≈ 1.0736 corresponds exactly to the reduction
of ≈ 7.4 % observed in the Y measurements.

**Applied correction.** The linear scaling was replaced by a
tangent-aspect relation:

```
tan_half_fov_y = tan_half_fov_x × (IMG_H / IMG_W)
fy_px          = (IMG_H / 2) / tan_half_fov_y
                  → algebraically equal to fx_px
```

This is the formula consistent with square *pixels*, in which the
effective vertical aperture is `aperture_horizontal × (H/W)` and
the vertical *focal* in *pixels* equals the horizontal one. The
same correction was applied to both capture *scripts*. The XY
projection of the point cloud continues to be per-pixel
(section 18.6); only the intrinsics were corrected.

The *pipeline* now exposes:
- `intrinsics_model = "pinhole_tangent_aspect_corrected"` in
  `piece_metadata.json` and in `cavities_summary.json`;
- `fx_px` and `fy_px` in both metadata files;
- console line per capture:
  `[intrinsics] fx_px=..., fy_px=..., mpp_x=..., mpp_y=...`.

**Before/after comparison.**

| Piece | Y before (mm) | Y after (mm) | Y CAD (mm) | error after |
|-------|---------------|--------------|------------|-------------|
| square    | 46.4 | **49.8** | 50 | −0.4 %  |
| circle    | 46.0 | **49.4** | 50 | −1.2 %  |
| triangle  | 46.0 | **49.4** | 50 | −1.2 %  |
| rectangle | 69.4 | **74.5** | 75 | −0.7 %  |

`fx_px = fy_px = 426.6667` confirmed in the four
`piece_metadata.json` files; X/Y symmetry restored (square and
circle now measure `49.8 × 49.8` mm and `49.4 × 49.4` mm
respectively).

**Validation result after the correction.**

`scripts/validate_piece_captures.py`: 4/4 pieces pass all
structural criteria. Key metrics:

| Piece     | X (mm) | Y (mm) | Z span (mm) | `piece_height_median` (mm) |
|-----------|--------|--------|-------------|----------------------------|
| rectangle | 49.8   | 74.5   | 0           | 104.5                      |
| square    | 49.8   | 49.8   | 0           | 104.5                      |
| circle    | 49.4   | 49.4   | 0           | 104.5                      |
| triangle  | 49.4   | 49.4   | 0           | 104.5                      |

All dimensions within ≈ 1.2 % of the CAD; median height at
104.5 mm vs 105 mm (≈ 0.5 %).

**Interpretation.**

- The correction closed the systematic Y bias. The `square` and
  `circle` pieces recovered exact X/Y symmetry on the footprint
  plane, and the `rectangle` now exhibits the expected aspect
  ratio (74.5 / 49.8 ≈ 1.50, vs CAD 75 / 50 = 1.50).
- The problem was **not** exclusively one of segmentation. The
  segmentation by depth layers was correct; the defect was
  strictly in the projective geometry. This distinction is
  important for the report, because it shows that the two
  subsystems (segmentation and projection) can fail independently
  and require separate audits.
- The `piece_height_median = 104.5 mm` remaining correct before
  and after the *fix* also confirms that the surface estimation
  by `auto_depth_layers` was already correct — the Y bias did
  not affect Z because Z is computed by direct subtraction of
  depth and does not pass through the *pixel*-scale.

**Remaining limitation.**

- Residual error of up to ≈ 1.2 % in XY, attributable to
  *pixel-by-pixel* quantization at the mask boundaries and to
  anti-*aliasing* in the Replicator depth rasterizer. This error
  is symmetric in X and Y, situated within the engineering
  tolerance, and uncorrelated with piece shape. No additional
  action is proposed for this limitation within the scope of
  Baseline 1.
- The Z `span` remains equal to zero per capture; it is a joint
  property of the observed geometry (strictly planar upper face)
  and of the `float32` quantization of the annotator. Acceptable
  for Baseline 1 (footprint matching); requires a complementary
  approach for vertical insertion verification.
- The correction was also applied to
  `scripts/capture_cavity_detection.py` but the cavities still
  need to be **recaptured** so that the saved point clouds reflect
  the new formula (the existing `.npy` files remain based on the
  old intrinsics). See doc 03 — section 16 for the re-execution
  protocol.

---

## Notes for the author

Items that should be recorded manually, outside this document, and
that are not captured in the validation files:

- **Independent verification of one piece**: physically measure one
  piece (preferably the rectangle) and confirm that the CAD
  dimensions in section 17 are correct — this verification anchors
  the scale audit that confronts the CAD dimensions with the
  spans measured in
  `data/pieces_detected/validation_summary.csv`.
- **Visual confirmation of `footprints_grid.png`** after the
  `fy_px` correction (see 18.10–18.11).
- **USD camera pose at the moment of the validated capture** (read
  from the `camera_pose` field in `piece_metadata.json`, but it is
  advisable to keep a screenshot of the *stage* inspector for the
  report).
- **Physical pose of the virtual camera in USD** (translation and
  orientation) at the moment of the validated capture.
- **Exact version of Isaac Sim** and of the container used.
- **Possible lighting changes** between captures, if relevant.
- **Justification of the initial set of pieces** (rectangle,
  square, circle, star) — why these and not others.
