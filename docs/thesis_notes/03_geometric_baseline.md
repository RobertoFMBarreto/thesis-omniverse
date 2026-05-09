# 03 — Baseline 1: deterministic geometric matching

> Implementation note for future conversion into a LaTeX section.
> Status: Baseline 1 — deterministic perception and matching, no
> learned component.
> Date: 2026-05-02.

---

## 1. Objective of the phase

Establish a deterministic baseline of piece-cavity matching based
on 2D geometry, on top of the artifacts of Phases 1 (pieces) and
2 (cavities). The baseline serves two explicit purposes:

- provide a comparison reference for any future learned method;
- expose perception problems (scale, segmentation, sample support)
  that may be masked by isolated visual inspection of individual
  footprints.

The baseline **does not** classify shapes, **does not** use
learning, **does not** depend on any manual piece → cavity mapping.
The correspondence is discovered exclusively from real-scale
geometry.

---

## 2. Inputs

The baseline reads the point clouds of Phases 1 and 2:

- `data/pieces_detected/{rectangle, square, circle, star}/piece_pointcloud.npy`
- `data/cavities_detected/cavity_{00,01,02,03}/cavity_pointcloud.npy`

Each cloud has *shape* `(2048, 3)` in metres, with X/Y centred on
the centroid and real-world scale preserved. Only the X and Y
columns are used; Z is discarded in this baseline (see section 9).

The `*_footprint.png` files produced by Phases 1 and 2 are **not**
used by the baseline: they are visualization artifacts. The baseline
rasterizes its own masks from the points, ensuring that both sides
(pieces and cavities) use the same canvas convention.

The folder names (`rectangle`, `square`, ..., `cavity_00`, ...) are
**organization labels** — they never enter the matching algorithm.

---

## 3. Configuration decisions effectively applied

| Constant | Value | Comment |
|---|---|---|
| `ROTATION_STEP_DEG` | 2 | 180 angles per piece-cavity pair. |
| `WORLD_CANVAS_M` | 0.080 m | 8 × 8 cm canvas, sufficient for the largest piece. |
| `FOOTPRINT_RESOLUTION_M_PER_PX` | 0.00025 m/px | 0.25 mm/px → 320 × 320 px canvas. |
| `CLEARANCE_DILATION_M` | 0.001 m | 1 mm = 4 px of cavity dilation. Perception/matching tolerance, **not** a CAD-validated mechanical clearance. |
| `W_IOU`, `W_INSIDE`, `W_OUTSIDE` | 0.55 / 0.35 / 0.10 | Composite score weights. |
| `COMPATIBLE_INSIDE_MIN` | 0.80 | Lower threshold for inside_ratio. |
| `COMPATIBLE_OUTSIDE_MAX` | 0.20 | Upper threshold for outside_ratio. |
| `COMPATIBLE_IOU_MIN` | 0.55 | Lower threshold for IoU. |
| `SUSPICIOUS_AREA_RATIO_MAX` | 0.50 | Suspicious-scale flag. |
| `LOW_RAW_SUPPORT_AREA_PX` | 200 | Cavities with fewer raw pixels than this receive `low_raw_support=True`. |
| `TIE_MARGIN` | 0.01 | Margin for tie in the selection of the best cavity. |

These decisions were made based on the proposal of the geometric
review and on the project constraints. In particular, **alpha-shape
was not adopted in this version**: a lightweight rasterization
*pipeline* was preferred (NumPy + OpenCV only) with convex *hull*
as an explicit *fallback*.

---

## 4. Rasterization *pipeline* (piece and cavity)

The same function is applied to both inputs, ensuring *pixel-by-pixel*
comparability:

1. Elimination of duplicates/near-duplicates at the half-pixel
   level (0.125 mm), neutralising the sampling with replacement
   performed in Phases 1 and 2 when the raw mask had fewer points
   than `N_POINTS = 2048`.
2. Projection of the XY points onto the pixel grid (320 × 320 px),
   with inversion of the Y axis to coincide with the convention of
   the footprint images of the previous phases.
3. Binary *splat* (each hit pixel set to 255).
4. Morphological *close* operation with a 3 × 3 element to join
   sparse *pixels*.
5. Filling of external contours with
   `cv2.findContours(... RETR_EXTERNAL, CHAIN_APPROX_NONE)` and
   `cv2.drawContours(..., thickness=cv2.FILLED)`. This choice
   preserves concavities of the external contour (important for
   the geometry of the star), while closing internal holes caused
   by sparse *splatting*.
6. Convex *hull* *fallback*: if the resulting mask is empty or
   contains fewer than 50 pixels, an alternative mask is built via
   `cv2.convexHull`. This occurrence is recorded in the metadata
   of the pair as `convex_hull_fallback=True`.

The intent is to keep the baseline with minimal dependencies
(NumPy + OpenCV) and avoid *alpha-shape* tunings in this first
iteration. If a very concave shape (e.g.: star) is unable to
differentiate itself from the others, *alpha-shape* will be
considered in a future iteration.

---

## 5. Rotation search

The search is a **uniform grid** from 0° to 360° exclusive with
2° step: 180 evaluations per piece-cavity pair. The rotation is
applied to the **XY points** before rasterization (not to the
already-rasterized masks), avoiding interpolation artifacts.

Computational cost: 4 pieces × 4 cavities × 180 rotations ≈ 2880
rasterizations; total observed time ≈ 5 seconds. No
*coarse-to-fine* strategy was needed. No symmetry exploitation was
used, because that would require prior shape classification —
which is expressly excluded.

---

## 6. Metrics and composite *score*

For each rotation θ, with `P(θ)` = mask of the rotated piece and
`C` = mask of the cavity dilated by `CLEARANCE_DILATION_M`:

```
inside_ratio  = |P ∩ C| / |P|
outside_ratio = |P ∩ ¬C| / |P|       (= 1 − inside_ratio)
IoU           = |P ∩ C| / |P ∪ C|

score = 0.55 × IoU + 0.35 × inside_ratio − 0.10 × outside_ratio
```

The `area_ratio` is computed relative to the **non-dilated** cavity
mask:

```
area_ratio = min(|P|, |C_undilated|) / max(|P|, |C_undilated|)
```

`area_ratio` **does not** enter the *score*. It is used only as a
diagnostic:

- `suspicious_scale = (area_ratio < 0.50)`.

Cavities with fewer than 200 raw pixels in the capture metadata
receive `low_raw_support = True` (currently only `cavity_00`).

---

## 7. Compatibility criterion in the current version

In the version currently implemented, a single `compatible` flag
is set to `True` if and only if, at the optimal angle:

- `inside_ratio ≥ 0.80`,
- `outside_ratio ≤ 0.20`,
- `IoU ≥ 0.55`.

The score, area-ratio and `suspicious_scale` / `low_raw_support`
flags are recorded separately. In case of tie between the two best
cavities for a piece (margin < 0.01), the piece is marked
`tie=True` and both candidates are listed.

The visual analysis (section 11) motivates a proposal to
**reformulate these flags** to separate "best geometric match" from
"physical scale plausibility" — see section 13.

---

## 8. Produced outputs

Root directory: `data/baseline1_geometric_matching/`.

**Globals:**

| File | Content |
|---|---|
| `results_all.json` | Full serialization (each piece × cavity × optimal rotation pair). |
| `results_matrix.csv` | 4 × 4 matrix of optimal *scores*. |
| `summary.txt` | Human-readable summary. |
| `run_metadata.json` | Parameters, *timestamps*, paths, success status. |
| `run_log.txt` | Copy of the console, overwritten on each execution. |
| `score_matrix_heatmap.png` | Annotated 4 × 4 heat map. |
| `best_match_grid.png` | 4 × 3 grid: piece | cavity | overlay at optimal rotation. |

**Per piece (`<piece>/`):**

- `best_match.json` — best cavity, best rotation, *flags*.
- `ranking.json` — all cavities ordered by optimal *score*.
- `all_cavities_comparison.png` — row with 4 overlays.

**Per pair (`<piece>/vs_<cavity>/`):**

- `rotation_scores.csv` — 180 lines: rotation, inside, outside, IoU, score.
- `pair_summary.json` — pair summary.
- `overlay_best.png` — coloured overlay at the optimal rotation.
- `score_curve.png` — curve of *score* and IoU vs rotation.

The write policy follows the rule of the previous phases: the
output directory is cleaned at the start so that files from
previous executions cannot be confused with the current result.

---

## 9. Summary of validation results

The execution produced a **clean diagonal** in the 4 × 4 matrix:
each piece preferred a distinct cavity (without any hypothesis
imposed *a priori*).

| Piece       | Best cavity | Rotation | *Score* | inside | outside | IoU   | area_ratio | suspicious_scale | low_raw_support | compatible |
|-------------|-------------|----------|---------|--------|---------|-------|------------|------------------|------------------|------------|
| rectangle   | cavity_01   | 90°      | 0.708   | 0.909  | 0.091   | 0.725 | 0.284      | True             | False            | True       |
| square      | cavity_03   | 180°     | 0.855   | 0.968  | 0.032   | 0.945 | 0.303      | True             | False            | True       |
| circle      | cavity_02   | 192°     | 0.837   | 0.969  | 0.031   | 0.911 | 0.315      | True             | False            | True       |
| star        | cavity_00   | 16°      | 0.663   | 0.803  | 0.197   | 0.730 | 0.222      | True             | True             | True       |

Margins between the best and the second-best *score* per piece:

- rectangle: 0.222 (strong);
- square: 0.180 (strong);
- circle: 0.080 (weak);
- star: 0.101 (weak).

All four pieces trigger `suspicious_scale = True` (area_ratio
between 0.22 and 0.32). Only `star ↔ cavity_00` adds
`low_raw_support = True`.

---

## 10. Visual inspection of the overlays

Visual inspection of the overlays produced in `overlay_best.png`
confirms:

- **Rectangle vs cavity_01 (90°)** — dominant green, thin
  dilation halo, red confined to the short ends. Long axes
  aligned. Physically plausible match.
- **Square vs cavity_03** — overlay almost entirely green, minimal
  dilation halo, residual reds in the corners. Cleanest case.
- **Circle vs cavity_02** — dominant green, thin circular halo,
  small reds in the perimeter. Correct match.
- **Star vs cavity_00 (16°)** — the **central body** of the star
  overlays the interior of the cavity (green inside, not in the
  halo); the **five points** are red outside the cavity. The
  match is **not** being "saved" by the dilation: it is real
  geometry, but partial geometry.

The global grid and the multi-cavity comparison of the star
additionally confirm that **the piece masks have approximately
3 × the area of the cavity masks** in all optimal pairs, which is
consistent with `area_ratio ≈ 0.3` and justifies the transversal
`suspicious_scale = True` flag.

---

## 11. Experimental decision: replacement of the star by triangle

This section documents a methodological decision adopted for the
*MVP* of the baseline.

### What happened with the star

The `star` piece was correctly detected in Phase 1, validated by
the *footprint* and point cloud files, and was correctly
associated by Baseline 1 to the only available small cavity,
`cavity_00`. However:

- the star piece has an XY span of approximately 20 mm, while
  `cavity_00` has only about 10.7 mm. The `area_ratio` is 0.222
  and `suspicious_scale` is active;
- `cavity_00` has only 114 raw *pixels* in the segmentation mask
  before resampling with replacement to `N_POINTS = 2048`, so
  `low_raw_support` is active;
- visually, the central body of the star fits in the cavity, but
  the five points of the star remain outside. The footprint
  effectively matched is the "convex core" of the star, not the
  full characteristic shape.

The match won in the matrix for two combined reasons:

1. `cavity_00` is the only small cavity, being geometrically the
   only candidate with scale close to the piece;
2. the remaining cavities are significantly larger, so the star
   fits entirely inside them with `inside_ratio = 1.0`, but with
   low `IoU` — the `outside_ratio` term in this case is zero
   (nothing of the piece lies outside), but the `area_ratio` term
   (not used in the *score*) reveals the size discrepancy that
   the IoU penalizes.

### Why this is a limitation of the current experiment

The combination `suspicious_scale + low_raw_support + weak margin`
means that:

- the match is the best available, but **it is not a physically
  plausible match** without a scale reconciliation between the
  piece side and the cavity side;
- the star simultaneously introduces **two** fragility factors
  (strongly concave geometry and low sample support of the
  target), making it difficult to isolate what is failing: the
  piece perception, the cavity perception, the absolute scale, or
  the footprint rasterization.

For an *MVP* of the baseline, this coupling is harmful: the
objective of the *MVP* is to demonstrate the full *pipeline*
(perception → representation → matching) under controlled and
well-dimensioned conditions. Keeping the star at this stage
confuses the reading of the results.

### Why the triangle is a better choice for the *MVP*

- Non-circular and non-rectangular geometry, which keeps the
  rotation search relevant (the IoU varies significantly with the
  angle, with no continuous rotational symmetry as in the circle
  nor 90° symmetry as in the square).
- **Convex** geometry, avoiding the rasterization problems
  associated with concave contours.
- Simpler to validate dimensionally in Fusion (three vertices,
  three edges, well-defined angles) and more predictable in the
  *top-down* capture.
- Allows assessing the robustness of the *score* to non-rectangular
  pieces without the noise of the thin star points.

### The star may return later

The star is geometrically an interesting test case for:

- robustness to concave contours (assesses whether the
  rasterization or a future *alpha-shape* preserves concavities);
- robustness to low-density representations of the target cavity
  (`low_raw_support`);
- study of the dependence of the composite *score* on shapes
  where `inside_ratio = 1.0` does not imply a good match.

The intent is to reintroduce it **after** the baseline is
validated with controlled geometry and after the scale problems
have been reconciled, as a ***stress* test case** of the
*pipeline*, possibly accompanied by an enlarged/reprojected cavity
so that the `area_ratio` is plausible.

### Points to retain for the report

- The baseline **did not** use any piece → cavity mapping. The
  diagonal result was obtained exclusively by the rasterized
  geometry and by the rotation search.
- The replacement of the star by the triangle is a *scope*
  decision of the *MVP*, not an admission that the star is
  intractable; it is a decision of variable reduction.
- **Before re-executing final results, it is necessary to
  validate the pieces and the cavities dimensionally in Fusion**
  against the XY spans measured in
  `data/pieces_detected/validation_summary.csv` and
  `data/cavities_detected/validation_summary.csv`. The
  transversal *suspicious_scale = True* suggests one of the
  following hypotheses, all to be audited:
  (a) different camera intrinsics between piece capture and
      cavity capture;
  (b) under-segmentation of the cavity contours by the depth
      threshold;
  (c) actual dimensional discrepancy between the CAD models of
      the pieces and the cavities;
  (d) systematic error in the surface depth used as metric
      reference.

### Final CAD dimensions for the scale audit

The CAD dimensions of the revised experimental set (star replaced
by triangle) are recorded in
`data/expected_cad_dimensions.json` and summarised below. **They
are not consumed by the matching algorithm** — they serve only for
the scale audit referred to above.

Pieces (nominal XY × Z extrusion):

| Piece       | XY (mm)                    | Extrusion (mm) |
|-------------|----------------------------|----------------|
| square      | 50 × 50                    | 105            |
| rectangle   | 50 × 75                    | 105            |
| triangle    | base 50, geom. height 50   | 105            |
| circle      | diameter 50                | 105            |

Cavities (nominal XY × Z depth):

| Cavity      | XY (mm)                    | Depth (mm) |
|-------------|----------------------------|------------|
| square      | 51 × 51                    | 75         |
| rectangular | 51 × 76                    | 75         |
| triangular  | base 51, geom. height 51   | 75         |
| circular    | diameter 51                | 75         |

Board: thickness 75 mm; external dimensions still to be recorded.

Nominal clearance: **1 mm total, 0.5 mm per side**.

Direct comparison with the Baseline 1 parameter: the constant
`CLEARANCE_DILATION_M = 0.001` (1 mm), originally justified as a
perception/matching tolerance, **coincides numerically** with the
total CAD clearance. This coincidence is favourable for the
*MVP* — the dilation applied on the cavity (used to compensate for
under-segmentation) has the same order of magnitude as the
physical mechanical clearance, so the compatibility threshold
represents a scenario close to the physical one. **It must not,
however, be treated as a validated mechanical justification**: it
is only a convenient coincidence; a different mechanically applied
clearance in the future will require readjustment.

**Confirmed experimental decision:** the star remains outside the
main set. It remains recorded in `expected_cad_dimensions.json`
under `optional_stress_test_shapes` for later reintroduction as a
concave *stress* case, likely accompanied by a dedicated
dimensionally compatible cavity (the `cavity_00` of the previous
bench was 10.7 mm — clearly undersized for the 20 mm star).

---

## 12. Known limitations

1. **Z discarded — no insertion-depth validation**. The baseline
   operates exclusively on 2D footprints in the XY plane. The
   final CAD dimensions make this limitation concrete: the piece
   extrusion is 105 mm and the nominal cavity depth is 75 mm; a
   piece inserted to the bottom protrudes 30 mm above the top of
   the board. Baseline 1 **does not detect or score** this
   protrusion. The Z information is already present in
   `piece_pointcloud.npy` and `cavity_pointcloud.npy`; a
   subsequent *Baseline 1.5* could add a vertical compatibility
   check without altering the perception *pipeline*, comparing
   the observed piece height against `cavity.depth_m` and against
   the contact dynamics to be modelled.
2. **`suspicious_scale = True` in all pairs** — strong symptom
   of a perception problem to be investigated (see section 11
   "Points to retain").
3. **Weak margins on two pieces** — `circle` (0.080) and `star`
   (0.101). The small margin reflects the geometric proximity
   between square and circle at this resolution, and the
   non-existence of a truly star-compatible cavity. The first is
   solved with more resolution; the second with the
   methodological replacement described in section 11.
4. **Cavity identity is positional** — `cavity_NN` is not
   semantic. Reorderings of the scene change the identifiers; the
   baseline is insensitive to this, but the human *output* may
   look different between executions.
5. **`alpha-shape` not used in this version** — internal
   concavities may be lost if they appear in future cases.
6. **`CLEARANCE_DILATION_M = 1 mm` is a matching tolerance**,
   **not** a CAD-validated mechanical clearance.
7. **Composite *score* weights were fixed *a priori* and were
   not tuned to this set of 4 pieces** — there is no
   overfitting, but no sensitivity study of these weights was
   performed either.

---

## 13. Proposed reformulation of the compatibility *flags*

Following the visual inspection (section 10), it is proposed to
separate the `compatible` *flag* into three non-conflicting
concepts:

| *Flag* | Definition | Current state in the 4 pieces |
|---|---|---|
| `geometric_best_match` | This cavity is the first in this piece's *ranking* (matrix diagonal). | True for all. |
| `physical_scale_plausible` | `area_ratio ≥ 0.50` and `suspicious_scale = False`. Indicates that the real-world scale between piece and cavity is coherent. | False for all. |
| `margin_weak` | `best_score − second_score < 0.10`. | True for `circle` and `star`. |

This reformulation prevents `compatible = True` from being
interpreted as "ready for robotic insertion" when, in reality, the
absolute scale is still unreconciled. The change affects only the
output metadata and does not invalidate already-recorded results;
it can be applied in the next iteration of the baseline.

---

## 14. Relevance for the thesis objective

Baseline 1 is not the learned approach intended by the thesis, but
it serves three specific functions in the global plan:

- **Comparison reference** for any future learned matching method.
  The learned method is expected to equal or surpass Baseline 1
  on well-dimensioned pieces and to specifically improve the
  cases where Baseline 1 shows weak margins.
- **Diagnostic of the perception *pipeline***. The fact that
  `suspicious_scale = True` is triggered on all matches is a
  signal that only emerges when both sides are directly compared
  in real-world scale — a signal that is not detectable from the
  isolated inspection of Phases 1 and 2 alone.
- **Operational definition of what counts as "compatibility"** of
  insertion in purely geometric terms. The method and the
  thresholds proposed here will be the basis against which any
  more sophisticated notion will be confronted.

---

## 15. Figures to include later in LaTeX

See `docs/thesis_notes/figures_index.md` for the consolidated
table. The most relevant figures of this phase are:

- `data/baseline1_geometric_matching/score_matrix_heatmap.png`
- `data/baseline1_geometric_matching/best_match_grid.png`
- `data/baseline1_geometric_matching/star/all_cavities_comparison.png`
- `data/baseline1_geometric_matching/star/vs_cavity_00/overlay_best.png`
- `data/baseline1_geometric_matching/rectangle/vs_cavity_01/overlay_best.png`
- `data/baseline1_geometric_matching/square/vs_cavity_03/overlay_best.png`
- `data/baseline1_geometric_matching/circle/vs_cavity_02/overlay_best.png`

---

## 16. Current state and re-execution protocol

The results of this section (4×4 matrix, *overlays*, compatibility
*flags*) were produced with the **previous** experimental set —
`rectangle, square, circle, star` — and before the corrections
introduced in the piece-capture *script* (camera control via the
*stage*, estimation by `auto_depth_layers`, depth-dependent
per-pixel projection). **They should be treated as intermediate
diagnostic**, and not as the final result of Baseline 1.

### 16.1 Consolidated decisions

- **Main set**: `rectangle, square, circle, triangle`.
- **`star`**: removed from the main set for being excessively
  sensitive to segmentation and to absolute scale (see
  section 11). It remains recorded in
  `data/expected_cad_dimensions.json` under
  `optional_stress_test_shapes` as a concave *stress* case
  reserved for future work. **It does not** enter the
  re-execution of Baseline 1.
- **No piece→cavity mapping** in any part of the algorithm
  (principle maintained).

### 16.2 Preconditions for re-execution

Re-execution should occur **only after** the following sequence
(see doc 01 — section 18.11 for the complete protocol):

1. Correct the computation of the vertical *focal* in *pixels*
   (`fy_px`) in
   `scripts/capture_piece_detection.py` (and in
   `scripts/capture_cavity_detection.py` if it shares the same
   formula — see doc 02 — section 19).
2. Recapture the four pieces of the main set and re-validate with
   `scripts/validate_piece_captures.py`.
3. Recapture the cavities, if applicable, and re-validate.
4. Scale audit against
   `data/expected_cad_dimensions.json`: XY spans and
   `piece_height_median_m` should fall within a pre-defined
   tolerance margin (suggestion: ≤ 2 % relative error, once the
   current systematic Y bias has been removed).
5. Only then re-execute
   `scripts/baseline1_geometric_matching.py` with the set
   `rectangle, square, circle, triangle`.

For the cavity-side scale audit and CAD correction episode that
covers item 3 of this list and its associated CAD-vs-measured
comparison, see doc 02 — section 20.

### 16.3 Current result — preserve as diagnostic

The 4×4 matrix, the *overlays* and the discussion of sections
9–11 remain documented as an **intermediate result**. They serve
three purposes in the report:

1. show that Baseline 1 correctly discovered the diagonal
   (rectangle ↔ cavity_01, square ↔ cavity_03, etc.) without
   any manual mapping, proof-of-concept of the deterministic
   method;
2. expose the transversal `suspicious_scale = True` symptom,
   diagnostic that motivated the subsequent investigation
   (surface estimation, per-pixel projection);
3. ground the experimental decision to remove the `star` from
   the main set (section 11).

These results **should not** be cited as a measure of performance
of the deterministic approach in the final report. When the
subsequent re-execution is available, it should be published as
the **reference result** and this intermediate result should be
clearly labelled as "pre-corrections" or "intermediate
diagnostic".

### 16.3.bis Status of the precondition (update)

From the list in section 16.2:

1. ~~Correct `fy_px` in
   `scripts/capture_piece_detection.py` and in
   `scripts/capture_cavity_detection.py`.~~ — **done**. The
   linear scaling `fov_v = fov_h × (H/W)` was replaced by the
   tangent-aspect relation
   `tan_half_fov_y = tan_half_fov_x × (H/W)`. Full detail in
   doc 01 — section 18.12. Both *scripts* now expose
   `intrinsics_model = "pinhole_tangent_aspect_corrected"` and
   the fields `fx_px`, `fy_px` in metadata.
2. ~~Recapture and re-validate the four pieces.~~ — **done**.
   Results in doc 01 — section 18.12: 4/4 structurally valid,
   all dimensions within ≤ 1.2 % of the CAD,
   `piece_height_median = 104.5` mm vs CAD 105 mm.
3. **Pending**: recapture the cavities with the new formula and
   re-validate with `scripts/validate_cavity_captures.py`. The
   cavity point clouds in
   `data/cavities_detected/cavity_*/cavity_pointcloud.npy`
   remain based on the old intrinsics until this recapture
   occurs.
4. **Pending**: scale audit of the cavities against
   `data/expected_cad_dimensions.json` after the recapture.
5. **Pending**: re-execution of Baseline 1 with the `triangle`
   set (rectangle, square, circle, triangle) and with already
   corrected cavity metadata.

Baseline 1 **should not** be re-executed before at least item 3
is concluded, under penalty of comparing piece footprints
(correct scale) with cavity footprints (old scale, ~7 %
underestimated in Y) — which would produce an artificially low
*area_ratio* on the cavity side and could distort the discussion
of Baseline 1.

### 16.4 Outputs to re-generate

The next execution will produce a new set of the same artifacts
in `data/baseline1_geometric_matching/`. To avoid confusion
between executions, it is recommended to:

- before re-execution, copy `results_matrix.csv`,
  `results_all.json`, `summary.txt`, `score_matrix_heatmap.png`,
  `best_match_grid.png` and `run_log.txt` to an archive
  directory (for example
  `data/baseline1_geometric_matching/_archive/2026-05_star_set/`)
  to preserve the intermediate record;
- only then let `scripts/baseline1_geometric_matching.py`
  overwrite the canonical artifacts.

The `cavity_NN` nomenclature continues to be positional and
deterministic by construction; the replacement of the `star` by
the `triangle` on the piece side does not change the numbering of
the cavities, but may change which cavity wins each *match* if
the scale is substantially different.

---

## Notes for the author

Items to be recorded manually, outside this document:

- Nominal CAD dimensions of the pieces and the cavities in
  Fusion, with explicit units, to support the scale audit
  proposed in section 11.
- Justification of the revised initial set of pieces
  (rectangle, square, circle, **triangle**) — why these and not
  others.
- USD specification of the Phase 1 vs Phase 2 camera (focal,
  aperture, pose), to rule out/confirm the hypothesis of
  divergent intrinsics between the two captures.
- Result of the next execution of the baseline with triangle,
  with direct comparison to the numbers of this execution.
- Final decision on whether the star returns as a *stress* case
  and at which phase of the work.
