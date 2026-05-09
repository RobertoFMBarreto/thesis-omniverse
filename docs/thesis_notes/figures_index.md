# Figures index for the thesis report

> Organization document for future conversion to LaTeX.
> Does not copy, rename or generate files — only records which
> images should be considered for inclusion and with what caption.
> Status: initialised with Phase 1 (single-view piece detection).
> Date: 2026-05-01.

---

## Conventions

- **Figure ID**: internal identifier used in this document and
  reusable as `\label{...}` in LaTeX.
- **Source file**: path relative to the repository root.
- **Suggested LaTeX filename**: name proposed for the
  copied/renamed file in the `figures/` folder of the LaTeX
  project (to be created in the future). Convention:
  `figXX_<piece>_<type>.png`, always in lowercase, without spaces
  or accents.
- **Suggested caption**: caption in technical English, directly
  reusable in `\caption{...}`.
- **Related section**: reference to the corresponding section in
  `docs/thesis_notes/01_piece_detection_singleview.md` or to the
  thesis chapter.
- **Notes**: observations on quality, purpose or warnings relevant
  for inclusion.

The validation files (`validation_summary.csv` and
`validation_summary.json`) are not figures, but are recorded as
**data sources** for tables and metrics to be presented in the
report.

---

## Table of figures — Phase 1: single-view piece detection

| Figure ID | Source file | Suggested LaTeX filename | Suggested caption | Related section | Notes |
|---|---|---|---|---|---|
| `fig:footprints_grid` | `data/pieces_detected/footprints_grid.png` | `fig01_footprints_grid.png` | 2D *top-down* footprints of the four pieces captured in single view (rectangle, square, circle, star), projected from the point cloud with real-world scale preserved. | Phase 1 — sections 12 and 15 | Visual summary; candidate for opening figure of the validation section. |
| `fig:rectangle_debug` | `data/pieces_detected/rectangle/piece_debug.png` | `fig02_rectangle_debug.png` | Overlay of the selected component mask, bounding box and centroid for the rectangular piece. | Phase 1 — section 7 | Illustrates the result of the segmentation and selection of the connected component. |
| `fig:rectangle_footprint` | `data/pieces_detected/rectangle/piece_footprint.png` | `fig03_rectangle_footprint.png` | 2D *top-down* footprint of the rectangular piece, in real-world scale (0.5 mm/*pixel*). | Phase 1 — section 9 | Useful for discussing the geometric representation that will feed the deterministic baseline. |
| `fig:square_debug` | `data/pieces_detected/square/piece_debug.png` | `fig04_square_debug.png` | Overlay of the selected component mask, bounding box and centroid for the square piece. | Phase 1 — section 7 | Comparable to `fig:rectangle_debug` for discussing *pipeline* consistency across pieces. |
| `fig:square_footprint` | `data/pieces_detected/square/piece_footprint.png` | `fig05_square_footprint.png` | 2D *top-down* footprint of the square piece, in real-world scale. | Phase 1 — section 9 | Allows visual verification of the X≈Y symmetry reported in the validation metrics. |
| `fig:circle_debug` | `data/pieces_detected/circle/piece_debug.png` | `fig06_circle_debug.png` | Overlay of the selected component mask, bounding box and centroid for the circular piece. | Phase 1 — section 7 | Useful for discussing the *pipeline* behaviour with curved boundaries. |
| `fig:circle_footprint` | `data/pieces_detected/circle/piece_footprint.png` | `fig07_circle_footprint.png` | 2D *top-down* footprint of the circular piece, in real-world scale. | Phase 1 — section 9 | Inspect with respect to the contour discretization at the chosen resolution. |
| `fig:star_debug` | `data/pieces_detected/star/piece_debug.png` | `fig08_star_debug.png` | Overlay of the selected component mask, bounding box and centroid for the star-shaped piece. | Phase 1 — section 7 | Geometrically the most demanding case; useful for discussing concave vertices. |
| `fig:star_footprint` | `data/pieces_detected/star/piece_footprint.png` | `fig09_star_footprint.png` | 2D *top-down* footprint of the star-shaped piece, in real-world scale. | Phase 1 — section 9 | Most informative test case for the future comparison via IoU/Chamfer between candidate rotations. |

---

## Table of figures — Phase 2: board and cavity detection

| Figure ID | Source file | Suggested LaTeX filename | Suggested caption | Related section | Notes |
|---|---|---|---|---|---|
| `fig:cavity_rgb` | `data/cavities_detected/rgb.png` | `fig10_cavity_rgb.png` | RGB image of the scene with the board and the cavities, captured by the virtual camera. | Phase 2 — section 4 | Provides the visual reference of the scene before any processing. |
| `fig:cavity_depth_vis` | `data/cavities_detected/depth_vis.png` | `fig11_cavity_depth_vis.png` | Coloured visualization of the depth image of the same scene. | Phase 2 — section 4 | Allows commenting on the depth ordering between background, board and cavities. |
| `fig:board_mask` | `data/cavities_detected/board_mask.png` | `fig12_board_mask.png` | Binary mask of the upper face of the board, with holes corresponding to the cavities. | Phase 2 — section 6 | Shows that the detected board has holes geometrically coherent with the cavities. |
| `fig:board_region_mask` | `data/cavities_detected/board_region_mask.png` | `fig13_board_region_mask.png` | Filled board (`contour` mode), search domain for the cavity detection. | Phase 2 — section 6 | Pair with `fig:board_mask`: illustrates the difference between detected surface and filled search domain. |
| `fig:board_debug` | `data/cavities_detected/board_debug.png` | `fig14_board_debug.png` | RGB overlay with the detected board tinted, filled contour, bounding box and centroid. | Phase 2 — section 6 | Synthesis figure of the automatic detection of the board. |
| `fig:board_roi_auto_debug` | `data/cavities_detected/board_roi_auto_debug.png` | `fig15_board_roi_auto_debug.png` | Diagnostic of the automatic board detection process: depth candidates and parameters used. | Phase 2 — sections 6 and 14 | Useful for the discussion of the encountered problems and tuned parameters. |
| `fig:raw_cavity_mask` | `data/cavities_detected/raw_cavity_mask.png` | `fig16_raw_cavity_mask.png` | Binary mask after application of the depth criterion restricted to the board domain. | Phase 2 — section 8 | Pre-connected-components state; useful for discussing morphological cleanup. |
| `fig:cavities_debug` | `data/cavities_detected/cavities_debug.png` | `fig17_cavities_debug.png` | RGB overlay with each detected cavity tinted and numbered (cavity_00 to cavity_03). | Phase 2 — section 9 | Synthesis figure of the cavity detection with spatial identifiers. |
| `fig:cavities_footprints_grid` | `data/cavities_detected/footprints_grid.png` | `fig18_cavities_footprints_grid.png` | 2D *top-down* footprints of the detected cavities, in a grid labelled by identifier. | Phase 2 — sections 12 and 13 | Phase 1 figure analogue for the cavities. Useful for later visual comparison piece vs. cavity. |

---

## Table of figures — Baseline 1: deterministic geometric matching (star-set, historical)

> Methodological note: from the next execution onwards the set of
> pieces will be `rectangle, square, circle, triangle` — the star
> was removed from the MVP for matching-fragility reasons (see
> Baseline 1 — section 11). The figures below are from the
> execution validated with the star and should be used with
> captions that reflect that experimental decision.
>
> **Status:** historical / pre-correction (star-set Baseline 1 —
> preserved as intermediate diagnostic). The canonical paths
> `data/baseline1_geometric_matching/score_matrix_heatmap.png`,
> `.../best_match_grid.png` and the per-piece overlays under
> `rectangle/vs_cavity_01/`, `square/vs_cavity_03/`,
> `circle/vs_cavity_02/` were overwritten by the post-correction
> run C (see the table for the **final main set** below);
> consequently the file paths in this table no longer point to
> the star-set artefacts on disk and the entries are kept as a
> documentary record of the intermediate state described in
> Baseline 1 — sections 9–11. The `star/...` entries refer to
> files that were never overwritten by the post-correction run
> (the star is out of scope for run C) and may still exist on
> disk at the recorded paths.

| Figure ID | Source file | Suggested LaTeX filename | Suggested caption | Related section | Notes |
|---|---|---|---|---|---|
| `fig:baseline1_score_matrix` | `data/baseline1_geometric_matching/score_matrix_heatmap.png` | `fig19_baseline1_score_matrix.png` | 4 × 4 heat map with the composite *score* between each piece and each cavity at the optimal rotation. | Baseline 1 — section 9 | Clean diagonal: proof that the baseline discovers the matching without any manual mapping. **Status:** historical / pre-correction (star-set Baseline 1 — preserved as intermediate diagnostic). |
| `fig:baseline1_best_grid` | `data/baseline1_geometric_matching/best_match_grid.png` | `fig20_baseline1_best_grid.png` | Grid of the optimal matchings: for each piece, piece mask, cavity mask and overlay at the optimal rotation. | Baseline 1 — sections 9 and 10 | Synthesis figure for the report; simultaneously illustrates the matching and the scale discrepancy. **Status:** historical / pre-correction (star-set Baseline 1 — preserved as intermediate diagnostic). |
| `fig:baseline1_rectangle_overlay` | `data/baseline1_geometric_matching/rectangle/vs_cavity_01/overlay_best.png` | `fig21_baseline1_rectangle_overlay.png` | Piece-cavity overlay for `rectangle ↔ cavity_01` at the optimal 90° rotation. | Baseline 1 — section 10 | Cleanest case of long-axis alignment via rotation search. **Status:** historical / pre-correction (star-set Baseline 1 — preserved as intermediate diagnostic). |
| `fig:baseline1_square_overlay` | `data/baseline1_geometric_matching/square/vs_cavity_03/overlay_best.png` | `fig22_baseline1_square_overlay.png` | `square ↔ cavity_03` overlay at the optimal 180° rotation. | Baseline 1 — section 10 | Highest-IoU case (≈ 0.945). **Status:** historical / pre-correction (star-set Baseline 1 — preserved as intermediate diagnostic). |
| `fig:baseline1_circle_overlay` | `data/baseline1_geometric_matching/circle/vs_cavity_02/overlay_best.png` | `fig23_baseline1_circle_overlay.png` | `circle ↔ cavity_02` overlay at the optimal 192° rotation. | Baseline 1 — section 10 | Weak margin (0.080) over the second best. **Status:** historical / pre-correction (star-set Baseline 1 — preserved as intermediate diagnostic). |
| `fig:baseline1_star_overlay` | `data/baseline1_geometric_matching/star/vs_cavity_00/overlay_best.png` | `fig24_baseline1_star_overlay.png` | `star ↔ cavity_00` overlay at the optimal 16° rotation: central body of the star inside the cavity, points outside. | Baseline 1 — sections 10 and 11 | **Critical figure** for the justification of the decision to replace the star by the triangle in the MVP. **Status:** historical / pre-correction (star-set Baseline 1 — preserved as intermediate diagnostic). |
| `fig:baseline1_star_all_cavities` | `data/baseline1_geometric_matching/star/all_cavities_comparison.png` | `fig25_baseline1_star_all_cavities.png` | Comparison of the star against the four cavities: in the three large ones it fits entirely (`inside_ratio = 1.0`) with low IoU; in `cavity_00` it is geometric matching but with points outside. | Baseline 1 — section 11 | Key image to show **why** the isolated `inside_ratio` criterion is insufficient. **Status:** historical / pre-correction (star-set Baseline 1 — preserved as intermediate diagnostic). |

---

## Table of figures — Baseline 1, final main set (post-corrections, run C)

> Note: these figures correspond to the **final-state** Baseline 1
> execution on the post-correction main set
> (`rectangle, square, circle, triangle`), after closure of the
> piece intrinsics, cavity intrinsics, CAD circular-cavity scale
> and cavity-recapture corrections, and after the
> representation-normalisation correction (convex-hull fallback
> under fragmentation triggers) documented in Baseline 1 —
> section 17.3. They overwrite the canonical paths previously
> occupied by the star-set artefacts above. Prefer these for any
> discussion of the final-state Baseline 1 in the report.

| Figure ID | Source file | Suggested LaTeX filename | Suggested caption | Related section | Notes |
|---|---|---|---|---|---|
| `fig:baseline1_final_score_matrix` | `data/baseline1_geometric_matching/score_matrix_heatmap.png` | `fig37_baseline1_final_score_matrix.png` | 4 × 4 heat map of the composite *score* between each piece (`rectangle, square, circle, triangle`) and each cavity at the optimal rotation, after representation normalisation. | Baseline 1 — sections 17.4 and 17.5 | Final-state matrix; clean diagonal with `compatible = True` on every diagonal pair (first run for which the joint thresholds are met). |
| `fig:baseline1_final_best_grid` | `data/baseline1_geometric_matching/best_match_grid.png` | `fig38_baseline1_final_best_grid.png` | Grid of the optimal matchings on the final main set: per piece, piece mask, cavity mask and overlay at the optimal rotation. | Baseline 1 — sections 17.4 and 17.5 | Synthesis figure for the final-state report; replaces `fig:baseline1_best_grid` (canonical path overwritten by run C). |
| `fig:baseline1_final_rectangle_overlay` | `data/baseline1_geometric_matching/rectangle/vs_cavity_00/overlay_best.png` | `fig39_baseline1_final_rectangle_overlay.png` | Piece-cavity overlay for `rectangle ↔ cavity_00` at the optimal 180° rotation, run C (representation-normalised). Score 0.883, IoU 0.969, margin 0.293 over the second best. | Baseline 1 — section 17.4 | Strong-margin diagonal pair on the final main set. |
| `fig:baseline1_final_square_overlay` | `data/baseline1_geometric_matching/square/vs_cavity_02/overlay_best.png` | `fig40_baseline1_final_square_overlay.png` | `square ↔ cavity_02` overlay at the optimal 270° rotation, run C. Score 0.884, IoU 0.971, margin 0.168. | Baseline 1 — section 17.4 | Diagonal pair with intermediate margin on the final main set. |
| `fig:baseline1_final_circle_overlay` | `data/baseline1_geometric_matching/circle/vs_cavity_03/overlay_best.png` | `fig41_baseline1_final_circle_overlay.png` | `circle ↔ cavity_03` overlay at the optimal 254° rotation, run C. Score 0.889, IoU 0.980, margin 0.114 (smallest margin of the final main set). | Baseline 1 — section 17.4 | Smallest-margin diagonal pair on the final main set. |
| `fig:baseline1_final_triangle_overlay` | `data/baseline1_geometric_matching/triangle/vs_cavity_01/overlay_best.png` | `fig42_baseline1_final_triangle_overlay.png` | `triangle ↔ cavity_01` overlay at the optimal 0° rotation, run C. Score 0.886, IoU 0.975, margin 0.227. The triangle is the piece that replaces the star in the main set. | Baseline 1 — sections 11 and 17.4 | Diagonal pair on the final main set; also illustrates the role of the triangle as the rotation-relevant convex test case introduced in section 11. |

---

## Table of figures — Final main set (`triangle`-set pieces, post-corrections)

> Note: these figures correspond to the **current state** of the
> piece-capture *pipeline* after (a) camera control via the
> *stage*, (b) estimation by `auto_depth_layers`, (c)
> depth-dependent per-pixel projection and (d) **vertical
> intrinsics correction** (`intrinsics_model =
> "pinhole_tangent_aspect_corrected"`, see doc 01 — section
> 18.12). Prefer these instead of the figures `fig07_circle_*` /
> `fig08_star_*` for any discussion of the final main set in the
> report.

| Figure ID | Source file | Suggested LaTeX filename | Suggested caption | Related section | Notes |
|---|---|---|---|---|---|
| `fig:final_footprints_grid` | `data/pieces_detected/footprints_grid.png` | `fig26_final_footprints_grid.png` | 2D *top-down* footprints of the four pieces of the final main set (rectangle, square, circle, triangle), after replacement of the star and correction of the per-pixel projection. | doc 01 — section 18.8 | Replaces `fig:footprints_grid` for the post-corrections state. |
| `fig:final_rectangle_debug` | `data/pieces_detected/rectangle/piece_debug.png` | `fig27_final_rectangle_debug.png` | Overlay of the selected mask for the rectangle (75 × 50 mm) with bounding box and centroid, in the validated capture of the final main set. | doc 01 — section 18.5 | Final experimental set. |
| `fig:final_square_debug` | `data/pieces_detected/square/piece_debug.png` | `fig28_final_square_debug.png` | Overlay of the selected mask for the square (50 × 50 mm). | doc 01 — section 18.5 | Final experimental set. |
| `fig:final_circle_debug` | `data/pieces_detected/circle/piece_debug.png` | `fig29_final_circle_debug.png` | Overlay of the selected mask for the circle (Ø 50 mm). | doc 01 — section 18.5 | Final experimental set. |
| `fig:final_triangle_debug` | `data/pieces_detected/triangle/piece_debug.png` | `fig30_final_triangle_debug.png` | Overlay of the selected mask for the triangle (base 50 mm, geom. height 50 mm), the piece that replaces the star in the main set. | doc 01 — sections 18.1 and 18.5 | Central case of the experimental decision documented in doc 03 — section 11. |
| `fig:final_triangle_depth_layers` | `data/pieces_detected/triangle/depth_layers_debug.png` | `fig31_final_triangle_depth_layers.png` | Diagnostic of the `auto_depth_layers` estimation for the triangle capture: ROI (cyan), expanded segmentation ROI (yellow) and panel with the list of detected depth peaks, with the selected peak highlighted in green. | doc 01 — section 18.4 | Method figure for the section on automatic surface estimation. |
| `fig:final_triangle_raw_mask` | `data/pieces_detected/triangle/raw_piece_mask.png` | `fig32_final_triangle_raw_mask.png` | Binary mask resulting from the threshold `depth < surface_z − SURFACE_TOLERANCE` for the triangle, restricted to the expanded ROI. | doc 01 — section 18.5 | Useful for discussing the effect of the ROI restriction and of the tolerance threshold. |
| `fig:final_rectangle_footprint` | `data/pieces_detected/rectangle/piece_footprint.png` | `fig33_final_rectangle_footprint.png` | 2D *top-down* footprint of the rectangle (49.8 × 74.5 mm measured vs CAD 50 × 75 mm), in real-world scale after the intrinsics correction. | doc 01 — section 18.12 | Replaces `fig:rectangle_footprint` for the post-correction state. |
| `fig:final_square_footprint` | `data/pieces_detected/square/piece_footprint.png` | `fig34_final_square_footprint.png` | 2D *top-down* footprint of the square (49.8 × 49.8 mm measured vs CAD 50 × 50 mm), in real-world scale after the intrinsics correction. | doc 01 — section 18.12 | X/Y symmetry restored after the correction. |
| `fig:final_circle_footprint` | `data/pieces_detected/circle/piece_footprint.png` | `fig35_final_circle_footprint.png` | 2D *top-down* footprint of the circle (49.4 × 49.4 mm measured vs CAD Ø 50 mm), in real-world scale after the intrinsics correction. | doc 01 — section 18.12 | Inscribed in an almost-square bounding box. |
| `fig:final_triangle_footprint` | `data/pieces_detected/triangle/piece_footprint.png` | `fig36_final_triangle_footprint.png` | 2D *top-down* footprint of the triangle (base 49.4 mm, geom. height ≈ 49.4 mm vs CAD 50 × 50 mm), in real-world scale after the intrinsics correction. | doc 01 — section 18.12 | Replaces the star as a test case with non-circular non-rectangular convex geometry. |

---

## Data sources (non-figures) for tables and metrics

| Internal ID | Source file | Suggested use | Notes |
|---|---|---|---|
| `data:validation_csv_pieces` | `data/pieces_detected/validation_summary.csv` | Source of the table of spans and point counts of the pieces, **final main set** (rectangle, square, circle, triangle). | Flat format, easy to transform into `\begin{tabular}`. Overwrites the historical version that included `star`. |
| `data:validation_json_pieces` | `data/pieces_detected/validation_summary.json` | Detailed source (exact X/Y/Z bounds, validation *flags*) of the final main set. | Canonical source for the current state of the main set. |
| `data:piece_metadata_triangle` | `data/pieces_detected/triangle/piece_metadata.json` | Source of the diagnostics of the `auto_depth_layers` estimation and of the per-pixel projection (`projection_depth_mode`, `support_surface_depth_m`, `piece_depth_median_m`, `piece_height_median_m`). | Per-capture metadata. Analogues exist for `rectangle`, `square`, `circle`. |
| `data:validation_csv_cavities` | `data/cavities_detected/validation_summary.csv` | Source of the table of areas and spans of the cavities. | Flat format, easy to transform into `\begin{tabular}`. |
| `data:validation_json_cavities` | `data/cavities_detected/validation_summary.json` | Detailed source of the validation of the cavities (X/Y/Z bounds, *flags*). | More complete than the CSV; prefer it as canonical source for Phase 2. |
| `data:cavities_summary_json` | `data/cavities_detected/cavities_summary.json` | Source of the *pipeline* parameters (board, table depth, detection *flags*, list of rejected components). | Useful for the section on encountered problems / tuned parameters. |
| `data:cavities_run_log` | `data/cavities_detected/run_log.txt` | Console log of the validated execution (overwritten in each execution). | Useful for literal citation in the report, with the care of being saved outside the current *log* before overwriting it. |
| `data:baseline1_results_matrix` | `data/baseline1_geometric_matching/results_matrix.csv` | Source of the 4 × 4 piece × cavity *score* table. | Flat; direct conversion to `\begin{tabular}`. |
| `data:baseline1_results_all` | `data/baseline1_geometric_matching/results_all.json` | Detailed source of all pairs (optimal rotation, *flags*, *area_ratio*, fallbacks). | Canonical source for the discussion of margins, *flags* and diagnostics. |
| `data:baseline1_summary_txt` | `data/baseline1_geometric_matching/summary.txt` | Human-readable summary of the validated execution. | Useful for literal citation of the state of the MVP. |
| `data:baseline1_run_metadata` | `data/baseline1_geometric_matching/run_metadata.json` | Parameters used (canvas, resolution, dilation, weights). | Necessary to ensure reproducibility in the report. |
| `data:baseline1_run_log` | `data/baseline1_geometric_matching/run_log.txt` | Console log (overwritten in each execution). | Same prior-copy care as for Phase 2. |
| `data:expected_cad_dimensions` | `data/expected_cad_dimensions.json` | Canonical reference of the nominal CAD dimensions (pieces, cavities, board, clearance). Main set: square, rectangle, circle, **triangle**; star in `optional_stress_test_shapes`. | For validation/reporting only: **NOT** consumed by the matching algorithm. Useful for the scale audit referred to in doc 03 — section 11 and for confronting `validation_summary.csv` of Phases 1 and 2. |

---

## Gaps and figures to consider later

Items **not** available yet but potentially relevant for the
report, to be recorded when produced:

- Comparative figure **before/after segmentation** (original RGB
  alongside the selected mask) per piece.
- Figure illustrating the support surface estimation (depth
  histogram with annotated peak), both for Phase 1 (piece table)
  and for Phase 2 (table/background and top of the board).
- *Per-cavity* dedicated figures (`cavity_NN/cavity_debug.png`
  and `cavity_NN/cavity_footprint.png`), to be recorded when the
  decision is made to highlight an individual cavity in the
  report.
- Comparative figure piece vs. corresponding cavity — already
  partially available in `best_match_grid.png` (Baseline 1,
  execution with the star). Replace by an equivalent execution
  with the **triangle** piece as soon as the CAD-vs-capture
  scale audit has been done and the baseline has been re-executed
  (see Baseline 1 — section 11).
- **Pending:** post-intrinsics-correction version of the Phase 2
  figures (`fig:cavity_*`, `fig:board_*`, `fig:cavities_debug`,
  `fig:cavities_footprints_grid`). The files currently on disk
  in `data/cavities_detected/` were still generated with the old
  `fy_px` formula; they will be replaced automatically after the
  recapture of the cavities (see doc 03 — section 16.3.bis).
- *Score matrix* table with triangle in place of the star, for
  direct comparison with the current 4 × 4 table of this
  document.
- Possible *stress test* figure of the star (with a
  scale-compatible star cavity), reserved for a phase later than
  the validation of the MVP.
- Diagram of the perception *pipeline* (to be drawn separately,
  for example in TikZ or in a vector tool), to be referenced as
  `fig:pipeline_overview`.

---

## Conventions for future migration to LaTeX

When this index is materialised in the LaTeX project:

1. Create the `figures/` folder in the LaTeX project.
2. Copy each `Source file` to `figures/<Suggested LaTeX filename>`.
   Do not modify the image; only rename it.
3. Insert each figure with:

   ```latex
   \begin{figure}[ht]
     \centering
     \includegraphics[width=0.7\linewidth]{figures/figXX_<piece>_<type>.png}
     \caption{<caption in technical English>}
     \label{<Figure ID>}
   \end{figure}
   ```

4. Keep the `Figure ID` of this document as `\label`, so that
   cross-references in the thesis text are stable even if the
   names of the image files are reorganised.
5. Do not embed the images at arbitrary real-world scale — prefer
   `width=0.7\linewidth` or `width=0.45\linewidth` for grids of
   two side-by-side figures.
