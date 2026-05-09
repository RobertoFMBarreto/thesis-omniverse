# 05 — Baseline 2 Phase B: Deterministic Multi-View Geometric Matching

> Implementation note for future conversion into a LaTeX section.
> Status: design proposal — not yet implemented.
> Date: 2026-05-09.

---

## 1. Motivation

### 1.1 Limitations of single-view matching

Baseline 1 (doc 03 — section 17) operates on a single top-down RGB-D
capture per piece and per cavity. That premise carries three
structural limitations that multi-view observation may partially
address.

**Depth-edge bias.** The `distance_to_image_plane` annotator
generates a depth map whose edges are geometrically ambiguous:
pixels at the boundary between a piece face and the surrounding
board belong to a mixed depth band, and the exact threshold used to
segment the piece from the board propagates into the perimeter of
the rasterised footprint. The convex-hull fallback invoked in run C
of section 17 converts this perimeter-only signal into a solid
convex region, which works for the current convex main set but does
not distinguish a rectangle from a square when both produce a convex
silhouette from above.

**Partial cavity-wall observation.** The top-down view images the
opening plane of each cavity but cannot observe the lateral walls or
the cavity depth. The depth annotator's coverage of the interior of
the cavity depends on the viewing angle; in doc 03 — section 17.2 the
raw cavity point clouds were documented as near-empty interior splats,
which required the convex-hull fallback for all four cavities. An
oblique view adds side-wall signal that is structurally absent from
the top-down view.

**Perimeter-only sampling after the fallback.** Because the fallback
fills via convex hull, the resulting piece and cavity representations
lose any concavity that was present in the original point-cloud
boundary. The current main set is entirely convex (doc 03 — section
17.7), but this means that the existing fallback cannot scale to
non-convex pieces without modification.

**Loss of concavities.** The convex-hull fallback explicitly
discards concave features. The star piece was excluded from the main
set for this reason among others (doc 03 — section 11). A fused
multi-view representation that samples the piece boundary from
several orientations may recover lateral concavities that are
invisible to a single top-down camera.

### 1.2 Viewpoint ambiguity between similar shapes

The rectangle and the square are the canonical hard case for any
geometric matching approach applied from a single viewpoint. Both
produce a four-sided convex footprint from above. The discriminating
signal is the aspect ratio of the bounding box, but this signal is
weak from exactly top-down because the two horizontal dimensions of
the piece are equally visible and their relative magnitude is
compressed by the perspective. In the Baseline 1 results
(doc 03 — section 17.4), the rectangle-cavity and square-cavity
margins were strong (0.293 and 0.168 respectively), which suggests
the current scale and aspect ratio were sufficient for the specific
CAD dimensions in the MVP set (50 x 75 mm for the rectangle,
50 x 50 mm for the square). However, this margin may degrade if
the pieces are more similar in size or if the viewing angle slightly
tilts the top-down camera.

The circle and the square also exhibit ambiguity under certain
conditions: when the cavity span is similar, a square cavity and a
circular cavity viewed from a slightly oblique angle can produce
comparable bounding-box areas and similar IoU scores against a
circular piece footprint. The circle margin in Baseline 1 was the
smallest among the four pieces (0.114).

### 1.3 Multi-view as a hypothesis for ambiguity reduction

The hypothesis motivating Phase B is: observations from
complementary viewpoints may improve the discriminability of
otherwise ambiguous shape pairs. This is not a guarantee. Specific
conditions must hold for a multi-view system to offer better
discrimination than a single view:

- the additional views must expose geometric features that are
  structurally absent from the top-down view (e.g. the aspect
  ratio of a lateral face, the shoulder height of the piece);
- the per-view matching signal must be consistent with the
  geometric truth (e.g. the oblique view must not be dominated
  by noise or shadow);
- the aggregation strategy must combine the views in a way that
  reinforces correct assignments and does not average away the
  discriminating signal.

Where these conditions are not met, a multi-view system will
perform no better than Baseline 1 and may perform worse if the
aggregation introduces cancellation. This risk is documented
explicitly in section 6.

### 1.4 Value of deterministic methods before any learned approach

Deterministic geometric baselines serve two explicit functions in
the project plan (doc 03 — section 1): they act as a comparison
reference for any future learned method, and they expose perception
problems that may be masked by visual inspection of individual
footprints. A Phase B deterministic multi-view baseline extends
this function to the multi-view regime before any learned
descriptor, any embedding, or any data-driven aggregation is
introduced.

This sequencing is deliberate: if a learned multi-view method is
proposed later, the deterministic baseline defines the performance
floor that the learned method must exceed to justify its complexity.
It also isolates the representation-level improvement (from
single-view to multi-view) from the representation-learning question
(from handcrafted to learned descriptors) — a distinction that the
thesis rationale in doc 03 — section 14 and doc 04 — section 2.7
records explicitly as a scientific hygiene requirement.

---

## 2. Design goals

### 2.1 Goals

- **Deterministic pipeline.** No random number generator, no
  sampling, no training. The same camera poses and the same Phase A
  artifacts always produce the same matching output.
- **Explainable scoring.** Every term in the aggregate score has a
  named geometric meaning that can be stated in one sentence. No
  term is an opaque embedding or a learned weight.
- **Reproducibility.** A fresh re-execution on the same Phase A
  outputs must produce bit-identical results (modulo floating-point
  order-of-operations on the same hardware).
- **Modularity.** Per-view descriptor extractors are implemented as
  independent, swappable functions. The aggregation step is a
  separate function that receives a fixed-length vector of per-view
  scores. The Baseline 1 rasteriser and scoring head are reused
  where applicable; they are not modified.
- **Low computational overhead.** The MVP set is four pieces and
  four cavities; three views per piece and per cavity. The total
  number of comparison operations must remain tractable within the
  same order of magnitude as Baseline 1 (doc 03 — section 5:
  approximately 2880 rasterisations, total observed time ~5 s).
- **Infrastructure compatibility.** Phase B consumes the per-view
  artifacts produced by Phase A
  (`data/multiview_captures/pieces/<piece>/view_NN_<name>/`) and,
  where cavity-side multi-view captures are produced, the equivalent
  cavity artifacts. It does not modify Phase A scripts or outputs.

### 2.2 Non-goals

- No learned representations, descriptors, embeddings, or
  classifier heads.
- No pose estimation in 3D.
- No grasp planning, trajectory planning, or motion synthesis.
- No insertion policy or robotic execution.
- No 3D reconstruction, depth fusion, or volumetric integration.
- No neural networks of any kind.
- No SLAM, no calibrated rig extrinsics beyond the per-view pose
  metadata already written by Phase A.
- No ROS integration.

---

## 3. Proposed pipeline

### 3.1 Overview

The Phase B pipeline takes as input the per-view artifacts of Phase A
(RGB image, depth map, per-view metadata JSON) for each piece, and
the equivalent per-view or single-view artifacts for each cavity. It
produces a ranked list of (cavity, rotation, score) tuples per piece,
along with per-view subscores, an aggregate score, a confidence proxy,
and debug overlays.

The pipeline is structured in six stages. Each stage is a pure
function with explicitly typed inputs and outputs; no global state
is shared between stages.

### 3.2 Pipeline stages

**Stage 1 — Per-view descriptor extraction.**
Input: one view's (rgb.png, depth.npy, metadata.json) for a given
piece or cavity.
Output: a descriptor dict containing a fixed set of scalar geometric
descriptors (see section 4).
Determinism: fully deterministic given the input files.
Relation to Baseline 1: this is a new stage; Baseline 1 has no
equivalent because it operates on the fused 2D footprint, not on
per-view crops.

**Stage 2 — Per-view geometric comparison.**
Input: the descriptor dict of a piece view and the descriptor dict
of a candidate cavity view of the same viewpoint type, plus the
rasterised footprint masks (piece and cavity) at the standard
canvas (320 x 320 px, 0.25 mm/px — inherited from Baseline 1,
doc 03 — section 3).
Output: a per-view score vector containing inside_ratio,
outside_ratio, IoU, and any descriptor-level distances (see
section 4) for the optimal rotation found by the existing rotation
search of Baseline 1.
Determinism: the rotation search is a uniform grid (0 deg to
360 deg, 2 deg step, 180 evaluations); fully deterministic.
Relation to Baseline 1: the rasteriser function
`rasterise_xy_to_mask` and the rotation-search loop are reused
unchanged. Only the cavity mask source changes: instead of a single
capture, the cavity mask may come from a matched viewpoint.

**Stage 3 — Score normalisation.**
Input: the per-view score vector from Stage 2.
Output: all score components scaled to [0, 1] so that cross-view
aggregation is not dominated by a single component's numeric range.
Determinism: linear or min-max normalisation against analytically
known bounds (e.g. IoU is already in [0, 1]; descriptor distances
may require clipping or inversion).
Relation to Baseline 1: the Baseline 1 composite score formula
(doc 03 — section 6) is applied per view after normalisation, with
the same weights W_IOU=0.55, W_INSIDE=0.35, W_OUTSIDE=0.10.

**Stage 4 — Cross-view aggregation.**
Input: a vector of K per-view composite scores (one per view, K=3
for the current three-view layout).
Output: a single aggregate score per (piece, cavity) pair.
Determinism: the aggregation function is a closed-form expression
of the input scores (see section 5 for candidates).
Relation to Baseline 1: this stage has no direct counterpart in
Baseline 1. The aggregate score is the Phase B analogue of the
Baseline 1 composite score.

**Stage 5 — Candidate ranking.**
Input: the aggregate scores for all (piece, cavity) pairs.
Output: a ranked list of cavities for each piece, with the top-1
and top-2 scores and their margin (best_score - second_score).
Determinism: rank ordering is by descending aggregate score; ties
are broken by cavity index (as in Baseline 1, doc 03 — section 7).
Relation to Baseline 1: same ranking logic, applied to the aggregate
score instead of the single-view score.

**Stage 6 — Confidence estimation.**
Input: the per-view scores and the aggregate score for the top-1
candidate.
Output: a deterministic uncertainty proxy: e.g. the inverse of the
rank-1 vs rank-2 margin, the view-disagreement entropy, or a
boolean flag for view conflict (when one view disagrees with the
majority assignment).
Determinism: fully deterministic given the input scores.
Relation to Baseline 1: Baseline 1 records a boolean `tie` flag
(doc 03 — section 7) when the margin is below TIE_MARGIN=0.01.
Phase B generalises this to a continuous proxy and an optional
conflict flag.

### 3.3 Architecture diagram

The diagram below shows the pipeline for one (piece x cavity) pair
evaluated across three views. One instance of this diagram runs for
each of the 4 x 4 = 16 pairs; Stage 5 collects all 16 aggregate
scores.

```
  Piece artifacts (Phase A)        Cavity artifacts (Phase A or B)
  -------------------------        --------------------------------
  view_00_top_down/                view_00_top_down/
    rgb.png                          rgb.png
    depth.npy         ---->  [Stage 1: descriptor extraction]
    metadata.json            |
                             |  piece_desc_top      cavity_desc_top
                             +-->  [Stage 2: per-view comparison]
                                      |
                                  score_top  (inside, outside, IoU, ...)
                                  [Stage 3: normalise]  -->  s_top in [0,1]

  view_01_front_oblique/           view_01_front_oblique/
    rgb.png                          rgb.png
    depth.npy         ---->  [Stage 1: descriptor extraction]
    metadata.json            |
                             |  piece_desc_front    cavity_desc_front
                             +-->  [Stage 2: per-view comparison]
                                      |
                                  score_front
                                  [Stage 3: normalise]  -->  s_front in [0,1]

  view_02_side_oblique/            view_02_side_oblique/
    rgb.png                          rgb.png
    depth.npy         ---->  [Stage 1: descriptor extraction]
    metadata.json            |
                             |  piece_desc_side     cavity_desc_side
                             +-->  [Stage 2: per-view comparison]
                                      |
                                  score_side
                                  [Stage 3: normalise]  -->  s_side in [0,1]

                             [Stage 4: cross-view aggregation]
                             aggregate_score = f(s_top, s_front, s_side)

                             [Stage 5: candidate ranking]
                             rank this (piece x cavity) pair

                             [Stage 6: confidence proxy]
                             margin, view_entropy, conflict_flag
```

All per-view scores, the aggregate score, and the confidence proxy
are written to `data/baseline2_geometric_matching/<piece>/vs_<cavity>/`
following the same artefact layout convention as Baseline 1
(doc 03 — section 8).

---

## 4. Candidate descriptor strategies

For each descriptor below, the discussion covers: definition, what
it measures, viewpoint sensitivity, advantages, weaknesses, and
where in the pipeline it would attach.

### 4.1 Contour area

Definition: the number of pixels (or equivalent area in mm^2) inside
the outermost contour of the rasterised mask.
Measures: cross-sectional size of the piece or cavity opening as seen
from a given viewpoint.
Viewpoint sensitivity: high for oblique views, where the projected
area is a foreshortened version of the true cross-section.
Advantages: trivially computed; scale-preserving if the canvas
resolution is fixed.
Weaknesses: does not distinguish shape type; two geometrically
different shapes can have identical area. For oblique views, area
changes with viewpoint angle and is not directly comparable across
views without a geometric correction.
Pipeline attachment: Stage 1 output; used as a normalisation factor
in Stage 2 and as a sanity check on the area_ratio diagnostic
(inherited from Baseline 1, doc 03 — section 6).

### 4.2 Aspect ratio of the bounding box

Definition: max(width, height) / min(width, height) of the
axis-aligned bounding box of the rasterised mask.
Measures: elongation of the piece or cavity footprint.
Viewpoint sensitivity: most informative from the top-down view;
from oblique views the bounding box reflects both the actual shape
and the projection geometry.
Advantages: simple; directly discriminates rectangle from square
(aspect ratio ~1.5 vs ~1.0 for the MVP set).
Weaknesses: insensitive to the difference between a circle and a
square of similar extent (both have aspect ratio ~1.0 from above).
The foreshortening of oblique views adds noise for the lateral views.
Pipeline attachment: Stage 1; used as a secondary comparator in
Stage 2 (distance |AR_piece - AR_cavity|).

### 4.3 Hu moments

Definition: the seven rotation-invariant and scale-invariant moment
invariants computed by `cv2.HuMoments(cv2.moments(mask))`.
Measures: shape content in a rotation- and scale-normalised sense.
Viewpoint sensitivity: the invariance is valid for a planar binary
image; for oblique views the projected shape is not the true
cross-section, so the invariants lose their geometric meaning.
Advantages: compact (7 scalars); well-defined; rotation-invariant,
which may allow bypassing some of the rotation search in Stage 2.
Weaknesses: instability under low-pixel-count masks (observed in
the star's cavity_00 case, doc 03 — section 9); high sensitivity
to mask quality. The log-transform commonly applied (log|h_i|) can
amplify noise when h_i is near zero. For the convex MVP set, several
Hu moments are nearly identical between the square and the circle,
reducing discriminability.
Pipeline attachment: Stage 1; compared by a weighted Euclidean
distance in descriptor space in Stage 2 (before rotation search,
as a pre-filter candidate gate).

### 4.4 Contour similarity via cv2.matchShapes

Definition: the Hu-moment-based shape distance returned by
`cv2.matchShapes(contour_piece, contour_cavity, cv2.CONTOURS_MATCH_I1, 0)`.
Measures: overall shape difference between two contours, invariant
to translation, rotation, and scale (by design of the Hu-basis).
Viewpoint sensitivity: same caveats as Hu moments.
Advantages: produces a single non-negative scalar per pair per view;
zero means identical shape.
Weaknesses: scale invariance is a disadvantage here — two shapes
that differ only in scale (a known discriminator in this project)
are reported as similar. Phase B should use this descriptor only
alongside a scale-sensitive comparator such as area or aspect ratio,
not as a standalone gate. It is also sensitive to contour
fragmentation.
Pipeline attachment: Stage 1 / Stage 2; supplement to area-based
comparison, not a replacement.

### 4.5 Convexity

Definition: mask_area_px / convex_hull_area_px, where
convex_hull_area_px is the pixel count of the convex hull of the
outer contour.
Measures: the fraction of the convex hull that is filled by the
actual mask. A convex shape has convexity = 1.0; concavities reduce
it below 1.0.
Viewpoint sensitivity: moderate; foreshortening changes the apparent
concavity of oblique surfaces.
Advantages: useful for distinguishing the star (convexity << 1.0)
from the convex MVP set (convexity ~= 1.0 for all four).
Weaknesses: for the current main set (all convex) all pieces
saturate near 1.0, making convexity nearly uninformative as a
discriminator. It becomes useful only when non-convex shapes enter
the evaluation.
Redundancy note: for convex shapes, convexity and circularity both
saturate near their respective maxima and carry minimal differential
information relative to each other. Using both simultaneously adds
computation without adding discriminability for the convex MVP set.
Pipeline attachment: Stage 1; included in the descriptor vector but
its weight in Stage 2 should be reduced (or zeroed) when all
candidates are confirmed convex.

### 4.6 Circularity

Definition: 4 * pi * area / perimeter^2, bounded in [0, 1] with
1.0 for a perfect circle.
Measures: how closely the outline approximates a circle.
Viewpoint sensitivity: sensitive to perimeter estimation quality;
fragmented contours produce erroneously low values.
Advantages: directly discriminates circle (circularity ~= 1.0) from
non-circular shapes for clean masks.
Weaknesses: numerically unstable when the perimeter is estimated
from a sparse or fragmented contour; degrades under the pixelation
of a 320 x 320 canvas at piece scales of ~50 mm (approximately
200 px diameter). For convex shapes, circularity and convexity
carry overlapping information — both approach their maxima for a
circle and both are bounded for a polygon. Using both is redundant
for the convex set.
Redundancy note (versus convexity): circularity is
perimeter-sensitive (O(P^2) denominator); convexity is
area-sensitive (ratio of two areas). They are not identical, but
for the convex MVP set they are sufficiently correlated that
including both in a simple weighted sum risks double-counting the
circle-vs-polygon signal. Prefer one: circularity is more
interpretable for this use case.
Pipeline attachment: Stage 1; recommended over convexity for the
convex MVP set.

### 4.7 Occupancy ratios

Definition: two ratios — mask_area / bounding_box_area (bbox
occupancy) and mask_area / convex_hull_area (hull occupancy).
Measures: compactness within the bounding rectangle and within the
convex envelope respectively.
Viewpoint sensitivity: moderate for the bounding-box variant
(foreshortening); low for the hull variant (both numerator and
denominator are affected similarly).
Advantages: bbox occupancy directly discriminates circle from square
(pi/4 ~= 0.785 vs 1.0) and is robust to mild noise.
Weaknesses: bbox occupancy is rotation-sensitive (a tilted rectangle
fills less of its axis-aligned bounding box than an aligned one).
If the rotation search is deferred to Stage 2, Stage 1 occupancy
descriptors are computed at the capture rotation, which may not be
the optimal alignment.
Pipeline attachment: Stage 1; useful as a pre-filter. Hull occupancy
overlaps with convexity (they differ only in normalisaton convention).

### 4.8 Silhouette consistency

Definition: for a given viewpoint (e.g. top_down), the IoU between
the rasterised piece footprint and the rasterised cavity opening
mask, after the rotation search finds the optimal in-plane angle.
This is structurally equivalent to the Stage 2 per-view comparison
but applied to the silhouette contour rather than the solid-filled
mask.
Measures: alignment of the outline, not just the filled area.
Viewpoint sensitivity: directly reflects the viewpoint.
Advantages: sensitive to contour-level mismatches that inside_ratio
and outside_ratio can mask (e.g. a piece that sits inside a larger
cavity with inside_ratio=1.0 but poor outline similarity).
Weaknesses: requires a clean, closed contour; noisy for low-pixel
masks.
Pipeline attachment: Stage 2; an optional complement to the solid
IoU already computed in Stage 2. Not a Stage 1 descriptor.

### 4.9 Recommended starting point

For the convex MVP set, the combination most likely to provide
useful discrimination without redundancy is: aspect ratio (for
rectangle vs square), circularity (for circle vs polygon), and area
(for scale consistency). Contour similarity via `cv2.matchShapes`
is worth including as a compact shape-difference proxy, with the
caveat that its scale invariance must be countered by the area term.
Hu moments are theoretically attractive but may add noise for the
small masks in this setup; they are lower priority for the first
implementation.

This is a starting-point assessment, not a fixed commitment. Final
descriptor selection is a Phase B implementation choice contingent
on the per-view mask quality observed after segmentation.

---

## 5. Multi-view aggregation strategies

### 5.1 Weighted average

Definition: aggregate_score = sum(w_v * s_v for v in views), where
w_v >= 0 and sum(w_v) = 1.
Mathematical form: S_agg = w_top * s_top + w_front * s_front + w_side * s_side.
Advantages: simple; interpretable; allows the top-down view to
receive higher weight (it is the view closest to the validated
Baseline 1 setup); all views contribute.
Failure modes: a single bad view (e.g. the scene intruder in the
top-down view noted in doc 04 — section 1.7, item 2) dilutes the
aggregate by its contribution, but does not block an incorrect
assignment if the bad view's score is high for the wrong cavity.
If all views have similar weights, one noisy view can shift the
aggregate enough to change the rank-1 assignment.
Interpretability: high; each weight has a named geometric rationale.
Suitability for top + front + side: suitable. The top view is
geometrically privileged (same orientation as Baseline 1) and can
receive the dominant weight.

### 5.2 Minimum-score gating

Definition: aggregate_score = min(s_top, s_front, s_side).
Mathematical form: S_agg = min_v(s_v).
Advantages: conservative; a (piece, cavity) pair is only highly
scored if every view agrees. Maximally insensitive to a single
over-optimistic view.
Failure modes: any one view with a genuinely low score for a correct
pair (e.g. a partially occluded oblique view) will suppress the
correct assignment, even if the other two views strongly agree.
Interpretability: high; "the worst view determines viability".
Suitability: appropriate when the three views are equally reliable
and the capture is clean. Risky under the current Phase A setup
where the oblique views have known centring issues (doc 04 —
section 1.7, item 3) and the top-down view has an intruder.

### 5.3 Voting

Definition: each view independently selects the rank-1 cavity; the
aggregate assigns a pair one vote per view that ranks that cavity
first.
Mathematical form: vote_c = |{v : rank_1_v == c}|; S_agg = vote_c / K
(K = 3 views).
Advantages: robust to one view that gives a different rank-1 cavity,
as long as the majority of views agree.
Failure modes: with K=3 views, votes can result in a three-way
split (all different), a 2-1 split, or unanimous agreement.
A three-way split is non-informative. A 2-1 split selects the
majority but discards magnitude information. Ties in voting are
broken by average score, reintroducing the weighted-average problem.
Interpretability: moderate; "majority of views agree on this cavity".
Suitability: works when at least two of three views are reliable
and the cavities are sufficiently distinct. With K=3 it is the
minimum viable voting scheme; adding a fourth view would improve
resilience.

### 5.4 Confidence-weighted fusion

Definition: aggregate_score = sum(c_v * s_v) / sum(c_v), where c_v
is a per-view confidence proxy (e.g. the pixel count of the
segmented piece mask, or the rank-1 vs rank-2 margin per view).
Mathematical form: S_agg = (sum c_v * s_v) / (sum c_v).
Advantages: views with clear geometric signal receive higher weight
automatically; a degraded view receives low weight.
Failure modes: the confidence proxy must be defined without access
to the ground truth. If the proxy is correlated with the error
(e.g. a large mask area does not imply a correct assignment), the
fusion can systematically downweight correct views.
Interpretability: lower than weighted average; the effective weights
are data-dependent and vary per pair.
Suitability: theoretically attractive; practically requires a
reliable per-view confidence signal, which the current Phase A
artifacts do not yet provide explicitly.

### 5.5 View-priority heuristics

Definition: the top-down view dominates; oblique views act as
tie-breakers. Formally: rank by s_top; within ties (margin <
TIE_MARGIN), break by s_front, then s_side.
Mathematical form: lexicographic ordering (s_top, s_front, s_side).
Advantages: preserves the Baseline 1 result for pairs where the
top-down view is unambiguous; oblique views only alter the outcome
when the single-view result was already uncertain.
Failure modes: if the top-down view is systematically biased (e.g.
the intruder object in doc 04 — section 1.6 contaminates the
top-down mask), the heuristic propagates that bias without
correction.
Interpretability: very high; the decision logic is a documented
priority list.
Suitability: conservative and compatible with the Baseline 1
philosophy; a natural starting point before more complex
aggregations are tested.

### 5.6 Comparative analysis

The deterministic, geometry-only philosophy of the project (doc 03
— section 1) favours aggregation strategies with explicit geometric
meaning and predictable failure modes over strategies that adapt
their behaviour to the data in ways that are harder to audit.

On that basis, weighted average with a top-down-dominant weighting
has the strongest theoretical fit: it has a named geometric
rationale per weight, it is closed-form, it degrades gracefully
under one noisy view, and it extends naturally to a larger view set
by adjusting the weight vector. View-priority heuristics are a
close second — they are maximally interpretable and compatible with
Baseline 1 outputs, but they lose magnitude information in the
tie-breaking step.

Minimum-score gating and voting are both worth testing as
sensitivity checks, but neither is recommended as the primary
aggregation given the known capture limitations of Phase A (oblique
centring, intruder).

Confidence-weighted fusion is deferred until a reliable per-view
confidence signal is available.

The final selection among these strategies is a Phase B
implementation decision contingent on the per-view score
distributions observed after segmentation. It is not fixed by
this design document.

---

## 6. Ambiguity analysis

### 6.1 Rectangle vs square

The rectangle's aspect ratio is nominally 75/50 = 1.5; the square's
is 1.0 (doc 03 — section 11). From a clean top-down view at
sufficient resolution, the two are distinguishable by bounding-box
aspect ratio alone. However, the discrimination degrades when:

- the piece is rotated to an orientation that is not aligned with
  the image axes (the axis-aligned bounding box grows for a tilted
  rectangle);
- the viewpoint is slightly oblique, foreshortening one dimension;
- the rasterised canvas pixel count for the shorter dimension of
  the rectangle is small (50 mm at 0.25 mm/px = 200 px; the
  foreshortened version may be fewer).

The front_oblique view (offset along -Y in world coordinates,
doc 04 — section 1.4) will expose the 50 x 75 mm face of the
rectangle as an elongated lateral rectangle, providing a second
independent elongation signal. This is the strongest expected
benefit of multi-view for the rectangle-square pair.

### 6.2 Circle and partial symmetry

The circle has continuous rotational symmetry. Every in-plane
rotation of the circle's footprint is geometrically equivalent.
The rotation search in Stage 2 will find a near-flat score curve
with respect to rotation angle (similarly to Baseline 1, where the
optimal rotation for the circle was reported but is physically
arbitrary). This is not a failure — it correctly represents the
circle's symmetry. The aggregation should not treat the 180
equivalent rotations as distinct best results.

The square has C4 symmetry: rotations of 0 deg, 90 deg, 180 deg,
270 deg are geometrically equivalent. The rotation search will find
four equivalent optima. Neither situation introduces a problem for
the ranking, but both may inflate the effective search space.

### 6.3 Viewpoint degeneracy

A view whose camera axis is aligned with a symmetry plane of the
piece recovers less discriminating information than a view that is
not aligned with any symmetry. The side_oblique view (offset along
+X, doc 04 — section 1.4) observes the piece from a direction
perpendicular to the front_oblique view; for a square piece (equal
in both X and Y) these two oblique views observe structurally
identical cross-sections, providing redundant rather than
complementary information. For the rectangle (unequal X and Y),
the two oblique views observe the 50 mm face and the 75 mm face
respectively, which is complementary. This asymmetry should be
acknowledged when interpreting the aggregated score for the
rectangle-square pair.

### 6.4 Descriptor instability at low pixel counts

Hu moments and circularity can become numerically unstable when the
mask area is small. The star's cavity_00 was documented as having
only 114 raw pixels before resampling (doc 03 — section 9). For the
MVP set at the standard canvas resolution, the smallest expected
mask area is the triangle (estimated at approximately 170 x 200 px
of filled area from a 50 x 50 mm base at 0.25 mm/px), which is
adequate for area and aspect ratio but marginal for higher-order
moments. If the oblique-view masks are smaller (foreshortened), the
instability risk increases.

### 6.5 Missing-view handling

If one view fails to capture (no valid depth pixels, segmentation
returns an empty mask, or the piece is not visible in that view),
the aggregation must degrade gracefully. Proposed conventions:

- for weighted average: treat the missing view as absent and
  renormalise the weights over the remaining views;
- for minimum-score gating: a missing view cannot be scored and
  cannot be the minimum; exclude it and take the minimum over the
  remaining views, with a flag `n_views_used < K` in the output;
- for voting: exclude the missing view from the vote;
- in all cases: record `view_failed = True` and the failure reason
  in the per-pair JSON output.

A missing view does not abort the pipeline; it reduces the
evidence available and increases the reported uncertainty (via the
confidence proxy of Stage 6).

### 6.6 Conditions for Phase B to improve over Baseline 1

Phase B may improve on Baseline 1 when:

- the shape pair being compared has a lateral dimension that differs
  from the top-down silhouette (especially rectangle vs square,
  where the lateral face of the rectangle is elongated);
- the top-down view has a known confound (e.g. the scene intruder
  in doc 04 — section 1.6) and the oblique views are clean.

Phase B is unlikely to improve on Baseline 1 when:

- all views agree on an incorrect assignment (systematic bias, not
  a random per-view error);
- the viewpoints chosen expose the same ambiguity from every angle
  (e.g. two pieces that differ only in height, invisible to a
  camera imaging the top face);
- the per-view masks are all of poor quality due to a shared scene
  defect (e.g. global illumination error, global depth dropout).

---

## 7. Evaluation plan

### 7.1 Comparison against Baseline 1 final state

The reference is the run C result of doc 03 — section 17.4: four
diagonal assignments, all compatible, margins 0.114 – 0.293, all
suspicious_scale = False. Phase B evaluation will report, for each
aggregation strategy and descriptor combination:

- whether the diagonal assignment is preserved;
- the change in aggregate score for each diagonal pair relative to
  the Baseline 1 composite score;
- the change in rank-1 vs rank-2 margin;
- whether any previously compatible pair becomes non-compatible
  or vice versa;
- whether suspicious_scale is triggered for any pair;
- whether the convex_hull_fallback is triggered (it is expected,
  since the rasteriser is unchanged).

### 7.2 Ambiguity reduction analysis

For the rectangle-square pair specifically, the evaluation will
report the rank-1 vs rank-2 margin under each aggregation strategy.
A wider margin than Baseline 1 (0.293 for rectangle, 0.168 for
square) would be evidence that the additional views improve
disambiguation. A narrower margin would indicate that the oblique
views introduce noise. The circle-square pair (Baseline 1 circle
margin 0.114) is the second ambiguity of interest.

### 7.3 Ranking stability — view-dropout test

For each oblique view (front_oblique, side_oblique) independently,
re-run the aggregation excluding that view (simulating a failed
capture). The rank-1 assignment should not change for shape pairs
where the top-down view is already deterministic. A change in
rank-1 assignment on view dropout indicates that the aggregation
is relying on a single oblique view for a critical decision, which
is a fragility worth documenting.

### 7.4 Failure-case catalogue

The following failure conditions are pre-specified for documentation
regardless of whether they occur in the first evaluation run:

- **Intruder contamination**: the scene intruder in the top-down
  view (doc 04 — section 1.6) is included in the piece segment.
  Expected symptom: inflated piece mask area, increased
  outside_ratio, possible change in rank-1.
- **Oblique centring loss**: the side_oblique view has the piece
  off-centre (doc 04 — section 1.7, item 3). Expected symptom:
  reduced pixel count in the piece segment, descriptor instability,
  increased outside_ratio for that view.
- **Depth dropout**: one view returns no valid depth pixels for the
  piece region. Expected symptom: empty mask, Stage 1 outputs
  undefined; aggregation uses n_views_used = K-1.
- **Aspect-ratio inversion**: an oblique view foreshortens the long
  axis of the rectangle to appear shorter than its width, reversing
  the aspect ratio. Expected symptom: aspect-ratio descriptor for
  that view matches the square cavity better than the rectangle
  cavity.

These failure conditions are documented before the evaluation runs
so that any observed anomaly can be attributed to a pre-identified
cause rather than diagnosed post hoc.

---

## 8. Limitations

### 8.1 Deterministic constraints

The pipeline has no adaptive behaviour beyond the configured
trigger constants inherited from Baseline 1 (doc 03 — section 3).
The aggregation weights, the descriptor selection, and the
compatibility thresholds are fixed before execution and are not
updated based on the observed scores. A different piece or a
different scene may require different weight values.

### 8.2 Viewpoint dependence

The three viewpoints are fixed by the Phase A constants
(`TOP_DOWN_HEIGHT = 0.50 m`, `OBLIQUE_HEIGHT = 0.40 m`,
`OBLIQUE_OFFSET = 0.30 m`, doc 04 — section 1.4). The matching
performance will reflect this specific choice. Viewpoints chosen
for the MVP set (convex, axis-aligned pieces) may not be optimal
for other piece geometries.

### 8.3 Handcrafted-descriptor limitations

The descriptors in section 4 are defined analytically from
pixel-level operations. They do not generalise to novel shapes
beyond what their geometric definition captures. A square-like
polygon with slightly rounded corners may not trigger the
circularity threshold intended to separate circles from polygons.

### 8.4 Sequential vs synchronised capture

Phase A uses sequential single-camera relocation (doc 04 —
sections 1.4 and 2.2): a single virtual camera is moved between
three programmatic poses within one script execution. This means
the three views are not simultaneous snapshots of the same scene
state. Any simulation element that changes between captures (e.g.
an object moving slightly, a lighting update) may introduce
inconsistencies between views that would not occur in a true
synchronised multi-camera rig. The intended architecture for Phase
B onwards is multiple static cameras authored in USD (doc 04 —
sections 1.7 and 2.7), which would eliminate this concern. Phase B
results are therefore specific to the sequential-capture regime and
may differ from results obtained on a synchronised rig.

### 8.5 Lack of semantic understanding

The system reasons over 2D geometric descriptors derived from
projected masks. It does not access material properties, topology,
or 3D volumetric geometry. Two shapes that are geometrically
distinguishable in 3D but project to similar 2D footprints from
all three viewpoints cannot be separated by Phase B.

### 8.6 Scalability

Each new piece requires a separate three-view capture round with
per-piece visibility control (doc 04 — section 2.3). Each new
cavity adds another scoring loop at Stage 2. The total number of
Stage 2 evaluations grows as O(N_pieces x N_cavities x K_views x
N_rotations). For the MVP set (4 x 4 x 3 x 180 = 8640 evaluations)
this is tractable, but a larger piece-cavity set requires an
explicit computational budget decision.

### 8.7 Convex geometry assumption

The convex-hull fallback in the rasteriser (Baseline 1, doc 03 —
sections 4 and 17.7) and the descriptor choices recommended in
section 4.9 are calibrated for the convex MVP set. The star piece
was excluded from the main set precisely because the convex-hull
fallback misrepresents its footprint (doc 03 — section 11). Phase B
inherits this limitation unchanged.

---

## 9. Future directions

### 9.1 Learned descriptors

The handcrafted descriptors of section 4 could be replaced by or
combined with learned feature vectors extracted from per-view crops
(e.g. CNN-based or transformer-based spatial descriptors). This
direction directly addresses the limitation of section 8.3
(handcrafted descriptors do not generalise beyond their definition)
and the limitation of section 8.5 (no semantic understanding).
It is out of scope for Phase B and would constitute a distinct
Baseline 3 or a learned component of the system.

### 9.2 Latent embeddings for piece-cavity matching

Deep metric learning on piece-cavity pairs could produce a latent
space where geometrically compatible pairs cluster closer than
incompatible ones, without requiring explicit handcrafted
descriptors. This addresses the same limitations as section 9.1
and additionally provides a natural extension to multi-view
aggregation (view embeddings can be pooled in the latent space).
This is the longer-term learned approach referenced in the project
plan and is explicitly out of scope for the deterministic baseline
phases.

### 9.3 Geometric feature learning from supervised or
self-supervised signals

Supervised pairing labels (compatible vs incompatible, possibly
generated by the deterministic baseline itself as a teacher signal)
could drive a feature learning stage that identifies which 2D
geometric cues are most predictive of compatibility. Self-supervised
signals (e.g. reconstruction of the cross-section from multiple
views) could provide representation learning without manual labels.
This direction addresses the scalability limitation of section 8.6
by amortising the per-pair matching cost into an embedding lookup.

### 9.4 True static-multi-camera synchronised capture

Replacing the sequential single-camera relocation with multiple
static cameras authored in USD (already identified in doc 04 —
sections 1.7 and 2.7 as the intended architecture) directly
addresses the limitation of section 8.4. Joint extrinsic
calibration between views would also enable geometric consistency
checks across views (e.g. verifying that the observed piece centre
is consistent across cameras), which the current sequential setup
cannot support.

### 9.5 Probabilistic matching with uncertainty propagation

The confidence proxy of Stage 6 is currently a deterministic scalar
(margin, view entropy, conflict flag). Replacing it with a formal
probabilistic model (e.g. a Bayesian update over cavity hypotheses
as each view is observed) would allow uncertainty to propagate
through the pipeline in a principled way. This addresses the
limitation of section 8.1 (no adaptive behaviour) while remaining
interpretable.

### 9.6 Downstream robotic integration

Perception-to-motion-planning integration — using the Phase B
output (cavity ID, optimal rotation, compatibility score) as input
to a pick-and-place or insertion planner — is explicitly out of
scope for the current thesis phase. It is recorded here as a future
direction that depends on Phase B producing reliable, low-latency
cavity and rotation estimates, and on a separate insertion policy
or trajectory generator being available. Any claim of readiness for
robotic execution must be preceded by an end-to-end validation that
Phase B does not attempt.
