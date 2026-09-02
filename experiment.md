# Tracking and mask flicker experiments

The remaining flicker can come from two different parts of the pipeline:

1. **Geometry flicker:** the weather map jumps, scales, or warps for one or a
   few frames because the tracked TV polygon briefly moves incorrectly.
2. **Mask flicker:** pixels shimmer around the presenter or green-screen edge
   while the tracked polygon itself remains stable.

These need separate experiments. Chroma-assisted tracking can help with the
first problem, but it does not directly stabilize noisy pixel classification
at the foreground boundary.

## Current baseline

The TAPNext++ stage already follows this pipeline:

```text
raw TAPNext points
  -> RANSAC geometry estimation
  -> confidence and geometry validation
  -> temporal outlier rejection
  -> gap and off-screen recovery
  -> corner-trajectory smoothing
  -> fresh homography per frame
```

It tracks 32 points across the TV plane rather than relying on four raw corner
tracks. It rejects low-inlier and invalid quadrilaterals, rejects abrupt corner
displacement and area changes, interpolates rejected frames, and smooths the
four resolved corner trajectories with a five-frame median and bidirectional
EMA. Homographies are calculated after the corners are stabilized; homography
matrices are not averaged directly.

The OpenCV mask stage already measures green coverage inside each tracked
polygon. That value is currently diagnostic only and does not influence the
tracking result. It also has an optional soft-boundary mode, but each frame is
still classified independently.

## Experiment 1: chroma-assisted geometry arbitration

### Problem

A single-frame or short-run tracking outlier can make the complete weather map
jump even when the surrounding trajectory is correct. Because a homography is
sensitive to its corners, a small geometry error can be visually obvious.

### Hypothesis

When the existing temporal checks identify suspicious geometry, the visible
green pixels can act as an independent measurement and help choose between the
raw tracked quad and a temporally predicted or interpolated quad.

Chroma should be a conservative tie-breaker, not a tracker on its own. Low
green coverage may be legitimate when the presenter, motion blur, a shadow, or
another foreground object obscures the screen.

### Proposed method

For every frame already marked suspicious by tracking or geometry validation:

1. Keep the raw TAPNext-derived quad as one candidate.
2. Produce a second candidate from the surrounding good frames using the
   existing interpolation or motion prediction.
3. Measure chroma agreement for both candidates against the source frame.
4. Select the predicted candidate only when its chroma agreement is better by
   a deliberately conservative margin and its temporal and polygon geometry
   remain valid.
5. Otherwise retain the existing TAPNext result.

The comparison should consider both green pixels captured inside the candidate
polygon and candidate area unsupported by green. Absolute green coverage alone
is not sufficient because foreground occlusion can lower it for a valid quad.

```text
tracking-suspect frame
  -> raw quad + predicted quad
  -> temporal and polygon validity checks
  -> comparative chroma agreement
  -> conservative arbitration
  -> stabilized corners
  -> homography
```

### Evaluation

Use a fixed set of clips containing known one-frame or short-run map jumps.
Compare the current tracker with chroma arbitration enabled and disabled.

Record:

- which frames triggered arbitration;
- the raw and predicted chroma scores and winning margin;
- maximum corner displacement and polygon-area change;
- whether the visible map jump was removed;
- any valid camera movement incorrectly replaced;
- any failure caused by foreground occlusion or low green coverage.

Success means obvious isolated geometry jumps are removed without adding lag,
rejecting genuine camera motion, or degrading off-screen recovery.

### First evaluation: 2026-08-28

The first score-only evaluation used a 124-frame green-display clip with a
presenter, a strong camera push-in, and partial off-screen geometry. The
unchanged TAPNext++ V2 baseline produced only 30 direct-good frames and 94
recovered frames. Its maximum corner displacement was 197.4828 pixels on frame
123, where the review evidence shows a malformed quad.

A chroma projection calibrated from a trusted direct-good frame visually
followed the display better during much of the late zoom. However, automatic
scoring did not pass the safety bar. It detected 162 of 240 controlled
perturbations (67.5%), missed every tested bottom-right corner spike, usually
missed downward translations, and could still prefer malformed raw quads that
enclosed large amounts of green.

The result supports chroma as useful independent evidence but rejects green
coverage or component overlap as a sufficient arbitration rule. The next
iteration must add coherent-transform and temporal candidate gates, calibrate
the inset between the outer bezel and green surface, score edges separately,
and abstain when presenter occlusion truncates the measured green component.
Production coordinates remain unchanged after this evaluation.

### Second evaluation: 2026-08-28

The second iteration implements the four required safeguards. It calibrates a
normalized bezel-to-green inset from robust direct-good samples, scores the
four edges independently, gates consecutive candidates with a coupled
similarity transform and constant-velocity prediction, and refuses a candidate
when fewer than two edges or too little connected-component support survives.
After any rejection gap, re-acquisition is stricter: all four edges, at least
80% precision, and a score of at least 0.75 are required.

On the same 240 controlled perturbations, the new evaluator detected 230
(`95.83%`), compared with 162 (`67.5%`) in the first pass. It selected 52
chroma-projection frames, selected no raw candidates, and abstained for 42
frames during destructive presenter occlusion. The four-edge gate reacquired
the chroma trajectory at frame 92.

Abstention alone was not temporally safe: switching to baseline at frame 50
and back at frame 92 would introduce 107 px and 231 px corner jumps. The
offline applied result therefore ignores all chroma proposals inside the
abstention interval and bridges the baseline-relative correction between its
trusted endpoints. A six-frame blend eases the first baseline-to-chroma
handoff. This produces 30 baseline, 6 transition-blend, 42 occlusion-bridge,
and 46 directly accepted chroma frames.

The applied experimental coordinates reduce maximum corner displacement from
`197.4822 px` at frame 123 to `23.5346 px` at frame 34, and maximum polygon-area
change from `17.30%` to `5.75%`. Visual review confirms that the broken frame-50
component is never used and that the frame-123 result follows the display
instead of the malformed baseline quad.

The remaining 10 controlled misses and the need for a future anchor during an
occlusion bridge keep this out of production for now. Next validation should
use several clips with different presenter poses, screen colors, camera moves,
and occlusion lengths, with thresholds fixed before evaluation.

### Weather-composite validation: 2026-08-28

The diagnostic tracking review overstated the visible failure because it drew
the bad baseline and rejected chroma candidates alongside the applied result.
To test the output that viewers would actually see, the iteration-2 corners
were retargeted to the weather app's 900×1280 `WeatherScreenInsert` surface and
rendered through `StudioWeatherCompositeMask` with a lossless 124-frame OpenCV
replacement mask.

The resulting 768×1376, 24 fps composite passed its complete 124-frame media
contract. Four overview sheets covering every frame show the weather remaining
inside the bezel, following the push-in, and staying behind the presenter. No
geometry jump comparable to the original frame-123 baseline failure is visible
in the actual composite. This makes the current geometry result materially
better than the multi-candidate diagnostic video suggested.

The composite does expose green contamination and hard transitions around some
presenter edges. That is a replacement-mask boundary problem addressed by
Experiment 2; it is distinct from projective tracking. The single-clip and
future-anchor limitations still apply, so the result remains experimental even
though this clip's final composition is visually coherent.

An apples-to-apples Remotion render was also produced from the untouched
`coordinates-baseline.json`, using its own polygon-derived replacement mask.
The original baseline and assisted render are similar in the early direct-good
section. During the late push-in, the baseline increasingly exposes green
wedges and misplaces the weather surface, culminating in its known frame-123
deformation; the assisted version remains closer to the bezel. A slowed
side-by-side review preserves all 124 frames with baseline on the left and
iteration 2 on the right. If an earlier production render looked better than
this saved baseline, it came from a different tracking artifact and should be
compared separately.

### Controlled-source H3 trial: 2026-08-28

A no-reference five-second H3 clip was generated specifically for easier
tracking (`2093447850816499714`). The prompt requested an empty studio, a rigid
fully visible display, slow controlled camera motion, and a fixed asymmetric
green-on-green feature pattern. The verified result is 768×1376, 24 fps, and
124 frames.

The first frame, last frame, one-frame-per-second sheet, and a denser sample at
five-frame intervals show a complete rectangular bezel with four visible
corners and a highly trackable pattern that appears stable in the sampled
frames. H3 did not fill the rectangular active display as requested: it made
the green texture a large rounded, pixel-stepped shape surrounded by black
inside the bezel. That makes this candidate useful as a tracking-pattern proof
of concept but unsuitable for the current full-screen chroma replacement
pipeline. It was not used as the source for a tracking run.

After explicit authorization, a second no-reference candidate used a much
simpler request: a normal studio television whose active surface is filled
edge-to-edge with semi-dark green, relying on natural illumination for texture
(`2093449726949421058`). The verified result is also 768×1376, 24 fps, and 124
frames. Its first and last frames show the complete rectangular active area and
all four bezel corners. The one-second sheet and a denser sample at five-frame
intervals show no obvious cut, bezel deformation, or loss of rectangular fill.
This candidate passed the sampled visual preflight and was then run through the
unchanged TAPNext++ V2 tracker. Frame-zero display detection scored `0.978642`
combined confidence and correctly selected the active green surface.

Continuous tracking was dramatically cleaner than on the presenter clip. All
`124/124` frames were direct-good, with no recovered, bridged, held,
prediction-only, or invalid frames. The tracker retained 31–32 of its 32 inset
edge queries on every frame. Mean reprojection error was `0.6148 px`, p95 was
`1.0577 px`, maximum per-frame corner displacement was `1.7743 px`, and maximum
per-frame area change was `0.6412%`.

The low per-frame motion is not caused by a frozen result: the four corners
move `56.2–68.4 px` from start to finish and the tracked display area grows by
`35.06%` during the push-in. Full-frame review shows the quad following the
active-display boundary through frame 123. That last frame is flagged as a QA
suspect only because it contains the largest relative step in an otherwise
very stable sequence; it is visually aligned.

Compared with the earlier presenter's `30/124` direct-good frames, 94 recovery
frames, and malformed `197.48 px` final step, this result confirms that source
control can remove most of the tracking difficulty without changing the
tracker or adding more points. It does not prove that semi-dark green variation
alone is responsible: this clip also keeps the screen rigid and fully visible,
uses coherent mild camera motion, and contains no presenter occlusion. A fair
next stress test should preserve this display design while adding controlled
partial occlusion.

### Harder controlled-source stress trial: 2026-08-29

Two authorized no-reference H3 generations attempted to combine an opaque
left-to-right presenter crossing with a rightward camera pan that pushes half
the display outside the frame and then returns to center. H3 did not satisfy
both variables in one source. The five-second candidate
(`2093454879979270146`) produced destructive opaque presenter occlusion but no
meaningful camera pan. The ten-second retry (`2093456417573138434`) produced a
real partial-offscreen pan and return, but changed the screen to landscape and
placed the presenter behind the green surface as a tinted silhouette.

The two clips were retained as separate stress tests so opaque occlusion and
partial-offscreen motion could be measured independently with unchanged
TAPNext++ QA V2. Frame-zero detection correctly selected the active surface in
both cases, with confidence `0.983781` and `0.975113` respectively.

The opaque-occlusion clip produced `63/124` direct-good and 61 recovered frames,
including 52 partial-affine and 9 bridged frames. Maximum per-frame corner
movement was `5.1584 px`, maximum area change was `0.3570%`, and p95
reprojection error was `1.2812 px`. Full review shows a coherent recovered quad
through the presenter, including the maximum-step frame. However, direct
tracking never returns after frame 63; prediction age reaches 60 at the end.

The partial-offscreen pan clip produced `167/243` direct-good and 76
partial-affine frames, with no bridging, invalid geometry, prediction-only, or
held frames. Maximum per-frame corner movement was `8.4027 px` on the genuine
return pan, maximum area change was `0.9218%`, and p95 reprojection error was
`1.5032 px`. Frames 116–144 use recovery during the pan and direct tracking
re-acquires at frame 145. A later terminal recovery interval spans frames
206–242 and reaches prediction age 37 despite the display being visible.

Every frame in both all-frame review videos was checked. Neither run contains a
malformed geometry jump comparable with the original `197.48 px` failure. The
important remaining weakness is long-horizon re-acquisition: recovery can stay
visually plausible for 37–60 frames without a new direct anchor, but it is not
safe to extrapolate that behavior through continued camera motion. A production
policy should require strong re-acquisition or a future anchor rather than
treating a long partial-affine interval as indefinitely trustworthy.

The longer run also exposed a review-evidence bug: its coordinates and review
video correctly contain all 243 frames, but the built-in overview artifact list
still stops after four 30-frame sheets at frame 119. Continuation sheets for
frames 120–242 were generated locally from the checksum-verified review video.
The TAPNext++ Modal app was updated and deployed on 2026-08-29 to remove that
fixed four-sheet assumption. New runs accept up to 720 frames, emit one overview
sheet for each 30-frame block, and correctly name and pad the final partial
sheet. The exact simultaneous opaque-occlusion plus half-offscreen test remains
unmeasured because neither source met the complete visual contract.

The 243-frame candidate was then replayed through that deployed image under the
fresh run ID `controlled-green-occlusion-v1-candidate-2-qa5` and input hash
`38faed6dfbe2401bff8096dbf5434ac4983a3046fb2886f064053db268b05828`.
The deployment reported implementation suffix
`qa5-dynamic-evidence-offscreen-v2`. Its coordinates, metrics, and all-frame
review video are byte-for-byte identical to the QA4 artifacts, confirming that
the deployment did not alter geometry. The server produced nine overview
sheets covering every frame from 0 through 242; all were inspected, and the
last sheet correctly contains frames 240–242 followed by black padding. No
malformed jump is visible. This verifies the longer-run evidence fix end to end,
while leaving the terminal 37-frame partial-affine re-acquisition caveat
unchanged.

That deployed QA5 replay did **not** exercise Experiment 1's chroma-assisted
code: the production stage does not import `chroma_arbitration.py`. A separate
offline iteration-2 replay was therefore run on the QA5 coordinates and all 243
source frames. It proposed 206 baseline, 24 chroma-projection, and 13 raw
frames, rejected 2 chroma and 63 raw candidates, and recorded no abstentions.
The lack of abstention is expected for this source mismatch: the presenter is
rendered behind the green display rather than destructively occluding it.

After stabilization, 206 frames remain baseline, 21 are transition blends, 13
use raw TAPNext geometry, and 3 use direct chroma projection. All nine applied
overview sheets were inspected. The result remains visually attached to the
display, but candidate alternation during frames 206–241 makes it temporally
worse than the already-coherent baseline. Maximum corner displacement rises
from `8.4026 px` to `19.7161 px`, maximum polygon-area change rises from
`0.9218%` to `3.3669%`, and eight transitions exceed the baseline's maximum
step. This replay therefore rejects iteration 2 as an improvement on this
clip. The next arbitration revision needs a baseline-is-already-good/no-switch
gate and coherence across raw/chroma source handoffs, not only within each
candidate trajectory.

### Third evaluation: simpler repair-only arbitration, 2026-08-29

A smaller arbitration policy was tested on both the original positive case and
the newer stable 243-frame control. It keeps baseline geometry by default,
ignores raw TAPNext candidates, and considers chroma only when all of the
following persist for at least three frames: the baseline is already marked
suspicious, the existing chroma evidence and trajectory gates accept the
candidate, chroma beats baseline by the fixed `0.04` score margin, and their
maximum corner disagreement exceeds `2.5%` of the source diagonal (`39.40 px`).
Finally, a qualifying segment must contain an actual baseline movement failure
above `1.5%` of the diagonal (`23.64 px`); large chroma disagreement alone does
not prove that baseline is wrong. Repair segments use an adaptive smoothstep
handoff. There is no future-anchor occlusion bridge.

On the original 124-frame failure, two sustained chroma-win segments are found,
but only frames 92–123 contain a baseline failure and become a repair. The
applied result contains 92 baseline, 19 repair-blend, and 13 direct
chroma-repair frames. Maximum corner displacement falls from `197.4822 px` to
`21.9076 px`, and maximum polygon-area change falls from `17.30%` to `4.85%`.
All 124 frames were inspected; the frame-123 malformed baseline quad is removed
and the applied outline remains coherent. This is smoother than iteration 2's
`23.5346 px` maximum and no longer depends on an offline future-anchor bridge.

The opaque left-to-right presenter clip is the decisive second negative
control. Its evaluator abstains on 29 destructive-occlusion frames and finds
two later chroma-win segments, but the baseline's maximum movement is only
`5.1582 px`. The baseline-failure gate rejects both repairs, returning all 124
corner arrays exactly. Full visual review confirms unchanged coherent geometry
through the presenter crossing.

On the 243-frame stable control, zero frames satisfy the material-disagreement
gate. The policy returns all 243 baseline corner arrays exactly, preserving the
`8.4026 px` maximum movement and `0.9218%` maximum area change. This resolves
the regression caused by broad multi-candidate arbitration: chroma now acts as
a sustained failure repair, not as a generally competing tracker. The result
is still experimental because the thresholds have only been exercised on one
positive clip and two negative controls.

To eliminate clip-selection ambiguity, the 243-frame negative control was run
once more in a fresh `latest-h3-simple` namespace from a newly downloaded S3
object, not from the old local candidate path. The object key is
`runninghub/h3/h3-ref-to-video-2093456417573138434.mp4`, and its verified
SHA-256 is
`1d49acc0c01e8694b73e5afe4d08c4213642e82af67ad85f2841bbe177f7e9fd`.
The fresh evaluator again found no repair segment. All 243 output corner arrays
match the QA5 baseline exactly, and the newly rendered all-frame review is
byte-for-byte identical to the earlier checksum-matching review.

### Baseline weather composition on the pan clip: 2026-08-29

Because the 243-frame partial-offscreen run has no malformed geometry jumps,
its unchanged QA5 baseline was used for an end-to-end weather-map composition.
The coordinates were retargeted to the weather app's 900×1280
`WeatherScreenInsert` surface and rendered through
`StudioWeatherCompositeMask` with the standard binary HSV replacement mask.

The resulting video stream is 768×1376 at 24 fps and contains all 243 source
frames. Nine overview sheets covering frames 0–242 show stable placement
through the rightward pan, the half-visible display interval, the return to
center, and the terminal recovery interval. No visible map jump or detachment
from the bezel was found. The presenter in this H3 generation walks behind the
green display, so its green-tinted silhouette is part of the replaced screen
content and correctly disappears in the weather composite; this is not a
foreground-occlusion test.

The normal-speed result is
`experiments/controlled-green-occlusion-v1/latest-h3-simple/results/weather-composite-baseline.mp4`.
An all-frame 6 fps review and the nine contact sheets are stored beside it.

Full-resolution playback subsequently exposed a localized defect that the
downscaled sheets hid. At frame 206, direct tracking drops from 20/32 inliers
to 18/32 (`56.25%`), below the 60% acceptance threshold, and all eight
right-edge queries are rejected. The QA fallback enters partial-affine recovery
for frames 206–242. It remains temporally smooth, but a similarity transform
cannot independently correct the lower-right perspective; the resolved edge is
approximately 18 px above the measured green boundary. This leaves a visible
green wedge in the weather composite.

A narrow offline tail-recovery policy was tested in response. It requires a
terminal non-direct run plus three consecutive frames with four supported
chroma edges, score ≥0.90, precision ≥0.95, and a score advantage of at least
0.04. Raw candidates are ignored. After confirmation it performs one
six-frame handoff and stays on the calibrated chroma-projection trajectory,
avoiding the raw/chroma alternation that degraded the broader iteration-2
result.

The rule finds the terminal run at frame 206, confirms evidence at frame 208,
starts its offline blend at frame 203, and remains fully latched for frames
208–242. Maximum corner movement and polygon-area change remain equal to the
baseline (`8.4026 px` and `0.9218%`). Source-green pixels that still match the
original plate fall by `20.06%` over frames 206–242, from 41,143 to 32,888.
This source-retention measurement replaces the earlier HSV-only count, which
could also classify greenish pixels in the weather artwork. Full-resolution
comparison confirms that the large lower-right wedge is removed, although a
thin calibration border remains. This is a successful repair for this clip,
not yet a general production policy.

The corrected composition is
`experiments/controlled-green-occlusion-v1/latest-h3-simple/results/weather-composite-chroma-tail.mp4`.

A weather-only overscan variant then expanded every destination edge outward
by 5 source pixels while keeping the corrected tracking and replacement mask
unchanged. The black bezel remains protected by foreground alpha, and the thin
left/bottom calibration border is visibly covered at full resolution. Matching
source-green pixels in frames 206–242 fall a further `1.66%`, from 32,888 to
32,341 (`21.39%` below the original baseline). Every overscanned tail frame was
inspected without visible spill beyond the bezel. The preferred render is
`experiments/controlled-green-occlusion-v1/latest-h3-simple/results/weather-composite-chroma-tail-overscan5.mp4`.

The narrow tail policy is now implemented in `tapnextpp-modal-app` as the
opt-in `parameters.chromaTailRecovery` QA V2 setting. The production path uses
the decoded source frames to calibrate the bezel-to-green inset, runs the same
four-edge score plus coherent-transform and temporal gates, and publishes the
final corners and homographies only when the full terminal-tail latch passes.
It records calibration, per-frame evidence, decision reason, source, blend
weight, and recomputed final-geometry metrics. The default is disabled;
destructive occlusion or discontinuous post-confirmation support leaves the
baseline geometry unchanged.

### Confidence-weighted per-edge fusion: 2026-08-29

The successful tail latch treats the calibrated chroma projection as one
all-or-nothing quadrilateral. A follow-up offline experiment instead represents
each top, right, bottom, and left observation as a line with its own confidence.
Confidence is the calibrated candidate precision multiplied by the geometric
mean of that edge's directional score and unambiguous support. Adjacent line
observations act on shared corner variables, so the solver still produces one
coherent quadrilateral. The TAPNext++ result remains a point prior, and a
temporal term smooths only the correction relative to that moving baseline.

The safety contract stays deliberately narrower than the fusion mathematics.
It uses the same terminal recovery, three-frame acquisition, score, precision,
and six-frame offline handoff as the latch. All four edges must remain supported
through the end; loss of one edge causes whole-tail abstention rather than a
temporal guess through destructive presenter occlusion. Raw candidates remain
ignored. Final correction magnitude, frame movement, polygon area change, and
convexity are gated before coordinates are published.

An initial `8:1` chroma-to-baseline weight with temporal weight `12` reduced
matching source-green pixels by `13.30%`, but left the bottom edge a few pixels
conservative. A bounded parameter check found `24:1` with temporal weight `8`
to be the better balance. It brings the mean frame-208–242 corner distance to
the calibrated chroma trajectory down to `0.446 px`, and the frame-206
bottom-right residual to `1.378 px`, without exceeding any baseline motion
metric.

On the 243-frame weather render, frames 206–242 fall from 41,143 matching
source-green pixels at baseline to 33,577, an `18.39%` reduction. The full
chroma latch remains slightly better at 32,888 pixels (`20.06%`); edge fusion is
only `2.09%` above it. Full-resolution frame 206 shows the large lower-right
wedge removed, and frame 239 is visually equivalent to the latch. Maximum
corner movement and polygon-area change remain exactly at the baseline limits
of `8.4026 px` and `0.9218%`; the largest correction from baseline is
`19.4356 px`.

The 124-frame opaque-presenter control briefly meets acquisition at frames
101–103, but later discontinuous edge support triggers the hard occlusion rule.
The result contains 124 baseline frames, zero fused frames, and a `0.0 px`
maximum coordinate difference from baseline. This is a successful conservative
alternative to the full latch on the two available cases, but it does not beat
the latch on the positive clip and remains offline pending broader validation.

Implementation, synthetic tests, reports, and the three-way weather review are
in `experiments/chroma-arbitration-v1/apply_edge_fusion.py`,
`experiments/chroma-arbitration-v1/test_apply_edge_fusion.py`, and
`experiments/controlled-green-occlusion-v1/latest-h3-simple/results/`.

### Hybrid tracking-point layout: 2026-08-29

The same verified 243-frame clip was rerun with an opt-in
`hybrid-24-edge-8-interior` query layout: six points near each TV edge plus a
projected 4×2 interior lattice. Chroma tail recovery was disabled, so the only
experimental variable was the location of the 32 TAPNext++ plane queries. The
run ID is `controlled-green-occlusion-v1-hybrid-32`, and the source SHA-256 is
unchanged.

The hybrid layout regressed rather than improving the original failure.
Direct frames fell from 167 to 138, recovered frames rose from 76 to 105, and
the longest recovery interval expanded from frames 206–242 to frames 141–242.
Temporal rejects increased from 60 to 92. Maximum corner movement rose from
`8.4027 px` to `12.5869 px`, and maximum polygon-area change rose from
`0.9218%` to `2.7160%`.

Frame 206 remains at exactly 18/32 inliers (`56.25%`) in both layouts, so the
hybrid does not fix the right-edge loss. It is substantially worse
geometrically: the hybrid and perimeter results differ by up to `181.222 px`
at frame 206, and the full-resolution hybrid outline extends far outside the
top and bottom bezel.

The interior points remained visible and frequently joined RANSAC: their
inlier rate is `80.09%` overall and `70.47%` over frames 141–242. This is
misleading evidence rather than missing evidence. On the mostly uniform green
surface, interior points can support a coherent but incorrect projective fit;
reducing each edge from eight to six points also removes useful boundary
leverage. The raw hybrid fit repeatedly fails temporal consistency after frame
140 and never returns to direct geometry.

The decision is to keep `perimeter-32` as the production default. The hybrid
layout remains available only as an explicit experiment option. Results and
the four-times-slower perimeter-left/hybrid-right review are in
`experiments/controlled-green-occlusion-v1/hybrid-32/`.

### Additive 40-point layout: 2026-08-29

A fourth layout experiment retained all 32 original perimeter points and
appended the same eight projected interior points. Queries 0–31 are exactly
identical to the baseline, and chroma recovery remained disabled. This tests
whether RANSAC can ignore ambiguous interior points normally and use them only
when an edge disappears.

The additive layout is less damaging than replacing edge points, but it still
regresses. Direct frames fall from 167 to 144, recovered frames rise from 76
to 99, and the terminal partial-affine interval expands from frames 206–242 to
frames 145–242. Temporal rejects increase from 60 to 86. Maximum corner
movement rises from `8.4027 px` to `11.9174 px`, and maximum area change rises
from `0.9218%` to `2.6007%`.

Frame 206 diagnoses the acceptance question cleanly. The additive fit has
24/40 inliers (`60.0%`): 20 perimeter points and four interior points. It
therefore passes the existing inlier-ratio threshold but is still classified
`temporal_reject`. The failure is the geometry selected by RANSAC, not the
larger denominator. Its resolved corners differ from the perimeter baseline by
up to `168.213 px` at frame 206 and visibly extend beyond the top and bottom
bezel. Maximum disagreement reaches `186.422 px` at frame 223.

Interior points remain fully visible and join the RANSAC consensus `77.98%`
of the time overall (`67.03%` over frames 141–242). On this mostly uniform
green fill, the extra observations are plentiful but systematically
ambiguous. The production decision remains `perimeter-32`; both fixed-grid
interior layouts remain explicit experiment options only. Full results are in
`experiments/controlled-green-occlusion-v1/additive-40/`.

## Experiment 2: temporal mask stabilization

### Problem

Even with stable geometry, compression noise, shadows, motion blur, and small
HSV changes can make pixels alternate between replace and preserve on adjacent
frames. This appears as shimmer or flicker around the presenter or screen
boundary. Geometry arbitration will not fix it.

### Hypothesis

A small amount of temporal consistency applied only to ambiguous boundary
pixels can remove isolated mask flashes while preserving confident foreground,
confident green regions, and real motion.

### Proposed method

Start with a conservative offline three-frame filter:

1. Generate the existing hard mask and continuous soft-green score for each
   frame.
2. Leave confident black and confident white pixels unchanged.
3. Restrict temporal processing to ambiguous pixels near the hard foreground
   boundary inside the tracked polygon.
4. Use the previous and next frames to reject isolated one-frame mask changes,
   or apply hysteresis with separate enter-green and leave-green thresholds.
5. Keep pixels outside the visible tracked polygon black, including fully
   off-screen frames.

The first version should avoid global temporal smoothing. Broad averaging can
create trails behind a moving presenter or delay real mask changes.

```text
stable tracked polygon + source frames
  -> per-frame HSV masks
  -> identify ambiguous boundary pixels
  -> three-frame temporal vote or hysteresis
  -> lossless grayscale mask video
```

### Evaluation

Use clips with stable tracking but visible edge shimmer, including fast hand
movement, motion blur, partial occlusion, and partial/off-screen TV geometry.

Record:

- the number of pixels changed by temporal stabilization per frame;
- isolated black-to-white-to-black and white-to-black-to-white transitions;
- mask-edge temporal variation;
- any visible trails, delayed motion, or lost translucent detail;
- behavior on partially and fully off-screen frames;
- hard-mask and soft-boundary results separately.

Success means visibly reduced boundary shimmer without trails, temporal lag,
changes outside the tracked polygon, or violations of the existing mask-video
contract.

## Recommended order

Run the experiments independently so their effects are measurable:

1. Evaluate Experiment 1 on clips where the complete map visibly jumps.
2. Evaluate Experiment 2 on clips where geometry is stable but mask edges
   shimmer.
3. Enable both only after each improvement beats the current baseline on its
   own test set.
