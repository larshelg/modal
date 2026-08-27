The one piece I'd steal is this:

raw TAPNext track → temporal validation → gap/outlier repair → smoothing → homography

rather than:

raw TAPNext track → homography

And because you have the colored TV, you can make that validation stronger than WanGP's face tracking: use the chroma pixels as an independent measurement of whether the TAPNext geometry still makes sense.

That might be the small improvement worth experimenting with—not a new tracker, SAM, DINO, or another model.

You are spot on here. My soliton worked well, but I had some frames where the weather map were jumping a bit

Exactly. That kind of occasional map jump is probably the easiest remaining problem to fix without changing the tracking system.

What you likely have is not a persistent tracking failure, but a single-frame or short-run homography outlier. TAPNext gives you four corner trajectories, and one point moves slightly wrong for 1–3 frames. Because a homography is sensitive to its corners, a small corner error can produce a surprisingly visible jump in the inserted map.

I’d put a tiny temporal stabilization stage between TAPNext and the homography:

TAPNext raw points → outlier detection → interpolate bad points → light smoothing → homography → Remotion

The important part is not to aggressively smooth everything. That could make a genuine camera pan or zoom lag behind the TV. Instead, first detect implausible deviations.

For each tracked point, you already have positions over time:

p(t-2), p(t-1), p(t), p(t+1), p(t+2)

From those you can predict roughly where p(t) should be based on its neighbors. If the actual point suddenly differs by, say, 8–15 pixels while the surrounding frames form a smooth trajectory, mark that observation as an outlier and replace it with interpolation.

Even better, do this at the TV geometry level rather than simply looking at individual point distance. Check things such as:

sudden change in TV polygon area,
sudden change in edge lengths,
unusual corner acceleration,
homography movement relative to previous frames,
disagreement with the visible chroma region.

The chroma check is especially valuable because it's an independent measurement. Imagine TAPNext says the screen shifted 12 pixels right for one frame, but the green TV pixels are still overwhelmingly where the previous trajectory predicts them. That's very strong evidence that TAPNext had a one-frame wobble.

You can therefore assign each frame something like:

confidence = tracking_temporal_consistency × polygon_consistency × chroma_agreement

For high-confidence frames, use TAPNext exactly as-is.

For one low-confidence frame surrounded by good frames:

frame 99 good → frame 100 BAD → frame 101 good

replace frame 100 with an interpolation between 99 and 101.

For a slightly longer problem:

99 good → 100 bad → 101 bad → 102 good

interpolate the two bad frames across the gap.

Only if the bad period lasts substantially longer would you need an actual re-track/recovery mechanism.

There is one additional trick I'd test: smooth the corner trajectories rather than the resulting homography matrix. Homography matrices aren't particularly nice quantities to average directly because their coefficients interact nonlinearly. Stabilize the four (x,y) trajectories first, then calculate a fresh homography from the stabilized corners for every frame.

So you end up with something very small:

TAPNext
   ↓
4 corner trajectories
   ↓
outlier rejection
   ↓
short-gap interpolation
   ↓
very light temporal smoothing
   ↓
H(t)
   ↓
weather map

And I'd start extremely conservatively. Fix only obvious spikes first. Your current tracking already works well, so the goal isn't to make TAPNext's path artificially silky — it's simply to remove those “why did the map jump for one frame?” moments.

This is also where WanGP's implementation gave us the useful idea: it doesn't assume a tracker's raw temporal output is production-ready. Tracking and stabilization are two separate stages. I think adding that distinction to your pipeline could remove most of the remaining visual glitches with surprisingly little code.