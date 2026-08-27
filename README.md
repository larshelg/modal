# Modal apps

This checkout contains independently deployable Modal applications:

- `my-wangpt-modal-app`: WanGP MCP server.
- `rest-wangpt-modal-app`: protected asynchronous generation/training API.
- `fizgig-modal-app`: Fizgig GPU training worker.
- `cotracker-modal-app`: CoTracker3 point-inference worker for the weather
  studio homography experiment.
- `tapnextpp-modal-app`: S3-native TAPNext++ tracking and homography stage for
  weather-studio experiments. Raw point inference remains an internal GPU
  boundary of the staged workflow.
- `opencv-mask-modal-app`: CPU-only TAPNext-guided green-screen replacement-mask
  worker. It writes a lossless grayscale mask, selected debug frames, and a
  compact quality summary to S3.
- `sam3-modal-app`: SAM 3.1 presenter-mask worker plus S3-native still-image
  display detection. The detector returns TAPNext-ready corners and a verified
  annotated overlay ArtifactRef.

Each application has its own README, Python project, deployment command, and
tests.
