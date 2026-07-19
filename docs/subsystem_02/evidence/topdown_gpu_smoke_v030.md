# SLEAP-NN 0.3.0 Top-Down GPU Smoke Evidence

A real GPU smoke test completed successfully using SLEAP-NN 0.3.0 and a
two-stage top-down bundle containing a centroid model and a centered-instance
model.

Observed result:

- inference status: `success`;
- `pose.slp`, `pose.parquet`, and `overlay.mp4` generated;
- technical QC computed with outcome `pass`;
- run discovery reported inference mode `topdown`;
- run discovery classified the run as `complete_reviewable`.

The test exercised the shared S1 timing, Parquet, technical-QC, overlay, and
run-discovery pipeline. SLEAP-NN 0.3.x `predict` is the currently supported
Behavior Suite execution interface. Machine-specific model and user paths are
intentionally omitted from this general repository evidence note.
