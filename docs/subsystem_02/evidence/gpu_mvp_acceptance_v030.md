# Subsystem 02 MVP Full GPU Acceptance Evidence

The first full integrated Subsystem 02 MVP acceptance workflow completed
successfully on a real GPU machine. Machine-specific user, model, environment,
and session paths are intentionally omitted.

## Validated Runtime

- SLEAP-NN: `0.3.0`
- `sleap-io`: `0.8.0`
- inference modes: bottom-up and top-down
- top-down bundle: centroid plus centered-instance models

The external SLEAP-NN executable performed inference. The Behavior Suite GUI
Python environment provided `sleap-io` for Parquet export and SLEAP provenance
extraction in the shared artifact pipeline.

## Installation Closure

After the S2 runtime dependency fix, the supported one-click Windows installer
successfully recreated the GUI environment, installed the repository with its
S2 dependency set, and passed the Python, PySide6, `sleap-io`, application
dependency, `pip check`, and Behavior Suite doctor validations. No manual
machine repair is part of the supported installation workflow.

## Validated Workflow

The real-GPU acceptance exercised the integrated user workflow:

1. Complete the Subsystem 01 preprocessing handoff.
2. Enter Subsystem 02 through the main UI.
3. Run bottom-up inference.
4. Run top-down inference with the centroid plus centered-instance bundle.
5. Generate `pose.slp`, `pose.parquet`, `overlay.mp4`, and the required metadata
   and log artifacts for each mode.
6. Verify SLEAP provenance extraction and S1 frame/timing propagation.
7. Compute technical QC; both accepted runs produced outcome `pass`.
8. Discover and select the completed runs in the S2 UI.
9. Hand the selected completed run to the S3 interface successfully.

Both inference modes completed the same standardized Parquet, technical-QC,
overlay, metadata/provenance, and run-discovery pipeline. The integrated
workflow also verified downstream handoff from a selected completed run.

## Scope Boundary

This evidence is the acceptance basis for the finalized S2 MVP. S2 provides
UI-based pose inference, bottom-up and top-down model support, S1 integration,
standardized pose artifacts, metadata/provenance, technical QC, overlay
generation, run discovery, and an S3 handoff interface.

S2 technical QC validates inference execution and artifact integrity, detects
extreme abnormal failures, and may recommend review. It does not determine
final scientific usability and does not replace tracking validation or identity
verification. Identity correctness, tracking usability, and final scientific-
usability assessment remain S3 responsibilities.

UI-assisted model/parameter optimization, expanded pose-quality review tools,
and richer QC visualization remain non-blocking future enhancements rather than
missing S2 MVP requirements.
