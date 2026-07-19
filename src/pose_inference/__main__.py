"""Command-line entry point for Subsystem 02 pose inference."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pose_inference.runner import (
    PoseInferenceError,
    PoseInferenceModelSpec,
    PoseInferenceRequest,
    run_pose_inference,
)

app = typer.Typer(
    name="pose-inference",
    help="Run Subsystem 02 SLEAP-NN pose inference.",
    no_args_is_help=True,
)


@app.command("doctor")
def doctor() -> None:
    """Show that the Subsystem 02 wrapper is importable."""

    typer.echo("pose-inference wrapper: available")


@app.command("run")
def run(
    session_root: Annotated[
        Path,
        typer.Option(
            "--session-root",
            help="Session root containing preprocess/ or the preprocess directory itself.",
        ),
    ],
    model_path: Annotated[
        Path | None,
        typer.Option("--model-path", help="Bottom-up SLEAP model path."),
    ] = None,
    inference_mode: Annotated[
        str | None,
        typer.Option("--inference-mode", help="Inference mode: bottomup or topdown."),
    ] = None,
    centroid_model_path: Annotated[
        Path | None,
        typer.Option(
            "--centroid-model-path",
            help="Top-down centroid SLEAP model path.",
        ),
    ] = None,
    centered_instance_model_path: Annotated[
        Path | None,
        typer.Option(
            "--centered-instance-model-path",
            help="Top-down centered-instance SLEAP model path.",
        ),
    ] = None,
    profile: Annotated[
        Path | None,
        typer.Option("--profile", help="SLEAP-NN inference profile YAML."),
    ] = None,
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="Optional parent directory for run outputs."),
    ] = None,
    run_purpose: Annotated[str, typer.Option("--run-purpose")] = "development",
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Write records without executing SLEAP-NN."),
    ] = False,
) -> None:
    """Validate S1 handoff files and run or dry-run one SLEAP-NN inference."""

    try:
        has_topdown_component = (
            centroid_model_path is not None
            or centered_instance_model_path is not None
        )
        if inference_mode is None and has_topdown_component:
            raise PoseInferenceError(
                "Top-down component paths require --inference-mode topdown."
            )
        model_spec = None
        legacy_model_path = model_path
        if inference_mode is not None:
            normalized_mode = inference_mode.strip().lower().replace("-", "")
            if normalized_mode == "topdown" and model_path is not None:
                raise PoseInferenceError(
                    "Top-down inference cannot be combined with --model-path; use only "
                    "the two top-down component flags."
                )
            model_spec = PoseInferenceModelSpec(
                inference_mode=inference_mode,
                bottomup_model_path=model_path
                if normalized_mode == "bottomup"
                else None,
                centroid_model_path=centroid_model_path,
                centered_instance_model_path=centered_instance_model_path,
            )
            legacy_model_path = None
        selected_profile = profile or Path(
            "configs/subsystem_02/sleapnn_topdown_default_profile.yaml"
            if inference_mode is not None
            and inference_mode.strip().lower().replace("-", "") == "topdown"
            else "configs/subsystem_02/sleapnn_default_profile.yaml"
        )
        result = run_pose_inference(
            PoseInferenceRequest(
                session_root=session_root,
                model_path=legacy_model_path,
                profile_path=selected_profile,
                output_root=output_root,
                run_purpose=run_purpose,
                dry_run=dry_run,
                model_spec=model_spec,
            )
        )
    except PoseInferenceError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except Exception as exc:
        typer.echo(f"Internal error: {exc}", err=True)
        raise typer.Exit(code=2) from None

    typer.echo(f"Status: {result.status}")
    typer.echo(f"Run directory: {result.run_dir}")
    typer.echo(f"pose.slp: {result.pose_slp_path}")
    typer.echo(f"Metadata: {result.pose_meta_path}")
    typer.echo(f"Settings: {result.settings_used_path}")
    typer.echo(f"Manifest: {result.job_manifest_path}")
    typer.echo(f"Log: {result.processing_log_path}")
    if not result.success:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
