"""Command-line entry point for Subsystem 02 pose inference."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from pose_inference.runner import (
    PoseInferenceError,
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
    model_path: Annotated[Path, typer.Option("--model-path", help="SLEAP model path.")],
    profile: Annotated[
        Path,
        typer.Option("--profile", help="SLEAP-NN inference profile YAML."),
    ] = Path("configs/subsystem_02/sleapnn_default_profile.yaml"),
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
        result = run_pose_inference(
            PoseInferenceRequest(
                session_root=session_root,
                model_path=model_path,
                profile_path=profile,
                output_root=output_root,
                run_purpose=run_purpose,
                dry_run=dry_run,
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
