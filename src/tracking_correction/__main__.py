"""Command-line entry point for the Subsystem 03 tracking-correction shell."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from tracking_correction.contracts import TrackingCorrectionError, TrackingCorrectionRequest
from tracking_correction.runner import run_tracking_correction

app = typer.Typer(
    name="tracking-correction",
    help="Validate a completed Subsystem 02 run and run Subsystem 03 tracking correction.",
    no_args_is_help=True,
)


@app.command("doctor")
def doctor() -> None:
    """Show that the Subsystem 03 package and current-lab backend are importable."""

    typer.echo("tracking-correction shell: available")
    typer.echo("backend: current_lab_corrector")
    typer.echo("profile: current_lab_two_mouse_headstage_v1")
    typer.echo("review workspace: available")


@app.command("run")
def run(
    s2_run: Annotated[
        Path,
        typer.Option(
            "--s2-run",
            help="Completed Subsystem 02 run directory containing pose.parquet.",
        ),
    ],
    output_root: Annotated[
        Path | None,
        typer.Option("--output-root", help="Optional parent directory for the S3 run."),
    ] = None,
    run_purpose: Annotated[str, typer.Option("--run-purpose")] = "development",
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run",
            help="Validate the S2 handoff and write S3 records without invoking the corrector.",
        ),
    ] = False,
) -> None:
    """Validate an S2 handoff and run the current-lab corrector into an S3 workspace."""

    try:
        result = run_tracking_correction(
            TrackingCorrectionRequest(
                s2_run_dir=s2_run,
                output_root=output_root,
                run_purpose=run_purpose,
                dry_run=dry_run,
            )
        )
    except TrackingCorrectionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except Exception as exc:
        typer.echo(f"Internal error: {exc}", err=True)
        raise typer.Exit(code=2) from None

    typer.echo(f"Status: {result.status}")
    typer.echo(f"Run directory: {result.run_dir}")
    typer.echo(f"Profile: {result.profile_id}")
    typer.echo(f"Backend: {result.backend_id}")
    typer.echo(f"Backend status: {result.backend_status}")
    typer.echo(f"pose.parquet: {result.pose_parquet_path}")
    typer.echo(f"Prepared video: {result.prepared_video_path}")
    if result.working_tracked_pose_path is not None:
        typer.echo(f"Working tracked pose: {result.working_tracked_pose_path}")
    if result.machine_corrections_path is not None:
        typer.echo(f"Machine corrections: {result.machine_corrections_path}")
    typer.echo(f"Metadata: {result.run_meta_path}")
    typer.echo(f"Settings: {result.settings_used_path}")
    typer.echo(f"Log: {result.processing_log_path}")
    if not result.success:
        raise typer.Exit(code=1)


@app.command("review")
def review(
    s3_run: Annotated[
        Path,
        typer.Option(
            "--s3-run",
            help="Completed automatic Subsystem 03 run directory to open for review.",
        ),
    ],
) -> None:
    """Open the video-centered tracking review workspace for one S3 run."""

    from ui.tracking_review_app import (  # noqa: I001
        GUI_INSTALL_GUIDANCE,
        GuiDependencyError,
        launch_tracking_review,
    )

    try:
        exit_code = launch_tracking_review(s3_run)
    except GuiDependencyError:
        typer.echo(GUI_INSTALL_GUIDANCE, err=True)
        raise typer.Exit(code=1) from None
    except TrackingCorrectionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except Exception as exc:
        typer.echo(f"Internal error: {exc}", err=True)
        raise typer.Exit(code=2) from None
    if exit_code != 0:
        raise typer.Exit(code=exit_code)


@app.command("accept")
def accept(
    s3_run: Annotated[
        Path,
        typer.Option(
            "--s3-run",
            help="Completed automatic Subsystem 03 run directory to accept.",
        ),
    ],
) -> None:
    """Accept the working corrected pose without opening the GUI."""

    from tracking_correction.review import load_review_session

    try:
        session = load_review_session(s3_run)
        result = session.accept_tracking()
    except TrackingCorrectionError as exc:
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from None
    except Exception as exc:
        typer.echo(f"Internal error: {exc}", err=True)
        raise typer.Exit(code=2) from None

    if result.already_accepted:
        typer.echo("Status: already_accepted")
    else:
        typer.echo("Status: accepted")
    typer.echo(f"tracked_pose.parquet: {result.tracked_pose_path}")
    typer.echo(f"Accepted at: {result.accepted_at}")
    typer.echo(f"Working sha256: {result.working_sha256}")


if __name__ == "__main__":
    app()
