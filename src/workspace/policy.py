"""Context-change and async-job safeguards for D0-A (no GUI controller wiring).

Full MainWindow / page integration belongs to D0-D. These helpers define the
policy surface those integrations must honor:

- Do not silently retarget active jobs.
- Preserve S10 draft-lock release / stale-result rejection semantics.
- Separate global workspace selection from per-subsystem run selection.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from workspace.errors import WorkspaceNavigationBlocked
from workspace.models import normalize_path

ContextChangeKind = Literal["project", "session", "subsession", "clear", "restore"]

__all__ = [
    "ContextChangeDecision",
    "ContextChangeKind",
    "ContextChangeRequest",
    "JobBinding",
    "evaluate_context_change",
    "is_stale_job_result",
    "require_context_change_allowed",
    "retarget_active_job_forbidden_message",
]


@dataclass(frozen=True)
class JobBinding:
    """Immutable binding captured when an async GUI job starts.

    Workers must keep this binding for their lifetime. Completion callbacks
    compare against the *current* context and drop stale results on mismatch.
    """

    analysis_target_id: str
    discovery_root: Path
    generation: int
    job_kind: str = "unspecified"

    def __post_init__(self) -> None:
        object.__setattr__(self, "discovery_root", normalize_path(self.discovery_root))


@dataclass(frozen=True)
class ContextChangeRequest:
    """Caller-supplied navigation / job state for a proposed context change."""

    kind: ContextChangeKind
    hard_locked: bool = False
    """True when S2/S4-style hard navigation lock is active."""

    page_task_running: bool = False
    """True when the page that owns the change request has a soft-nav task running."""

    has_open_s10_draft: bool = False
    """True when an S10 editing session currently holds a draft writer lock."""


@dataclass(frozen=True)
class ContextChangeDecision:
    """Policy outcome for a proposed context change."""

    allowed: bool
    reason: str | None = None
    release_s10_draft_lock: bool = False
    bump_generation: bool = False
    clear_session_scoped_run_selections: bool = False
    retain_active_job_bindings: bool = True
    """Active jobs keep their original JobBinding; never retarget mid-flight."""


def evaluate_context_change(request: ContextChangeRequest) -> ContextChangeDecision:
    """Evaluate whether a workspace context change may proceed.

    Does not mutate WorkspaceContext or start scientific processing.
    """

    if request.hard_locked:
        return ContextChangeDecision(
            allowed=False,
            reason="Context change blocked while a hard-locked task is running.",
            retain_active_job_bindings=True,
        )
    if request.page_task_running:
        return ContextChangeDecision(
            allowed=False,
            reason=(
                "Context change blocked for this page while its task is running "
                "(soft-nav safeguard)."
            ),
            retain_active_job_bindings=True,
        )

    return ContextChangeDecision(
        allowed=True,
        release_s10_draft_lock=request.has_open_s10_draft,
        bump_generation=True,
        clear_session_scoped_run_selections=request.kind
        in {"project", "session", "clear", "restore"},
        retain_active_job_bindings=True,
    )


def require_context_change_allowed(request: ContextChangeRequest) -> ContextChangeDecision:
    decision = evaluate_context_change(request)
    if not decision.allowed:
        raise WorkspaceNavigationBlocked(decision.reason or "Context change blocked.")
    return decision


def is_stale_job_result(
    binding: JobBinding,
    *,
    current_analysis_target_id: str | None,
    current_generation: int,
) -> bool:
    """Return True when an async callback must be discarded.

    Mirrors existing discovery-generation / request-id rejection patterns.
    """

    if binding.generation != current_generation:
        return True
    if current_analysis_target_id is None:
        return True
    return binding.analysis_target_id != current_analysis_target_id


def retarget_active_job_forbidden_message() -> str:
    return (
        "Active jobs must not be silently retargeted; keep the original "
        "analysis_target_id / discovery_root binding until completion."
    )
