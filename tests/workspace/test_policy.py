from pathlib import Path

import pytest

from workspace.errors import WorkspaceNavigationBlocked
from workspace.policy import (
    ContextChangeRequest,
    JobBinding,
    evaluate_context_change,
    is_stale_job_result,
    require_context_change_allowed,
    retarget_active_job_forbidden_message,
)


def test_soft_and_hard_nav_policy() -> None:
    hard = evaluate_context_change(ContextChangeRequest(kind="session", hard_locked=True))
    assert hard.allowed is False

    soft = evaluate_context_change(
        ContextChangeRequest(kind="session", page_task_running=True)
    )
    assert soft.allowed is False

    ok = evaluate_context_change(
        ContextChangeRequest(kind="session", has_open_s10_draft=True)
    )
    assert ok.allowed is True
    assert ok.release_s10_draft_lock is True
    assert ok.bump_generation is True
    assert ok.clear_session_scoped_run_selections is True
    assert ok.retain_active_job_bindings is True


def test_require_raises() -> None:
    with pytest.raises(WorkspaceNavigationBlocked):
        require_context_change_allowed(
            ContextChangeRequest(kind="project", hard_locked=True)
        )


def test_stale_job_detection(tmp_path: Path) -> None:
    binding = JobBinding(
        analysis_target_id="session:abc",
        discovery_root=tmp_path,
        generation=1,
    )
    assert (
        is_stale_job_result(
            binding,
            current_analysis_target_id="session:abc",
            current_generation=1,
        )
        is False
    )
    assert (
        is_stale_job_result(
            binding,
            current_analysis_target_id="session:abc",
            current_generation=2,
        )
        is True
    )
    assert (
        is_stale_job_result(
            binding,
            current_analysis_target_id="session:other",
            current_generation=1,
        )
        is True
    )
    assert "must not be silently retargeted" in retarget_active_job_forbidden_message()
