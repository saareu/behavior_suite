"""S10 GUI responsiveness audit — measure GUI-blocking vs worker latency.

Read-only measurement harness. Does not change science or correction semantics.
Uses an eligible E-session package. Prints structured timings for architectural review.
"""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from uuid import uuid4

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication  # noqa: E402

from behavioral_review.editing import open_editing_session  # noqa: E402
from behavioral_review.source import open_review_source  # noqa: E402
from project.file_digest import sha256_file  # noqa: E402
from results_packaging.publication import is_s9_run_eligible_for_downstream  # noqa: E402
from ui.controllers.behavioral_review_controller import (  # noqa: E402
    BehavioralReviewController,
)
from ui.pages.behavioral_review_page import BehavioralReviewPage  # noqa: E402

SESSION = Path(r"C:\behavior_suite\test_cases\E_full_session_integration")
PACKAGE_ROOT = SESSION / "results_package"
REVIEW_ROOT = Path(r"C:\behavior_suite\tmp_s10_responsiveness_audit") / "drafts"


@dataclass
class Phase:
    name: str
    elapsed_ms: float
    kind: str  # gui | worker | e2e | decode | render | io
    detail: dict[str, Any] = field(default_factory=dict)


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


def _pick_package() -> Path:
    preferred = PACKAGE_ROOT / "results_package__20261028T181714"
    candidates = [preferred] if preferred.is_dir() else []
    candidates.extend(sorted(PACKAGE_ROOT.glob("results_package__*"), reverse=True))
    seen: set[Path] = set()
    for pkg in candidates:
        if pkg in seen:
            continue
        seen.add(pkg)
        try:
            if is_s9_run_eligible_for_downstream(pkg):
                return pkg
        except Exception:  # noqa: BLE001
            continue
    raise SystemExit("No eligible E-session S9 package found for audit.")


def _wait_until(predicate: Callable[[], bool], timeout_sec: float = 180.0) -> float:
    t0 = time.perf_counter()
    app = QApplication.instance()
    assert app is not None
    deadline = time.monotonic() + timeout_sec
    while not predicate():
        app.processEvents()
        if time.monotonic() >= deadline:
            raise TimeoutError("wait_until timed out")
        time.sleep(0.005)
    app.processEvents()
    return _ms(t0)


def _print_phases(title: str, phases: list[Phase]) -> None:
    print(f"\n=== {title} ===")
    for p in phases:
        extra = ""
        if p.detail:
            extra = " | " + ", ".join(f"{k}={v}" for k, v in p.detail.items())
        print(f"  [{p.kind:7}] {p.name:42} {p.elapsed_ms:8.1f} ms{extra}")


def measure_source_breakdown(package: Path) -> list[Phase]:
    phases: list[Phase] = []
    # Video hash alone
    src_probe = open_review_source(package, require_video=False, require_s8_states=False)
    video = src_probe.source_video
    if (
        video is not None
        and video.path is not None
        and video.path.is_file()
        and video.sha256
    ):
        t0 = time.perf_counter()
        live = sha256_file(video.path)
        phases.append(
            Phase(
                "sha256_prepared_video",
                _ms(t0),
                "io",
                {
                    "mb": round(video.path.stat().st_size / 1e6, 1),
                    "match": live.lower() == str(video.sha256).lower(),
                },
            )
        )
    states = src_probe.source_s8.states_path
    if states is not None and states.is_file():
        import pandas as pd

        t0 = time.perf_counter()
        frame = pd.read_parquet(states)
        phases.append(
            Phase(
                "read_full_s8_states_parquet",
                _ms(t0),
                "io",
                {"rows": len(frame), "mb": round(states.stat().st_size / 1e6, 2)},
            )
        )
        t0 = time.perf_counter()
        _ = frame.copy()
        phases.append(Phase("copy_full_s8_states_dataframe", _ms(t0), "io"))
    t0 = time.perf_counter()
    src = open_review_source(package, require_video=False, require_s8_states=True)
    phases.append(
        Phase(
            "open_review_source(require_video=False)",
            _ms(t0),
            "worker",
            {
                "video_available": bool(src.source_video and src.source_video.available),
                "s8_states": bool(src.s8_states_available),
            },
        )
    )
    t0 = time.perf_counter()
    session = open_editing_session(
        package, require_video=False, require_s8_states=True, initial_sample_idx=0
    )
    phases.append(
        Phase(
            "open_editing_session(+eager baseline)",
            _ms(t0),
            "worker",
            {
                "baseline_loaded": session.baseline_states is not None,
                "n_dense": session.n_dense_samples,
            },
        )
    )
    del session
    return phases


def measure_controller_flows(package: Path) -> list[Phase]:
    phases: list[Phase] = []
    controller = BehavioralReviewController()

    # Discovery (worker-equivalent call)
    t0 = time.perf_counter()
    identity = controller.prepare_session(SESSION)
    snap = controller.collect_discovery_snapshot(identity)
    assert controller.apply_discovery_snapshot(snap)
    phases.append(Phase("discovery_collect+apply", _ms(t0), "worker"))

    controller.select_s9_package(package.name)

    # Open package (worker body)
    open_id = controller.begin_open_automated()
    t0 = time.perf_counter()
    open_snap = controller.open_source_worker(open_id)
    worker_ms = _ms(t0)
    t1 = time.perf_counter()
    assert controller.apply_open_snapshot(open_snap)
    gui_ms = _ms(t1)
    phases.append(Phase("open_source_worker", worker_ms, "worker"))
    phases.append(
        Phase(
            "apply_open_snapshot(gui)",
            gui_ms,
            "gui",
            {"baseline_loaded": controller.state.editing_session.baseline_states is not None},  # type: ignore[union-attr]
        )
    )

    session = controller.state.editing_session
    assert session is not None
    n = controller.state.n_dense
    samples = [0, n // 2, n - 1]

    # Seek mapping only (GUI)
    for sample in samples:
        t0 = time.perf_counter()
        decode_id = controller.seek_sample(sample)
        phases.append(
            Phase(
                f"seek_sample({sample})",
                _ms(t0),
                "gui",
                {"prepared": controller.state.prepared_frame_idx},
            )
        )
        if decode_id is None:
            continue
        t0 = time.perf_counter()
        dec = controller.decode_worker(decode_id)
        phases.append(
            Phase(
                f"decode_worker(sample={sample})",
                _ms(t0),
                "decode",
                {
                    "unchanged": dec.prepared_frame_unchanged,
                    "has_frame": dec.frame_bgr is not None,
                },
            )
        )
        t0 = time.perf_counter()
        controller.apply_decode_snapshot(dec)
        phases.append(Phase(f"apply_decode(sample={sample})", _ms(t0), "gui"))

        win_id = controller.begin_window_request(half_window=120, include_s7=False)
        t0 = time.perf_counter()
        win = controller.window_worker(win_id)
        phases.append(
            Phase(
                f"window_worker(sample={sample})",
                _ms(t0),
                "worker",
                {
                    "lanes": len(win.timeline.lanes) if win.timeline else 0,
                    "bout_rows": len(win.bout_rows),
                },
            )
        )
        t0 = time.perf_counter()
        controller.apply_window_snapshot(win)
        phases.append(Phase(f"apply_window(sample={sample})", _ms(t0), "gui"))

    # Rapid repeated seeks (coalesce-relevant)
    rapid = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500]
    t0 = time.perf_counter()
    last_decode = None
    last_win = None
    for sample in rapid:
        did = controller.seek_sample(min(sample, n - 1))
        if did is not None:
            last_decode = controller.decode_worker(did)
            controller.apply_decode_snapshot(last_decode)
        wid = controller.begin_window_request(half_window=60, include_s7=False)
        last_win = controller.window_worker(wid)
        controller.apply_window_snapshot(last_win)
    phases.append(
        Phase(
            f"rapid_{len(rapid)}_seek+decode+window",
            _ms(t0),
            "e2e",
            {
                "avg_ms": round(_ms(t0) / len(rapid), 1),
                "cache": (
                    session.inspection.video.cache_stats()
                    if session.inspection.video
                    else None
                ),
            },
        )
    )
    del last_decode, last_win

    # Direction change (select + window)
    dirs = controller.state.directions
    if len(dirs) >= 2:
        t0 = time.perf_counter()
        controller.select_direction(dirs[1][0], dirs[1][1])
        sel_ms = _ms(t0)
        phases.append(Phase("select_direction", sel_ms, "gui"))
        wid = controller.begin_window_request(half_window=120)
        t0 = time.perf_counter()
        win = controller.window_worker(wid)
        phases.append(Phase("window_after_direction", _ms(t0), "worker"))
        controller.apply_window_snapshot(win)

    # Bout selection
    if controller.state.bout_rows:
        bout_id = str(controller.state.bout_rows[0]["bout_id"])
        t0 = time.perf_counter()
        controller.select_bout(bout_id)
        phases.append(Phase("select_bout(+jump)", _ms(t0), "gui"))

    # Editing path — begin_* is GUI-thread initiation; edit_worker is projection work.
    review_root = REVIEW_ROOT / f"audit_{uuid4().hex[:10]}"
    review_root.mkdir(parents=True, exist_ok=True)
    controller.set_reviewer_id(f"audit_{uuid4().hex[:8]}")
    t0 = time.perf_counter()
    enter_id = controller.begin_enter_editing()
    phases.append(
        Phase(
            "begin_enter_editing(gui)",
            _ms(t0),
            "gui",
            {"baseline_was_eager": session.baseline_states is not None},
        )
    )
    t0 = time.perf_counter()
    enter_snap = controller.edit_worker(enter_id)
    phases.append(Phase("enter_editing_edit_worker", _ms(t0), "worker"))
    assert controller.apply_edit_snapshot(enter_snap)

    t0 = time.perf_counter()
    draft_id = controller.begin_create_draft(review_root=review_root)
    phases.append(Phase("begin_create_draft(gui)", _ms(t0), "gui"))
    t0 = time.perf_counter()
    draft_snap = controller.edit_worker(draft_id)
    phases.append(Phase("create_draft_edit_worker(+rebuild)", _ms(t0), "worker"))
    assert controller.apply_edit_snapshot(draft_snap)

    # Need bout rows for subsequent corrections.
    wid = controller.begin_window_request(half_window=120)
    win = controller.window_worker(wid)
    controller.apply_window_snapshot(win)

    rows = [r for r in controller.state.bout_rows if r["channel"] == "exclusive"]
    if not rows:
        rows = list(controller.state.bout_rows)
    assert rows
    first = rows[0]
    start = int(first["start_sample_idx"])
    end = int(first["end_sample_idx"])
    focal = int(first["focal_id"]) if first["focal_id"] is not None else 0
    partner = int(first["partner_id"]) if first["partner_id"] is not None else 1

    t0 = time.perf_counter()
    excl_id = controller.begin_set_exclusive_label(
        start_sample_idx=start,
        end_sample_idx=min(start + 5, end),
        label="nose_to_body",
        focal_id=focal,
        partner_id=partner,
    )
    phases.append(Phase("begin_set_exclusive_label(gui)", _ms(t0), "gui"))
    t0 = time.perf_counter()
    excl_snap = controller.edit_worker(excl_id)
    phases.append(Phase("set_exclusive_label_edit_worker", _ms(t0), "worker"))
    assert controller.apply_edit_snapshot(excl_snap)

    t0 = time.perf_counter()
    contact_id = controller.begin_set_parallel_pred(
        channel="contact",
        start_sample_idx=start,
        end_sample_idx=min(start + 3, end),
        pred_value=0,
    )
    phases.append(Phase("begin_set_parallel_pred(gui)", _ms(t0), "gui"))
    t0 = time.perf_counter()
    contact_snap = controller.edit_worker(contact_id)
    phases.append(Phase("set_parallel_pred_edit_worker", _ms(t0), "worker"))
    assert controller.apply_edit_snapshot(contact_snap)

    t0 = time.perf_counter()
    undo_id = controller.begin_undo()
    phases.append(Phase("begin_undo(gui)", _ms(t0), "gui"))
    t0 = time.perf_counter()
    undo_snap = controller.edit_worker(undo_id)
    phases.append(Phase("undo_edit_worker", _ms(t0), "worker"))
    assert controller.apply_edit_snapshot(undo_snap)

    t0 = time.perf_counter()
    redo_id = controller.begin_redo()
    phases.append(Phase("begin_redo(gui)", _ms(t0), "gui"))
    t0 = time.perf_counter()
    redo_snap = controller.edit_worker(redo_id)
    phases.append(Phase("redo_edit_worker", _ms(t0), "worker"))
    assert controller.apply_edit_snapshot(redo_snap)

    # Reopen draft path: simulate apply_open_snapshot draft resume cost
    from ui.controllers.behavioral_review_controller import (
        VIEW_SOURCE_WORKING,
        OpenSourceIdentity,
    )

    draft_dir = controller.state.review_run_dir
    assert draft_dir is not None
    controller.close_session(release_lock=True)

    controller2 = BehavioralReviewController()
    identity = controller2.prepare_session(SESSION)
    assert controller2.apply_discovery_snapshot(
        controller2.collect_discovery_snapshot(identity)
    )
    controller2.select_s9_package(package.name)
    reviewer = f"audit_{uuid4().hex[:8]}"
    controller2.set_reviewer_id(reviewer)
    controller2.state.open_generation += 1
    open_id = OpenSourceIdentity(
        request_id=str(uuid4()),
        generation=controller2.state.open_generation,
        session_root=SESSION,
        package_dir=package,
        source_kind=VIEW_SOURCE_WORKING,
        review_run_dir=draft_dir,
    )
    controller2.state.active_open_identity = open_id
    controller2.state.open_running = True
    t0 = time.perf_counter()
    open_snap = controller2.open_source_worker(
        open_id, reviewer_id=reviewer, force_stale_lock=True
    )
    phases.append(
        Phase(
            "reopen_draft_open_worker(+draft resume)",
            _ms(t0),
            "worker",
            {"draft_resumed": open_snap.draft_resumed},
        )
    )
    t0 = time.perf_counter()
    assert controller2.apply_open_snapshot(open_snap, reviewer_id=reviewer)
    phases.append(
        Phase(
            "reopen_draft_apply_open_snapshot(gui)",
            _ms(t0),
            "gui",
            {"draft_open": controller2.state.draft_open},
        )
    )
    controller2.close_session(release_lock=True)
    return phases


def measure_page_e2e(package: Path) -> list[Phase]:
    """Page-level E2E with real GuiTaskRunners; estimate GUI-thread via processEvents gaps."""

    phases: list[Phase] = []
    app = QApplication.instance() or QApplication(sys.argv)
    page = BehavioralReviewPage()
    page.show()
    app.processEvents()

    t0 = time.perf_counter()
    page.set_session(SESSION)
    wait_ms = _wait_until(
        lambda: (not page.discovery_runner.is_running)
        and page.controller.state.session_root is not None
        and bool(page.controller.state.s9_options or page.controller.state.run_error)
    )
    phases.append(Phase("page_set_session+discovery_e2e", _ms(t0), "e2e", {"wait_ms": round(wait_ms, 1)}))

    # Select package and open
    page.controller.select_s9_package(package.name)
    page.refresh_from_state()
    app.processEvents()

    t0 = time.perf_counter()
    page._open_automated()  # noqa: SLF001 — audit harness
    wait_ms = _wait_until(
        lambda: (not page.open_runner.is_running)
        and page.controller.state.editing_session is not None
        and not page.controller.state.open_running
    )
    # Also wait initial decode/window to settle
    settle = _wait_until(
        lambda: (not page.decode_runner.is_running) and (not page.window_runner.is_running),
        timeout_sec=180.0,
    )
    phases.append(
        Phase(
            "page_open_package_e2e(+initial seek workers)",
            _ms(t0),
            "e2e",
            {"open_wait_ms": round(wait_ms, 1), "settle_ms": round(settle, 1)},
        )
    )

    n = page.controller.state.n_dense
    for label, sample in (("first", 0), ("middle", n // 2), ("last", n - 1)):
        t0 = time.perf_counter()
        page._seek_sample(sample)  # noqa: SLF001
        # GUI mapping should return immediately; wait for workers
        gui_return = _ms(t0)
        target = int(sample)
        wait_ms = _wait_until(
            lambda target=target: (not page.decode_runner.is_running)
            and (not page.window_runner.is_running)
            and page.controller.state.sample_idx == target
        )
        phases.append(
            Phase(
                f"page_seek_{label}",
                _ms(t0),
                "e2e",
                {
                    "gui_return_ms": round(gui_return, 2),
                    "worker_settle_ms": round(wait_ms, 1),
                    "sample": sample,
                    "prepared": page.controller.state.prepared_frame_idx,
                },
            )
        )

    # Rapid scrub: fire many seeks without waiting between; measure GUI return + final settle
    scrub_targets = [i * (n // 20) for i in range(21)]
    t0 = time.perf_counter()
    gui_blocks: list[float] = []
    for sample in scrub_targets:
        g0 = time.perf_counter()
        page._seek_sample(int(sample))  # noqa: SLF001
        gui_blocks.append(_ms(g0))
        app.processEvents()
    scrub_fire_ms = _ms(t0)
    settle = _wait_until(
        lambda: (not page.decode_runner.is_running) and (not page.window_runner.is_running)
    )
    phases.append(
        Phase(
            "page_rapid_scrub_21_seeks",
            scrub_fire_ms + settle,
            "e2e",
            {
                "fire_ms": round(scrub_fire_ms, 1),
                "settle_ms": round(settle, 1),
                "max_gui_seek_ms": round(max(gui_blocks), 2),
                "avg_gui_seek_ms": round(sum(gui_blocks) / len(gui_blocks), 2),
                "stale_coalesce": True,
            },
        )
    )

    # Direction change
    if len(page.controller.state.directions) >= 2:
        t0 = time.perf_counter()
        page.controller.select_direction(
            page.controller.state.directions[1][0],
            page.controller.state.directions[1][1],
        )
        page._request_seek_workers()  # noqa: SLF001
        gui_return = _ms(t0)
        settle = _wait_until(
            lambda: (not page.decode_runner.is_running) and (not page.window_runner.is_running)
        )
        phases.append(
            Phase(
                "page_direction_change",
                gui_return + settle,
                "e2e",
                {"gui_return_ms": round(gui_return, 2), "settle_ms": round(settle, 1)},
            )
        )

    # Bout select
    if page.controller.state.bout_rows:
        bout_id = str(page.controller.state.bout_rows[0]["bout_id"])
        t0 = time.perf_counter()
        page.controller.select_bout(bout_id)
        page._request_seek_workers()  # noqa: SLF001
        gui_return = _ms(t0)
        settle = _wait_until(
            lambda: (not page.decode_runner.is_running) and (not page.window_runner.is_running)
        )
        phases.append(
            Phase(
                "page_bout_select",
                gui_return + settle,
                "e2e",
                {"gui_return_ms": round(gui_return, 2), "settle_ms": round(settle, 1)},
            )
        )

    # Nonblocking edit path via GuiTaskRunner
    review_root = REVIEW_ROOT / f"page_audit_{uuid4().hex[:10]}"
    review_root.mkdir(parents=True, exist_ok=True)
    page.reviewer_edit.setText(f"audit_{uuid4().hex[:8]}")
    page._reviewer_changed()  # noqa: SLF001
    t0 = time.perf_counter()
    page._enter_editing()  # noqa: SLF001
    gui_return = _ms(t0)
    settle = _wait_until(
        lambda: (not page.edit_runner.is_running)
        and page.controller.state.editing_active
        and not page.controller.state.edit_running
    )
    phases.append(
        Phase(
            "page_enter_editing_e2e",
            gui_return + settle,
            "e2e",
            {"gui_return_ms": round(gui_return, 2), "settle_ms": round(settle, 1)},
        )
    )
    t0 = time.perf_counter()
    # begin_create_draft via controller so we can pin review_root for the audit.
    draft_id = page.controller.begin_create_draft(review_root=review_root)
    gui_return = _ms(t0)
    page._start_edit_task(draft_id)  # noqa: SLF001
    settle = _wait_until(
        lambda: (not page.edit_runner.is_running)
        and page.controller.state.draft_open
        and not page.controller.state.edit_running
    )
    phases.append(
        Phase(
            "page_create_draft_e2e",
            gui_return + settle,
            "e2e",
            {"gui_return_ms": round(gui_return, 2), "settle_ms": round(settle, 1)},
        )
    )
    if page.controller.state.bout_rows:
        page.edit_start_spin.setValue(0)
        page.edit_end_spin.setValue(1)
        t0 = time.perf_counter()
        page._apply_exclusive_label()  # noqa: SLF001
        gui_return = _ms(t0)
        settle = _wait_until(
            lambda: (not page.edit_runner.is_running)
            and not page.controller.state.edit_running
        )
        phases.append(
            Phase(
                "page_exclusive_correction_e2e",
                gui_return + settle,
                "e2e",
                {"gui_return_ms": round(gui_return, 2), "settle_ms": round(settle, 1)},
            )
        )

    page.close()
    return phases


def main() -> None:
    package = _pick_package()
    print(f"PACKAGE={package}")
    print(f"eligible={is_s9_run_eligible_for_downstream(package)}")

    phases = measure_source_breakdown(package)
    _print_phases("Source / IO breakdown", phases)

    phases = measure_controller_flows(package)
    _print_phases("Controller flows (sync worker calls; GUI = apply/edit)", phases)

    _app = QApplication.instance() or QApplication(sys.argv)
    del _app
    phases = measure_page_e2e(package)
    _print_phases("Page E2E with GuiTaskRunners", phases)


if __name__ == "__main__":
    main()
