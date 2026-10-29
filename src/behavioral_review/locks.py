"""Single-writer lock for behavioral review working drafts."""

from __future__ import annotations

import contextlib
import json
import os
import socket
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from behavioral_review.contracts import (
    LOCK_STALE_SECONDS_DEFAULT,
    LOCKS_DIRNAME,
    WRITER_LOCK_FILENAME,
    LockError,
)
from behavioral_review.events import utc_now_iso


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if os.name == "nt":
            import ctypes

            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(
                PROCESS_QUERY_LIMITED_INFORMATION, 0, int(pid)
            )
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except OSError:
        return False
    except Exception:  # noqa: BLE001 - treat unknown as not alive for recovery
        return False


@dataclass(slots=True)
class WriterLockInfo:
    holder_id: str
    pid: int
    reviewer_id: str
    acquired_at_utc: str
    hostname: str
    review_run_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "holder_id": self.holder_id,
            "pid": int(self.pid),
            "reviewer_id": self.reviewer_id,
            "acquired_at_utc": self.acquired_at_utc,
            "hostname": self.hostname,
            "review_run_id": self.review_run_id,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> WriterLockInfo:
        return cls(
            holder_id=str(raw["holder_id"]),
            pid=int(raw["pid"]),
            reviewer_id=str(raw.get("reviewer_id") or ""),
            acquired_at_utc=str(raw.get("acquired_at_utc") or ""),
            hostname=str(raw.get("hostname") or ""),
            review_run_id=str(raw.get("review_run_id") or ""),
        )


class WriterLock:
    """Exclusive writer lock under ``working/locks/writer.lock``."""

    def __init__(
        self,
        working_dir: Path,
        *,
        review_run_id: str,
        stale_seconds: float = LOCK_STALE_SECONDS_DEFAULT,
    ) -> None:
        self.working_dir = Path(working_dir)
        self.review_run_id = review_run_id
        self.stale_seconds = float(stale_seconds)
        self.locks_dir = self.working_dir / LOCKS_DIRNAME
        self.lock_path = self.locks_dir / WRITER_LOCK_FILENAME
        self.holder_id: str | None = None
        self._held = False

    @property
    def is_held(self) -> bool:
        return self._held

    def read(self) -> WriterLockInfo | None:
        if not self.lock_path.is_file():
            return None
        try:
            raw = json.loads(self.lock_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        if not isinstance(raw, dict):
            return None
        try:
            return WriterLockInfo.from_dict(raw)
        except (KeyError, TypeError, ValueError):
            return None

    def is_stale(self, info: WriterLockInfo | None = None) -> bool:
        info = info if info is not None else self.read()
        if info is None:
            return False
        if not _pid_alive(info.pid):
            return True
        try:
            acquired = datetime.fromisoformat(
                info.acquired_at_utc.replace("Z", "+00:00")
            )
            age = (datetime.now(UTC) - acquired).total_seconds()
            if age > self.stale_seconds and not _pid_alive(info.pid):
                return True
        except ValueError:
            return not _pid_alive(info.pid)
        return False

    def acquire(self, *, reviewer_id: str, force_stale: bool = False) -> WriterLockInfo:
        self.locks_dir.mkdir(parents=True, exist_ok=True)
        existing = self.read()
        if existing is not None:
            mine = (
                self.holder_id is not None
                and existing.holder_id == self.holder_id
                and existing.pid == os.getpid()
            )
            if mine:
                self._held = True
                return existing
            stale = self.is_stale(existing)
            if stale and force_stale:
                self._write_break_marker(existing, reviewer_id=reviewer_id)
            elif stale and not force_stale:
                raise LockError(
                    "Stale writer lock detected. Re-acquire with force_stale=True "
                    f"after operator confirmation (holder_id={existing.holder_id}, "
                    f"pid={existing.pid})."
                )
            elif not stale:
                raise LockError(
                    "Working draft is locked by another writer "
                    f"(holder_id={existing.holder_id}, pid={existing.pid}, "
                    f"reviewer_id={existing.reviewer_id!r})."
                )

        info = WriterLockInfo(
            holder_id=str(uuid4()),
            pid=int(os.getpid()),
            reviewer_id=str(reviewer_id).strip(),
            acquired_at_utc=utc_now_iso(),
            hostname=socket.gethostname(),
            review_run_id=self.review_run_id,
        )
        tmp = self.lock_path.with_suffix(".lock.tmp")
        text = json.dumps(info.to_dict(), indent=2, sort_keys=True) + "\n"
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            # Exclusive create of tmp then replace — best-effort single writer.
            if self.lock_path.is_file() and not force_stale:
                # Re-check race.
                existing = self.read()
                if existing is not None and not self.is_stale(existing):
                    raise LockError(
                        "Working draft is locked by another writer "
                        f"(holder_id={existing.holder_id}, pid={existing.pid})."
                    )
            fd = os.open(tmp, flags)
            try:
                with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
                    handle.write(text)
                    handle.flush()
                    os.fsync(handle.fileno())
            except Exception:
                with contextlib.suppress(OSError):
                    os.close(fd)
                raise
            os.replace(tmp, self.lock_path)
        except FileExistsError as exc:
            raise LockError("Concurrent lock acquisition rejected.") from exc
        finally:
            if tmp.is_file():
                with contextlib.suppress(OSError):
                    tmp.unlink()

        self.holder_id = info.holder_id
        self._held = True
        return info

    def release(self) -> None:
        if not self._held:
            return
        existing = self.read()
        if existing is not None and existing.holder_id == self.holder_id:
            with contextlib.suppress(OSError):
                self.lock_path.unlink()
        self._held = False
        self.holder_id = None

    def _write_break_marker(self, previous: WriterLockInfo, *, reviewer_id: str) -> None:
        marker = self.locks_dir / f"stale_lock_break_{int(time.time())}.json"
        payload = {
            "broken_at_utc": utc_now_iso(),
            "broken_by_reviewer_id": reviewer_id,
            "previous_lock": previous.to_dict(),
            "reason": "stale_lock_recovery",
        }
        marker.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        with contextlib.suppress(OSError):
            self.lock_path.unlink()
