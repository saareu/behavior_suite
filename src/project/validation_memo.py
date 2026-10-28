"""Process-local memoization for expensive downstream validation results.

Caches successful validation payloads keyed by a fingerprint of the run
directory's primary metadata and artifact (path, size, mtime). Digests and
provenance checks still execute on cache miss; unchanged inputs reuse the
prior result instead of re-walking nested S4/S5/S6/S7 trees.

Not a substitute for content digests — ``project.file_digest.sha256_file``
remains authoritative for byte-level staleness.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, TypeVar

_CACHE_MAX = 128
_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
_lock = threading.Lock()

T = TypeVar("T")


def artifact_fingerprint(
    path: Path | str | None, *, missing_token: str = "missing"
) -> tuple[str, int, int] | tuple[str]:
    """Return ``(resolved, size, mtime_ns)`` or ``(missing_token,)`` if absent."""

    if path is None:
        return (missing_token,)
    resolved = Path(path).expanduser().resolve(strict=False)
    try:
        stat = resolved.stat()
    except OSError:
        return (f"{missing_token}:{resolved}",)
    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
    return (str(resolved), int(stat.st_size), mtime_ns)


def run_dir_fingerprint(
    run_dir: Path | str,
    relative_artifacts: Sequence[str],
) -> tuple[Any, ...]:
    """Fingerprint a run directory via run_meta plus named relative artifacts."""

    root = Path(run_dir).expanduser().resolve(strict=False)
    parts: list[Any] = [str(root)]
    parts.append(artifact_fingerprint(root / "run_meta.json"))
    for name in relative_artifacts:
        parts.append(artifact_fingerprint(root / name))
    return tuple(parts)


def memoized_validation(
    *,
    kind: str,
    fingerprint: tuple[Any, ...],
    compute: Callable[[], T],
) -> T:
    """Return a cached validation result or compute and store one."""

    key = (kind, fingerprint)
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached  # type: ignore[no-any-return]

    value = compute()
    with _lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return value


def clear_validation_memo() -> None:
    """Drop all memoized validation results (tests / forced revalidation)."""

    with _lock:
        _cache.clear()


def validation_memo_stats() -> tuple[int, int]:
    """Return ``(entries, max_entries)`` for diagnostics."""

    with _lock:
        return len(_cache), _CACHE_MAX
