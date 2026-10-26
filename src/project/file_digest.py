"""Process-local SHA-256 digests with (path, size, mtime) invalidation.

Used by discovery and staleness checks so repeated reads of unchanged Parquet
or model files do not re-hash multi-hundred-megabyte artifacts on every GUI
refresh. Digests remain content-correct: any size or mtime change recomputes.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from pathlib import Path

_CACHE_MAX = 512
_cache: OrderedDict[tuple[str, int, int], str] = OrderedDict()
_lock = threading.Lock()


def sha256_file(path: Path | str) -> str:
    """Return the SHA-256 hex digest of ``path``, caching unchanged files."""

    resolved = Path(path).expanduser().resolve(strict=False)
    stat = resolved.stat()
    mtime_ns = int(getattr(stat, "st_mtime_ns", int(stat.st_mtime * 1_000_000_000)))
    key = (str(resolved), int(stat.st_size), mtime_ns)
    with _lock:
        cached = _cache.get(key)
        if cached is not None:
            _cache.move_to_end(key)
            return cached

    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    value = digest.hexdigest()
    with _lock:
        _cache[key] = value
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return value


def clear_digest_cache() -> None:
    """Drop all cached digests (tests / forced revalidation)."""

    with _lock:
        _cache.clear()


def digest_cache_stats() -> tuple[int, int]:
    """Return ``(entries, max_entries)`` for diagnostics."""

    with _lock:
        return len(_cache), _CACHE_MAX
