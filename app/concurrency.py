"""Concurrency limiting for heavy analysis jobs. Default OFF (unchanged behaviour).

`AI_MAX_CONCURRENT_JOBS` caps how many heavy analysis pipelines run at once so a
small Render instance cannot be overwhelmed. Unset, 0, or an invalid value means
disabled (current behaviour, no cap).

`AI_POST_TIMEOUT_DRAIN_SECONDS` bounds how long a timed-out job keeps holding its
concurrency slot while its (unkillable) worker thread drains, so a runaway thread
cannot be multiplied by newly started jobs. Default 0 = release immediately
(current timing). This module is pure and side-effect free.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

DEFAULT_MAX_CONCURRENT_JOBS = 0            # 0 = disabled (no cap)
DEFAULT_POST_TIMEOUT_DRAIN_SECONDS = 0.0  # 0 = release immediately (current timing)
MAX_DRAIN_SECONDS_CAP = 600.0             # hard safety ceiling for the drain wait


def max_concurrent_jobs(env: Optional[Mapping[str, str]] = None) -> int:
    """Return the concurrency cap. 0 = disabled (no cap).

    Unset, non-integer, or non-positive values all safely return 0 (disabled),
    so a misconfiguration preserves current behaviour rather than blocking work.
    """
    src = os.environ if env is None else env
    raw = src.get("AI_MAX_CONCURRENT_JOBS")
    if raw is None:
        return DEFAULT_MAX_CONCURRENT_JOBS
    try:
        value = int(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_MAX_CONCURRENT_JOBS
    return value if value > 0 else DEFAULT_MAX_CONCURRENT_JOBS


def concurrency_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    return max_concurrent_jobs(env) > 0


def post_timeout_drain_seconds(env: Optional[Mapping[str, str]] = None) -> float:
    """Seconds a timed-out job waits for its blocking thread before releasing its
    slot. 0 (default) = release immediately. Invalid/negative => 0. Clamped to a
    safe ceiling so a misconfiguration cannot hold a slot indefinitely.
    """
    src = os.environ if env is None else env
    raw = src.get("AI_POST_TIMEOUT_DRAIN_SECONDS")
    if raw is None:
        return DEFAULT_POST_TIMEOUT_DRAIN_SECONDS
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return DEFAULT_POST_TIMEOUT_DRAIN_SECONDS
    if value < 0:
        return DEFAULT_POST_TIMEOUT_DRAIN_SECONDS
    return min(value, MAX_DRAIN_SECONDS_CAP)
