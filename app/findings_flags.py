"""Feature flags for optional/experimental finding generation.

Default OFF. With EXTENDED_STROKE_FINDINGS unset/false the worker emits exactly
the current backstroke/butterfly findings, so production behaviour is unchanged.
"""

from __future__ import annotations

import os
from typing import Mapping, Optional

_TRUTHY = {"1", "true", "yes", "on"}


def extended_stroke_findings_enabled(env: Optional[Mapping[str, str]] = None) -> bool:
    """Whether experimental backstroke/butterfly findings are enabled (default false).

    These are additional 2D-heuristic, coach-review-required draft findings that
    reuse existing signal helpers. They must be validated on labelled clips
    before being enabled in production.
    """
    src = os.environ if env is None else env
    return str(src.get("EXTENDED_STROKE_FINDINGS", "false")).strip().lower() in _TRUTHY
