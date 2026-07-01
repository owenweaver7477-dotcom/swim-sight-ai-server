"""Pure helpers for optional inbound authentication on POST /process-video.

Default OFF: with AI_INBOUND_AUTH_MODE unset (or "off") the worker accepts jobs
exactly as before, so deploying this module changes no production behaviour until
the env vars are set.

Security notes:
- AI_INBOUND_SECRET is the app -> worker job-submission secret. It is deliberately
  SEPARATE from AI_WEBHOOK_SECRET (the worker -> app outbound callback secret) so
  the two rotate independently and a leak of one does not compromise the other.
- This module never logs and never returns a secret value. Comparison is
  constant-time (hmac.compare_digest).
"""

from __future__ import annotations

import hmac
import os
from typing import Mapping, Optional

VALID_MODES = ("off", "monitor", "enforce")
DEFAULT_MODE = "off"


def inbound_auth_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """Return the inbound auth mode: off | monitor | enforce (unknown -> off)."""
    src = os.environ if env is None else env
    mode = str(src.get("AI_INBOUND_AUTH_MODE", DEFAULT_MODE)).strip().lower()
    return mode if mode in VALID_MODES else DEFAULT_MODE


def inbound_secret(env: Optional[Mapping[str, str]] = None) -> str:
    """Return the configured inbound secret (empty string if unset)."""
    src = os.environ if env is None else env
    return str(src.get("AI_INBOUND_SECRET", "") or "").strip()


def inbound_secret_configured(env: Optional[Mapping[str, str]] = None) -> bool:
    return bool(inbound_secret(env))


def verify_inbound_secret(
    provided: Optional[str],
    env: Optional[Mapping[str, str]] = None,
) -> str:
    """Return an outcome code only: "ok" | "missing" | "invalid".

    Never returns or logs the secret value. Uses constant-time comparison.
      missing -> no header supplied
      invalid -> header supplied but does not match, or no secret configured
    """
    if provided is None or provided == "":
        return "missing"
    expected = inbound_secret(env)
    if not expected:
        return "invalid"
    return "ok" if hmac.compare_digest(expected, provided) else "invalid"
