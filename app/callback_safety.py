"""Pure helpers for optional callback_url host safety.

Default mode "monitor": callbacks send exactly as before; main.py only logs
whether the host WOULD be allowed. In "enforce", main.py blocks the callback
BEFORE send so the outbound webhook secret is never delivered to a
non-allowlisted host.

Rules: require https and an EXACT host match (no suffix/wildcard logic). This
module never logs and never handles secrets; callers log host + outcome only and
must never log the full callback URL (it may carry query parameters).
"""

from __future__ import annotations

import os
from typing import List, Mapping, Optional, Tuple
from urllib.parse import urlsplit

VALID_MODES = ("monitor", "enforce")
DEFAULT_MODE = "monitor"


def callback_host_mode(env: Optional[Mapping[str, str]] = None) -> str:
    """Return the callback host mode: monitor | enforce (unknown -> monitor)."""
    src = os.environ if env is None else env
    mode = str(src.get("AI_CALLBACK_HOST_MODE", DEFAULT_MODE)).strip().lower()
    return mode if mode in VALID_MODES else DEFAULT_MODE


def allowed_callback_hosts(env: Optional[Mapping[str, str]] = None) -> List[str]:
    """Parse AI_CALLBACK_ALLOWED_HOSTS (comma-separated) to lowercased hosts."""
    src = os.environ if env is None else env
    raw = str(src.get("AI_CALLBACK_ALLOWED_HOSTS", "") or "")
    hosts: List[str] = []
    for part in raw.split(","):
        host = part.strip().lower()
        if host and host not in hosts:
            hosts.append(host)
    return hosts


def is_callback_allowed(
    url: str,
    env: Optional[Mapping[str, str]] = None,
) -> Tuple[bool, str, str]:
    """Return (allowed, host, reason).

    `host` is safe to log (a bare hostname). `reason` is a short code:
      ok | scheme_not_https | missing_host | no_allowlist_configured |
      host_not_allowed | unparseable_url

    Matching is EXACT host, case-insensitive. Never returns the full URL,
    query string, or any secret.
    """
    try:
        parts = urlsplit(url or "")
    except Exception:
        return (False, "", "unparseable_url")

    host = (parts.hostname or "").strip().lower()

    if parts.scheme.lower() != "https":
        return (False, host, "scheme_not_https")
    if not host:
        return (False, host, "missing_host")

    allowlist = allowed_callback_hosts(env)
    if not allowlist:
        return (False, host, "no_allowlist_configured")
    if host in allowlist:
        return (True, host, "ok")
    return (False, host, "host_not_allowed")
