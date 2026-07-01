"""Callback URL host-safety tests.

Unit-tests the pure validator (app.callback_safety) and the main.py send guard
(main._guard_callback_host) across monitor/enforce. No network, no deploy. The
validator returns/logs a bare host only -- never the full URL or a secret.

Run:  python3 scripts/test_callback_host_allowlist.py
"""

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.callback_safety import (  # noqa: E402
    allowed_callback_hosts,
    callback_host_mode,
    is_callback_allowed,
)
import main as worker  # noqa: E402

ALLOWED = "swim-sight-3d-v1.vercel.app"
GOOD_URL = f"https://{ALLOWED}/api/ai/callback?job=123"
ENV_ONE = {"AI_CALLBACK_ALLOWED_HOSTS": ALLOWED}


def check(cond, label):
    if not cond:
        raise AssertionError(label)


def test_validator():
    # allowed exact host, https, ignores query
    ok, host, reason = is_callback_allowed(GOOD_URL, env=ENV_ONE)
    check(ok and host == ALLOWED and reason == "ok", "exact https host should be allowed")

    # case-insensitive host match
    ok, host, _ = is_callback_allowed(f"https://{ALLOWED.upper()}/api/ai/callback", env=ENV_ONE)
    check(ok and host == ALLOWED, "host match must be case-insensitive")

    # http rejected
    ok, _, reason = is_callback_allowed(f"http://{ALLOWED}/api/ai/callback", env=ENV_ONE)
    check(not ok and reason == "scheme_not_https", "http must be rejected")

    # different host rejected
    ok, _, reason = is_callback_allowed("https://attacker.tld/api/ai/callback", env=ENV_ONE)
    check(not ok and reason == "host_not_allowed", "unlisted host must be rejected")

    # suffix look-alike rejected (no wildcard/suffix logic)
    ok, _, reason = is_callback_allowed(f"https://{ALLOWED}.attacker.tld/api", env=ENV_ONE)
    check(not ok and reason == "host_not_allowed", "suffix look-alike must be rejected")

    # empty allowlist -> nothing allowed
    ok, _, reason = is_callback_allowed(GOOD_URL, env={})
    check(not ok and reason == "no_allowlist_configured", "empty allowlist must reject")

    # missing host
    ok, _, reason = is_callback_allowed("https:///api/ai/callback", env=ENV_ONE)
    check(not ok and reason == "missing_host", "missing host must be rejected")

    # multiple hosts, one matches
    multi = {"AI_CALLBACK_ALLOWED_HOSTS": f"a.example.com, {ALLOWED} , b.example.com"}
    ok, host, _ = is_callback_allowed(GOOD_URL, env=multi)
    check(ok and host == ALLOWED, "comma list should match")
    check(allowed_callback_hosts(multi) == ["a.example.com", ALLOWED, "b.example.com"], "host list parse")

    # mode parsing
    check(callback_host_mode(env={}) == "monitor", "default mode is monitor")
    check(callback_host_mode(env={"AI_CALLBACK_HOST_MODE": "ENFORCE"}) == "enforce", "enforce parsed")
    check(callback_host_mode(env={"AI_CALLBACK_HOST_MODE": "junk"}) == "monitor", "junk -> monitor")

    # returned host never carries the query string / secret
    _, host, _ = is_callback_allowed(f"https://{ALLOWED}/cb?token=should-not-appear", env=ENV_ONE)
    check("token" not in host and "?" not in host, "host must not include query/secret")


def _set_host_env(mode, hosts):
    os.environ["AI_CALLBACK_HOST_MODE"] = mode
    if hosts is None:
        os.environ.pop("AI_CALLBACK_ALLOWED_HOSTS", None)
    else:
        os.environ["AI_CALLBACK_ALLOWED_HOSTS"] = hosts


def test_main_guard():
    try:
        # monitor: allowed and disallowed BOTH proceed to send (returns True)
        _set_host_env("monitor", ALLOWED)
        check(worker._guard_callback_host(GOOD_URL, "job") is True, "monitor allowed -> send")
        check(worker._guard_callback_host("https://attacker.tld/cb", "job") is True, "monitor disallowed -> still send")

        # enforce: allowed proceeds, disallowed is blocked BEFORE send
        _set_host_env("enforce", ALLOWED)
        check(worker._guard_callback_host(GOOD_URL, "job") is True, "enforce allowed -> send")
        check(worker._guard_callback_host("https://attacker.tld/cb", "job") is False, "enforce disallowed -> blocked")
        check(worker._guard_callback_host(f"http://{ALLOWED}/cb", "job") is False, "enforce http -> blocked")

        # enforce with no allowlist -> fail closed (block everything)
        _set_host_env("enforce", None)
        check(worker._guard_callback_host(GOOD_URL, "job") is False, "enforce no-allowlist -> blocked")
    finally:
        os.environ.pop("AI_CALLBACK_HOST_MODE", None)
        os.environ.pop("AI_CALLBACK_ALLOWED_HOSTS", None)


def main() -> int:
    test_validator()
    test_main_guard()
    print("callback host allowlist tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
