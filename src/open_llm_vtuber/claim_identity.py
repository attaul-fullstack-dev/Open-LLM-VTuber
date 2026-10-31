"""Claim-owner identity used to recover reminders after a process crash.

A reminder claim is persisted in the character-state file so a second
connection cannot deliver the same row. The lease (600 s) bounds how long a
claim left by a crashed process can block delivery -- but waiting the whole
lease is unnecessary when the owner is *provably* dead.

Every claim owner therefore carries a process token::

    linux:<boot-id>:<pid-ns>:<pid>:<process-start-ticks>|<client_uid>:<uuid>

Recovery clears a claim only when that token proves the owning process is
gone (different boot, missing PID, or a reused PID). When the process is
still alive -- including a second instance sharing the same data directory --
a different PID namespace is in play, or the identity cannot be read, the
claim is left to the lease. The design is fail-safe: the worst case is the
previous 600 s wait, never a double delivery. Claims written by older versions
have no token and also fall back to the lease.
"""

from __future__ import annotations

import os
from typing import Optional

_EPOCH: Optional[str] = None
_BOOT_ID: Optional[str] = None
_PID_NS: Optional[str] = None


def _read(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read().strip()
    except OSError:
        return None


def _boot_id() -> Optional[str]:
    global _BOOT_ID
    if _BOOT_ID is None:
        _BOOT_ID = _read("/proc/sys/kernel/random/boot_id") or ""
    return _BOOT_ID or None


def _pid_namespace() -> Optional[str]:
    """Inode of this process' PID namespace, e.g. ``4026531836``."""
    global _PID_NS
    if _PID_NS is None:
        try:
            target = os.readlink("/proc/self/ns/pid")
        except OSError:
            target = ""
        digits = "".join(char for char in target if char.isdigit())
        _PID_NS = digits
    return _PID_NS or None


def _starttime(pid: int) -> Optional[int]:
    """Field 22 (starttime) of ``/proc/<pid>/stat``, in clock ticks."""
    raw = _read(f"/proc/{int(pid)}/stat")
    if not raw:
        return None
    # comm (field 2) may contain spaces and parentheses; fields after the
    # last ')' start at field 3 (state), so starttime is index 19 there.
    try:
        tail = raw[raw.rfind(")") + 2 :]
        return int(tail.split()[19])
    except (IndexError, ValueError):
        return None


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
        return True
    except ProcessLookupError:
        return False
    except (PermissionError, OSError):
        # Cannot probe it: treat as alive (fail safe, never steal a claim).
        return True


def process_token() -> str:
    """Stable identity of THIS process, or ``""`` when /proc is unavailable."""
    global _EPOCH
    if _EPOCH is None:
        boot = _boot_id()
        pid_ns = _pid_namespace()
        start = _starttime(os.getpid())
        if boot and pid_ns and start is not None:
            _EPOCH = f"linux:{boot}:{pid_ns}:{os.getpid()}:{start}"
        else:
            _EPOCH = ""
    return _EPOCH


def claim_owner(suffix: str) -> str:
    """Prefix a claim-owner suffix with this process token (when available)."""
    token = process_token()
    suffix = str(suffix or "").strip()
    return f"{token}|{suffix}" if token else suffix


def owner_is_dead(owner: str, *, current_token: Optional[str] = None) -> bool:
    """True only when ``owner`` provably belongs to a dead process.

    Unknown/legacy tokens, unreadable /proc data, a different PID namespace
    and live processes all return False so the caller keeps the claim until
    its lease expires.
    """
    raw = str(owner or "")
    token, separator, _suffix = raw.partition("|")
    if not separator or not token:
        return False  # legacy owner format: let the lease decide
    current = process_token() if current_token is None else str(current_token)
    if not current or token == current:
        return False
    parts = token.split(":")
    if len(parts) != 5 or parts[0] != "linux":
        return False
    boot, pid_ns, pid_text, start_text = parts[1], parts[2], parts[3], parts[4]
    current_boot = _boot_id()
    current_ns = _pid_namespace()
    if not current_boot or not current_ns:
        return False
    if boot != current_boot:
        return True  # the machine rebooted: that process cannot be alive
    if pid_ns != current_ns:
        # A PID from another namespace cannot be probed safely here.
        return False
    try:
        pid = int(pid_text)
        start = int(start_text)
    except ValueError:
        return False
    if pid <= 0:
        return False
    observed = _starttime(pid)
    if observed is None:
        return not _pid_alive(pid)
    return observed != start  # PID reused by a different process


__all__ = ["claim_owner", "owner_is_dead", "process_token"]
