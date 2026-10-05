"""Conversation-history safety net.

The stores under ``chat_history/``, ``episodic/``, ``character_state/``,
``world_state/`` and ``proactive_state/`` are the only record of a real
companion's shared past. They are plain JSON files with no version control and
no trash, so a single mistaken ``rm`` is unrecoverable -- which is exactly what
happened once already during a test campaign.

This module takes a cheap, timestamped, size-bounded snapshot of those stores
before the server starts and prunes only the snapshots it owns. It is purely
defensive:

- it never writes inside the live stores, only in ``backups/``;
- it never deletes anything except its own old snapshots, and only after a
  successful new one exists;
- every failure is logged and swallowed, because a backup problem must never
  stop the server from starting.
"""

import os
import shutil
import time
from typing import List

from loguru import logger

# Everything that represents "what happened between us".
PROTECTED_STORES = (
    "chat_history",
    "episodic",
    "character_state",
    "world_state",
    "proactive_state",
)

BACKUP_ROOT = "backups"
# Keep a handful of recent snapshots; each is small (JSON), so this is cheap.
MAX_SNAPSHOTS = int(os.environ.get("MILI_BACKUP_KEEP", "10"))
# Skip the whole thing when there is nothing worth saving.
MIN_FILE_COUNT = 1


def _iter_files(root: str) -> List[str]:
    out: List[str] = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.endswith((".json", ".jsonl")):
                out.append(os.path.join(dirpath, name))
    return out


def _prune(keep: int = MAX_SNAPSHOTS) -> None:
    """Delete only our own snapshots, oldest first, never the live stores."""
    try:
        if not os.path.isdir(BACKUP_ROOT):
            return
        entries = sorted(
            name
            for name in os.listdir(BACKUP_ROOT)
            if name.startswith("snapshot-") and os.path.isdir(
                os.path.join(BACKUP_ROOT, name)
            )
        )
        for name in entries[: max(0, len(entries) - max(0, keep))]:
            target = os.path.join(BACKUP_ROOT, name)
            # belt and braces: never step outside the backup root
            if os.path.abspath(target).startswith(os.path.abspath(BACKUP_ROOT)):
                shutil.rmtree(target, ignore_errors=True)
    except Exception as error:  # pragma: no cover - defensive
        logger.debug("Backup prune skipped: type={}", type(error).__name__)


def take_snapshot(reason: str = "startup") -> str:
    """Copy the protected stores into a timestamped backup. Returns its path.

    Fail-soft by design: on any problem it logs at debug level and returns an
    empty string rather than raising, because losing a backup must never be a
    reason the companion fails to start.
    """
    try:
        present = [
            store
            for store in PROTECTED_STORES
            if os.path.isdir(store) and len(_iter_files(store)) >= MIN_FILE_COUNT
        ]
        if not present:
            return ""

        # Microseconds matter: two restarts inside the same second would
        # otherwise reuse one directory, and ``copytree(dirs_exist_ok=True)``
        # would then overwrite the very copy we took to stay safe.
        stamp = time.strftime("%Y%m%d-%H%M%S") + f"-{time.time_ns() % 1_000_000:06d}"
        destination = os.path.join(BACKUP_ROOT, f"snapshot-{stamp}")
        os.makedirs(destination, exist_ok=False)
        copied = 0
        for store in present:
            shutil.copytree(store, os.path.join(destination, store), dirs_exist_ok=True)
            copied += 1
        # Keep a note of why this snapshot exists, so an old one is identifiable.
        with open(
            os.path.join(destination, "SNAPSHOT.txt"), "w", encoding="utf-8"
        ) as handle:
            handle.write(f"reason={reason}\ncreated={stamp}\nstores={copied}\n")
        logger.info(
            "Conversation snapshot: path={} stores={} reason={}",
            destination,
            copied,
            reason,
        )
        _prune()
        return destination
    except Exception as error:
        logger.debug("Conversation snapshot skipped: type={}", type(error).__name__)
        return ""


def restore_snapshot(snapshot_name: str) -> List[str]:
    """Copy a snapshot back over the live stores. Returns the stores restored.

    Deliberately explicit and never automatic: restoring must be a human
    decision. The live stores are moved aside first, so a mistaken restore is
    itself recoverable.
    """
    source = os.path.join(BACKUP_ROOT, os.path.basename(snapshot_name))
    if not os.path.isdir(source):
        raise FileNotFoundError(f"no such snapshot: {source}")
    restored: List[str] = []
    for store in PROTECTED_STORES:
        origin = os.path.join(source, store)
        if not os.path.isdir(origin):
            continue
        if os.path.isdir(store):
            aside = f"{store}.replaced-{time.strftime('%Y%m%d-%H%M%S')}"
            os.replace(store, aside)
            logger.warning(
                "Previous store moved aside: store={} path={}", store, aside
            )
        shutil.copytree(origin, store)
        restored.append(store)
    logger.info("Snapshot restored: snapshot={} stores={}", snapshot_name, restored)
    return restored
