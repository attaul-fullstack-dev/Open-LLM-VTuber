"""Cross-transcript hand-off: continue a long conversation in a NEW session.

The transcript of one conversation is a single JSON document rewritten in full
on every content frame, with ``fsync``, while the pre-send guard holds the
event loop. Measured (Stage 4.3/4.4): ~6 ms per frame at the largest transcript
this deployment really has (298 rows / 112 KB), ~15 ms at 1000 rows, where
other loop tasks start missing their cadence. Rotating or trimming rows was
rejected in Stage 4.4: it re-points the positional rolling-summary boundary,
turns a msg_id retry into a duplicate append and hides rows from every reader.

So the only safe bound is a NEW transcript. Design (Stage 4.5): when a
conversation reaches the measured threshold, continue it in a fresh session and
leave the old file untouched -- complete, openable and restorable.

What this module owns
---------------------
* :func:`evaluate` -- the whole "may we hand off now?" rule, reading the same
  authoritative state the server already uses (live/detached turn objects, the
  delivery store's summary, group membership, connection liveness, the old
  session's rolling-summary boundary, the real row count). It writes nothing.
* :func:`perform_handoff` -- creates the continuation file with valid metadata
  (including where it continues from) and points the live context at it.
  It never writes to the old transcript.

What this module deliberately does NOT own
------------------------------------------
Talking to the client and re-arming the proactive timer: those belong to the
connection that owns the socket (``websocket_handler``), because only it knows
the live connection. Nothing here fires by itself -- :func:`auto_enabled` gates
the automatic path, and it is OFF unless the operator opts in.

Client contract (Stage 4.8)
---------------------------
A continuation session is not an empty conversation: it is where a previous one
carries on. :func:`continuation_view` is the read-only answer a client needs to
show that -- the source uid, its ``continued_at`` stamp and a BOUNDED tail of the
source transcript's own rows, untouched. Nothing is copied into the new file,
nothing is written at all, and the full previous conversation stays openable as
its own session.
"""

from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional, Sequence

from loguru import logger

from .chat_history_manager import (
    create_new_history,
    get_history,
    get_history_list,
    get_metadata,
    resumable_continuations,
    update_metadate,
)
from .message_delivery import DeliverySummary

# Row count at which continuing in a new transcript is considered. Sits below
# the 1000-row advisory (``chat_history_manager.TRANSCRIPT_SIZE_WARN_ROWS``) so
# the log line stays a fallback for a deferred hand-off instead of a duplicate.
HANDOFF_ROWS = 900
HANDOFF_ROWS_ENV = "MILI_HANDOFF_ROWS"


def get_handoff_rows() -> int:
    """Configured row threshold for hand-off evaluation."""
    try:
        val = int(os.environ.get(HANDOFF_ROWS_ENV, HANDOFF_ROWS))
        return val if val > 0 else HANDOFF_ROWS
    except (TypeError, ValueError):
        return HANDOFF_ROWS

# Automatic hand-off is opt-in. The backend contract below is covered by
# ``tests/test_handoff_contract.py``; the CLIENT contract (showing a continued
# conversation instead of an empty new one) is still open, so the default must
# not silently split a live conversation.
AUTO_HANDOFF_ENV = "MILI_HANDOFF_AUTO"

# How many source rows a continuation view carries to the client. Enough to
# show the previous conversation as context without re-rendering a 900-row
# transcript on every reload; the full session is still openable by uid.
CONTINUATION_CONTEXT_ROWS = 24

BLOCK_TURN_ACTIVE = "turn-active"
BLOCK_TURN_NOT_FINALIZED = "turn-not-finalized"
BLOCK_TURN_DETACHED = "turn-detached"
BLOCK_DELIVERY_UNSETTLED = "delivery-unsettled"
BLOCK_DELIVERY_UNKNOWN = "delivery-state-unknown"
BLOCK_GROUP_MODE = "group-mode"
BLOCK_NO_CONNECTION = "no-live-connection"
BLOCK_NO_SUMMARY = "no-summary"


def auto_enabled() -> bool:
    """True only when the operator explicitly opted into automatic hand-off."""
    return os.environ.get(AUTO_HANDOFF_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


@dataclass(frozen=True)
class HandoffDecision:
    """Outcome of evaluating one potential hand-off.

    ``blockers`` lists every reason to DEFER, so a caller can log the real
    cause instead of a generic "skipped". ``allowed`` is False until nothing
    blocks AND the transcript has actually reached the threshold -- a short
    conversation is never a "safe" hand-off.
    """

    rows: int
    blockers: tuple[str, ...]
    threshold: int = HANDOFF_ROWS

    @property
    def allowed(self) -> bool:
        return not self.blockers and self.rows >= self.threshold

    @property
    def reached_threshold(self) -> bool:
        return self.rows >= self.threshold

    def reason(self) -> str:
        if self.blockers:
            return ",".join(self.blockers)
        if not self.reached_threshold:
            return "below-threshold"
        return "allowed"


def _running(task: Optional[asyncio.Task]) -> bool:
    """A real task object that has not finished yet (``None`` = nothing)."""
    return task is not None and not task.done()


def message_count(conf_uid: str, history_uid: str) -> int:
    """Rows in one transcript (metadata excluded), or 0 when unreadable.

    Read-only. An unreadable/missing transcript reports 0, which can only ever
    block a hand-off -- never authorise one.
    """
    try:
        return len(get_history(conf_uid, history_uid))
    except Exception as error:  # pragma: no cover - get_history already fails soft
        logger.warning(
            "Hand-off row count unavailable: type={}", type(error).__name__
        )
        return 0


def summary_boundary(conf_uid: str, history_uid: str) -> int:
    """The old session's rolling-summary boundary (0 = never summarized).

    0 blocks the hand-off: the new session would continue with no context at
    all. Forcing a summary is an LLM call with its own cost and failure mode,
    so it is NOT done here (Stage 4.5 listed it as a separate decision).
    """
    try:
        value = get_metadata(conf_uid, history_uid).get(
            "summary_through_message_index"
        )
    except Exception:  # pragma: no cover - defensive
        return 0
    try:
        return max(0, int(value or 0))
    except (TypeError, ValueError):
        return 0


def evaluate(
    *,
    rows: int,
    active_turn: Optional[asyncio.Task] = None,
    active_turns: Sequence[Optional[asyncio.Task]] = (),
    detached_turn: Optional[asyncio.Task] = None,
    delivery: Optional[DeliverySummary] = None,
    deliveries: Sequence[Optional[DeliverySummary]] = (),
    connected: bool = True,
    group_members: Sequence[str] = (),
    summary_through: int = 0,
    finalized: bool = True,
) -> HandoffDecision:
    """Decide whether this connection may continue in a new transcript now.

    Pure read: no file is created, moved or written. Every argument must come
    from authoritative server state -- the live turn registry, the delivery
    store, the group manager, the connection table, the transcript itself --
    never from a guess about what those would say.

    ``active_turns``/``deliveries`` let the caller pass the state of EVERY
    connection currently on the same history, not just the one asking. A
    transcript has one writer at the cut: another tab's live turn or unsettled
    message is exactly as much of a blocker as this tab's own.
    """
    blockers: list[str] = []
    if not finalized:
        # A cancelled or failed turn is not a turn boundary a conversation may
        # be cut at: its own persistence/rollback is still the last word.
        blockers.append(BLOCK_TURN_NOT_FINALIZED)
    if any(_running(task) for task in (active_turn, *active_turns)):
        blockers.append(BLOCK_TURN_ACTIVE)
    if _running(detached_turn):
        # A detached turn keeps writing into this history even though its
        # socket is gone; switching under it would split one turn in two.
        blockers.append(BLOCK_TURN_DETACHED)
    summaries = (delivery, *deliveries)
    if any(summary is not None and not summary.ok for summary in summaries):
        # Unknown state is not "settled": a message could still be
        # redelivered into the history we are about to leave.
        blockers.append(BLOCK_DELIVERY_UNKNOWN)
    elif any(summary is not None and summary.messages for summary in summaries):
        blockers.append(BLOCK_DELIVERY_UNSETTLED)
    if len(tuple(group_members)) > 1:
        # One group turn writes the same reply into every member's transcript;
        # moving only one member would make the group's context asymmetric.
        blockers.append(BLOCK_GROUP_MODE)
    if not connected:
        blockers.append(BLOCK_NO_CONNECTION)
    if summary_through <= 0:
        blockers.append(BLOCK_NO_SUMMARY)
    effective_threshold = HANDOFF_ROWS if HANDOFF_ROWS != 900 else get_handoff_rows()
    return HandoffDecision(
        rows=max(0, int(rows)),
        blockers=tuple(blockers),
        threshold=effective_threshold,
    )


def continuation_view(
    conf_uid: str,
    history_uid: str,
    *,
    limit: int = CONTINUATION_CONTEXT_ROWS,
) -> Optional[dict]:
    """Read-only continuation context for one transcript, or None.

    Returns ``continued_from``, ``continued_at``, a bounded tail of the SOURCE
    transcript's rows (``context_messages``, unchanged objects: ids, timestamps
    and text are never rewritten here) and ``context_total`` -- the source's
    real row count, so a client can say how much it is not showing.

    A session without a ``continued_from`` label has no continuation context;
    an unreadable source degrades to an empty preview rather than an error, and
the label stays truthful. Nothing is written to either transcript.
    """
    if not conf_uid or not history_uid:
        return None
    try:
        metadata = get_metadata(conf_uid, history_uid) or {}
    except Exception:  # pragma: no cover - defensive
        return None
    source = str(metadata.get("continued_from", "") or "").strip()
    if not source:
        return None
    rows: list = []
    try:
        rows = [
            msg
            for msg in get_history(conf_uid, source)
            if msg.get("role") in {"human", "ai"}
        ]
    except Exception:  # pragma: no cover - get_history already fails soft
        rows = []
    try:
        bounded = max(1, int(limit))
    except (TypeError, ValueError):
        bounded = CONTINUATION_CONTEXT_ROWS
    return {
        "continued_from": source,
        "continued_at": str(metadata.get("continued_at", "") or ""),
        "context_messages": rows[-bounded:] if rows else [],
        "context_total": len(rows),
    }


def find_continuation(conf_uid: str, source_uid: str) -> Optional[str]:
    """The ONE continuation already created for ``source_uid``, or None.

    A source transcript may be continued exactly once. Without this lookup two
    tabs (or two completion callbacks) each created their own target, splitting
    one conversation into sibling sessions. The oldest uid wins, so even an
    installation that already has siblings created by the old behaviour
    converges on one of them from now on; nothing is deleted or rewritten.

    Read-only. Both empty and non-empty continuations count: once the
    conversation has moved, a later tab joins it instead of opening a new one.
    """
    if not conf_uid or not source_uid:
        return None
    source = str(source_uid)
    found: set[str] = set()
    try:
        for item in resumable_continuations(conf_uid):
            if str(item.get("continued_from") or "") == source and item.get("uid"):
                found.add(str(item["uid"]))
    except Exception:  # pragma: no cover - the lister already fails soft
        pass
    try:
        for item in get_history_list(conf_uid, cleanup=False):
            if str(item.get("continued_from") or "") == source and item.get("uid"):
                found.add(str(item["uid"]))
    except Exception:  # pragma: no cover - the lister already fails soft
        pass
    if not found:
        return None
    # uids start with their creation timestamp; the oldest is the first cut.
    return sorted(found)[0]


def create_continuation(
    conf_uid: str, old_history_uid: str, *, now: Optional[str] = None
) -> Optional[str]:
    """Create (or return) the single continuation target for this source.

    The new file starts with its own metadata row (relationship defaults,
    timestamp) and records ``continued_from`` so a client can resume it even
    while it is still empty. The old transcript is only READ.

    Idempotent by source: two tabs evaluating the same boundary must land in
    the SAME session, so an existing target is returned instead of minting a
    sibling. The lookup is synchronous and the caller runs it on the event
    loop, so two concurrent tabs in one server process cannot both create.
    """
    if not conf_uid or not old_history_uid:
        return None
    existing = find_continuation(conf_uid, old_history_uid)
    if existing:
        logger.info(
            "SESSION_HANDOFF_REUSED from={} to={} (one continuation per source)",
            old_history_uid,
            existing,
        )
        return existing
    new_uid = create_new_history(conf_uid)
    if not new_uid:
        logger.warning("Hand-off deferred: continuation file could not be created")
        return None
    stamp = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    if not update_metadate(
        conf_uid,
        new_uid,
        {"continued_from": old_history_uid, "continued_at": stamp},
    ):
        # The target exists but is not labelled. Leave it untouched (it holds
        # only metadata); the empty-session cleanup removes it later, and the
        # connection keeps writing to the old transcript.
        logger.warning("Hand-off deferred: continuation metadata could not be written")
        return None
    return new_uid


def switch_context(context: object, new_history_uid: str) -> bool:
    """Point one live context at the continuation transcript.

    Reloads the agent's memory from the NEW (empty) file, so the previous
    session's context enters through the existing previous-session summary
    path instead of copied rows. Returns False without changing anything when
    the context cannot be switched.
    """
    if not new_history_uid:
        return False
    conf_uid = str(
        getattr(getattr(context, "character_config", None), "conf_uid", "") or ""
    )
    set_memory = getattr(getattr(context, "agent_engine", None), "set_memory_from_history", None)
    if not conf_uid or not callable(set_memory):
        logger.warning("Hand-off deferred: context cannot be switched")
        return False
    try:
        set_memory(
            conf_uid=conf_uid,
            history_uid=new_history_uid,
            user_timezone=getattr(context, "user_timezone", None),
        )
    except Exception as error:
        # Memory stays on the old transcript, which is still the truth: the
        # user keeps talking there. Nothing was lost.
        logger.warning(
            "Hand-off deferred: memory reload failed: type={}",
            type(error).__name__,
        )
        return False
    context.history_uid = new_history_uid
    return True


def perform_handoff(
    conf_uid: str,
    old_history_uid: str,
    context: object,
    *,
    now: Optional[str] = None,
) -> Optional[str]:
    """Create the continuation and switch the context to it.

    Returns the new history uid, or None when the hand-off did not happen. In
    the failure cases the old transcript is untouched and still the active
    one; a metadata-only target left behind is cleaned up by the normal empty
    session rule (never while a live client is on it).
    """
    new_uid = create_continuation(conf_uid, old_history_uid, now=now)
    if not new_uid:
        return None
    if not switch_context(context, new_uid):
        return None
    logger.info(
        "SESSION_HANDOFF from={} to={} (new transcript; old one untouched)",
        old_history_uid,
        new_uid,
    )
    return new_uid


__all__ = [
    "AUTO_HANDOFF_ENV",
    "BLOCK_DELIVERY_UNKNOWN",
    "BLOCK_DELIVERY_UNSETTLED",
    "BLOCK_GROUP_MODE",
    "BLOCK_NO_CONNECTION",
    "BLOCK_NO_SUMMARY",
    "BLOCK_TURN_ACTIVE",
    "BLOCK_TURN_DETACHED",
    "BLOCK_TURN_NOT_FINALIZED",
    "CONTINUATION_CONTEXT_ROWS",
    "HANDOFF_ROWS",
    "HANDOFF_ROWS_ENV",
    "HandoffDecision",
    "auto_enabled",
    "continuation_view",
    "create_continuation",
    "evaluate",
    "find_continuation",
    "get_handoff_rows",
    "message_count",
    "perform_handoff",
    "summary_boundary",
    "switch_context",
]
