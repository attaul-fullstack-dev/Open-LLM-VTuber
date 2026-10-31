"""Reproducible event-loop-blocking benchmark for the persist-before-send guard.

Stage 4.2 measured the *cost* of the per-frame durable write (3-7 ms at real
transcript sizes) but not what that cost does to the asyncio loop the whole
server shares. The guard's ``before_content`` is declared ``async`` but its
body calls ``store_message`` synchronously, so the loop is frozen for the
whole read + serialize + fsync + replace transaction. Stage 4.3 asks:

1. how long is the loop actually blocked, per frame and per turn, and
2. is that measurable above the loop's own scheduling jitter at the transcript
   sizes this deployment really has?

So the benchmark reports loop lag three ways:

``none``      a turn with the guard disabled: the noise floor of the harness
              (sleep jitter, GC, allocator) with no disk work at all.
``current``   the shipped guard, untouched: synchronous store inside the loop.
``offload``   a candidate that awaits ``asyncio.to_thread(store_message)`` but
              still waits for the write before allowing the frame. Persistence
              order is unchanged; only the thread that performs it changes.

Plus two derived numbers:

* turn time (wall clock for ``frames`` content frames + finalize), and
* the inter-frame gap of a SECOND, independent async task (a stand-in for the
  audio/websocket traffic sharing the loop) while the turn persists. That is
  the "impact on the next frame" the loop block actually causes.

Every number is measured, nothing is modelled. Not a ``test_*`` file, so
``unittest discover`` does not collect it.

Usage
-----
    python tests/bench_loop_blocking.py
    python tests/bench_loop_blocking.py --impl offload --rows 300 --frames 12
    python tests/bench_loop_blocking.py --json
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import statistics
import sys
import tempfile
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

from bench_history_persist import (  # noqa: E402
    CONF_UID,
    FRAME_TEXT,
    HISTORY_UID,
    build_transcript,
    percentile,
)
from src.open_llm_vtuber import chat_history_manager as chm  # noqa: E402
from src.open_llm_vtuber.conversations.single_conversation import (  # noqa: E402
    _DurableAIContent,
)

# Heartbeat cadence. 1 ms is finer than any schedulable frame and coarse
# enough that the loop is not pure timer thrash.
TICK_S = 0.001


class _OffloadGuard(_DurableAIContent):
    """Candidate: same order, same fsync, executed on a worker thread."""

    async def before_content(self, text_delta: str) -> bool:
        if not self.enabled:
            return True
        if self.failed:
            return False
        delta = str(text_delta or "")
        if not delta:
            return True
        candidate = self.accepted + delta
        if candidate == self.persisted:
            return True
        if await asyncio.to_thread(self._store, candidate):
            self.persisted = candidate
            return True
        self.failed = True
        return False

    async def finalize(self, canonical_text: str = "") -> tuple[bool, str]:
        if not self.enabled:
            return True, ""
        text = str(canonical_text or "")
        if self.failed:
            stored = await asyncio.to_thread(self._restore, self.accepted)
            return stored, self.accepted if stored else ""
        if not text:
            stored = await asyncio.to_thread(self._restore, "")
            return stored, ""
        if self.persisted == text:
            return True, text
        stored = await asyncio.to_thread(self._store, text)
        if not stored:
            self.failed = True
            return False, ""
        self.persisted = text
        return True, text


class _DisabledGuard(_OffloadGuard):
    """No-op turn: the harness' own scheduling jitter, no disk work."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.enabled = False

    async def before_content(self, text_delta: str) -> bool:  # type: ignore[override]
        return True

    async def finalize(self, canonical_text: str = "") -> tuple[bool, str]:  # type: ignore[override]
        return True, str(canonical_text or "")


IMPLS = {
    "none": _DisabledGuard,
    "current": _DurableAIContent,
    "offload": _OffloadGuard,
}


class _Heartbeat:
    """Sample how late a 1 ms timer actually fires: that lateness IS loop lag."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, float]] = []
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        while not self._stop.is_set():
            started = time.perf_counter()
            await asyncio.sleep(TICK_S)
            finished = time.perf_counter()
            # Everything past the requested 1 ms is time the loop could not
            # run this timer, i.e. time it spent blocked elsewhere.
            lag = (finished - started - TICK_S) * 1000
            self.samples.append((finished, lag))

    async def __aenter__(self):
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *exc_info):
        self._stop.set()
        if self._task is not None:
            await self._task
        return False

    def lags_between(self, start: float, end: float) -> list[float]:
        return [lag for stamp, lag in self.samples if start <= stamp <= end]


class _ConcurrentTraffic:
    """A second async task sending a frame every 100 ms, as the server does.

    Its inter-frame gaps show the loop block from the outside: a task that only
    sleeps cannot be delayed except by the loop being busy.
    """

    INTERVAL_S = 0.1

    def __init__(self) -> None:
        self.gaps: list[float] = []
        self._stop = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def _run(self) -> None:
        last = None
        while not self._stop.is_set():
            await asyncio.sleep(self.INTERVAL_S)
            now = time.perf_counter()
            if last is not None:
                self.gaps.append((now - last) * 1000)
            last = now

    async def __aenter__(self):
        self._task = asyncio.create_task(self._run())
        # Let it establish its cadence before the turn starts.
        await asyncio.sleep(self.INTERVAL_S * 3)
        self.gaps.clear()
        return self

    async def __aexit__(self, *exc_info):
        self._stop.set()
        if self._task is not None:
            await self._task
        return False


async def run_turn(impl: str, rows: int, frames: int, interval_s: float) -> dict:
    """One turn of ``frames`` content frames over a fresh transcript.

    ``interval_s`` models the gap before each frame: in production the sender
    only reaches the next content frame after the TTS payload is synthesised,
    so the loop gets to run other work between frames. Without that gap every
    frame of the turn is one contiguous block, which overstates the stall.
    """
    os.makedirs("chat_history", exist_ok=True)
    path = build_transcript(rows)
    guard_cls = IMPLS[impl]

    async with _Heartbeat() as heartbeat, _ConcurrentTraffic() as traffic:
        started = time.perf_counter()
        guard = guard_cls(
            enabled=True,
            conf_uid=CONF_UID,
            history_uid=HISTORY_UID,
            msg_id="bench-loop-0001",
            name="Mili",
            avatar="mili.png",
        )
        init_ms = (time.perf_counter() - started) * 1000

        frame_ms: list[float] = []
        guard_ms: list[float] = []
        starts: list[float] = []
        for _ in range(frames):
            await asyncio.sleep(interval_s)
            starts.append(time.perf_counter())
            frame_started = time.perf_counter()
            allowed = await guard.before_content(FRAME_TEXT)
            guard_ms.append((time.perf_counter() - frame_started) * 1000)
            if not allowed:
                raise RuntimeError(f"{impl}: guard blocked a frame")
            await guard.after_content(FRAME_TEXT, True)
            frame_ms.append((time.perf_counter() - frame_started) * 1000)

        final_started = time.perf_counter()
        await guard.finalize(FRAME_TEXT * frames)
        finalize_ms = (time.perf_counter() - final_started) * 1000
        turn_ms = (time.perf_counter() - started) * 1000
        end = time.perf_counter()

    lags = heartbeat.lags_between(started, end)
    cadence = [
        (later - earlier - interval_s) * 1000
        for earlier, later in zip(starts, starts[1:])
    ]
    return {
        "impl": impl,
        "rows": rows,
        "frames": frames,
        "init_ms": init_ms,
        "guard_ms": guard_ms,
        "frame_ms": frame_ms,
        "finalize_ms": finalize_ms,
        "turn_ms": turn_ms,
        "lags": lags,
        "cadence": cadence,
        "gaps": list(traffic.gaps),
        "transcript_bytes": os.path.getsize(path),
    }


def _stats(values: list[float], budget_ms: float = 0.0) -> dict:
    if not values:
        return {
            "med": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "over": 0.0,
            "count": 0,
        }
    return {
        "med": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
        "over": 100 * sum(1 for value in values if value > budget_ms) / len(values),
        "count": len(values),
    }


async def measure_noise(duration_s: float = 0.3) -> dict:
    """Loop lag with nothing to do: the harness' own scheduling jitter."""
    async with _Heartbeat() as heartbeat:
        await asyncio.sleep(duration_s)
    lags = [lag for _stamp, lag in heartbeat.samples]
    return _stats(lags, 5.0)


def summarize(samples: list[dict]) -> dict:
    guard = _stats([value for sample in samples for value in sample["guard_ms"]], 5.0)
    turn = _stats([sample["turn_ms"] for sample in samples], 100.0)
    lags = _stats([value for sample in samples for value in sample["lags"]], 5.0)
    gaps = _stats([value for sample in samples for value in sample["gaps"]], 110.0)
    cadence = _stats([value for sample in samples for value in sample["cadence"]], 20.0)
    return {
        "impl": samples[0]["impl"],
        "rows": samples[0]["rows"],
        "frames": samples[0]["frames"],
        "bytes": samples[0]["transcript_bytes"],
        "guard_med": guard["med"],
        "guard_p95": guard["p95"],
        "guard_over_5ms": guard["over"],
        "turn_med": turn["med"],
        "turn_p95": turn["p95"],
        "lag_med": lags["med"],
        "lag_p95": lags["p95"],
        "lag_max": lags["max"],
        "lag_over_5ms": lags["over"],
        "cadence_med": cadence["med"],
        "cadence_p95": cadence["p95"],
        "cadence_max": cadence["max"],
        "gap_med": gaps["med"],
        "gap_p95": gaps["p95"],
        "gap_max": gaps["max"],
        "gap_over_budget": gaps["over"],
        "blocked_share": 100
        * statistics.median(
            [sum(sample["guard_ms"]) / sample["turn_ms"] for sample in samples]
        ),
        "lag_samples": lags["count"],
    }


async def _run_scenario(impl: str, accepted: list[bool], msg_id: str) -> dict:
    """Replay one guard scenario and report what reached the disk."""
    os.makedirs("chat_history", exist_ok=True)
    build_transcript(0)
    writes: list[str] = []
    real_write = chm._write_history_atomic

    def counting_write(filepath, history_data):
        writes.append(filepath)
        return real_write(filepath, history_data)

    guard = IMPLS[impl](
        enabled=True,
        conf_uid=CONF_UID,
        history_uid=HISTORY_UID,
        msg_id=msg_id,
        name="Mili",
        avatar="mili.png",
    )
    with contextlib.nullcontext():
        chm._write_history_atomic = counting_write
        try:
            for allow in accepted:
                if not await guard.before_content(FRAME_TEXT):
                    break
                await guard.after_content(FRAME_TEXT, allow)
            await guard.finalize(FRAME_TEXT * len(accepted))
        finally:
            chm._write_history_atomic = real_write

    path = chm._get_safe_history_path(CONF_UID, HISTORY_UID)
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    row = next((item for item in rows if item.get("msg_id") == msg_id), None)
    return {
        "impl": impl,
        "accepted": "".join(str(value) for value in accepted),
        "writes": len(writes),
        "content": (row or {}).get("content"),
        "rows": len(rows),
    }


async def _run_concurrent(impl: str, frames: int = 6) -> dict:
    """Two guards writing the same transcript at the same time."""
    os.makedirs("chat_history", exist_ok=True)
    build_transcript(0)

    async def one(msg_id: str) -> None:
        guard = IMPLS[impl](
            enabled=True,
            conf_uid=CONF_UID,
            history_uid=HISTORY_UID,
            msg_id=msg_id,
            name="Mili",
            avatar="mili.png",
        )
        for _ in range(frames):
            await guard.before_content(FRAME_TEXT)
            await guard.after_content(FRAME_TEXT, True)
        await guard.finalize(FRAME_TEXT * frames)

    await asyncio.gather(one("conc-a"), one("conc-b"))
    path = chm._get_safe_history_path(CONF_UID, HISTORY_UID)
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    contents = {
        item.get("msg_id"): item.get("content")
        for item in rows
        if item.get("role") == "ai"
    }
    return {
        "impl": impl,
        "rows": len(rows),
        "a_ok": contents.get("conc-a") == FRAME_TEXT * frames,
        "b_ok": contents.get("conc-b") == FRAME_TEXT * frames,
        "lengths": {key: len(value or "") for key, value in contents.items()},
    }


async def _run_cancellation(impl: str, delay_s: float = 0.02) -> dict:
    """Cancel a frame guard while its durable write is in flight, then abort.

    This is the one hazard a thread offload could introduce that the current
    synchronous store cannot have: the loop is free while the worker writes,
    so cancellation can arrive mid-write. The row must not end up holding text
    the transport never saw.
    """
    from src.open_llm_vtuber.conversations import single_conversation

    os.makedirs("chat_history", exist_ok=True)
    build_transcript(0)
    real_store = single_conversation.store_message

    def slow_store(*args, **kwargs):
        time.sleep(delay_s)
        return real_store(*args, **kwargs)

    guard = IMPLS[impl](
        enabled=True,
        conf_uid=CONF_UID,
        history_uid=HISTORY_UID,
        msg_id="cancel-1",
        name="Mili",
        avatar="mili.png",
    )
    single_conversation.store_message = slow_store
    try:
        task = asyncio.ensure_future(guard.before_content(FRAME_TEXT))
        await asyncio.sleep(0.005)
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        # Cleanup path: the turn is gone, so the provisional text is dropped.
        await guard.abort()
        # Give any orphaned worker write time to land.
        await asyncio.sleep(delay_s * 4)
    finally:
        single_conversation.store_message = real_store

    path = chm._get_safe_history_path(CONF_UID, HISTORY_UID)
    with open(path, "r", encoding="utf-8") as handle:
        rows = json.load(handle)
    row = next((item for item in rows if item.get("msg_id") == "cancel-1"), None)
    return {
        "impl": impl,
        "row": None if row is None else row.get("content"),
        "leaked_unsent_text": bool(row and row.get("content")),
    }


async def verify(args) -> None:
    """Is a thread offload safe for the CURRENT design? Measure, don't assume."""
    with tempfile.TemporaryDirectory(prefix="mili-verify-") as workdir:
        os.chdir(workdir)
        scenarios = {
            "all": [True, True, True, True],
            "refused_middle": [True, True, False, True],
            "refused_tail": [True, True, False],
        }
        baseline = {}
        for impl in ("current", "offload"):
            for name, accepted in scenarios.items():
                result = await _run_scenario(impl, accepted, f"{impl}-{name}")
                baseline.setdefault(name, []).append(result)
                print(
                    "scenario {impl:>8} {scenario:<15} writes={writes} "
                    "rows={rows} content={content!r}".format(scenario=name, **result),
                    flush=True,
                )
        for name, results in baseline.items():
            same = results[0]["content"] == results[1]["content"] and (
                results[0]["writes"] == results[1]["writes"]
            )
            print(
                f"equivalence {name:<15} identical={same}",
                flush=True,
            )
        for impl in ("current", "offload"):
            print(
                "concurrent {impl:>8} {rows} rows a_ok={a_ok} b_ok={b_ok}".format(
                    **await _run_concurrent(impl)
                ),
                flush=True,
            )
        for impl in ("current", "offload"):
            for _ in range(3):
                result = await _run_cancellation(impl)
                print(
                    "cancellation {impl:>8} row={row!r} "
                    "leaked_unsent_text={leaked_unsent_text}".format(**result),
                    flush=True,
                )


async def main_async(args) -> list[dict]:
    results = []
    impls = args.impl or list(IMPLS)
    rows_list = args.rows or [20, 300, 1000, 2000]
    frames = args.frames or 12
    with tempfile.TemporaryDirectory(prefix="mili-loop-") as workdir:
        os.chdir(workdir)
        noise = await measure_noise()
        print(
            "noise    (idle 300ms) lag_med={med:5.2f}ms lag_p95={p95:5.2f}ms "
            "lag_max={max:6.2f}ms lag>5ms={over:5.1f}% n={count}".format(**noise),
            flush=True,
        )
        for impl in impls:
            for rows in rows_list:
                samples = [
                    await run_turn(impl, rows, frames, args.interval / 1000)
                    for _ in range(args.repeat)
                ]
                summary = summarize(samples)
                results.append(summary)
                print(
                    "{impl:>8} rows={rows:>5} bytes={bytes:>7} frames={frames:>3} "
                    "guard_med={guard_med:6.2f}ms guard_p95={guard_p95:6.2f}ms "
                    "turn_med={turn_med:7.2f}ms "
                    "lag_med={lag_med:5.2f}ms lag_p95={lag_p95:5.2f}ms "
                    "lag_max={lag_max:6.2f}ms lag>5ms={lag_over_5ms:5.1f}% "
                    "cadence_p95={cadence_p95:6.2f}ms "
                    "gap_p95={gap_p95:6.2f}ms gap>110ms={gap_over_budget:5.1f}% "
                    "blocked={blocked_share:5.1f}% n={lag_samples}".format(**summary),
                    flush=True,
                )
    return results


def main(argv=None) -> int:
    from loguru import logger

    logger.remove()  # the store logs every write at DEBUG; the table is output
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", action="append", choices=sorted(IMPLS))
    parser.add_argument("--rows", action="append", type=int)
    parser.add_argument("--frames", type=int)
    parser.add_argument(
        "--interval",
        type=float,
        default=50.0,
        help="ms between content frames (0 = back-to-back, one long block)",
    )
    parser.add_argument("--repeat", type=int, default=7)
    parser.add_argument("--json", action="store_true")
    parser.add_argument(
        "--verify",
        action="store_true",
        help="replay the durability contract for current vs offload",
    )
    args = parser.parse_args(argv)
    if args.verify:
        asyncio.run(verify(args))
        return 0
    results = asyncio.run(main_async(args))
    if args.json:
        summary = [{key: value for key, value in item.items()} for item in results]
        print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
