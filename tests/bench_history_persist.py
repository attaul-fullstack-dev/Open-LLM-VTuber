"""Reproducible I/O benchmark for the persist-before-send history guard.

Stage 4.1 made the canonical AI row durable *before* every content frame can
leave the process. Production code does that with a full read-modify-write of
the transcript file and an ``fsync`` per frame. Stage 4.2 asks one question
that only a measurement can answer: is that per-frame cost worth replacing
with something incremental?

This script is the measurement, not an optimisation. It drives the REAL guard
(``single_conversation._DurableAIContent``), the REAL store
(``chat_history_manager.store_message`` and its atomic ``os.replace``) and the
REAL cross-process lock, on transcripts of three sizes and turns of three
frame counts, and reports:

* open() calls split into reads and writes
* bytes read and bytes written
* atomic writes (each one is a ``flush`` + ``fsync`` + ``os.replace``)
* median and p95 latency per frame, and the per-turn total

``--impl`` selects what is measured:

``current``    the shipped guard, unchanged.
``readcache``  the same full durable write per frame, plus an in-process parse
               cache so consecutive frames of one turn do not re-read the file.
               Invalidated whenever the file changes underneath (other process).
``sidecar``    per frame only a small draft row is written+fsynced, folded into
               the transcript once at the end of the turn. Cheap per frame, but
               a reload can only see the draft if every reader merges it, so a
               crash mid-turn hides the reply unless the read path changes.

Not a ``test_*`` file: ``unittest discover`` must not collect it.

Usage
-----
    python tests/bench_history_persist.py
    python tests/bench_history_persist.py --impl readcache --repeat 20
    python tests/bench_history_persist.py --rows 200 --frames 4 --json
"""

from __future__ import annotations

import argparse
import asyncio
import builtins
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

from src.open_llm_vtuber import chat_history_manager as chm  # noqa: E402
from src.open_llm_vtuber.conversations.single_conversation import (  # noqa: E402
    _DurableAIContent,
)

CONF_UID = "bench_conf"
HISTORY_UID = "bench_history"

# A sentence's worth of display text: the unit one content frame carries.
FRAME_TEXT = "Aku masih di sini, kok. Kamu sudah minum obat belum hari ini? "


class _Counters(dict):
    def __init__(self) -> None:
        super().__init__(
            opens_read=0,
            opens_write=0,
            read_bytes=0,
            write_bytes=0,
            atomic_writes=0,
        )


class _CountingHandle:
    """File proxy that counts bytes and open() modes, nothing else."""

    def __init__(self, handle, counters: _Counters, mode: str) -> None:
        self._handle = handle
        self._counters = counters
        self._writing = any(flag in mode for flag in ("w", "a", "x", "+"))

    def read(self, *args):
        data = self._handle.read(*args)
        if isinstance(data, str):
            self._counters["read_bytes"] += len(data.encode("utf-8"))
        elif data:
            self._counters["read_bytes"] += len(data)
        return data

    def write(self, data):
        written = self._handle.write(data)
        self._counters["write_bytes"] += (
            len(data.encode("utf-8")) if isinstance(data, str) else len(data)
        )
        return written

    def __enter__(self):
        self._handle.__enter__()
        return self

    def __exit__(self, *exc_info):
        return self._handle.__exit__(*exc_info)

    def __getattr__(self, name):
        return getattr(self._handle, name)


class _Instrumentation:
    """Patch the store's open()/atomic-write so the benchmark can count them."""

    def __init__(self) -> None:
        self.counters = _Counters()
        self._real_open = None
        self._real_atomic = None

    def __enter__(self):
        real_open = chm.__dict__.get("open", builtins.open)
        real_atomic = chm._write_history_atomic
        counters = self.counters

        def counting_open(file, mode="r", *args, **kwargs):
            if any(flag in mode for flag in ("w", "a", "x", "+")):
                counters["opens_write"] += 1
            else:
                counters["opens_read"] += 1
            return _CountingHandle(
                real_open(file, mode, *args, **kwargs), counters, mode
            )

        def counting_atomic(filepath, history_data):
            counters["atomic_writes"] += 1
            return real_atomic(filepath, history_data)

        self._real_open = real_open
        self._real_atomic = real_atomic
        chm.open = counting_open
        chm._write_history_atomic = counting_atomic
        return self

    def __exit__(self, *exc_info):
        if self._real_open is builtins.open:
            chm.__dict__.pop("open", None)
        else:
            chm.open = self._real_open
        chm._write_history_atomic = self._real_atomic
        return False


def build_transcript(rows: int) -> str:
    """Write a transcript of ``rows`` messages and return its path."""
    conf_dir = os.path.join("chat_history", CONF_UID)
    os.makedirs(conf_dir, exist_ok=True)
    path = os.path.join(conf_dir, f"{HISTORY_UID}.json")
    data = [
        {
            "role": "metadata",
            "timestamp": "2026-10-28T00:00:00+00:00",
            "relationship_status": "close",
            "relationship_reason": "benchmark",
        }
    ]
    for index in range(rows):
        role = "human" if index % 2 == 0 else "ai"
        data.append(
            {
                "role": role,
                "timestamp": "2026-10-28T00:00:00+00:00",
                "content": (
                    f"Pesan lama nomor {index}: aku masih inget waktu kita "
                    "ngobrol soal obat dan jadwal tidur kamu."
                ),
                "msg_id": f"legacy{index:05d}",
                "name": "User" if role == "human" else "Mili",
            }
        )
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)
    return path


class _ReadCacheGuard(_DurableAIContent):
    """Candidate: keep the parsed transcript in memory between frames."""

    _cache: dict = {}

    def _store(self, content, *, name=None, avatar=None) -> bool:  # type: ignore[override]
        path = chm._get_safe_history_path(self.conf_uid, self.history_uid)
        try:
            lock = chm._get_history_lock(path)
            with lock:
                stat = os.stat(path)
                key = (path, stat.st_mtime_ns, stat.st_size)
                cached = self._cache.get(path)
                if cached is None or cached[0] != key:
                    with open(path, "r", encoding="utf-8") as handle:
                        cached = (key, json.load(handle))
                    self._cache[path] = cached
                data = cached[1]
                stable_id = chm._normalize_msg_id(self.msg_id)
                existing = next(
                    (
                        item
                        for item in data
                        if item.get("msg_id") == stable_id and item.get("role") == "ai"
                    ),
                    None,
                )
                if existing is not None:
                    existing["content"] = content
                else:
                    data.append(
                        {
                            "role": "ai",
                            "timestamp": "2026-10-28T00:00:00+00:00",
                            "content": content,
                            "msg_id": stable_id,
                            "name": self.name if name is None else name,
                        }
                    )
                chm._write_history_atomic(path, data)
                stat = os.stat(path)
                self._cache[path] = ((path, stat.st_mtime_ns, stat.st_size), data)
        except Exception:
            return False
        return True


class _SidecarGuard(_DurableAIContent):
    """Candidate: durable small draft per frame, folded in once at the end."""

    def _draft_path(self) -> str:
        path = chm._get_safe_history_path(self.conf_uid, self.history_uid)
        return f"{path}.draft.{self.msg_id}"

    def _store(self, content, *, name=None, avatar=None) -> bool:  # type: ignore[override]
        try:
            chm._write_history_atomic(
                self._draft_path(),
                [
                    {
                        "role": "ai",
                        "content": content,
                        "msg_id": chm._normalize_msg_id(self.msg_id),
                        "name": self.name if name is None else name,
                    }
                ],
            )
        except Exception:
            return False
        return True


IMPLS = {
    "current": _DurableAIContent,
    "readcache": _ReadCacheGuard,
    "sidecar": _SidecarGuard,
}


async def run_turn(
    impl: str, rows: int, frames: int, *, instrument: bool = False
) -> dict:
    """One simulated turn of ``frames`` content frames over a fresh transcript.

    ``instrument`` counts open()/bytes. It is a SEPARATE pass on purpose: the
    counting file proxy sits between ``json.dump`` and the disk, and
    ``json.dump`` emits thousands of tiny writes for an indented document, so
    counting during the timed pass would inflate the latency it reports.
    """
    os.makedirs("chat_history", exist_ok=True)
    path = build_transcript(rows)
    _ReadCacheGuard._cache.clear()
    guard_cls = IMPLS[impl]
    scope = _Instrumentation() if instrument else contextlib.nullcontext()

    with scope as instrumentation:
        counters = instrumentation.counters if instrumentation else _Counters()
        started = time.perf_counter()
        guard = guard_cls(
            enabled=True,
            conf_uid=CONF_UID,
            history_uid=HISTORY_UID,
            msg_id="bench-msg-0001",
            name="Mili",
            avatar="mili.png",
        )
        init_ms = (time.perf_counter() - started) * 1000

        frame_times = []
        for _ in range(frames):
            frame_started = time.perf_counter()
            allowed = await guard.before_content(FRAME_TEXT)
            frame_times.append((time.perf_counter() - frame_started) * 1000)
            if not allowed:
                raise RuntimeError(f"{impl}: guard blocked a frame")
            await guard.after_content(FRAME_TEXT, True)

        final_started = time.perf_counter()
        await guard.finalize(FRAME_TEXT * frames)
        finalize_ms = (time.perf_counter() - final_started) * 1000
        turn_ms = (time.perf_counter() - started) * 1000
        transcript_bytes = os.path.getsize(path)

    return {
        "impl": impl,
        "rows": rows,
        "frames": frames,
        "init_ms": init_ms,
        "frame_ms": frame_times,
        "finalize_ms": finalize_ms,
        "turn_ms": turn_ms,
        "opens_read": counters["opens_read"],
        "opens_write": counters["opens_write"],
        "read_bytes": counters["read_bytes"],
        "write_bytes": counters["write_bytes"],
        "atomic_writes": counters["atomic_writes"],
        "transcript_bytes": transcript_bytes,
    }


def breakdown(rows: int, repeat: int) -> dict:
    """Where one durable write's time goes, for the decision.

    ``serialize``  json.dump of the transcript plus flush/close, no fsync
    ``fsync``      the fsync of those bytes only
    ``full``       the store's atomic write (serialize + fsync + replace)

    A persisting writer cannot skip the fsync (that is what makes the row
    durable), so only the serialize part can ever be optimised away.
    """
    path = build_transcript(rows)
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    serialize = []
    sync = []
    full = []
    for _ in range(repeat):
        temporary = f"{path}.{os.getpid()}.tmp"
        started = time.perf_counter()
        handle = open(temporary, "w", encoding="utf-8")
        json.dump(data, handle, ensure_ascii=False, indent=2)
        handle.flush()
        serialize.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        os.fsync(handle.fileno())
        sync.append((time.perf_counter() - started) * 1000)
        handle.close()
        os.replace(temporary, path)
    for _ in range(repeat):
        started = time.perf_counter()
        chm._write_history_atomic(path, data)
        full.append((time.perf_counter() - started) * 1000)
    return {
        "rows": rows,
        "serialize_ms": statistics.median(serialize),
        "fsync_ms": statistics.median(sync),
        "full_ms": statistics.median(full),
        "bytes": os.path.getsize(path),
    }


def percentile(values, fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, round(fraction * (len(ordered) - 1))))
    return ordered[index]


def summarize(samples) -> dict:
    frame_ms = [value for sample in samples for value in sample["frame_ms"]]
    turn_ms = [sample["turn_ms"] for sample in samples]
    return {
        "frame_median_ms": statistics.median(frame_ms),
        "frame_p95_ms": percentile(frame_ms, 0.95),
        "frame_max_ms": max(frame_ms),
        "turn_median_ms": statistics.median(turn_ms),
        "turn_p95_ms": percentile(turn_ms, 0.95),
        "turn_max_ms": max(turn_ms),
        "init_median_ms": statistics.median([s["init_ms"] for s in samples]),
    }


async def main_async(args) -> list:
    results = []
    impls = args.impl or list(IMPLS)
    rows_list = args.rows or [20, 200, 2000]
    frames_list = args.frames or [1, 4, 12]
    with tempfile.TemporaryDirectory(prefix="mili-bench-") as workdir:
        os.chdir(workdir)
        for impl in impls:
            for rows in rows_list:
                for frames in frames_list:
                    samples = [
                        await run_turn(impl, rows, frames) for _ in range(args.repeat)
                    ]
                    counted = await run_turn(impl, rows, frames, instrument=True)
                    summary = summarize(samples)
                    summary.update(
                        {
                            key: counted[key]
                            for key in (
                                "opens_read",
                                "opens_write",
                                "read_bytes",
                                "write_bytes",
                                "atomic_writes",
                                "transcript_bytes",
                            )
                        }
                    )
                    summary.update({"impl": impl, "rows": rows, "frames": frames})
                    results.append(summary)
                    print(
                        "{impl:>9} rows={rows:>5} frames={frames:>3} "
                        "frame_med={frame_median_ms:7.2f}ms "
                        "frame_p95={frame_p95_ms:7.2f}ms "
                        "turn_med={turn_median_ms:8.2f}ms "
                        "reads={opens_read} writes={opens_write} "
                        "read={read_bytes}B write={write_bytes}B".format(**summary),
                        flush=True,
                    )
    return results


def main(argv=None) -> int:
    from loguru import logger

    logger.remove()  # the store logs every write at DEBUG; the table is the output
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--impl", action="append", choices=sorted(IMPLS))
    parser.add_argument("--rows", action="append", type=int)
    parser.add_argument("--frames", action="append", type=int)
    parser.add_argument("--repeat", type=int, default=10)
    parser.add_argument("--json", action="store_true", help="emit raw summaries")
    parser.add_argument(
        "--breakdown",
        action="store_true",
        help="also time serialize vs fsync for one durable write",
    )
    args = parser.parse_args(argv)
    if args.breakdown:
        with tempfile.TemporaryDirectory(prefix="mili-bench-") as workdir:
            os.chdir(workdir)
            for rows in args.rows or [20, 200, 2000]:
                parts = breakdown(rows, args.repeat)
                print(
                    "breakdown rows={rows:>5} bytes={bytes:>7} "
                    "serialize={serialize_ms:6.2f}ms fsync={fsync_ms:6.2f}ms "
                    "full={full_ms:6.2f}ms".format(**parts),
                    flush=True,
                )
        return 0
    results = asyncio.run(main_async(args))
    if args.json:
        print(json.dumps(results, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
