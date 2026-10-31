"""Two server processes, one character state AND one chat-history file.

``test_reminder_cross_process`` already pins claim/lease/receipt exclusivity on
the character-state store. It never touches the other file a delivered
reminder writes: ``chat_history/<conf>/<history_uid>.json``. That file is a
whole-document JSON store rewritten by read-modify-write, so this module
answers the remaining question -- can two competing processes corrupt the
transcript while claiming, delivering, storing and completing the same
reminder?

Every test here drives REAL child interpreters through ``subprocess`` (never
threads, never a mocked lock): the only things shared between them are the
files on disk. The child scripts are static strings so production functions
are exercised exactly as the server calls them.

No exactly-once claim is made anywhere: a content frame accepted by the
transport is not proof the user saw it, and a crash before the durable
receipt leaves no evidence at all.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
import uuid
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
)
from src.open_llm_vtuber.character_state import (
    load_character_state,
    save_character_state,
)
from src.open_llm_vtuber.reminder_delivery import delivered_ids, record_delivery

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = "xproc-history-char"
NOW = datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)


# One full delivery attempt in a separate interpreter:
# claim -> (transport send) -> transcript row -> durable receipt -> completion.
# ``slow_read`` widens the history read-modify-write window so a missing
# cross-process lock reliably loses a row instead of only sometimes.
DELIVERY_CHILD = r"""
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

import src.open_llm_vtuber.character_state as cs
import src.open_llm_vtuber.chat_history_manager as chm
from src.open_llm_vtuber.reminder_delivery import record_delivery

if payload.get("slow_read"):
    _real_load = json.load

    def slow_load(*args, **kwargs):
        data = _real_load(*args, **kwargs)
        # Inside store_message's read-modify-write window.
        time.sleep(payload["slow_read"])
        return data

    chm.json.load = slow_load

ready_dir = payload.get("ready_dir")
if ready_dir:
    open(os.path.join(ready_dir, payload["slot"] + ".ready"), "w").close()
    deadline = time.time() + 20
    while not os.path.exists(os.path.join(ready_dir, "start")):
        if time.time() > deadline:
            break
        time.sleep(0.01)

conf = payload["conf"]
now = datetime.fromisoformat(payload["now"])
owner = payload["owner"]
result = {
    "owner": owner,
    "text": payload["text"],
    "claimed": [],
    "stored": False,
    "receipt": False,
}

rows = cs.claim_due_future_intentions(
    conf,
    owner=owner,
    now=now,
    lease_seconds=payload.get("lease", 600),
    exclude_ids=payload.get("exclude_ids"),
)
result["claimed"] = [row["id"] for row in rows]
if rows:
    # The transcript row this process would persist for what it sent.
    chm.store_message(conf, payload["history"], "ai", payload["text"])
    result["stored"] = True
    if payload.get("receipt", True):
        result["receipt"] = record_delivery(
            conf, result["claimed"], owner=owner, now=now
        )
    if payload.get("complete", True):
        result["completed"] = cs.complete_claimed_future_intentions(
            conf, owner=owner, intention_ids=result["claimed"]
        )

with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"""

# Lease heartbeat against a live competitor: one process keeps a claim alive
# with real wall-clock renewals while the other keeps trying to take it.
HEARTBEAT_CHILD = r"""
import json
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

import src.open_llm_vtuber.character_state as cs

ready_dir = payload["ready_dir"]
open(os.path.join(ready_dir, payload["slot"] + ".ready"), "w").close()
deadline = time.time() + 20
while not os.path.exists(os.path.join(ready_dir, "start")):
    if time.time() > deadline:
        break
    time.sleep(0.01)

conf = payload["conf"]
owner = payload["owner"]
lease = payload["lease"]
interval = payload["interval"]
duration = payload["duration"]
result = {"owner": owner, "claimed": [], "renewals": 0, "attempts": 0}

marker = os.path.join(ready_dir, "claimed")

if payload["mode"] == "hold":
    rows = cs.claim_due_future_intentions(
        conf, owner=owner, now=datetime.now(timezone.utc), lease_seconds=lease
    )
    result["claimed"] = [row["id"] for row in rows]
    open(marker, "w").close()
    end = time.time() + duration
    while time.time() < end and result["claimed"]:
        time.sleep(interval)
        result["renewals"] += cs.renew_claimed_future_intentions(
            conf,
            owner=owner,
            intention_ids=result["claimed"],
            lease_seconds=lease,
            now=datetime.now(timezone.utc),
        )
else:
    # Do not compete before the holder has actually claimed, otherwise the
    # chaser could legitimately win the very first claim.
    wait_until = time.time() + 20
    while not os.path.exists(marker) and time.time() < wait_until:
        time.sleep(0.01)
    end = time.time() + duration
    while time.time() < end:
        result["attempts"] += 1
        rows = cs.claim_due_future_intentions(
            conf, owner=owner, now=datetime.now(timezone.utc), lease_seconds=lease
        )
        if rows:
            result["claimed"].extend(row["id"] for row in rows)
        time.sleep(interval)

with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"""

# Plain concurrent transcript append, nothing else.
APPEND_CHILD = r"""
import json
import os
import sys
import time

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

import src.open_llm_vtuber.chat_history_manager as chm

_real_load = json.load


def slow_load(*args, **kwargs):
    data = _real_load(*args, **kwargs)
    time.sleep(payload["slow_read"])
    return data


chm.json.load = slow_load

ready_dir = payload["ready_dir"]
open(os.path.join(ready_dir, payload["slot"] + ".ready"), "w").close()
deadline = time.time() + 20
while not os.path.exists(os.path.join(ready_dir, "start")):
    if time.time() > deadline:
        break
    time.sleep(0.01)

chm.store_message(payload["conf"], payload["history"], "ai", payload["text"])
with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump({"stored": True}, handle)
"""


class CrossProcessHistoryRaceTest(unittest.TestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.workdir = os.getcwd()
        self._counter = 0
        self.history = create_new_history(CONF)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    def _seed(self, *, due_offset_minutes: int = -1, receipt: bool = False):
        state = load_character_state(CONF)
        row = {
            "id": uuid.uuid4().hex,
            "text": "ingetin aku minum obat ya",
            "due_at": (NOW + timedelta(minutes=due_offset_minutes)).isoformat(),
            "created_at": (NOW - timedelta(days=1)).isoformat(),
            "status": "pending",
            "source": "conversation",
            "kind": "reminder",
            "tz": "Asia/Jakarta",
            "delivery_text": "Aku ingetin minum obat ya.",
        }
        state.future_intentions = [row]
        self.assertTrue(save_character_state(CONF, state))
        if receipt:
            self.assertTrue(record_delivery(CONF, [row["id"]], owner="seed"))
        return row

    def _payload(self, **overrides):
        self._counter += 1
        payload = {
            "conf": CONF,
            "history": self.history,
            "now": NOW.isoformat(),
            "owner": f"owner-{self._counter}",
            "text": f"delivery-{self._counter}",
            "result_path": os.path.join(self.workdir, f"result-{self._counter}.json"),
        }
        payload.update(overrides)
        return payload

    def _run_together(self, script, payloads, *, ready_timeout=25.0):
        ready_dir = os.path.join(self.workdir, f"ready-{uuid.uuid4().hex}")
        os.makedirs(ready_dir, exist_ok=True)
        for payload in payloads:
            payload["ready_dir"] = ready_dir
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", script, ROOT, self.workdir, json.dumps(payload)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for payload in payloads
        ]
        deadline = time.time() + ready_timeout
        while len(os.listdir(ready_dir)) < len(payloads) and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(
            len(os.listdir(ready_dir)),
            len(payloads),
            "children never became ready",
        )
        # The contestants are genuinely independent interpreters: distinct
        # OS processes, none of them this test runner (so no in-process RLock
        # or shared memory can explain a pass).
        child_pids = {process.pid for process in processes}
        self.assertEqual(len(child_pids), len(payloads), child_pids)
        self.assertNotIn(os.getpid(), child_pids)
        # Release every child at the same instant.
        open(os.path.join(ready_dir, "start"), "w").close()
        results = []
        for process, payload in zip(processes, payloads):
            stdout, stderr = process.communicate(timeout=60)
            self.assertEqual(
                process.returncode,
                0,
                f"child failed: {stderr[-2000:]} {stdout[-500:]}",
            )
            with open(payload["result_path"], "r", encoding="utf-8") as handle:
                results.append(json.load(handle))
        return results

    # ------------------------------------------------------------------
    # The transcript file itself: two processes, one document.
    # ------------------------------------------------------------------
    def test_participants_are_independent_interpreters(self):
        """Guard the premise of every other test in this module."""
        payloads = [self._payload(slot=slot, slow_read=0.05) for slot in ("a", "b")]
        self._run_together(APPEND_CHILD, payloads)
        self.assertEqual(len(get_history(CONF, self.history)), 2)

    def test_two_processes_appending_to_one_history_file_keep_both_rows(self):
        """A delivery in process A must never erase the row process B stored.

        Both processes append one AI row to the SAME history file whose
        read-modify-write window is held open for 0.35 s. Without a
        cross-process lock both read the same snapshot and the second atomic
        replace drops the first process's row -- a delivered reminder that
        vanishes from the transcript, i.e. exactly the "streamed but missing
        from history" inconsistency, caused by process concurrency instead of
        the streaming pipeline.
        """
        payloads = [
            self._payload(slot=slot, slow_read=0.35) for slot in ("a", "b")
        ]
        self._run_together(APPEND_CHILD, payloads)

        rows = get_history(CONF, self.history)
        self.assertEqual(
            sorted(row["content"] for row in rows),
            ["delivery-1", "delivery-2"],
            rows,
        )

    # ------------------------------------------------------------------
    # The same reminder, two processes: one delivery, one transcript row.
    # ------------------------------------------------------------------
    def test_same_reminder_delivered_by_two_processes_writes_one_row(self):
        """claim -> send -> row -> receipt -> done, raced across processes."""
        row = self._seed()
        payloads = [
            self._payload(slot=slot, slow_read=0.25) for slot in ("a", "b")
        ]
        results = self._run_together(DELIVERY_CHILD, payloads)

        winners = [item for item in results if item["claimed"]]
        self.assertEqual(len(winners), 1, results)
        self.assertEqual(winners[0]["claimed"], [row["id"]])
        self.assertTrue(winners[0]["receipt"])
        self.assertEqual(winners[0]["completed"], 1)
        # Exactly one process stored exactly one transcript row.
        self.assertTrue(winners[0]["stored"])
        self.assertEqual([item["stored"] for item in results].count(True), 1)
        self.assertEqual(
            [item["content"] for item in get_history(CONF, self.history)],
            [winners[0]["text"]],
        )
        # One durable receipt, one done row.
        self.assertEqual(delivered_ids(CONF), {row["id"]})
        stored = load_character_state(CONF).future_intentions[0]
        self.assertEqual(stored["status"], "done")
        self.assertNotIn("claimed_by", stored)

    def test_both_processes_reconciling_one_receipt_neither_sends_nor_writes(self):
        """Receipt durable + completion never written: both settle to done once."""
        row = self._seed(receipt=True)
        payloads = [self._payload(slot=slot) for slot in ("a", "b")]
        results = self._run_together(DELIVERY_CHILD, payloads)

        self.assertEqual([item["claimed"] for item in results], [[], []])
        self.assertEqual([item["stored"] for item in results], [False, False])
        self.assertEqual(get_history(CONF, self.history), [])
        stored = load_character_state(CONF).future_intentions[0]
        self.assertEqual(stored["status"], "done")
        self.assertNotIn("claimed_by", stored)
        # Reconciliation never appends a second receipt line.
        with open(
            os.path.join("character_state", f"{CONF}.deliveries.jsonl"),
            "r",
            encoding="utf-8",
        ) as handle:
            receipts = [json.loads(line) for line in handle if line.strip()]
        self.assertEqual([item["id"] for item in receipts], [row["id"]])

    # ------------------------------------------------------------------
    # Live lease boundary: renewal vs. a real competitor.
    # ------------------------------------------------------------------
    def test_live_heartbeat_keeps_expiring_lease_from_a_competing_process(self):
        """A working process must not lose its claim mid-turn, ever.

        Process A holds a 3 s lease and renews it every 0.5 s for 4 s of REAL
        wall-clock time (no fake clock); process B attempts a claim every
        0.3 s for the same 4 s. B must never win while A is alive, including
        across the original lease boundary: the on-disk renewal is the only
        thing that can express "still in flight" across processes.
        """
        row = self._seed()
        payloads = [
            self._payload(
                slot="a",
                mode="hold",
                lease=3,
                interval=0.5,
                duration=4.0,
                result_path=os.path.join(self.workdir, "hold.json"),
            ),
            self._payload(
                slot="b",
                mode="chase",
                lease=3,
                interval=0.3,
                duration=4.0,
                result_path=os.path.join(self.workdir, "chase.json"),
            ),
        ]
        holder, chaser = self._run_together(HEARTBEAT_CHILD, payloads)

        self.assertEqual(holder["claimed"], [row["id"]])
        self.assertGreaterEqual(holder["renewals"], 4, holder)
        self.assertGreaterEqual(chaser["attempts"], 5, chaser)
        self.assertEqual(chaser["claimed"], [], chaser)
        stored = load_character_state(CONF).future_intentions[0]
        self.assertEqual(stored["status"], "pending")
        self.assertEqual(stored["claimed_by"], "owner-1")

    # ------------------------------------------------------------------
    # Documented boundaries -- no exactly-once claim.
    # ------------------------------------------------------------------
    def test_crash_after_send_before_receipt_is_resent_by_the_other_process(self):
        """No receipt, lease expired -> the restart delivers again.

        A content frame accepted by a dead process' transport is not evidence
        the user ever saw it, and nothing durable records it. The takeover
        therefore writes a SECOND transcript row. This is the at-least-once
        window, asserted so nobody mistakes it for exactly-once delivery.
        """
        row = self._seed()
        # First process claims and stores the row, then dies before the receipt
        # (the child skips the journal append when ``receipt`` is false).
        first = self._run_together(
            DELIVERY_CHILD,
            [self._payload(slot="a", receipt=False, complete=False)],
        )[0]
        self.assertEqual(first["claimed"], [row["id"]])
        self.assertTrue(first["stored"])
        self.assertFalse(first["receipt"])
        self.assertEqual(delivered_ids(CONF), set())

        later = (NOW + timedelta(seconds=700)).isoformat()
        second = self._run_together(
            DELIVERY_CHILD, [self._payload(slot="b", now=later)]
        )[0]
        self.assertEqual(second["claimed"], [row["id"]])
        self.assertTrue(second["receipt"])
        self.assertEqual(
            [item["content"] for item in get_history(CONF, self.history)],
            [first["text"], second["text"]],
        )
        self.assertEqual(
            load_character_state(CONF).future_intentions[0]["status"], "done"
        )


if __name__ == "__main__":
    unittest.main()
