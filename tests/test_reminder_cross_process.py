"""Two server processes sharing one character-state directory.

These tests run real child interpreters (not threads), so the in-process
RLock cannot help them: only the advisory ``flock`` on
``character_state/<conf>.json.lock`` and the on-disk delivery journal are
shared. They cover:

* claim exclusion: two processes racing on the same due reminder -> one owner;
* receipt visibility: a receipt written by process A blocks process B after
  the claim lease, even when the completion write failed or A exited before
  completing (crash/restart between delivery and completion);
* retry before receipt: an undelivered attempt (released, no receipt) remains
  claimable by the other process;
* no exactly-once claim: the crash window before a receipt stays documented.
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

from src.open_llm_vtuber.character_state import (
    load_character_state,
    save_character_state,
)
from src.open_llm_vtuber.reminder_delivery import delivered_ids

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = "xproc-char"
NOW = datetime(2026, 10, 20, 9, 0, tzinfo=timezone.utc)

# Static child script: every input arrives as JSON in argv so braces stay
# literal and the script exercises the real production functions.
CHILD_SCRIPT = r"""
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

import src.open_llm_vtuber.character_state as cs

if payload.get("slow_load"):
    _original_load = cs.load_character_state

    def slow_load(conf_uid):
        state = _original_load(conf_uid)
        # Widen the read-modify-write window so a missing cross-process lock
        # reliably lets both processes load the same unclaimed snapshot.
        time.sleep(payload["slow_load"])
        return state

    cs.load_character_state = slow_load

if payload.get("wait_ready"):
    ready_dir = payload["ready_dir"]
    open(os.path.join(ready_dir, payload["slot"] + ".ready"), "w").close()
    deadline = time.time() + 20
    while not os.path.exists(os.path.join(ready_dir, "start")):
        if time.time() > deadline:
            break
        time.sleep(0.01)

conf = payload["conf"]
now = datetime.fromisoformat(payload["now"])
owner = payload["owner"]
result = {"owner": owner, "mode": payload["mode"]}

if payload["mode"] == "claim":
    rows = cs.claim_due_future_intentions(
        conf,
        owner=owner,
        now=now,
        lease_seconds=payload.get("lease", 600),
        max_claims=payload.get("max_claims"),
    )
    result["claimed"] = [row["id"] for row in rows]
    if rows and payload.get("renew"):
        result["renewed"] = cs.renew_claimed_future_intentions(
            conf,
            owner=owner,
            intention_ids=result["claimed"],
            lease_seconds=payload["renew"],
            now=datetime.fromisoformat(payload["renew_now"]),
        )
        result["claim_until"] = next(
            row.get("claim_until")
            for row in cs.load_character_state(conf).future_intentions
            if row.get("id") in result["claimed"]
        )
    if rows and payload.get("receipt"):
        from src.open_llm_vtuber.reminder_delivery import record_delivery

        result["receipt"] = record_delivery(conf, result["claimed"], owner=owner, now=now)
    if rows and payload.get("release"):
        result["released"] = cs.release_claimed_future_intentions(
            conf, owner=owner, intention_ids=result["claimed"]
        )
    if rows and payload.get("complete") == "fail":
        original_save = cs.save_character_state
        cs.save_character_state = lambda *args, **kwargs: False
        try:
            result["completed"] = cs.complete_claimed_future_intentions(
                conf, owner=owner, intention_ids=result["claimed"]
            )
        finally:
            cs.save_character_state = original_save
    elif rows and payload.get("complete"):
        result["completed"] = cs.complete_claimed_future_intentions(
            conf, owner=owner, intention_ids=result["claimed"]
        )

with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"""

# Child script for the end-to-end delivery race: claim -> "send" -> durable
# receipt -> completion, all against the shared state directory. Each child
# appends one line per message it would put on the wire, so the sends file is
# the observable count of deliveries (the transport itself is not part of the
# test).
DELIVER_SCRIPT = r"""
import json
import os
import sys
import time
from datetime import datetime

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

import src.open_llm_vtuber.character_state as cs
from src.open_llm_vtuber.reminder_delivery import record_delivery

if payload.get("slow_load"):
    _original_load = cs.load_character_state

    def slow_load(conf_uid):
        state = _original_load(conf_uid)
        time.sleep(payload["slow_load"])
        return state

    cs.load_character_state = slow_load

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
result = {"owner": owner, "sent": [], "completed": 0, "receipt": False}

rows = cs.claim_due_future_intentions(conf, owner=owner, now=now)
result["claimed"] = [row["id"] for row in rows]
if rows:
    ids = result["claimed"]
    # The delivery boundary: one line per message actually sent.
    with open(payload["sends_path"], "a", encoding="utf-8") as handle:
        handle.write("\\n".join(f"{owner} {row_id}" for row_id in ids) + "\\n")
        handle.flush()
        os.fsync(handle.fileno())
    result["sent"] = ids
    if payload.get("receipt", True):
        result["receipt"] = record_delivery(conf, ids, owner=owner, now=now)
    if payload.get("complete", "ok") == "fail":
        original_save = cs.save_character_state
        cs.save_character_state = lambda *args, **kwargs: False
        try:
            result["completed"] = cs.complete_claimed_future_intentions(
                conf, owner=owner, intention_ids=ids
            )
        finally:
            cs.save_character_state = original_save
    else:
        result["completed"] = cs.complete_claimed_future_intentions(
            conf, owner=owner, intention_ids=ids
        )

with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"""


class CrossProcessReminderTest(unittest.TestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.workdir = os.getcwd()
        self._counter = 0

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    def _seed(self):
        state = load_character_state(CONF)
        row = {
            "id": uuid.uuid4().hex,
            "text": "ingetin aku minum obat ya",
            "due_at": (NOW - timedelta(minutes=1)).isoformat(),
            "created_at": (NOW - timedelta(days=1)).isoformat(),
            "status": "pending",
            "source": "conversation",
            "kind": "reminder",
            "tz": "Asia/Jakarta",
        }
        state.future_intentions = [row]
        self.assertTrue(save_character_state(CONF, state))
        return row

    def _payload(self, **overrides):
        self._counter += 1
        payload = {
            "mode": "claim",
            "conf": CONF,
            "now": NOW.isoformat(),
            "owner": f"owner-{self._counter}",
            "result_path": os.path.join(self.workdir, f"result-{self._counter}.json"),
        }
        payload.update(overrides)
        return payload

    def _run(self, payload):
        process = subprocess.run(
            [sys.executable, "-c", CHILD_SCRIPT, ROOT, self.workdir, json.dumps(payload)],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            process.returncode,
            0,
            f"child failed: {process.stderr[-2000:]}",
        )
        with open(payload["result_path"], "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _spawn(self, payload):
        return subprocess.Popen(
            [sys.executable, "-c", CHILD_SCRIPT, ROOT, self.workdir, json.dumps(payload)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _collect(self, process, payload):
        stdout, stderr = process.communicate(timeout=60)
        self.assertEqual(
            process.returncode, 0, f"child failed: {stderr[-2000:]} {stdout[-500:]}"
        )
        with open(payload["result_path"], "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _run_deliver(self, payload):
        process = subprocess.run(
            [
                sys.executable,
                "-c",
                DELIVER_SCRIPT,
                ROOT,
                self.workdir,
                json.dumps(payload),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(
            process.returncode, 0, f"child failed: {process.stderr[-2000:]}"
        )
        with open(payload["result_path"], "r", encoding="utf-8") as handle:
            return json.load(handle)

    def _spawn_deliver(self, payload):
        return subprocess.Popen(
            [
                sys.executable,
                "-c",
                DELIVER_SCRIPT,
                ROOT,
                self.workdir,
                json.dumps(payload),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def test_two_processes_delivering_concurrently_send_once(self):
        """Full boundary across processes: claim -> send -> receipt -> done."""
        row = self._seed()
        sends_path = os.path.join(self.workdir, "sends.log")
        ready_dir = os.path.join(self.workdir, "ready-deliver")
        os.makedirs(ready_dir, exist_ok=True)
        payloads = [
            self._payload(
                owner=f"deliver-{slot}",
                slot=slot,
                wait_ready=True,
                ready_dir=ready_dir,
                slow_load=0.35,
                sends_path=sends_path,
            )
            for slot in ("a", "b")
        ]
        processes = [self._spawn_deliver(payload) for payload in payloads]
        deadline = time.time() + 20
        while len(os.listdir(ready_dir)) < 2 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(os.listdir(ready_dir)), 2, "children never became ready")
        open(os.path.join(ready_dir, "start"), "w").close()
        results = [
            self._collect(process, payload)
            for process, payload in zip(processes, payloads)
        ]

        # Exactly one process reached the delivery boundary, and the on-disk
        # journal has one receipt for one message.
        with open(sends_path, "r", encoding="utf-8") as handle:
            sends = [line for line in handle if line.strip()]
        self.assertEqual(len(sends), 1, sends)
        self.assertEqual(len([item for item in results if item["sent"]]), 1)
        self.assertEqual(delivered_ids(CONF), {row["id"]})
        stored = load_character_state(CONF).future_intentions[0]
        self.assertEqual(stored["status"], "done")
        # The loser is guaranteed to have completed nothing.
        losers = [item for item in results if not item["claimed"]]
        self.assertEqual(len(losers), 1)
        self.assertEqual(losers[0]["completed"], 0)

    def test_renewed_lease_blocks_cross_process_takeover(self):
        """A live long turn stays exclusively owned from another process."""
        row = self._seed()
        first = self._run(
            self._payload(
                owner="owner-a",
                lease=60,
                renew=60,
                renew_now=(NOW + timedelta(seconds=50)).isoformat(),
            )
        )
        self.assertEqual(first["claimed"], [row["id"]])
        self.assertEqual(first["renewed"], 1)
        # Lease was originally until +60 and renewal pushed it past +110.
        self.assertEqual(
            datetime.fromisoformat(first["claim_until"]),
            NOW + timedelta(seconds=110),
        )
        # Without the renewal this claim (at +100) would have succeeded.
        second = self._run(
            self._payload(
                owner="owner-b", now=(NOW + timedelta(seconds=100)).isoformat()
            )
        )
        self.assertEqual(second["claimed"], [])
        # Renewal never completes the row: once the lease really expires and no
        # receipt exists, the reminder is still retryable instead of lost.
        third = self._run(
            self._payload(
                owner="owner-c", now=(NOW + timedelta(seconds=120)).isoformat()
            )
        )
        self.assertEqual(third["claimed"], [row["id"]])

    def test_lease_expiry_without_renewal_is_the_documented_window(self):
        """No heartbeat -> takeover after expiry; receipt still stops resends."""
        row = self._seed()
        first = self._run(self._payload(owner="owner-a", lease=5))
        self.assertEqual(first["claimed"], [row["id"]])
        second = self._run(
            self._payload(
                owner="owner-b",
                now=(NOW + timedelta(seconds=10)).isoformat(),
                receipt=True,
            )
        )
        # Documented limitation: a sender whose lease lapsed (stalled process)
        # can be taken over while it is still streaming. The heartbeat in the
        # live dispatcher exists to make this not happen for a running turn.
        self.assertEqual(second["claimed"], [row["id"]])
        self.assertTrue(second["receipt"])
        third = self._run(
            self._payload(
                owner="owner-c", now=(NOW + timedelta(seconds=900)).isoformat()
            )
        )
        self.assertEqual(third["claimed"], [])
        self.assertEqual(
            load_character_state(CONF).future_intentions[0]["status"], "done"
        )

    def test_two_processes_racing_claim_exactly_one(self):
        row = self._seed()
        ready_dir = os.path.join(self.workdir, "ready")
        os.makedirs(ready_dir, exist_ok=True)
        payloads = [
            self._payload(
                owner=f"owner-{slot}",
                slot=slot,
                wait_ready=True,
                ready_dir=ready_dir,
                slow_load=0.35,
            )
            for slot in ("a", "b")
        ]
        processes = [self._spawn(payload) for payload in payloads]
        deadline = time.time() + 20
        while len(os.listdir(ready_dir)) < 2 and time.time() < deadline:
            time.sleep(0.01)
        self.assertEqual(len(os.listdir(ready_dir)), 2, "children never became ready")
        # Release both children at the same instant.
        open(os.path.join(ready_dir, "start"), "w").close()
        results = [
            self._collect(process, payload)
            for process, payload in zip(processes, payloads)
        ]

        claimed_id_lists = [result["claimed"] for result in results]
        winners = [items for items in claimed_id_lists if items]
        self.assertEqual(len(winners), 1, claimed_id_lists)
        self.assertEqual(winners[0], [row["id"]])
        stored = load_character_state(CONF).future_intentions[0]
        self.assertEqual(stored["status"], "pending")
        self.assertIn(stored.get("claimed_by"), {"owner-a", "owner-b"})

    def test_receipt_from_one_process_blocks_another_after_completion_failure(self):
        row = self._seed()
        first = self._run(
            self._payload(owner="owner-a", receipt=True, complete="fail")
        )
        self.assertEqual(first["claimed"], [row["id"]])
        self.assertTrue(first["receipt"])
        self.assertEqual(first["completed"], 0)
        self.assertEqual(delivered_ids(CONF), {row["id"]})
        stored = load_character_state(CONF).future_intentions[0]
        self.assertEqual(stored["status"], "pending")
        self.assertEqual(stored["claimed_by"], "owner-a")

        later = (NOW + timedelta(seconds=700)).isoformat()
        second = self._run(self._payload(owner="owner-b", now=later))
        self.assertEqual(second["claimed"], [])
        stored = load_character_state(CONF).future_intentions[0]
        self.assertEqual(stored["status"], "done")
        self.assertNotIn("claimed_by", stored)

    def test_receipt_survives_restart_between_delivery_and_completion(self):
        row = self._seed()
        first = self._run(self._payload(owner="owner-a", receipt=True))
        self.assertEqual(first["claimed"], [row["id"]])
        self.assertTrue(first["receipt"])
        # Process exits without completion: only disk survives.
        later = (NOW + timedelta(seconds=700)).isoformat()
        second = self._run(self._payload(owner="owner-restarted", now=later))
        self.assertEqual(second["claimed"], [])
        stored = load_character_state(CONF).future_intentions[0]
        self.assertEqual(stored["status"], "done")

    def test_failed_delivery_before_receipt_stays_retryable_across_processes(self):
        row = self._seed()
        first = self._run(self._payload(owner="owner-a", release=True))
        self.assertEqual(first["claimed"], [row["id"]])
        self.assertNotIn("receipt", first)
        self.assertEqual(delivered_ids(CONF), set())

        second = self._run(self._payload(owner="owner-b"))
        self.assertEqual(second["claimed"], [row["id"]])

    def test_crash_before_receipt_remains_at_least_once(self):
        """Documented boundary: no receipt -> the restart may resend."""
        row = self._seed()
        first = self._run(self._payload(owner="owner-a"))
        self.assertEqual(first["claimed"], [row["id"]])
        later = (NOW + timedelta(seconds=700)).isoformat()
        second = self._run(self._payload(owner="owner-restarted", now=later))
        self.assertEqual(second["claimed"], [row["id"]])


if __name__ == "__main__":
    unittest.main()
