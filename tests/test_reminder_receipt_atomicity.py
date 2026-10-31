"""Is the delivery-receipt journal atomic across server processes?

``reminder_delivery.record_delivery`` appends one JSON line per delivered
reminder with ``O_APPEND`` + ``fsync``; ``delivered_ids`` reads the whole file
*witHOUT* taking any lock. The claim path trusts that read: a receipt it fails
to see is a reminder it may send again.

Every test here runs real child interpreters (``subprocess``), so neither the
in-process ``RLock`` nor any shared memory can mask a race: the only shared
state is ``character_state/<conf>.deliveries.jsonl`` on disk.

Two questions are answered separately and honestly:

* append/read concurrency -- can a writer corrupt a line, or a reader miss a
  receipt that is already durable? (measured, not assumed)
* crash before the receipt -- can that window be closed without a client ACK
  or receiver-side idempotency? (asserted to stay OPEN, so nobody mistakes it
  for exactly-once delivery)
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

from src.open_llm_vtuber.character_state import load_character_state, save_character_state
from src.open_llm_vtuber.reminder_delivery import (
    delivered_ids,
    get_delivery_journal_path,
    record_delivery,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = "journal-atomicity"
LEASE = 600

# Writer: barrier, then append receipt batches until its time slice ends.
WRITER_CHILD = r"""
import json
import os
import sys
import time

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

from src.open_llm_vtuber.reminder_delivery import record_delivery

ready_dir = payload["ready_dir"]
open(os.path.join(ready_dir, payload["slot"] + ".ready"), "w").close()
deadline = time.time() + 20
while not os.path.exists(os.path.join(ready_dir, "start")):
    if time.time() > deadline:
        break
    time.sleep(0.005)

conf = payload["conf"]
batches = payload.get("batches")
per_batch = payload["per_batch"]
end = time.time() + payload["duration"]
written = []
batch = 0
while (batch < batches) if batches is not None else (time.time() < end):
    ids = [
        "%s-%05d-%03d" % (payload["slot"], batch, index)
        for index in range(per_batch)
    ]
    written.append(bool(record_delivery(conf, ids, owner=payload["slot"])))
    batch += 1
    if batches is None:
        time.sleep(0.002)

with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump({"batch": batch, "ok": written}, handle)
"""

# Reader: parse the journal and call delivered_ids() in a tight loop while the
# writers append. Detects torn lines (undecodable/partial JSON) and any read
# that loses an id it had already observed (a missed durable receipt).
READER_CHILD = r"""
import fcntl
import json
import os
import sys
import time

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

from src.open_llm_vtuber.reminder_delivery import delivered_ids
from src.open_llm_vtuber.reminder_delivery import get_delivery_journal_path

ready_dir = payload["ready_dir"]
open(os.path.join(ready_dir, payload["slot"] + ".ready"), "w").close()
deadline = time.time() + 20
while not os.path.exists(os.path.join(ready_dir, "start")):
    if time.time() > deadline:
        break
    time.sleep(0.005)

conf = payload["conf"]
path = get_delivery_journal_path(conf)
readable = set()
torn = []
shrinks = []
reads = 0
end = time.time() + payload["duration"]
while time.time() < end:
    try:
        with open(path, "rb") as handle:
            # The probe asks for the same shared lock a production reader uses,
            # so a torn line here means appends are not serialised at all.
            fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
            raw = handle.read()
    except FileNotFoundError:
        continue
    reads += 1
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        torn.append("undecodable: %s" % error)
        continue
    local = set()
    for line in text.split("\n"):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except Exception:
            # A line that exists on disk but is not valid JSON: either a torn
            # append, or a read that cannot see a durable receipt.
            torn.append(line[:120])
            continue
        local.add(str(parsed.get("id", "")))
    current = delivered_ids(conf)
    if not readable.issubset(current):
        # An id this reader already saw on disk is missing from the set the
        # claim path would use: a durable receipt the reader could not see.
        shrinks.append(sorted(readable - current)[:3])
    readable |= local

with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump(
        {"reads": reads, "torn": torn, "shrinks": shrinks, "seen": len(readable)},
        handle,
    )
"""

# Writer frozen mid-append: holds the journal lock with half a line on disk,
# then finishes it. A reader must not read through that window.
PARTIAL_WRITER_CHILD = r"""
import fcntl
import json
import os
import sys
import time

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

from src.open_llm_vtuber.reminder_delivery import get_delivery_journal_path

path = get_delivery_journal_path(payload["conf"])
os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
fcntl.flock(descriptor, fcntl.LOCK_EX)
# Exactly what a lock-free reader could catch: a receipt cut in half.
os.write(descriptor, ('{"id":"%s","owner":"half' % payload["id"]).encode("utf-8"))
open(os.path.join(payload["ready_dir"], "partial"), "w").close()
time.sleep(payload["hold"])
os.write(
    descriptor,
    b'","delivered_at":"2026-01-01T00:00:00+00:00"}\n',
)
os.fsync(descriptor)
os.close(descriptor)
"""

PARTIAL_READER_CHILD = r"""
import json
import os
import sys
import time

sys.path.insert(0, sys.argv[1])
os.chdir(sys.argv[2])
payload = json.loads(sys.argv[3])

from src.open_llm_vtuber.reminder_delivery import delivered_ids

started = time.time()
ids = delivered_ids(payload["conf"])
elapsed = time.time() - started
with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump({"ids": sorted(ids), "elapsed": elapsed}, handle)
"""

# Hard crash: claim, optionally write the receipt, then os._exit. No cleanup,
# no completion write, no interpreter shutdown -- nothing but the fsynced
# journal and the state file survive.
CRASH_CHILD = r"""
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

ready_dir = payload["ready_dir"]
open(os.path.join(ready_dir, payload["slot"] + ".ready"), "w").close()
deadline = time.time() + 20
while not os.path.exists(os.path.join(ready_dir, "start")):
    if time.time() > deadline:
        break
    time.sleep(0.005)

conf = payload["conf"]
owner = payload["owner"]
now = datetime.fromisoformat(payload["now"])
rows = cs.claim_due_future_intentions(
    conf, owner=owner, now=now, lease_seconds=payload["lease"]
)
ids = [row["id"] for row in rows]
if rows and payload["stage"] == "before_receipt":
    # The transport accepted a content frame: the transcript row is written.
    chm.store_message(conf, payload["history"], "ai", payload["text"])
if rows and payload["stage"] == "after_receipt":
    record_delivery(conf, ids, owner=owner, now=now)
os._exit(0)
"""

# Ordinary restart: claim (which reconciles receipts) and report.
RESTART_CHILD = r"""
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

ready_dir = payload["ready_dir"]
open(os.path.join(ready_dir, payload["slot"] + ".ready"), "w").close()
deadline = time.time() + 20
while not os.path.exists(os.path.join(ready_dir, "start")):
    if time.time() > deadline:
        break
    time.sleep(0.005)

conf = payload["conf"]
owner = payload["owner"]
now = datetime.fromisoformat(payload["now"])
rows = cs.claim_due_future_intentions(
    conf, owner=owner, now=now, lease_seconds=payload["lease"]
)
result = {"owner": owner, "claimed": [row["id"] for row in rows], "receipt": False}
if rows:
    ids = result["claimed"]
    chm.store_message(conf, payload["history"], "ai", payload["text"])
    result["receipt"] = record_delivery(conf, ids, owner=owner, now=now)
    result["completed"] = cs.complete_claimed_future_intentions(
        conf, owner=owner, intention_ids=ids
    )
with open(payload["result_path"], "w", encoding="utf-8") as handle:
    json.dump(result, handle)
"""


class JournalAtomicityTest(unittest.TestCase):
    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.workdir = os.getcwd()
        self.journal = get_delivery_journal_path(CONF)
        self._counter = 0

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    # ------------------------------------------------------------------
    def _payload(self, **overrides):
        self._counter += 1
        payload = {
            "conf": CONF,
            "slot": f"s{self._counter}",
            "result_path": os.path.join(self.workdir, f"r{self._counter}.json"),
            "duration": 1.5,
            "per_batch": 100,
            "lease": LEASE,
        }
        payload.update(overrides)
        return payload

    def _run_together(self, participants, *, ready_timeout=25.0):
        """Start every ``(script, payload)`` pair behind one shared barrier."""
        ready_dir = os.path.join(self.workdir, f"ready-{uuid.uuid4().hex}")
        os.makedirs(ready_dir, exist_ok=True)
        for _script, payload in participants:
            payload["ready_dir"] = ready_dir
        processes = [
            subprocess.Popen(
                [sys.executable, "-c", script, ROOT, self.workdir, json.dumps(payload)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            for script, payload in participants
        ]
        deadline = time.time() + ready_timeout
        while len(os.listdir(ready_dir)) < len(participants) and time.time() < deadline:
            time.sleep(0.005)
        self.assertEqual(
            len(os.listdir(ready_dir)),
            len(participants),
            "children never became ready",
        )
        child_pids = {process.pid for process in processes}
        self.assertEqual(len(child_pids), len(participants), child_pids)
        self.assertNotIn(os.getpid(), child_pids)
        open(os.path.join(ready_dir, "start"), "w").close()
        results = []
        for process, (_script, payload) in zip(processes, participants):
            stdout, stderr = process.communicate(timeout=180)
            self.assertEqual(
                process.returncode,
                0,
                f"child failed: {stderr[-2000:]} {stdout[-500:]}",
            )
            if payload.get("result_path"):
                with open(payload["result_path"], "r", encoding="utf-8") as handle:
                    results.append(json.load(handle))
        return results

    def _journal_lines(self):
        with open(self.journal, "r", encoding="utf-8") as handle:
            return [line for line in handle if line.strip()]

    def _journal_ids(self):
        return [json.loads(line)["id"] for line in self._journal_lines()]

    def _seed(self):
        state = load_character_state(CONF)
        row = {
            "id": uuid.uuid4().hex,
            "text": "ingetin aku minum obat ya",
            "due_at": (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat(),
            "created_at": (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(),
            "status": "pending",
            "source": "conversation",
            "kind": "reminder",
            "tz": "Asia/Jakarta",
            "delivery_text": "Aku ingetin minum obat ya.",
        }
        state.future_intentions = [row]
        self.assertTrue(save_character_state(CONF, state))
        return row

    def _history(self):
        from src.open_llm_vtuber.chat_history_manager import create_new_history

        return create_new_history(CONF)

    def _history_messages(self, history_uid):
        from src.open_llm_vtuber.chat_history_manager import get_history

        return get_history(CONF, history_uid)

    # ------------------------------------------------------------------
    # Concurrent append
    # ------------------------------------------------------------------
    def test_concurrent_batches_from_two_processes_write_only_complete_lines(self):
        """Two writers, 100 ids per append: every line must stay parseable.

        Each batch is one ``O_APPEND`` write of ~100 JSON lines. If the two
        writers could interleave inside a line, ``delivered_ids`` would discard
        the corrupt line -- a durable receipt that silently disappears and
        lets the reminder be sent again.
        """
        participants = [
            (WRITER_CHILD, self._payload(slot=slot, batches=40, duration=60.0))
            for slot in ("alpha", "beta")
        ]
        results = self._run_together(participants)

        for result in results:
            self.assertTrue(all(result["ok"]), result)
        self.assertEqual(len(self._journal_lines()), 80 * 100)
        ids = self._journal_ids()  # raises on any torn line
        self.assertEqual(len(ids), len(set(ids)), "a receipt was written twice")
        expected = {
            f"{slot}-{batch:05d}-{index:03d}"
            for slot in ("alpha", "beta")
            for batch in range(40)
            for index in range(100)
        }
        self.assertEqual(set(ids), expected)
        self.assertEqual(delivered_ids(CONF), expected)

    def test_two_different_receipts_appended_at_the_same_instant_are_both_durable(self):
        participants = [
            (WRITER_CHILD, self._payload(slot=slot, batches=1, duration=60.0))
            for slot in ("alpha", "beta")
        ]
        results = self._run_together(participants)

        self.assertEqual([result["ok"] for result in results], [[True], [True]])
        ids = self._journal_ids()
        self.assertEqual(len(ids), 200)
        self.assertEqual(len(set(ids)), 200)
        self.assertEqual(len(delivered_ids(CONF)), 200)

    def test_receipt_appended_after_a_crashed_partial_line_is_still_durable(self):
        """A dead writer's truncated line must not poison the NEXT receipt.

        The journal is append-only, so a process killed mid-append can leave a
        line without its newline. Appending the next receipt straight after it
        glues the two together into one unparseable line, and
        ``delivered_ids`` silently discards it -- the fresh receipt is lost and
        the reminder can be sent again after a failed completion.
        """
        os.makedirs(os.path.dirname(self.journal), exist_ok=True)
        with open(self.journal, "w", encoding="utf-8") as handle:
            handle.write('{"id":"torn","owner":"dead","delivered')

        self.assertTrue(record_delivery(CONF, ["fresh-receipt"], owner="alive"))
        self.assertEqual(delivered_ids(CONF), {"fresh-receipt"})
        # The surviving file is still line-delimited: one ignored fragment
        # plus one complete receipt.
        with open(self.journal, "r", encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[-1])["id"], "fresh-receipt")

    def test_concurrent_appends_after_a_crashed_partial_line_keep_every_receipt(
        self,
    ):
        """Two live writers after a torn tail: both receipts must survive."""
        os.makedirs(os.path.dirname(self.journal), exist_ok=True)
        with open(self.journal, "w", encoding="utf-8") as handle:
            handle.write('{"id":"torn","owner":"dead","delivered')

        participants = [
            (WRITER_CHILD, self._payload(slot=slot, batches=20, duration=60.0))
            for slot in ("alpha", "beta")
        ]
        results = self._run_together(participants)

        for result in results:
            self.assertTrue(all(result["ok"]), result)
        parsed = []
        torn = []
        for line in self._journal_lines():
            try:
                parsed.append(json.loads(line)["id"])
            except ValueError:
                torn.append(line)
        # Exactly the dead writer's fragment stays unparseable; every receipt
        # written by a live process survives intact.
        self.assertEqual(len(torn), 1, torn)
        self.assertEqual(len(parsed), len(set(parsed)), "a receipt was written twice")
        self.assertEqual(len(parsed), 40 * 100)
        self.assertEqual(delivered_ids(CONF), set(parsed))

    # ------------------------------------------------------------------
    # Concurrent read
    # ------------------------------------------------------------------
    def test_delivered_ids_never_loses_a_durable_receipt_while_another_process_writes(
        self,
    ):
        """The claim path's journal read against a live writer.

        ``delivered_ids`` takes no lock, so the question is empirical: while
        two processes append hundreds of lines, a third must never observe a
        torn line, an unreadable file, or an id it had already read disappear.
        """
        participants = [
            (
                WRITER_CHILD,
                self._payload(slot=slot, batches=None, duration=1.5, per_batch=100),
            )
            for slot in ("alpha", "beta")
        ]
        participants.append(
            (READER_CHILD, self._payload(slot="reader", duration=1.7))
        )
        results = self._run_together(participants)
        report = results[-1]

        self.assertGreater(report["reads"], 10, report)
        self.assertEqual(report["torn"], [], report)
        self.assertEqual(report["shrinks"], [], report)
        self.assertGreater(report["seen"], 0, report)
        # The writers finished fewer lines than the final file holds, so the
        # reader really did observe a growing journal.
        self.assertGreater(len(self._journal_ids()), 200)
        self.assertGreaterEqual(len(delivered_ids(CONF)), report["seen"])

    def test_delivered_ids_waits_for_an_in_progress_append(self):
        """Deterministic: a half-written receipt must not be read as absent.

        The writer holds the journal lock with half a receipt line on disk and
        finishes it 0.8 s later. A reader without a shared lock reads the torn
        line, discards it as corrupt and reports "no receipt" -- after which
        the claim path may deliver the reminder again. With the shared lock the
        reader waits and sees the complete receipt.
        """
        ready_dir = os.path.join(self.workdir, "partial-ready")
        os.makedirs(ready_dir, exist_ok=True)
        target = "in-progress-receipt"
        writer = subprocess.Popen(
            [
                sys.executable,
                "-c",
                PARTIAL_WRITER_CHILD,
                ROOT,
                self.workdir,
                json.dumps({"conf": CONF, "id": target, "hold": 0.8, "ready_dir": ready_dir}),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        marker = os.path.join(ready_dir, "partial")
        deadline = time.time() + 20
        while not os.path.exists(marker) and time.time() < deadline:
            time.sleep(0.005)
        self.assertTrue(os.path.exists(marker), "writer never published a partial line")

        reader_payload = {
            "conf": CONF,
            "result_path": os.path.join(self.workdir, "partial-reader.json"),
        }
        reader = subprocess.run(
            [
                sys.executable,
                "-c",
                PARTIAL_READER_CHILD,
                ROOT,
                self.workdir,
                json.dumps(reader_payload),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(reader.returncode, 0, reader.stderr[-2000:])
        stdout, stderr = writer.communicate(timeout=60)
        self.assertEqual(writer.returncode, 0, f"{stderr[-2000:]} {stdout[-200:]}")

        with open(reader_payload["result_path"], "r", encoding="utf-8") as handle:
            report = json.load(handle)
        self.assertIn(target, report["ids"], report)
        self.assertGreaterEqual(
            report["elapsed"], 0.4, "reader did not wait for the append to finish"
        )
        self.assertEqual(delivered_ids(CONF), {target})

    # ------------------------------------------------------------------
    # Crash windows
    # ------------------------------------------------------------------
    def test_crash_after_fsync_before_completion_is_reconciled_without_resend(self):
        """os._exit right after record_delivery: no completion write at all."""
        row = self._seed()
        history = self._history()
        crash = self._payload(
            slot="crash",
            stage="after_receipt",
            owner="crash-owner",
            now=datetime.now(timezone.utc).isoformat(),
            text="Aku ingetin minum obat ya.",
            history=history,
            result_path=None,
        )
        self._run_together([(CRASH_CHILD, crash)])

        stored = load_character_state(CONF).future_intentions[0]
        self.assertEqual(stored["status"], "pending")
        self.assertEqual(stored["claimed_by"], "crash-owner")
        self.assertEqual(delivered_ids(CONF), {row["id"]})

        later = (datetime.now(timezone.utc) + timedelta(seconds=900)).isoformat()
        restart = self._payload(
            slot="restart",
            owner="restart-owner",
            now=later,
            text="restart row",
            history=history,
        )
        results = self._run_together([(RESTART_CHILD, restart)])
        self.assertEqual(results[0]["claimed"], [])
        self.assertFalse(results[0]["receipt"])
        self.assertEqual(
            load_character_state(CONF).future_intentions[0]["status"], "done"
        )
        self.assertEqual(self._history_messages(history), [])
        # The receipt was accepted once and never rewritten.
        self.assertEqual(len(self._journal_ids()), 1)

    def test_crash_between_transport_frame_and_receipt_leaves_the_window_open(self):
        """No client ACK: a crash before the receipt cannot be recovered.

        The transcript row proves the words left this process, but nothing
        durable links them to the reminder, so the restart has no evidence and
        must re-deliver -- a second transcript row. Asserted so this is never
        mistaken for exactly-once delivery.
        """
        row = self._seed()
        history = self._history()
        crash = self._payload(
            slot="crash",
            stage="before_receipt",
            owner="crash-owner",
            now=datetime.now(timezone.utc).isoformat(),
            text="first delivery",
            history=history,
            result_path=None,
        )
        self._run_together([(CRASH_CHILD, crash)])

        self.assertEqual(delivered_ids(CONF), set())
        stored = load_character_state(CONF).future_intentions[0]
        self.assertEqual(stored["status"], "pending")
        self.assertEqual(
            [item["content"] for item in self._history_messages(history)],
            ["first delivery"],
        )

        later = (datetime.now(timezone.utc) + timedelta(seconds=900)).isoformat()
        restart = self._payload(
            slot="restart",
            owner="restart-owner",
            now=later,
            text="second delivery",
            history=history,
        )
        results = self._run_together([(RESTART_CHILD, restart)])
        self.assertEqual(results[0]["claimed"], [row["id"]])
        self.assertTrue(results[0]["receipt"])
        self.assertEqual(results[0]["completed"], 1)
        self.assertEqual(
            [item["content"] for item in self._history_messages(history)],
            ["first delivery", "second delivery"],
        )
        self.assertEqual(
            load_character_state(CONF).future_intentions[0]["status"], "done"
        )


if __name__ == "__main__":
    unittest.main()
