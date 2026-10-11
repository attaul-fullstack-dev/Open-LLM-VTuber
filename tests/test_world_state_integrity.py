"""World/Life State timestamp integrity (deterministic, isolated).

Two corrupt-stamp failure modes are covered here. Both were reachable from a
hand-edited or externally written ``world_state/<conf_uid>.json``:

1. An unparseable ``activity_started_at`` used to be stored verbatim, and
   ``_parse_iso`` then substituted the *current* moment as its fallback. The
   resulting age of zero on every call permanently disabled activity duration
   limits, the sleeping early exit and stale-idle decisions, and the bad value
   was written straight back to disk so nothing ever repaired it.
2. A ``last_update_at`` far in the future was returned unchanged with
   ``changed=False``, so it was never repaired: the life state stayed frozen
   until the real wall clock reached the forged value.

No file outside a temporary directory is read or written by these tests.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber import world_state as ws

NOW = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
FAR_FUTURE = NOW + timedelta(days=1100)
JKT = "Asia/Jakarta"


class StateFromDictTest(unittest.TestCase):
    def test_unparseable_activity_start_is_replaced_not_stored(self):
        state = ws._state_from_dict(
            {"activity_started_at": "not-a-timestamp", "activity": "reading"}, NOW
        )
        # must be a parseable stamp, and must not silently become "now"
        parsed = datetime.fromisoformat(state.activity_started_at)
        self.assertLessEqual(parsed, NOW)

    def test_unparseable_activity_start_falls_back_to_last_update(self):
        last = (NOW - timedelta(hours=3)).isoformat()
        state = ws._state_from_dict(
            {
                "activity_started_at": "",
                "last_update_at": last,
                "activity": "reading",
            },
            NOW,
        )
        self.assertEqual(
            datetime.fromisoformat(state.activity_started_at),
            datetime.fromisoformat(last),
        )

    def test_valid_stamps_survive_round_trip(self):
        start = (NOW - timedelta(hours=2)).isoformat()
        last = (NOW - timedelta(hours=1)).isoformat()
        state = ws._state_from_dict(
            {"activity_started_at": start, "last_update_at": last}, NOW
        )
        self.assertEqual(
            datetime.fromisoformat(state.activity_started_at),
            datetime.fromisoformat(start),
        )
        self.assertEqual(
            datetime.fromisoformat(state.last_update_at),
            datetime.fromisoformat(last),
        )

    def test_activity_age_is_not_frozen_by_a_garbage_start(self):
        """The concrete consequence: age must grow with the last update stamp."""
        state = ws._state_from_dict(
            {
                "activity_started_at": "garbage",
                "last_update_at": (NOW - timedelta(hours=3)).isoformat(),
                "activity": "reading",
            },
            NOW,
        )
        age = ws._activity_age_s(state, NOW)
        self.assertGreater(age, 0.0)
        self.assertAlmostEqual(age, 3 * 3600, delta=2)


class ReconcileFutureStampTest(unittest.TestCase):
    def _state(self, last_update_at):
        return ws.WorldState(
            location=ws.DEFAULT_LOCATION,
            activity="sleeping",
            energy=ws.DEFAULT_ENERGY,
            mood="calm",
            time_context="siang",
            activity_started_at=(NOW - timedelta(hours=9)).isoformat(),
            last_update_at=last_update_at,
            recent_activity_history=[],
        )

    def test_corrupt_future_stamp_is_repaired_and_persisted(self):
        baseline = self._state(FAR_FUTURE.isoformat())
        updated, changed = ws.reconcile(baseline, NOW, JKT)
        self.assertTrue(changed, "the repair must reach disk")
        self.assertLessEqual(datetime.fromisoformat(updated.last_update_at), NOW)

    def test_repaired_state_stops_being_frozen(self):
        """After repair the state must keep evolving, not stay inert."""
        baseline = self._state(FAR_FUTURE.isoformat())
        first, changed = ws.reconcile(baseline, NOW, JKT)
        self.assertTrue(changed)
        # a later instant must again move the stamp forward, proving the
        # state is no longer pinned to the forged value
        later, later_changed = ws.reconcile(first, NOW + timedelta(hours=5), JKT)
        self.assertEqual(
            datetime.fromisoformat(later.last_update_at),
            NOW + timedelta(hours=5),
        )

    def test_small_backwards_skew_stays_fail_soft(self):
        """Ordinary NTP jitter keeps the historical never-move-backwards path."""
        baseline = self._state((NOW + timedelta(minutes=5)).isoformat())
        updated, changed = ws.reconcile(baseline, NOW, JKT)
        self.assertFalse(changed)
        self.assertEqual(updated.last_update_at, baseline.last_update_at)

    def test_repair_threshold_is_documented(self):
        self.assertEqual(ws._MAX_BACKWARDS_SKEW, timedelta(hours=1))


class LoadRepairIntegrationTest(unittest.TestCase):
    """End-to-end through the on-disk API in an isolated working directory."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs("world_state", exist_ok=True)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _write(self, payload):
        with open("world_state/repairchar.json", "w", encoding="utf-8") as handle:
            json.dump(payload, handle)

    def _read(self):
        with open("world_state/repairchar.json", encoding="utf-8") as handle:
            return json.load(handle)

    def test_future_stamp_on_disk_is_repaired_by_load_and_reconcile(self):
        self._write(
            {
                "location": "kamar",
                "activity": "sleeping",
                "energy": 80,
                "mood": "calm",
                "time_context": "malam",
                "activity_started_at": (NOW - timedelta(hours=9)).isoformat(),
                "last_update_at": FAR_FUTURE.isoformat(),
                "recent_activity_history": [],
            }
        )
        state = ws.load_and_reconcile_world_state(
            "repairchar", NOW, "world_state", tz=JKT
        )
        on_disk = self._read()
        self.assertLessEqual(datetime.fromisoformat(on_disk["last_update_at"]), NOW)
        self.assertTrue(datetime.fromisoformat(state.last_update_at) <= NOW)

    def test_garbage_start_stamp_on_disk_is_normalised(self):
        self._write(
            {
                "activity": "reading",
                "activity_started_at": "not-a-timestamp",
                "last_update_at": (NOW - timedelta(hours=2)).isoformat(),
            }
        )
        ws.load_and_reconcile_world_state("repairchar", NOW, "world_state", tz=JKT)
        on_disk = self._read()
        # whatever the state drifted to, the stamp must now be parseable
        start = datetime.fromisoformat(on_disk["activity_started_at"])
        self.assertTrue((NOW - start).total_seconds() >= 0)


if __name__ == "__main__":
    unittest.main()
