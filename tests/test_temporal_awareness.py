"""Temporal awareness — minimal acceptance tests (fixed clock, no I/O outside tmp).

Covers: current-date anchor (A), memory Today/Yesterday/N-days (B-D),
dynamic age without rewriting storage (E), user-timezone day boundaries (F),
UTC-aware chat history timestamps (G), unchanged ordering (H), no LLM calls (I).
"""

import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber.agent.relationship_context import (
    build_relationship_context,
)
from src.open_llm_vtuber.character_state import (
    CharacterState,
    build_character_memory_context,
)
from src.open_llm_vtuber.world_state import (
    format_temporal_anchor,
    memory_age_label,
)
from src.open_llm_vtuber import chat_history_manager

JAKARTA = "Asia/Jakarta"
# Oct 1 2026 07:00 Jakarta == Sep 30 2026 24:00 UTC.
NOW_OCT1 = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)


def mem(text, added_at, explicit=True):
    return {"text": text, "added_at": added_at, "explicit": explicit}


class TemporalAnchorTest(unittest.TestCase):
    def test_a_current_date_weekday_timezone(self):
        anchor = format_temporal_anchor(
            datetime(2026, 9, 29, 2, 0, tzinfo=timezone.utc), JAKARTA
        )
        self.assertIn("September 29, 2026", anchor)
        self.assertIn("Tuesday", anchor)
        self.assertIn("Asia/Jakarta", anchor)

    def test_a_utc_fallback_without_timezone(self):
        anchor = format_temporal_anchor(
            datetime(2026, 9, 29, 2, 0, tzinfo=timezone.utc), None
        )
        self.assertIn("September 29, 2026", anchor)
        self.assertIn("UTC", anchor)

    def test_a_invalid_timezone_falls_back_to_utc(self):
        anchor = format_temporal_anchor(NOW_OCT1, "Not/AZone")
        self.assertIn("UTC", anchor)


class MemoryAgeLabelTest(unittest.TestCase):
    def test_b_memory_today(self):
        label = memory_age_label("2026-09-30T18:00:00+00:00", NOW_OCT1, JAKARTA)
        self.assertTrue(label.startswith("[Today | "), label)

    def test_c_memory_yesterday(self):
        label = memory_age_label("2026-09-30T10:00:00+00:00", NOW_OCT1, JAKARTA)
        self.assertTrue(label.startswith("[Yesterday | "), label)

    def test_d_memory_two_days_ago_with_date(self):
        label = memory_age_label("2026-09-29T07:00:00+00:00", NOW_OCT1, JAKARTA)
        self.assertEqual(label, "[2 days ago | Sep 29]")

    def test_d_memory_older(self):
        label = memory_age_label("2026-08-01T00:00:00+00:00", NOW_OCT1, JAKARTA)
        self.assertIn("days ago", label)
        self.assertIn("Aug 1", label)

    def test_e_same_storage_new_age_when_clock_advances(self):
        stored = "2026-09-29T07:00:00+00:00"
        day1 = memory_age_label(stored, NOW_OCT1, JAKARTA)
        day2 = memory_age_label(
            stored, datetime(2026, 10, 5, 0, 0, tzinfo=timezone.utc), JAKARTA
        )
        self.assertEqual(day1, "[2 days ago | Sep 29]")
        self.assertEqual(day2, "[6 days ago | Sep 29]")
        self.assertNotIn("today", stored.lower())

    def test_f_day_boundary_uses_user_timezone_not_server(self):
        # Same two instants read differently per zone: added Sep 29 17:00Z
        # is Sep 29 in UTC but Sep 30 00:00+ in Jakarta; now Sep 30 16:00Z
        # is Sep 30 in both zones.
        added = "2026-09-29T17:00:00+00:00"
        now = datetime(2026, 9, 30, 16, 0, tzinfo=timezone.utc)
        self.assertTrue(memory_age_label(added, now, "UTC").startswith("[Yesterday | "))
        self.assertTrue(memory_age_label(added, now, JAKARTA).startswith("[Today | "))

    def test_unparseable_renders_untagged(self):
        self.assertEqual(memory_age_label("not-a-date", NOW_OCT1, JAKARTA), "")
        self.assertEqual(memory_age_label("", NOW_OCT1, JAKARTA), "")

    def test_future_skew_reads_as_today(self):
        label = memory_age_label("2026-10-02T00:00:00+00:00", NOW_OCT1, JAKARTA)
        self.assertTrue(label.startswith("[Today | "), label)


class MemoryContextRenderTest(unittest.TestCase):
    def test_memory_bullets_carry_age_text_preserved_order_kept(self):
        state = CharacterState(
            memories=[
                mem(
                    "User worked on frontend for 3 hours because of a bug.",
                    "2026-09-29T07:00:00+00:00",
                ),
                mem("User likes tea.", "2026-09-30T10:00:00+00:00"),
            ]
        )
        block = build_character_memory_context(state, tz=JAKARTA, now=NOW_OCT1)
        self.assertIn("[2 days ago | Sep 29] User worked on frontend", block)
        self.assertIn("[Yesterday | Sep 30] User likes tea.", block)
        # Both explicit -> ascending added_at: Sep 29 line first (unchanged).
        self.assertLess(block.index("Sep 29"), block.index("Sep 30"))

    def test_h_explicit_priority_ordering_unchanged(self):
        state = CharacterState(
            memories=[
                mem("newer implicit fact", "2026-09-30T10:00:00+00:00", explicit=False),
                mem("older explicit fact", "2026-09-20T10:00:00+00:00", explicit=True),
            ]
        )
        block = build_character_memory_context(state, tz=JAKARTA, now=NOW_OCT1)
        self.assertLess(
            block.index("older explicit fact"), block.index("newer implicit fact")
        )

    def test_unparseable_memory_text_survives(self):
        state = CharacterState(memories=[mem("legacy fact", "someday")])
        block = build_character_memory_context(state, tz=JAKARTA, now=NOW_OCT1)
        self.assertIn("- legacy fact", block)

    def test_empty_memories_empty_block(self):
        self.assertEqual(
            build_character_memory_context(CharacterState(), tz=JAKARTA, now=NOW_OCT1),
            "",
        )


class RelationshipAgeTest(unittest.TestCase):
    def test_status_carries_update_age(self):
        ctx = build_relationship_context(
            "close", updated_at="2026-09-28T00:00:00+00:00", tz=JAKARTA, now=NOW_OCT1
        )
        self.assertIn("close (status updated 3 days ago, Sep 28)", ctx)

    def test_missing_updated_at_renders_legacy_line(self):
        ctx = build_relationship_context("familiar")
        self.assertIn("Current state: familiar.", ctx)


class ChatHistoryTimestampTest(unittest.TestCase):
    def test_g_new_records_are_utc_aware(self):
        with tempfile.TemporaryDirectory() as tmp:
            cwd = os.getcwd()
            os.chdir(tmp)
            try:
                uid = chat_history_manager.create_new_history("test-conf")
                self.assertTrue(uid)
                path = os.path.join("chat_history", "test-conf", f"{uid}.json")
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                self.assertTrue(data[0]["timestamp"].endswith("+00:00"))
                chat_history_manager.store_message("test-conf", uid, "human", "halo")
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                stamps = [m["timestamp"] for m in data if m.get("timestamp")]
                self.assertTrue(stamps)
                for stamp in stamps:
                    self.assertTrue(stamp.endswith("+00:00"), stamp)
                    datetime.fromisoformat(stamp)
            finally:
                os.chdir(cwd)


class NoLlmCallTest(unittest.TestCase):
    def test_i_temporal_helpers_are_synchronous_pure(self):
        for fn in (
            format_temporal_anchor,
            memory_age_label,
            build_character_memory_context,
            build_relationship_context,
        ):
            self.assertFalse(asyncio.iscoroutinefunction(fn), fn.__name__)


class VpsRestartOfflineTest(unittest.TestCase):
    """Tuesday 15:00 (user-local) shutdown -> Thursday 12:00 restart.

    Proves elapsed wall-clock time comes from persisted timestamp vs the
    runtime system clock (fake clock here), never from process startup.
    """

    TUE_15_JKT = datetime(2026, 9, 29, 8, 0, tzinfo=timezone.utc)
    THU_12_JKT = datetime(2026, 10, 1, 5, 0, tzinfo=timezone.utc)

    def test_restart_uses_persisted_timestamp_not_startup(self):
        from src.open_llm_vtuber import world_state as ws_mod
        from src.open_llm_vtuber.world_state import (
            default_state,
            format_temporal_anchor,
            load_and_reconcile_world_state,
            save_world_state,
        )

        with tempfile.TemporaryDirectory() as tmp:
            # 1-2. Tuesday 15:00 user-local: persist life state, then
            # "shut down" (drop every in-memory reference).
            fresh = default_state(self.TUE_15_JKT, JAKARTA)
            initial_energy = fresh.energy
            self.assertTrue(save_world_state("mili", fresh, tmp))
            del fresh

            # 3-4. Thursday 12:00 user-local: "restart", reconcile from disk.
            revived = load_and_reconcile_world_state(
                "mili", now=self.THU_12_JKT, base_dir=tmp, tz=JAKARTA
            )

            # 6a. Current date derives from Thursday 12:00, not Tuesday.
            anchor = format_temporal_anchor(self.THU_12_JKT, JAKARTA)
            self.assertIn("October 1, 2026 (Thursday)", anchor)
            self.assertIn("Asia/Jakarta", anchor)
            # 6b. Timezone conversion correct (05:00Z == 12:00 Jakarta).
            local = ws_mod.user_local_datetime(self.THU_12_JKT, JAKARTA)
            self.assertEqual((local.hour, local.day), (12, 1))
            # 6c. Elapsed 45h wall-clock from persisted stamp applied:
            # last_update_at advanced to restart moment, energy decayed.
            self.assertEqual(
                revived.last_update_at,
                self.THU_12_JKT.isoformat(timespec="seconds"),
            )
            self.assertLess(revived.energy, initial_energy)

    def test_memory_age_survives_restart_anchored_to_original_stamp(self):
        stored = "2026-09-29T08:00:00+00:00"  # Tue 15:00 Jakarta
        state = CharacterState(
            memories=[
                {
                    "text": "hari ini bikin frontend 3 jam gara-gara satu bug.",
                    "added_at": stored,
                    "explicit": True,
                }
            ]
        )
        block = build_character_memory_context(state, tz=JAKARTA, now=self.THU_12_JKT)
        self.assertIn("[2 days ago | Sep 29]", block)
        self.assertIn("hari ini bikin frontend", block)

    def test_default_clock_is_runtime_system_time(self):
        # No `now` passed anywhere: the stamp must come from the live
        # system clock at call time (proves no startup-time baseline).
        from src.open_llm_vtuber.world_state import default_state

        before = datetime.now(timezone.utc)
        stamp = default_state(None, JAKARTA).last_update_at
        after = datetime.now(timezone.utc)
        parsed = datetime.fromisoformat(stamp)
        # Stamps are second-precision; allow sub-second truncation skew.
        self.assertLessEqual(
            before.replace(microsecond=0), parsed + timedelta(seconds=1)
        )
        self.assertLessEqual(parsed, after)


if __name__ == "__main__":
    unittest.main()
