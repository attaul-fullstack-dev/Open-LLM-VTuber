"""Temporal session awareness — deterministic tests (fake clock, tmp dirs).

Covers the Temporal Awareness stage on top of test_temporal_awareness.py:
seed-goal stamp honors injected time (B1), reactive tz plumbing (B2),
previous-session recency rendering (M1), persisted user timezone (M2),
summary age tag (M3), restart/offline gap, and stale-clock regression.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.conversation_summary import build_summary_message
from src.open_llm_vtuber.character_state import (
    default_seed_goals,
    load_character_state,
    set_character_timezone,
)
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history_list,
    store_message,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.world_state import (
    WorldState,
    apply_reactive,
    format_session_recency,
    format_temporal_anchor,
    load_world_state,
    reconcile,
    save_world_state,
    utcnow,
)

JAKARTA = "Asia/Jakarta"
# Fixed instant: Oct 1 2026 00:00 UTC == 07:00 Jakarta (Wednesday).
T0 = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)


class _FakeLLM:
    model = "temporal-test"
    max_tokens = 100

    async def chat_completion(self, messages, system=None, tools=None):
        yield "oke."


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


def make_agent(conf_uid, history_uid, tz=None):
    agent = BasicMemoryAgent(
        llm=_FakeLLM(),
        system="persona",
        live2d_model=_FakeLive2D(),
        tts_preprocessor_config=TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={
                "translate_audio": False,
                "translate_provider": "deeplx",
            },
        ),
    )
    agent.set_memory_from_history(conf_uid, history_uid, user_timezone=tz)
    return agent


def playing_state(at):
    iso = at.isoformat(timespec="seconds")
    return WorldState(
        location="room",
        activity="playing",
        energy=80,
        mood="playful",
        time_context="evening",
        activity_started_at=iso,
        last_update_at=iso,
        recent_activity_history=[],
        mood_ttl_turns=0,
        mood_set_at=None,
    )


class SeedGoalStampTest(unittest.TestCase):
    def test_injected_utc_now_honored(self):
        goals = default_seed_goals(T0)
        for goal in goals:
            self.assertEqual(goal["created_at"], "2026-10-01T00:00:00+00:00")

    def test_injected_offset_converted_to_canonical_utc(self):
        plus7 = T0.astimezone(__import__("zoneinfo").ZoneInfo(JAKARTA))
        goals = default_seed_goals(plus7)
        for goal in goals:
            self.assertEqual(goal["created_at"], "2026-10-01T00:00:00+00:00")

    def test_naive_now_assumed_utc(self):
        goals = default_seed_goals(datetime(2026, 10, 1, 0, 0))
        for goal in goals:
            self.assertEqual(goal["created_at"], "2026-10-01T00:00:00+00:00")

    def test_none_uses_real_clock(self):
        before = utcnow()
        stamp = default_seed_goals()[0]["created_at"]
        after = utcnow()
        parsed = datetime.fromisoformat(stamp)
        self.assertIsNotNone(parsed.tzinfo)
        self.assertLessEqual((parsed - before).total_seconds(), 120)
        self.assertLessEqual((after - parsed).total_seconds(), 120)


class ApplyReactiveTzTest(unittest.TestCase):
    def test_tz_plumbing_parity_on_charged_playing_turn(self):
        # 11:00 UTC: day in UTC, 18:00 Jakarta (night). A charged turn
        # interrupts playing -> idle either way; tz must not change that.
        moment = datetime(2026, 9, 30, 11, 0, tzinfo=timezone.utc)
        plain, changed_plain = apply_reactive(playing_state(moment), ["joy"], moment)
        zoned, changed_zoned = apply_reactive(
            playing_state(moment), ["joy"], moment, tz=JAKARTA
        )
        self.assertTrue(changed_plain)
        self.assertTrue(changed_zoned)
        self.assertEqual(zoned.activity, plain.activity)
        self.assertEqual(zoned.activity, "idle")
        self.assertEqual(zoned.location, plain.location)
        self.assertEqual(zoned.energy, plain.energy)
        self.assertEqual(zoned.recent_activity_history, plain.recent_activity_history)


class SessionRecencyTest(unittest.TestCase):
    def test_same_day(self):
        line = format_session_recency(
            "2026-09-30T10:00:00+00:00",
            datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc),
            JAKARTA,
        )
        self.assertEqual(line, "Previous conversation: today (Sep 30).")

    def test_yesterday(self):
        line = format_session_recency(
            "2026-09-29T10:00:00+00:00",
            datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc),
            JAKARTA,
        )
        self.assertEqual(line, "Previous conversation: yesterday (Sep 29).")

    def test_n_days_ago(self):
        line = format_session_recency(
            "2026-09-27T10:00:00+00:00",
            datetime(2026, 9, 30, 15, 0, tzinfo=timezone.utc),
            None,
        )
        self.assertEqual(line, "Previous conversation: 3 days ago (Sep 27).")

    def test_midnight_boundary_uses_user_timezone(self):
        # 00:30 Jakarta Oct 1; previous 23:30 Jakarta Sep 30 -> yesterday
        # in Jakarta, but the same calendar day in UTC.
        line = format_session_recency(
            "2026-09-30T16:30:00+00:00",
            datetime(2026, 9, 30, 17, 30, tzinfo=timezone.utc),
            JAKARTA,
        )
        self.assertIn("yesterday", line)

    def test_missing_or_bad_timestamp_omits_line(self):
        self.assertEqual(format_session_recency(None, T0, JAKARTA), "")
        self.assertEqual(format_session_recency("", T0, JAKARTA), "")
        self.assertEqual(format_session_recency("not-a-time", T0, JAKARTA), "")

    def test_future_timestamp_clamps_to_today(self):
        line = format_session_recency("2026-10-05T00:00:00+00:00", T0, JAKARTA)
        self.assertIn("today", line)

    def test_naive_legacy_timestamp_read_as_utc(self):
        line = format_session_recency("2026-09-30T10:00:00", T0, None)
        self.assertIn("yesterday", line)


class TimezonePersistenceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        self.conf_uid = "tzchar"

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_roundtrip(self):
        state = set_character_timezone(self.conf_uid, JAKARTA)
        self.assertIsNotNone(state)
        self.assertEqual(state.user_timezone, JAKARTA)
        self.assertEqual(load_character_state(self.conf_uid).user_timezone, JAKARTA)

    def test_blank_never_wipes_stored_zone(self):
        set_character_timezone(self.conf_uid, JAKARTA)
        self.assertIsNone(set_character_timezone(self.conf_uid, ""))
        self.assertIsNone(set_character_timezone(self.conf_uid, None))
        self.assertEqual(load_character_state(self.conf_uid).user_timezone, JAKARTA)

    def test_agent_falls_back_to_stored_zone(self):
        set_character_timezone(self.conf_uid, JAKARTA)
        history_uid = create_new_history(self.conf_uid)
        agent = make_agent(self.conf_uid, history_uid, tz=None)
        self.assertEqual(agent._user_timezone, JAKARTA)

    def test_session_zone_wins_over_stored(self):
        set_character_timezone(self.conf_uid, JAKARTA)
        history_uid = create_new_history(self.conf_uid)
        agent = make_agent(self.conf_uid, history_uid, tz="UTC")
        self.assertEqual(agent._user_timezone, "UTC")


class PrevSessionScanTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        self.conf_uid = "scanchar"

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_returns_other_session_and_never_deletes(self):
        import json

        def write_history(uid, stamp):
            path = os.path.join("chat_history", self.conf_uid, f"{uid}.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    [
                        {"role": "metadata", "timestamp": stamp, "title": ""},
                        {"role": "human", "timestamp": stamp, "content": "halo"},
                    ],
                    handle,
                )

        current = create_new_history(self.conf_uid)
        older = create_new_history(self.conf_uid)
        old_stamp = "2026-09-29T10:00:00+00:00"
        new_stamp = "2026-09-30T10:00:00+00:00"
        write_history(older, old_stamp)
        write_history(current, new_stamp)
        found = BasicMemoryAgent._latest_other_session_at(self.conf_uid, current)
        self.assertEqual(found, old_stamp)
        # Read-only: the scan must not delete or rewrite anything.
        listing = {
            item["uid"]: item for item in get_history_list(self.conf_uid, cleanup=False)
        }
        self.assertIn(current, listing)
        self.assertIn(older, listing)
        with open(
            os.path.join("chat_history", self.conf_uid, f"{current}.json"),
            encoding="utf-8",
        ) as handle:
            self.assertEqual(json.load(handle)[1]["timestamp"], new_stamp)

    def test_none_when_no_previous_session(self):
        only = create_new_history(self.conf_uid)
        store_message(self.conf_uid, only, "human", "halo")
        self.assertIsNone(
            BasicMemoryAgent._latest_other_session_at(self.conf_uid, only)
        )


class RestartOfflineGapTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        self.conf_uid = "gapchar"

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_world_survives_restart_and_reconciles_gap(self):
        # Persist at T0, "restart" (fresh load), reconcile 26h later.
        self.assertTrue(save_world_state(self.conf_uid, playing_state(T0)))
        fresh = load_world_state(self.conf_uid)
        self.assertEqual(fresh.activity, "playing")
        later = T0 + timedelta(hours=26)
        updated, _ = reconcile(fresh, later, JAKARTA)
        self.assertIsNotNone(updated.last_update_at)
        self.assertGreater(
            datetime.fromisoformat(updated.last_update_at),
            datetime.fromisoformat(fresh.last_update_at),
        )
        # Recency across the gap renders from absolute timestamps.
        line = format_session_recency(fresh.last_update_at, later, JAKARTA)
        self.assertIn("yesterday", line)

    def test_timezone_survives_restart(self):
        set_character_timezone(self.conf_uid, JAKARTA)
        self.assertEqual(load_character_state(self.conf_uid).user_timezone, JAKARTA)


class StaleClockRegressionTest(unittest.TestCase):
    def test_default_anchor_uses_real_today(self):
        anchor = format_temporal_anchor()
        today = utcnow()
        self.assertIn(f"{today:%B} {today.day}, {today.year}", anchor)
        self.assertIn("UTC", anchor)

    def test_default_seed_stamp_is_current(self):
        stamp = default_seed_goals()[0]["created_at"]
        skew = abs((utcnow() - datetime.fromisoformat(stamp)).total_seconds())
        self.assertLess(skew, 120)


class SummaryAgeTagTest(unittest.TestCase):
    def test_tag_prefixes_summary(self):
        message = build_summary_message("ringkasan", age_tag="[Yesterday | Sep 30]")
        self.assertTrue(message["content"].endswith("[Yesterday | Sep 30] ringkasan"))

    def test_no_tag_keeps_legacy_format(self):
        message = build_summary_message("ringkasan")
        self.assertTrue(message["content"].endswith("ringkasan"))
        self.assertNotIn("[", message["content"])


if __name__ == "__main__":
    unittest.main()
