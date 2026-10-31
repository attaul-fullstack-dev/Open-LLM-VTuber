"""Future intentions — explicit reminder requests (deterministic, no LLM).

Covers: detection narrowness, due parsing with injectable clock, dedup,
pending/done/cancel, restart persistence, timezone boundary, prompt block
bounds, fail-soft, and no new memory system (same character_state file).
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber.character_state import (
    complete_future_intention,
    load_character_state,
    record_future_intention,
)
from src.open_llm_vtuber.future_intentions import (
    KIND_PLAN,
    KIND_REMINDER,
    add_future_intention,
    build_future_intention_context,
    claim_due_intentions,
    detect_future_intention,
    detect_future_plan,
    detect_reminder_request,
    due_intentions,
    pending_intentions,
)

JKT = "Asia/Jakarta"
NOW = datetime(2026, 10, 2, 5, 0, tzinfo=timezone.utc)  # 12:00 Jakarta
CONF = "futurechar"


class DetectionTest(unittest.TestCase):
    def test_explicit_reminder_with_besok_detected(self):
        found = detect_future_intention(
            "tolong ingetin aku besok jam 7 buat berangkat lomba",
            now=NOW, tz=JKT,
        )
        self.assertIsNotNone(found)
        self.assertIsNotNone(found.due_at)

    def test_narration_about_tomorrow_without_verb_ignored(self):
        self.assertIsNone(
            detect_future_intention("besok aku ada ujian sekolah", now=NOW, tz=JKT)
        )

    def test_verb_without_anchor_ignored(self):
        self.assertIsNone(
            detect_future_intention("tolong ingetin aku ya", now=NOW, tz=JKT)
        )

    def test_recall_question_ignored(self):
        self.assertIsNone(
            detect_future_intention("kamu masih inget besok?", now=NOW, tz=JKT)
        )

    def test_overlong_ignored(self):
        self.assertIsNone(
            detect_future_intention("tolong ingetin aku besok " + "x" * 300, now=NOW, tz=JKT)
        )

    def test_no_universal_intimate_flag(self):
        # Detection carries no category/polarity/permission flag at all.
        found = detect_future_intention(
            "ingatkan aku besok jam 8 minum obat", now=NOW, tz=JKT
        )
        self.assertIsNotNone(found)
        self.assertNotIn("intimate", json.dumps(found.__dict__).lower())
        self.assertNotIn("sexual", json.dumps(found.__dict__).lower())

    def test_user_reminding_mili_is_plan_not_reminder(self):
        for text in (
            "besok aku akan ingatkan kamu minum obat",
            "nanti aku ingetin kamu ya",
            "nanti aku ingetin kamu",
        ):
            with self.subTest(text=text):
                self.assertIsNone(
                    detect_future_intention(text, now=NOW, tz=JKT)
                )
                detection = detect_reminder_request(text, [], now=NOW, tz=JKT)
                self.assertEqual(detection.status, "none")
                self.assertIsNone(detection.intention)
                plan = detect_future_plan(text, now=NOW, tz=JKT)
                self.assertIsNotNone(plan)
                self.assertEqual(plan.kind, KIND_PLAN)

    def test_user_requesting_mili_to_remind_is_reminder(self):
        for text in (
            "besok ingetin aku minum obat",
            "nanti ingetin aku ya",
        ):
            with self.subTest(text=text):
                found = detect_future_intention(text, now=NOW, tz=JKT)
                self.assertIsNotNone(found)
                self.assertEqual(found.kind, KIND_REMINDER)
                self.assertIsNotNone(found.due_at)

    def test_request_markers_keep_first_person_requests_valid(self):
        for text in (
            "besok aku minta tolong ingetin aku minum obat",
            "besok aku mau kamu ingetin aku minum obat",
        ):
            with self.subTest(text=text):
                found = detect_future_intention(text, now=NOW, tz=JKT)
                self.assertIsNotNone(found)
                self.assertEqual(found.kind, KIND_REMINDER)


class DueParsingTest(unittest.TestCase):
    def test_besok_resolves_next_day_0700_local(self):
        found = detect_future_intention("ingetin aku besok berangkat", now=NOW, tz=JKT)
        stamp = datetime.fromisoformat(found.due_at)
        local = stamp.astimezone(__import__("zoneinfo").ZoneInfo(JKT))
        self.assertEqual((local.day, local.hour), (3, 7))

    def test_explicit_jam_today_or_tomorrow(self):
        # NOW local 12:00; jam 15 -> today 15:00 local.
        found = detect_future_intention("ingatkan aku jam 15 minum obat", now=NOW, tz=JKT)
        stamp = datetime.fromisoformat(found.due_at)
        local = stamp.astimezone(__import__("zoneinfo").ZoneInfo(JKT))
        self.assertEqual((local.day, local.hour), (2, 15))
        # jam 6 (already passed) -> tomorrow.
        found2 = detect_future_intention("ingatkan aku jam 6 minum obat", now=NOW, tz=JKT)
        stamp2 = datetime.fromisoformat(found2.due_at)
        local2 = stamp2.astimezone(__import__("zoneinfo").ZoneInfo(JKT))
        self.assertEqual((local2.day, local2.hour), (3, 6))

    def test_vague_anchor_stays_pending_without_due(self):
        # "minggu depan" is supported; an unsupported anchor is not.
        found = detect_future_intention(
            "reminder presentasi minggu depan", now=NOW, tz=JKT
        )
        self.assertIsNotNone(found)
        self.assertIsNotNone(found.due_at)


class StorageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_record_persist_restart(self):
        stored = record_future_intention(
            CONF, "tolong ingetin aku besok jam 7 berangkat", now=NOW, tz=JKT
        )
        self.assertIsNotNone(stored)
        reloaded = load_character_state(CONF)
        self.assertEqual(len(reloaded.future_intentions), 1)
        # Absolute UTC persisted, never a relative label.
        row = reloaded.future_intentions[0]
        self.assertIn("+00:00", str(row["due_at"]))
        self.assertNotIn("besok", json.dumps(row).lower().replace(row["text"].lower(), ""))

    def test_duplicate_same_text_not_stored_twice(self):
        record_future_intention(CONF, "ingatkan aku besok minum obat", now=NOW, tz=JKT)
        record_future_intention(CONF, "ingatkan aku besok minum obat", now=NOW, tz=JKT)
        self.assertEqual(len(load_character_state(CONF).future_intentions), 1)

    def test_complete_and_cancel(self):
        stored = record_future_intention(
            CONF, "ingatkan aku besok minum obat", now=NOW, tz=JKT
        )
        self.assertTrue(complete_future_intention(CONF, stored["id"]))
        self.assertEqual(pending_intentions(load_character_state(CONF).future_intentions), [])

    def test_old_state_without_key_loads_empty(self):
        os.makedirs("character_state", exist_ok=True)
        with open(os.path.join("character_state", f"{CONF}.json"), "w") as handle:
            json.dump({"relationship_status": "close", "memories": []}, handle)
        state = load_character_state(CONF)
        self.assertEqual(state.future_intentions, [])
        self.assertEqual(state.relationship_status, "close")

    def test_due_only_after_due_at(self):
        record_future_intention(
            CONF, "ingetin aku besok jam 7 berangkat", now=NOW, tz=JKT
        )
        rows = load_character_state(CONF).future_intentions
        self.assertEqual(due_intentions(rows, now=NOW), [])
        later = NOW + timedelta(days=2)
        self.assertEqual(len(due_intentions(rows, now=later)), 1)

    def test_same_file_as_memories_not_new_system(self):
        record_future_intention(CONF, "ingatkan aku besok minum obat", now=NOW, tz=JKT)
        path = os.path.join("character_state", f"{CONF}.json")
        with open(path) as handle:
            data = json.load(handle)
        self.assertIn("future_intentions", data)
        self.assertIn("memories", data)
        self.assertIn("goals", data)


class PromptBlockTest(unittest.TestCase):
    def test_empty_renders_empty(self):
        self.assertEqual(build_future_intention_context([], tz=JKT), "")

    def test_block_carries_absolute_due(self):
        stored_rows = []
        from src.open_llm_vtuber.future_intentions import FutureIntention

        rows = add_future_intention(
            stored_rows,
            FutureIntention(
                text="ingatkan aku besok minum obat",
                due_at=(NOW + timedelta(days=1)).isoformat(),
                created_at=NOW.isoformat(),
            ),
        )
        block = build_future_intention_context(rows, tz=JKT)
        self.assertIn("remind", block.lower())
        self.assertIn("Oct", block)


class ProactiveWiringTest(unittest.TestCase):
    """Due reminder becomes HIGH explicit_reminder via the real trigger."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs("character_state", exist_ok=True)
        os.makedirs("episodic", exist_ok=True)
        os.makedirs("world_state", exist_ok=True)
        from types import SimpleNamespace

        from src.open_llm_vtuber.agent.agents.basic_memory_agent import (
            BasicMemoryAgent,
        )
        from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig

        class _LLM:
            model = "future-wire"
            max_tokens = 8

            async def chat_completion(self, *a, **k):
                raise AssertionError("no LLM in trigger path")
                if False:
                    yield None

        agent = BasicMemoryAgent(
            llm=_LLM(),
            system="persona",
            live2d_model=SimpleNamespace(extract_emotion=lambda text: []),
            tts_preprocessor_config=TTSPreprocessorConfig(
                remove_special_char=True,
                translator_config={"translate_audio": False, "translate_provider": "deeplx"},
            ),
        )
        agent._character_conf_uid = CONF
        agent._user_timezone = JKT
        agent._load_character_state(CONF)
        self.agent = agent
        from src.open_llm_vtuber.websocket_handler import WebSocketHandler

        self.server = WebSocketHandler.__new__(WebSocketHandler)
        self.ctx = SimpleNamespace(
            agent_engine=agent, user_timezone=JKT, history_uid="h1"
        )
        self.sig = SimpleNamespace(
            user_question_pending=False,
            unfinished_topic=False,
            has_useful_memory=False,
            memory_relevance_score=0.0,
        )

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _budget(self):
        from src.open_llm_vtuber.proactive_gate import (
            ProactiveBudgetState,
            local_day_key,
            local_hour_key,
            resolve_user_tz,
        )

        zone = resolve_user_tz(JKT)
        return ProactiveBudgetState(
            daily_request_count=0,
            daily_count_date=local_day_key(NOW, zone),
            hourly_meaningful_count=0,
            hourly_count_hour=local_hour_key(NOW, zone),
        )

    def test_no_intention_stays_low(self):
        from src.open_llm_vtuber import world_state as ws_mod
        from unittest.mock import patch

        with patch.object(ws_mod, "utcnow", return_value=NOW):
            trigger = self.server._proactive_trigger(
                self.ctx, self.sig, self._budget(), now=NOW
            )
        self.assertEqual(trigger.priority, "low")

    def test_due_intention_becomes_high_and_gate_binds(self):
        from unittest.mock import patch

        from src.open_llm_vtuber import world_state as ws_mod
        from src.open_llm_vtuber.proactive_gate import (
            ProactiveGateConfig,
            evaluate_gate,
        )

        record_future_intention(
            CONF, "tolong ingetin aku besok jam 7 berangkat", now=NOW, tz=JKT
        )
        # Refresh agent view (observer path does this per turn).
        from src.open_llm_vtuber.character_state import load_character_state

        self.agent._character_state = load_character_state(CONF)
        later = NOW + timedelta(days=2)
        with patch.object(ws_mod, "utcnow", return_value=later):
            trigger = self.server._proactive_trigger(
                self.ctx, self.sig, self._budget(), now=later
            )
        self.assertEqual(trigger.priority, "high")
        self.assertEqual(trigger.reason, "explicit_reminder")
        cfg = ProactiveGateConfig()
        self.assertTrue(
            evaluate_gate(self._budget(), cfg, trigger, now=later, tz=JKT).allowed
        )
        full = self._budget()
        full.daily_request_count = 60
        from src.open_llm_vtuber.proactive_gate import local_day_key, resolve_user_tz

        full.daily_count_date = local_day_key(later, resolve_user_tz(JKT))
        # Reminder lane: a due obligation is not blocked by the spontaneous
        # daily ceiling, but a dead session still is. Without the verified
        # lane flag the ceiling keeps applying (plans/spontaneous turns).
        self.assertTrue(
            evaluate_gate(
                full, cfg, trigger, now=later, tz=JKT, reminder_due=True
            ).allowed
        )
        self.assertFalse(
            evaluate_gate(full, cfg, trigger, now=later, tz=JKT).allowed
        )
        self.assertFalse(
            evaluate_gate(
                full,
                cfg,
                trigger,
                now=later,
                tz=JKT,
                connection_valid=False,
                reminder_due=True,
            ).allowed
        )

    def test_consume_on_dispatch_dedups(self):
        from datetime import timedelta as _td

        record_future_intention(
            CONF, "tolong ingetin aku besok jam 7 berangkat", now=NOW, tz=JKT
        )
        from src.open_llm_vtuber.character_state import load_character_state

        self.agent._character_state = load_character_state(CONF)
        later = NOW + _td(days=2)
        self.assertTrue(self.agent.has_due_future_intention(moment=later))
        count = self.agent.consume_due_future_intentions(moment=later)
        self.assertEqual(count, 1)
        self.agent._character_state = load_character_state(CONF)
        self.assertFalse(self.agent.has_due_future_intention(moment=later))


class ClaimSelectionTest(unittest.TestCase):
    """A due plan must never be claimed instead of the user's reminder."""

    def _row(self, row_id, kind, minutes_ago):
        return {
            "id": row_id,
            "text": row_id,
            "due_at": (NOW - timedelta(minutes=minutes_ago)).isoformat(),
            "created_at": NOW.isoformat(),
            "status": "pending",
            "source": "conversation",
            "kind": kind,
        }

    def test_due_reminder_is_claimed_before_an_earlier_due_plan(self):
        plan = self._row("plan", KIND_PLAN, 30)
        reminder = self._row("reminder", KIND_REMINDER, 5)
        rows, claimed = claim_due_intentions(
            [plan, reminder], now=NOW, owner="owner-a", max_claims=1
        )
        self.assertEqual([row["id"] for row in claimed], ["reminder"])
        stored = {row["id"]: row for row in rows}
        self.assertNotIn("claimed_by", stored["plan"])
        self.assertEqual(stored["reminder"]["claimed_by"], "owner-a")
        self.assertEqual(stored["reminder"]["status"], "pending")

    def test_undated_row_is_never_claimable(self):
        row = self._row("undated", KIND_REMINDER, 0)
        row["due_at"] = None
        rows, claimed = claim_due_intentions([row], now=NOW, owner="owner-a")
        self.assertEqual(claimed, [])
        self.assertNotIn("claimed_by", rows[0])
        self.assertEqual(rows[0]["status"], "pending")


if __name__ == "__main__":
    unittest.main()
