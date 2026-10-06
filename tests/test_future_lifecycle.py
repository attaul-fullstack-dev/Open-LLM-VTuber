"""Phase 8 — future-intention full lifecycle (isolated, deterministic).

A. same-session recall (in-memory, no reload)
B. new-session recall (fresh agent, same disk)
C. backend restart (state file roundtrip)
D. history independence (keyed by conf, not history_uid)
E. temporal relevance (due-soonest first, undated last)
F. due vs future split
G. completed/cancelled excluded from pending + context
I. prompt token bounds
J. ADL trigger behavior (due plan -> HIGH explicit_reminder, gate binds)
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.open_llm_vtuber.character_state import (
    complete_future_intention,
    load_character_state,
    record_future_intention,
)
from src.open_llm_vtuber.future_intentions import (
    build_future_intention_context,
    due_intentions,
    pending_intentions,
)

JKT = "Asia/Jakarta"
NOW = datetime(2026, 10, 2, 5, 0, tzinfo=timezone.utc)  # Fri 12:00 WIB
CONF = "lifecyclechar"


def make_agent(conf_uid=CONF):
    from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
    from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig

    class _LLM:
        model = "lifecycle"
        max_tokens = 8

        async def chat_completion(self, *a, **k):
            raise AssertionError("no LLM in lifecycle path")
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
    agent._character_conf_uid = conf_uid
    agent._user_timezone = JKT
    agent._load_character_state(conf_uid)
    return agent


class LifecycleTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        for d in ("character_state", "episodic", "world_state"):
            os.makedirs(d, exist_ok=True)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_A_same_session_in_memory(self):
        agent = make_agent()
        agent.observe_character_events("besok aku ada ujian", "oke noted")
        rows = agent.list_future_intentions()
        self.assertEqual(len(rows), 1)
        self.assertIn("besok aku ada ujian", rows[0]["text"])

    def test_B_new_session_from_disk(self):
        agent = make_agent()
        agent.observe_character_events("besok aku ada ujian", "oke")
        fresh = make_agent()
        rows = fresh.list_future_intentions()
        self.assertEqual(len(rows), 1)
        self.assertIn("besok aku ada ujian", rows[0]["text"])

    def test_C_restart_roundtrip(self):
        record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        import json

        with open(f"character_state/{CONF}.json") as h:
            data = json.load(h)
        self.assertEqual(len(data["future_intentions"]), 1)
        reloaded = load_character_state(CONF)
        self.assertEqual(len(reloaded.future_intentions), 1)

    def test_D_history_independent(self):
        a = make_agent()
        a._history_uid = "history-A"
        a.observe_character_events("besok aku ada ujian", "oke")
        b = make_agent()
        b._history_uid = "history-B"
        self.assertEqual(len(b.list_future_intentions()), 1)

    def test_E_temporal_relevance_order(self):
        record_future_intention(CONF, "aku mau mulai project baru", now=NOW, tz=JKT)
        record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        rows = pending_intentions(load_character_state(CONF).future_intentions)
        self.assertEqual(len(rows), 2)
        # Dated first, undated last.
        self.assertIsNotNone(rows[0]["due_at"])
        self.assertIsNone(rows[1]["due_at"])

    def test_F_due_vs_future(self):
        record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        record_future_intention(CONF, "aku mau mulai project baru", now=NOW, tz=JKT)
        rows = load_character_state(CONF).future_intentions
        self.assertEqual(due_intentions(rows, now=NOW), [])
        later = NOW + timedelta(days=2)
        due = due_intentions(rows, now=later)
        self.assertEqual(len(due), 1)
        self.assertIn("ujian", due[0]["text"])

    def test_G_done_cancelled_excluded(self):
        stored = record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        record_future_intention(CONF, "lusa aku ada lomba", now=NOW, tz=JKT)
        self.assertTrue(complete_future_intention(CONF, stored["id"]))
        rows = load_character_state(CONF).future_intentions
        pending = pending_intentions(rows)
        self.assertEqual(len(pending), 1)
        block = build_future_intention_context(rows, tz=JKT)
        self.assertNotIn("ujian", block)
        self.assertIn("lomba", block)

    def test_I_prompt_bounds(self):
        for i in range(25):
            record_future_intention(CONF, f"besok aku ada acara nomor {i}", now=NOW, tz=JKT)
        rows = load_character_state(CONF).future_intentions
        block = build_future_intention_context(rows, tz=JKT)
        lines = [ln for ln in block.splitlines() if ln.startswith("- ")]
        self.assertLessEqual(len(lines), 8)
        self.assertGreater(len(lines), 0)

    def test_J_adl_trigger_and_gate(self):
        from unittest.mock import patch

        from src.open_llm_vtuber import world_state as ws_mod
        from src.open_llm_vtuber.proactive_gate import (
            ProactiveBudgetState,
            ProactiveGateConfig,
            evaluate_gate,
            local_day_key,
            local_hour_key,
            resolve_user_tz,
        )
        from src.open_llm_vtuber.websocket_handler import WebSocketHandler

        agent = make_agent()
        record_future_intention(CONF, "besok aku ada ujian", now=NOW, tz=JKT)
        from src.open_llm_vtuber.character_state import load_character_state as _load

        agent._character_state = _load(CONF)
        server = WebSocketHandler.__new__(WebSocketHandler)
        ctx = SimpleNamespace(agent_engine=agent, user_timezone=JKT, history_uid="h1")
        sig = SimpleNamespace(
            user_question_pending=False,
            unfinished_topic=False,
            has_useful_memory=False,
            memory_relevance_score=0.0,
        )
        zone = resolve_user_tz(JKT)
        later = NOW + timedelta(days=2)
        budget = ProactiveBudgetState(
            daily_request_count=0,
            daily_count_date=local_day_key(later, zone),
            hourly_meaningful_count=0,
            hourly_count_hour=local_hour_key(later, zone),
        )
        with patch.object(ws_mod, "utcnow", return_value=later):
            trigger = server._proactive_trigger(ctx, sig, budget, now=later)
        self.assertEqual(trigger.priority, "high")
        self.assertEqual(trigger.reason, "explicit_reminder")
        cfg = ProactiveGateConfig()
        self.assertTrue(evaluate_gate(budget, cfg, trigger, now=later, tz=JKT).allowed)
        # Consume-on-dispatch dedups: second trigger goes quiet.
        self.assertEqual(agent.consume_due_future_intentions(moment=later), 1)
        agent._character_state = _load(CONF)
        with patch.object(ws_mod, "utcnow", return_value=later):
            again = server._proactive_trigger(ctx, sig, budget, now=later)
        self.assertEqual(again.priority, "low")


if __name__ == "__main__":
    unittest.main()
