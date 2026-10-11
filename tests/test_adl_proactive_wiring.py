"""ADL -> Proactive V2 runtime wiring (integration, no network, no LLM).

Proves the decision layer is not just a taxonomy any more: a stored, explicitly
activated goal with fresh evidence becomes a HIGH trigger *inside the existing
Proactive V2 trigger builder*, and that record then has to clear the real gate.

Uses the real ``WebSocketServer._proactive_trigger`` / ``_goal_evidence_trigger``
against a real agent object and real on-disk character/episodic state.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from src.open_llm_vtuber.proactive_gate import (
    PRIORITY_HIGH,
    PRIORITY_LOW,
    ProactiveBudgetState,
    ProactiveGateConfig,
    evaluate_gate,
    local_day_key,
    local_hour_key,
    resolve_user_tz,
)

JKT = "Asia/Jakarta"
NOW = datetime(2026, 10, 2, 5, 0, tzinfo=timezone.utc)
ZONE = resolve_user_tz(JKT)
CONF = "wirechar"


def budget(daily: int = 0, last_proactive=None) -> ProactiveBudgetState:
    return ProactiveBudgetState(
        daily_request_count=daily,
        daily_count_date=local_day_key(NOW, ZONE),
        hourly_meaningful_count=0,
        hourly_count_hour=local_hour_key(NOW, ZONE),
        last_proactive_at=last_proactive,
    )


def write_events(events):
    os.makedirs("episodic", exist_ok=True)
    with open(
        os.path.join("episodic", f"{CONF}.json"), "w", encoding="utf-8"
    ) as handle:
        json.dump(events, handle)


def make_agent():
    from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
    from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig

    class _LLM:
        model = "adl-wire"
        max_tokens = 64

        async def chat_completion(self, messages, system=None, tools=None):
            if False:
                yield None

    agent = BasicMemoryAgent(
        llm=_LLM(),
        system="persona",
        live2d_model=SimpleNamespace(extract_emotion=lambda text: []),
        tts_preprocessor_config=TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={
                "translate_audio": False,
                "translate_provider": "deeplx",
            },
        ),
    )
    agent._character_conf_uid = CONF
    agent._user_timezone = JKT
    return agent


def server_for(agent):
    """The real handler object, built without running __init__.

    Only the trigger helpers are exercised, so the service-context container is
    never needed; constructing it would start engines and MCP clients.
    """
    from src.open_llm_vtuber.websocket_handler import WebSocketHandler

    return WebSocketHandler.__new__(WebSocketHandler)


NO_SIGNALS = SimpleNamespace(
    user_question_pending=False,
    unfinished_topic=False,
    has_useful_memory=False,
    memory_relevance_score=0.0,
)


class TriggerWiringTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs("character_state", exist_ok=True)
        os.makedirs("episodic", exist_ok=True)
        os.makedirs("world_state", exist_ok=True)
        self.agent = make_agent()
        self.agent._load_character_state(CONF)
        self.context = SimpleNamespace(
            agent_engine=self.agent,
            user_timezone=JKT,
            history_uid="h1",
        )
        self.server = server_for(self.agent)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _utcnow(self):
        from src.open_llm_vtuber import world_state as ws_mod

        return patch.object(ws_mod, "utcnow", return_value=NOW)

    def test_no_goals_no_evidence_stays_low(self):
        write_events(
            [
                {
                    "id": "e1",
                    "event_text": "gw coba masak chicken",
                    "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
                }
            ]
        )
        with self._utcnow():
            trigger = self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
        self.assertEqual(trigger.priority, PRIORITY_LOW)
        self.assertEqual(trigger.reason, "generic_idle")

    def test_seed_goal_alone_stays_low(self):
        write_events(
            [
                {
                    "id": "e1",
                    "event_text": "gw coba masak chicken",
                    "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
                }
            ]
        )
        with self._utcnow():
            trigger = self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
        # goals were seeded on load but none is active yet
        self.assertEqual(self.agent.goal_snapshot()["seed"], 3)
        self.assertEqual(trigger.priority, PRIORITY_LOW)

    def test_active_goal_with_evidence_becomes_high(self):
        write_events(
            [
                {
                    "id": "e1",
                    "event_text": "gw coba masak chicken",
                    "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
                }
            ]
        )
        self.agent.set_goal_status("try-three-dishes", "active")
        with self._utcnow():
            trigger = self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
        self.assertEqual(trigger.priority, PRIORITY_HIGH)
        self.assertEqual(trigger.reason, "goal_evidence")
        # The evidence anchor is DEFERRED, not pinned here: while the
        # evidence is still unconsumed the trigger must keep firing, exactly
        # like a pending explicit reminder. Pinning at trigger time used to
        # let a gate-suppressed turn consume the anchor without speaking,
        # which permanently silenced the goal.
        with self._utcnow():
            repeat = self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
        self.assertEqual(repeat.priority, PRIORITY_HIGH)
        self.assertEqual(repeat.reason, "goal_evidence")

    def _goal_row(self, goal_id="try-three-dishes"):
        from src.open_llm_vtuber.character_state import load_character_state

        for row in load_character_state(CONF).goals or []:
            if str(row.get("id")) == goal_id:
                return row
        return {}

    def test_trigger_detection_does_not_pin_the_anchor(self):
        write_events(
            [
                {
                    "id": "e1",
                    "event_text": "gw coba masak chicken",
                    "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
                }
            ]
        )
        self.agent.set_goal_status("try-three-dishes", "active")
        with self._utcnow():
            self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
        # detection alone must leave the authoritative file untouched
        self.assertEqual(self._goal_row().get("last_evidence_id", ""), "")

    def test_suppressed_turn_leaves_the_goal_pending(self):
        write_events(
            [
                {
                    "id": "e1",
                    "event_text": "gw coba masak chicken",
                    "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
                }
            ]
        )
        self.agent.set_goal_status("try-three-dishes", "active")
        cfg = ProactiveGateConfig()
        with self._utcnow():
            trigger = self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
        # daily ceiling: the gate suppresses the turn
        self.assertFalse(
            evaluate_gate(budget(60), cfg, trigger, now=NOW, tz=JKT).allowed
        )
        self.server._discard_pending_goal_evidence(self.context)
        self.assertEqual(self._goal_row().get("last_evidence_id", ""), "")
        # and the goal is therefore re-detected rather than lost forever
        with self._utcnow():
            again = self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
        self.assertEqual(again.reason, "goal_evidence")

    def test_dispatch_consumes_the_anchor_exactly_once(self):
        write_events(
            [
                {
                    "id": "e1",
                    "event_text": "gw coba masak chicken",
                    "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
                }
            ]
        )
        self.agent.set_goal_status("try-three-dishes", "active")
        with self._utcnow():
            self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
        self.server._consume_pending_goal_evidence(self.context)
        self.assertTrue(self._goal_row().get("last_evidence_id"))
        # consuming again is a no-op, so the same event cannot retrigger
        self.server._consume_pending_goal_evidence(self.context)
        with self._utcnow():
            after = self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
        self.assertEqual(after.priority, PRIORITY_LOW)

    def test_anchor_survives_a_concurrent_activation_by_another_session(self):
        """The deferred pin must be a read-modify-write, not a snapshot save.

        Another session activating the goal after this agent loaded state
        used to be silently reverted when the anchor was persisted.
        """
        from src.open_llm_vtuber.character_state import (
            load_character_state,
            save_character_state,
        )

        write_events(
            [
                {
                    "id": "e1",
                    "event_text": "gw coba masak chicken",
                    "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
                }
            ]
        )
        self.agent.set_goal_status("try-three-dishes", "active")
        with self._utcnow():
            self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
        # a second writer completes the goal on disk behind our back
        other = load_character_state(CONF)
        for row in other.goals or []:
            if row.get("id") == "morning-reading-week":
                row["status"] = "done"
        save_character_state(CONF, other)
        self.server._consume_pending_goal_evidence(self.context)
        statuses = {
            str(row.get("id")): str(row.get("status"))
            for row in load_character_state(CONF).goals or []
        }
        self.assertEqual(statuses.get("morning-reading-week"), "done")
        self.assertTrue(self._goal_row().get("last_evidence_id"))

    def test_goal_decision_still_has_to_clear_the_real_gate(self):
        write_events(
            [
                {
                    "id": "e1",
                    "event_text": "gw coba masak chicken",
                    "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
                }
            ]
        )
        self.agent.set_goal_status("try-three-dishes", "active")
        cfg = ProactiveGateConfig()
        with self._utcnow():
            trigger = self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
        self.assertTrue(trigger.is_meaningful)
        self.assertTrue(evaluate_gate(budget(), cfg, trigger, now=NOW, tz=JKT).allowed)
        # daily ceiling still binds the goal decision
        self.assertFalse(
            evaluate_gate(budget(60), cfg, trigger, now=NOW, tz=JKT).allowed
        )
        # minimum gap still binds it
        self.assertFalse(
            evaluate_gate(
                budget(last_proactive=(NOW - timedelta(seconds=30)).isoformat()),
                cfg,
                trigger,
                now=NOW,
                tz=JKT,
            ).allowed
        )

    def test_done_goal_no_longer_produces_evidence(self):
        write_events(
            [
                {
                    "id": "e1",
                    "event_text": "gw coba masak chicken",
                    "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
                }
            ]
        )
        self.agent.set_goal_status("try-three-dishes", "active")
        self.agent.set_goal_status("try-three-dishes", "done")
        with self._utcnow():
            trigger = self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
        self.assertEqual(trigger.priority, PRIORITY_LOW)

    def test_stale_evidence_does_not_trigger(self):
        write_events(
            [
                {
                    "id": "e1",
                    "event_text": "gw coba masak chicken",
                    "occurred_at": (NOW - timedelta(days=30)).isoformat(),
                }
            ]
        )
        self.agent.set_goal_status("try-three-dishes", "active")
        with self._utcnow():
            trigger = self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
        self.assertEqual(trigger.priority, PRIORITY_LOW)

    def test_agent_without_the_facade_is_fail_soft(self):
        bare = SimpleNamespace(
            history_uid="h1", user_timezone=JKT, agent_engine=object()
        )
        with self._utcnow():
            trigger = self.server._proactive_trigger(bare, NO_SIGNALS, budget())
        self.assertEqual(trigger.priority, PRIORITY_LOW)

    def test_broken_facade_is_fail_soft(self):
        def boom():
            raise RuntimeError("state file gone")

        broken = SimpleNamespace(
            history_uid="h1",
            user_timezone=JKT,
            agent_engine=SimpleNamespace(classify_goal_evidence=boom),
        )
        with self._utcnow():
            trigger = self.server._proactive_trigger(broken, NO_SIGNALS, budget())
        self.assertEqual(trigger.priority, PRIORITY_LOW)

    def test_missing_context_is_fail_soft(self):
        self.assertFalse(self.server._goal_evidence_trigger(None))

    def test_no_llm_call_is_made_by_the_decision_layer(self):
        """The trigger path must not touch the provider at all."""
        write_events(
            [
                {
                    "id": "e1",
                    "event_text": "gw coba masak chicken",
                    "occurred_at": (NOW - timedelta(hours=1)).isoformat(),
                }
            ]
        )
        self.agent.set_goal_status("try-three-dishes", "active")
        calls = []

        class _TripwireLLM:
            model = "tripwire"

            async def chat_completion(self, *a, **k):
                calls.append((a, k))
                if False:
                    yield None

        self.agent._llm = _TripwireLLM()
        with self._utcnow():
            self.server._proactive_trigger(self.context, NO_SIGNALS, budget())
            self.server._goal_evidence_trigger(self.context)
        self.assertEqual(calls, [], "decision layer must not call the model")


if __name__ == "__main__":
    unittest.main()
