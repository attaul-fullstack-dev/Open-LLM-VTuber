"""Unified Memory Manager: overview and per-store lifecycle (isolated).

Covers the acceptance contract for the cross-store Memory Manager:

- every persistent store the retrieval layer reads appears in ONE overview
  (facts, active preferences, pending intentions, bounded episodic events
  with total, read-only world snapshot), each from its authoritative file;
- each category deletes through its own store semantics (exact-text fact,
  id-based episodic delete, preference retract via the pre-existing terminal
  status, intention cancel), never one blanket operation;
- deletions invalidate the retrieval caches that could otherwise resurrect
  the row (proven episodic-cache resurrection vector);
- unknown/empty identifiers change nothing and report failure.

All in a temporary working directory; no production data is used.
"""

import os
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.character_state import (
    cancel_stored_future_intention,
    load_character_state,
    record_future_intention,
    record_interaction_preference,
    retract_interaction_preference,
)
from src.open_llm_vtuber.chat_history_manager import create_new_history
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber import episodic_memory as em

CONF = "memory-overview"
JKT = "Asia/Jakarta"


def seed_episodic(text, session="sess-ov"):
    return em.append_episodic_event(
        CONF,
        {
            "event_text": text,
            "occurred_at": datetime.now(timezone.utc).isoformat(),
            "session_uid": session,
            "source": "conversation",
        },
    )


class MemoryOverviewTest(unittest.TestCase):
    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self.agent = BasicMemoryAgent(
            llm=SimpleNamespace(model="ov", max_tokens=1, chat_completion=None),
            system="persona",
            live2d_model=SimpleNamespace(extract_emotion=lambda t: []),
            tts_preprocessor_config=TTSPreprocessorConfig(
                remove_special_char=True,
                translator_config={
                    "translate_audio": False,
                    "translate_provider": "deeplx",
                },
            ),
        )
        history_uid = create_new_history(CONF)
        self.agent.set_memory_from_history(CONF, history_uid)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _seed_all(self):
        self.assertTrue(self.agent.add_character_memory("OVERVIEW kopi susu"))
        pref = record_interaction_preference(
            CONF, "Mulai sekarang OVERVIEW jangan terlalu galak."
        )
        self.assertIsNotNone(pref)
        intention = record_future_intention(
            CONF, "Ingatkan OVERVIEW besok jam 7 pagi.", tz=JKT
        )
        self.assertIsNotNone(intention)
        seed_episodic("OVERVIEW festival layang")
        return pref["id"], intention["id"]

    def test_overview_covers_every_store(self):
        pref_id, intention_id = self._seed_all()
        overview = self.agent.list_memory_overview()
        self.assertEqual(
            [row["text"] for row in overview["facts"]], ["OVERVIEW kopi susu"]
        )
        self.assertEqual([row["id"] for row in overview["preferences"]], [pref_id])
        self.assertEqual([row["id"] for row in overview["intentions"]], [intention_id])
        self.assertEqual(overview["episodic"]["total"], 1)
        self.assertEqual(len(overview["episodic"]["events"]), 1)
        self.assertIn(
            "OVERVIEW festival layang",
            overview["episodic"]["events"][0]["event_text"],
        )
        self.assertIsNotNone(overview["world"])
        self.assertIn("activity", overview["world"])

    def test_overview_bounds_episodic_and_reports_total(self):
        seed_episodic("OVERVIEW festival layang sore")
        seed_episodic("OVERVIEW rapat kerja pagi")
        seed_episodic("OVERVIEW makan malam keluarga")
        seed_episodic("OVERVIEW hujan deras malam")
        seed_episodic("OVERVIEW kucing tetangga lahir")
        overview = self.agent.list_memory_overview(episodic_limit=2)
        self.assertEqual(overview["episodic"]["total"], 5)
        self.assertEqual(len(overview["episodic"]["events"]), 2)
        # most recent first
        self.assertIn(
            "kucing tetangga", overview["episodic"]["events"][0]["event_text"]
        )

    def test_overview_is_stable_across_a_fresh_session(self):
        self._seed_all()
        first = self.agent.list_memory_overview()
        fresh = object.__new__(BasicMemoryAgent)
        fresh._character_conf_uid = CONF
        fresh._character_state = load_character_state(CONF)
        second = fresh.list_memory_overview()
        self.assertEqual(first["facts"], second["facts"])
        self.assertEqual(first["preferences"], second["preferences"])
        self.assertEqual(first["intentions"], second["intentions"])
        self.assertEqual(first["episodic"], second["episodic"])

    def test_overview_episodic_ids_stable_and_delete_isolated(self):
        # Legacy store with rows predating ids: the overview must expose ONE
        # durable id per row (not a fresh one per load), and deleting the
        # chosen row must leave the others untouched.
        import json as _json

        rows = [
            {
                "event_text": "OVERVIEW legacy satu",
                "occurred_at": None,
                "session_uid": "sess-legacy",
                "source": "conversation",
                "created_at": "2026-01-01T00:00:00+00:00",
                "tz": JKT,
            },
            {
                "event_text": "OVERVIEW legacy dua",
                "occurred_at": None,
                "session_uid": "sess-legacy",
                "source": "conversation",
                "created_at": "2026-01-02T00:00:00+00:00",
                "tz": JKT,
            },
        ]
        os.makedirs("episodic", exist_ok=True)
        with open(f"episodic/{CONF}.json", "w", encoding="utf-8") as handle:
            _json.dump(rows, handle)

        first = self.agent.list_memory_overview()
        second = self.agent.list_memory_overview()
        ids_first = sorted(row["id"] for row in first["episodic"]["events"])
        ids_second = sorted(row["id"] for row in second["episodic"]["events"])
        self.assertTrue(all(ids_first))
        self.assertEqual(ids_first, ids_second)

        target = next(
            row["id"]
            for row in first["episodic"]["events"]
            if row["event_text"] == "OVERVIEW legacy satu"
        )
        survivor = next(
            row["id"]
            for row in first["episodic"]["events"]
            if row["event_text"] == "OVERVIEW legacy dua"
        )
        self.assertTrue(self.agent.delete_episodic_event(target))
        after = self.agent.list_memory_overview()
        self.assertEqual(after["episodic"]["total"], 1)
        self.assertEqual(
            [row["event_text"] for row in after["episodic"]["events"]],
            ["OVERVIEW legacy dua"],
        )
        self.assertEqual(after["episodic"]["events"][0]["id"], survivor)

    def test_overview_without_a_character_is_empty_not_broken(self):
        self.agent._character_conf_uid = ""
        overview = self.agent.list_memory_overview()
        self.assertEqual(overview["facts"], [])
        self.assertEqual(overview["preferences"], [])
        self.assertEqual(overview["intentions"], [])
        self.assertEqual(overview["episodic"], {"total": 0, "events": []})
        self.assertIsNone(overview["world"])


class PreferenceRetractTest(unittest.TestCase):
    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_retract_hides_from_renderer_and_list(self):
        from src.open_llm_vtuber.interaction_preferences import (
            active_preferences,
            build_interaction_preference_context,
        )

        pref = record_interaction_preference(
            CONF, "Mulai sekarang RETRACT jangan terlalu galak."
        )
        self.assertIsNotNone(pref)
        state, removed = retract_interaction_preference(CONF, pref["id"])
        self.assertIsNotNone(state)
        self.assertEqual(removed, 1)
        self.assertEqual(active_preferences(state.interaction_preferences), [])
        self.assertEqual(
            build_interaction_preference_context(state.interaction_preferences),
            "",
        )

    def test_retract_unknown_or_blank_changes_nothing(self):
        import json as _json

        pref = record_interaction_preference(
            CONF, "Mulai sekarang RETRACT jangan terlalu galak."
        )
        self.assertIsNotNone(pref)
        path = os.path.join("character_state", f"{CONF}.json")
        with open(path, encoding="utf-8") as handle:
            before = handle.read()
        state, removed = retract_interaction_preference(CONF, "no-such-id")
        self.assertIsNotNone(state)
        self.assertEqual(removed, 0)
        state, removed = retract_interaction_preference(CONF, "   ")
        self.assertEqual(removed, 0)
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(_json.loads(handle.read()), _json.loads(before))

    def test_retract_twice_reports_second_as_noop(self):
        pref = record_interaction_preference(
            CONF, "Mulai sekarang RETRACT jangan terlalu galak."
        )
        self.assertIsNotNone(pref)
        _, removed = retract_interaction_preference(CONF, pref["id"])
        self.assertEqual(removed, 1)
        _, removed = retract_interaction_preference(CONF, pref["id"])
        self.assertEqual(removed, 0)


class IntentionCancelTest(unittest.TestCase):
    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_cancel_removes_from_pending(self):
        from src.open_llm_vtuber.future_intentions import pending_intentions

        row = record_future_intention(CONF, "Ingatkan CANCEL besok jam 7 pagi.", tz=JKT)
        self.assertIsNotNone(row)
        self.assertTrue(cancel_stored_future_intention(CONF, row["id"]))
        remaining = pending_intentions(load_character_state(CONF).future_intentions)
        self.assertEqual(remaining, [])
        # cancelling again, or an unknown/blank id, changes nothing
        self.assertFalse(cancel_stored_future_intention(CONF, row["id"]))
        self.assertFalse(cancel_stored_future_intention(CONF, "no-such-id"))
        self.assertFalse(cancel_stored_future_intention(CONF, "  "))


class EpisodicDeleteTest(unittest.TestCase):
    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self.agent = BasicMemoryAgent(
            llm=SimpleNamespace(model="ov", max_tokens=1, chat_completion=None),
            system="persona",
            live2d_model=SimpleNamespace(extract_emotion=lambda t: []),
            tts_preprocessor_config=TTSPreprocessorConfig(
                remove_special_char=True,
                translator_config={
                    "translate_audio": False,
                    "translate_provider": "deeplx",
                },
            ),
        )
        history_uid = create_new_history(CONF)
        self.agent.set_memory_from_history(CONF, history_uid)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_delete_removes_only_the_target_and_clears_cache(self):
        keep = seed_episodic("EPDEL kopi tumpah")
        drop = seed_episodic("EPDEL teh manis")
        self.agent._episodic_query = "EPDEL"
        self.assertEqual(len(self.agent._episodic_selection()), 2)
        self.assertTrue(self.agent.delete_episodic_event(drop["id"]))
        remaining = em.load_episodic_events(CONF)
        self.assertEqual([e["id"] for e in remaining], [keep["id"]])
        # the proven resurrection vector: the cache must be gone, so the
        # deleted event can no longer be served or prompted
        self.assertIsNone(self.agent._episodic_selection_cache)
        reselected = self.agent._episodic_selection()
        self.assertEqual([e["id"] for e in reselected], [keep["id"]])

    def test_delete_unknown_or_blank_is_a_safe_noop(self):
        seed_episodic("EPDEL kopi tumpah")
        self.assertFalse(self.agent.delete_episodic_event("no-such-id"))
        self.assertFalse(self.agent.delete_episodic_event("   "))
        self.assertFalse(self.agent.delete_episodic_event(""))
        self.assertEqual(len(em.load_episodic_events(CONF)), 1)


class MemoryOverviewHandlerTest(unittest.TestCase):
    """The four new WS endpoints, against the real agent and stores."""

    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        from types import SimpleNamespace as _NS

        from src.open_llm_vtuber.websocket_handler import WebSocketHandler

        self.agent = BasicMemoryAgent(
            llm=SimpleNamespace(model="ov", max_tokens=1, chat_completion=None),
            system="persona",
            live2d_model=SimpleNamespace(extract_emotion=lambda t: []),
            tts_preprocessor_config=TTSPreprocessorConfig(
                remove_special_char=True,
                translator_config={
                    "translate_audio": False,
                    "translate_provider": "deeplx",
                },
            ),
        )
        history_uid = create_new_history(CONF)
        self.agent.set_memory_from_history(CONF, history_uid)
        self.handler = WebSocketHandler(default_context_cache=None)
        self.handler.client_contexts["client"] = _NS(
            history_uid=history_uid,
            agent_engine=self.agent,
            character_config=_NS(conf_uid=CONF),
            user_timezone=JKT,
        )
        self.history_uid = history_uid

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _socket(self):
        import json as _json

        class _FakeWS:
            def __init__(self):
                self.messages = []

            async def send_text(self, payload):
                self.messages.append(_json.loads(payload))

        return _FakeWS()

    def _run(self, coro):
        import asyncio as _asyncio

        return _asyncio.run(coro)

    def test_overview_round_trip(self):
        import asyncio as _asyncio

        self.assertTrue(self.agent.add_character_memory("WSOV kopi susu"))
        ws = self._socket()
        _asyncio.run(
            self.handler._handle_fetch_memory_overview(
                ws, "client", {"episodic_limit": 10}
            )
        )
        reply = ws.messages[-1]
        self.assertEqual(reply["type"], "memory-overview")
        overview = reply["overview"]
        self.assertEqual([row["text"] for row in overview["facts"]], ["WSOV kopi susu"])
        self.assertIn("preferences", overview)
        self.assertIn("intentions", overview)
        self.assertIn("total", overview["episodic"])
        self.assertIn("world", overview)

    def test_episodic_delete_round_trip(self):
        import asyncio as _asyncio

        drop = seed_episodic("WSOV teh manis")
        keep = seed_episodic("WSOV kopi tumpah")
        ws = self._socket()
        _asyncio.run(
            self.handler._handle_delete_episodic_event(
                ws, "client", {"event_id": drop["id"]}
            )
        )
        reply = ws.messages[-1]
        self.assertEqual(reply["type"], "episodic-event-deleted")
        self.assertTrue(reply["success"])
        self.assertEqual([e["id"] for e in em.load_episodic_events(CONF)], [keep["id"]])
        ws2 = self._socket()
        _asyncio.run(
            self.handler._handle_delete_episodic_event(ws2, "client", {"event_id": ""})
        )
        self.assertFalse(ws2.messages[-1]["success"])
        self.assertEqual(len(em.load_episodic_events(CONF)), 1)

    def test_preference_retract_round_trip(self):
        import asyncio as _asyncio

        pref = record_interaction_preference(
            CONF, "Mulai sekarang WSOV jangan terlalu galak."
        )
        self.assertIsNotNone(pref)
        ws = self._socket()
        _asyncio.run(
            self.handler._handle_delete_interaction_preference(
                ws, "client", {"preference_id": pref["id"]}
            )
        )
        reply = ws.messages[-1]
        self.assertEqual(reply["type"], "interaction-preference-deleted")
        self.assertTrue(reply["success"])
        self.assertEqual(self.agent.list_memory_overview()["preferences"], [])
        ws2 = self._socket()
        _asyncio.run(
            self.handler._handle_delete_interaction_preference(
                ws2, "client", {"preference_id": pref["id"]}
            )
        )
        self.assertFalse(ws2.messages[-1]["success"])

    def test_intention_cancel_round_trip(self):
        import asyncio as _asyncio

        row = record_future_intention(CONF, "Ingatkan WSOV besok jam 7 pagi.", tz=JKT)
        self.assertIsNotNone(row)
        ws = self._socket()
        _asyncio.run(
            self.handler._handle_cancel_future_intention(
                ws, "client", {"intention_id": row["id"]}
            )
        )
        reply = ws.messages[-1]
        self.assertEqual(reply["type"], "future-intention-cancelled")
        self.assertTrue(reply["success"])
        self.assertEqual(self.agent.list_memory_overview()["intentions"], [])
        ws2 = self._socket()
        _asyncio.run(
            self.handler._handle_cancel_future_intention(
                ws2, "client", {"intention_id": "no-such-id"}
            )
        )
        self.assertFalse(ws2.messages[-1]["success"])


if __name__ == "__main__":
    unittest.main()
