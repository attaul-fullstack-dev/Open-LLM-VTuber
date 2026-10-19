"""Memory Manager consistency: retrieval sources vs what the UI lists.

Read-only audit driven entirely through GENUINE production functions in an
isolated temporary working directory (no production memory is used). The
unified Memory Manager dialog displays five persistent stores, and Mili's
prompt draws from several stores. These tests pin, per store:

  write via the official path -> authoritative file -> Mili's retrieval block
  -> overview/UI -> a fresh connection -> unchanged;

plus the read-failure isolation contract and the two proven boundaries found
during the audit (episodic window; cross-connection snapshot staleness).

Nothing here changes production behaviour; the boundaries are pinned so that
any future change to them is a deliberate, reviewed edit.
"""

import os
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.character_state import (
    build_character_memory_context,
    load_character_state,
    record_future_intention,
)
from src.open_llm_vtuber.chat_history_manager import create_new_history
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber import episodic_memory as em
from src.open_llm_vtuber.interaction_preferences import (
    active_preferences,
    build_interaction_preference_context,
)
from src.open_llm_vtuber.future_intentions import (
    build_future_intention_context,
    pending_intentions,
)

CONF = "mm-consistency"
JKT = "Asia/Jakarta"


class FakeLLM:
    model = "mm-consistency"
    max_tokens = 64

    async def chat_completion(self, *args, **kwargs):
        if False:
            yield None


def make_agent():
    agent = BasicMemoryAgent(
        llm=FakeLLM(),
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
    history_uid = create_new_history(CONF)
    agent.set_memory_from_history(CONF, history_uid)
    return agent


class MemoryManagerConsistencyTest(unittest.TestCase):
    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self.agent = make_agent()

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    # --- per-store happy-path consistency -------------------------------

    def test_character_facts_recalled_and_listed_in_fresh_connection(self):
        self.agent.add_character_memory("CONS sumur biru", explicit=True)
        self.agent.add_character_memory("CONS menyiram anggrek", explicit=False)

        disk = [m["text"] for m in load_character_state(CONF).memories]
        block = build_character_memory_context(load_character_state(CONF), tz=JKT)
        overview = self.agent.list_memory_overview()
        listed = [f["text"] for f in overview["facts"]]

        self.assertEqual(sorted(disk), sorted(listed))
        for text in disk:
            self.assertIn(text, block)

        fresh = object.__new__(BasicMemoryAgent)
        fresh._character_conf_uid = CONF
        fresh._character_state = load_character_state(CONF)
        self.assertEqual(
            [f["text"] for f in fresh.list_memory_overview()["facts"]],
            listed,
        )

    def test_active_preference_recalled_and_listed_excludes_superseded(self):
        self.agent.observe_character_events(
            "Mulai sekarang CONS jangan terlalu galak.", "Baik."
        )
        state = load_character_state(CONF)
        active = active_preferences(state.interaction_preferences)
        self.assertTrue(active)

        block = build_interaction_preference_context(
            state.interaction_preferences, tz=JKT
        )
        overview = self.agent.list_memory_overview()
        listed_ids = [p["id"] for p in overview["preferences"]]
        self.assertEqual(listed_ids, [p["id"] for p in active])
        self.assertIn("jangan terlalu galak", block)

        # Retract (through the UI path) -> gone from both block and overview.
        self.assertTrue(self.agent.remove_interaction_preference(active[0]["id"]))
        state2 = load_character_state(CONF)
        self.assertEqual(active_preferences(state2.interaction_preferences), [])
        self.assertEqual(
            build_interaction_preference_context(
                state2.interaction_preferences, tz=JKT
            ),
            "",
        )
        self.assertEqual(self.agent.list_memory_overview()["preferences"], [])

    def test_pending_intention_recalled_and_listed_excludes_cancelled(self):
        self.agent._observe_future_intention(
            "tolong ingetin aku besok jam 7 buat berangkat lomba"
        )
        state = load_character_state(CONF)
        pending = pending_intentions(state.future_intentions)
        self.assertTrue(pending)

        block = build_future_intention_context(state.future_intentions, tz=JKT)
        overview = self.agent.list_memory_overview()
        self.assertEqual(
            [i["id"] for i in overview["intentions"]],
            [p["id"] for p in pending],
        )
        self.assertIn("berangkat lomba", block)

        self.assertTrue(self.agent.cancel_future_intention(pending[0]["id"]))
        state2 = load_character_state(CONF)
        self.assertEqual(pending_intentions(state2.future_intentions), [])
        self.assertEqual(
            build_future_intention_context(state2.future_intentions, tz=JKT), ""
        )
        self.assertEqual(self.agent.list_memory_overview()["intentions"], [])

    def test_episodic_event_recalled_and_listed(self):
        stored = em.append_episodic_event(
            CONF,
            {
                "event_text": "CONS festival layang kemarin sore",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "session_uid": "sess-cons",
                "source": "conversation",
            },
        )
        self.assertIsNotNone(stored)
        self.agent._episodic_query = "CONS festival layang"
        block = self.agent._episodic_context_for_prompt()
        self.assertIn("CONS festival layang", block)

        overview = self.agent.list_memory_overview()
        self.assertEqual(overview["episodic"]["total"], 1)
        self.assertIn(
            "CONS festival layang",
            overview["episodic"]["events"][0]["event_text"],
        )
        self.assertEqual(overview["episodic"]["events"][0]["id"], stored["id"])

    def test_world_state_present_in_prompt_and_overview(self):
        prompt = self.agent._relationship_system_prompt("PERSONA")
        self.assertIn("[Mili World State]", prompt)
        overview = self.agent.list_memory_overview()
        self.assertIsNotNone(overview["world"])

    # --- read-failure reporting + isolation -----------------------------

    def test_episodic_read_failure_is_reported_and_isolated(self):
        self.agent.add_character_memory("CONS fakta selamat", explicit=True)
        with unittest.mock.patch.object(
            em, "load_episodic_events", side_effect=RuntimeError("boom")
        ):
            overview = self.agent.list_memory_overview()
        # failed category: empty data AND an explicit error status
        self.assertEqual(overview["episodic"], {"total": 0, "events": []})
        self.assertEqual(overview["status"]["episodic"], "error")
        # other categories must survive a single-store failure
        self.assertEqual([f["text"] for f in overview["facts"]], ["CONS fakta selamat"])
        self.assertEqual(overview["status"]["facts"], "ok")
        self.assertIsNotNone(overview["world"])
        self.assertEqual(overview["status"]["world"], "ok")

    def test_character_state_read_failure_is_reported_and_isolated(self):
        import src.open_llm_vtuber.episodic_memory as epi
        from src.open_llm_vtuber import character_state as cs

        epi.append_episodic_event(
            CONF,
            {
                "event_text": "CONS event bertahan",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "session_uid": "s",
                "source": "conversation",
            },
        )
        with unittest.mock.patch.object(
            cs, "load_character_state", side_effect=RuntimeError("boom")
        ):
            overview = self.agent.list_memory_overview()
        # the failed store backs facts + preferences + intentions -> error
        for category in ("facts", "preferences", "intentions"):
            self.assertEqual(overview["status"][category], "error")
        self.assertEqual(overview["preferences"], [])
        self.assertEqual(overview["intentions"], [])
        # independent stores survive
        self.assertEqual(overview["episodic"]["total"], 1)
        self.assertEqual(overview["status"]["episodic"], "ok")
        self.assertIsNotNone(overview["world"])
        self.assertEqual(overview["status"]["world"], "ok")

    def test_corrupt_store_file_is_reported_as_error_not_empty(self):
        # A present-but-unparseable store must read as "error", never as a
        # legitimate empty category ("you have no data here").
        os.makedirs("episodic", exist_ok=True)
        with open(f"episodic/{CONF}.json", "w", encoding="utf-8") as handle:
            handle.write("{not valid json")
        overview = self.agent.list_memory_overview()
        self.assertEqual(overview["status"]["episodic"], "error")
        self.assertEqual(overview["episodic"], {"total": 0, "events": []})

    def test_status_recovers_after_the_store_is_readable_again(self):
        os.makedirs("episodic", exist_ok=True)
        with open(f"episodic/{CONF}.json", "w", encoding="utf-8") as handle:
            handle.write("{broken")
        self.assertEqual(
            self.agent.list_memory_overview()["status"]["episodic"], "error"
        )
        # repair the store (synthetic fixture, isolated dir)
        with open(f"episodic/{CONF}.json", "w", encoding="utf-8") as handle:
            handle.write("[]")
        recovered = self.agent.list_memory_overview()
        self.assertEqual(recovered["status"]["episodic"], "empty")

    def test_status_distinguishes_empty_from_ok(self):
        empty = self.agent.list_memory_overview()
        self.assertEqual(empty["status"]["episodic"], "empty")
        self.assertEqual(empty["status"]["facts"], "empty")
        self.agent.add_character_memory("CONS fakta ada", explicit=True)
        em.append_episodic_event(
            CONF,
            {
                "event_text": "CONS event ada",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "session_uid": "s",
                "source": "conversation",
            },
        )
        filled = self.agent.list_memory_overview()
        self.assertEqual(filled["status"]["facts"], "ok")
        self.assertEqual(filled["status"]["episodic"], "ok")


class KnownBoundaryTest(unittest.TestCase):
    """Proven boundaries and the cross-connection consistency fix."""

    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self.agent = make_agent()

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_episodic_retrieved_outside_latest_window_is_not_in_overview(self):
        # An event old by storage order but matched by the query IS recalled,
        # yet the overview only carries the latest 50. This is a real
        # retrieval-vs-UI coverage gap (disclosed via the "latest N of M"
        # label); pinned here. A "load more"/search control is the minimal
        # future remedy.
        old_text = "CONS zqx oldmarker kucing antah berantah"
        em.append_episodic_event(
            CONF,
            {
                "event_text": old_text,
                "occurred_at": "2020-01-01T00:00:00+00:00",
                "session_uid": "s-old",
                "source": "conversation",
            },
        )
        for i in range(60):
            em.append_episodic_event(
                CONF,
                {
                    "event_text": f"CONS zzz{i} unik{i} kata{i} pagi{i} malam{i}",
                    "occurred_at": (
                        datetime.now(timezone.utc) - timedelta(days=1)
                    ).isoformat(),
                    "session_uid": f"s-{i}",
                    "source": "conversation",
                },
            )
        self.agent._episodic_query = "zqx oldmarker kucing antah"
        block = self.agent._episodic_context_for_prompt()
        overview = self.agent.list_memory_overview()
        visible = [e["event_text"] for e in overview["episodic"]["events"]]

        self.assertIn("zqx oldmarker", block, "old event must be recalled")
        self.assertNotIn(old_text, visible, "…but it is outside the UI window")
        self.assertEqual(overview["episodic"]["total"], 61)
        self.assertEqual(len(visible), 50)

    def test_cross_connection_add_is_reflected_without_reconnect(self):
        # Two ACTIVE connections: B adds a fact; A's next retrieval turn must
        # pick it up without waiting for A to resume/switch history.
        other = make_agent()
        other.add_character_memory("CONS fakta dari koneksi lain", explicit=True)

        block = self.agent._relationship_system_prompt("PERSONA")
        self.assertIn("CONS fakta dari koneksi lain", block)
        listed = [f["text"] for f in self.agent.list_memory_overview()["facts"]]
        self.assertIn("CONS fakta dari koneksi lain", listed)

    def test_cross_connection_delete_is_reflected_without_reconnect(self):
        self.agent.add_character_memory("CONS fakta akan dihapus", explicit=True)
        other = make_agent()
        other.remove_character_memory("CONS fakta akan dihapus", exact=True)

        block = self.agent._relationship_system_prompt("PERSONA")
        self.assertNotIn("CONS fakta akan dihapus", block)
        listed = [f["text"] for f in self.agent.list_memory_overview()["facts"]]
        self.assertNotIn("CONS fakta akan dihapus", listed)

    def test_unchanged_store_is_not_reloaded_every_turn(self):
        # The stamp gate must avoid re-reading disk when nothing changed:
        # after the first turn the state object stays identical.
        self.agent._relationship_system_prompt("PERSONA")
        first = self.agent._character_state
        stamp = self.agent._character_state_stamp
        self.agent._relationship_system_prompt("PERSONA")
        self.assertIs(self.agent._character_state, first)
        self.assertEqual(self.agent._character_state_stamp, stamp)

    def test_cross_connection_due_reminder_triggers_without_a_turn(self):
        # Proactive reminder trigger: B stores an ALREADY-DUE reminder; A owns
        # the scheduler and must detect it BEFORE taking any turn. Otherwise a
        # reminder written by a sibling active connection never fires until an
        # unrelated turn happens to refresh A's snapshot.
        make_agent()  # second active connection
        past = datetime.now(timezone.utc) - timedelta(days=2)
        stored = record_future_intention(
            CONF,
            "Ingatkan aku besok jam 7 pagi untuk CONS konsistensi.",
            now=past,
            tz=JKT,
        )
        self.assertIsNotNone(stored)
        self.assertTrue(stored.get("due_at"))
        self.assertTrue(
            self.agent.has_due_future_intention(),
            "a sibling connection's due reminder must be visible to the trigger",
        )


if __name__ == "__main__":
    unittest.main()
