"""Phase 14 — future intention cross-session recall acceptance (isolated).

Proves the real Session A -> Session B boundary with temp dir + temp UID:

A. Session A stores a dated intention; Session B is a NEW agent with a
   DIFFERENT history_uid that never saw Session A's text; the recall query
   mentions no topic; the intention is found via the persistent mechanism.
B. Backend restart (disk roundtrip) + another new session: still recalled
   while not expired.
C. Transcript independence: Session B's in-memory transcript contains no
   Session A content, yet the context block carries the intention text.
D. Expired handling: a past-due row is classified DUE (eligible for
   consume-on-dispatch exactly once), not as an upcoming future row.
E. Timezone: "tomorrow 07:00" resolves 7h apart between Asia/Jakarta and
   UTC and renders in the user zone.
F. No fabrication: a query about a never-stored topic yields no invented
   row; an empty store renders an empty block.
G. Multiple intentions: two distinct dated rows are both retrieved,
   due-soonest first.
H. Stated-date grounding: the rendered block carries "stated <Mon DD>"
   from created_at so "kemarin aku bilang ... apa?" grounds correctly;
   rows without created_at render as before (fail-soft).
I. Eviction protection: on overflow, undated rows are evicted before
   dated ones, so a dated intention survives immediate-desire flood.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.open_llm_vtuber.character_state import (
    load_character_state,
    record_future_intention,
)
from src.open_llm_vtuber.future_intentions import (
    build_future_intention_context,
    detect_any_future_intention,
    due_intentions,
    pending_intentions,
)

JKT = "Asia/Jakarta"
# Monday 2026-10-05 12:00 Jakarta (matches the real incident week).
NOW_A = datetime(2026, 10, 5, 5, 0, tzinfo=timezone.utc)
CONF = "p14recall"
SESSION_A_TEXT = "Besok jam 07.00 aku mau nanya lagi sesuatu."
SESSION_B_QUERY = "Mil, kemarin aku bilang hari ini/besok mau nanya apa?"


def make_agent(conf_uid=CONF, history_uid="history-x"):
    from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
    from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig

    class _LLM:
        model = "p14"
        max_tokens = 8

        async def chat_completion(self, *a, **k):
            raise AssertionError("no LLM in recall path")

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
    agent._character_conf_uid = conf_uid
    agent._history_uid = history_uid
    agent._user_timezone = JKT
    agent._load_character_state(conf_uid)
    return agent


class CrossSessionRecallTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="p14-")
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        for directory in ("character_state", "episodic", "world_state"):
            os.makedirs(directory, exist_ok=True)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_A_session_a_to_session_b(self):
        agent_a = make_agent(history_uid="history-A")
        self.assertTrue(agent_a.observe_character_events(SESSION_A_TEXT, "oke noted"))
        # Session B: brand-new agent, different history, never saw A's text.
        agent_b = make_agent(history_uid="history-B")
        self.assertNotIn(SESSION_A_TEXT, str(getattr(agent_b, "_memory", "")))
        rows = agent_b.list_future_intentions()
        self.assertEqual(len(rows), 1)
        self.assertIn(SESSION_A_TEXT, rows[0]["text"])
        # The recall query itself must not create a new intention.
        self.assertIsNone(
            detect_any_future_intention(SESSION_B_QUERY, now=NOW_A, tz=JKT)
        )
        self.assertFalse(agent_b._observe_future_intention(SESSION_B_QUERY))
        self.assertEqual(len(agent_b.list_future_intentions()), 1)

    def test_B_restart_new_session_still_recalls(self):
        agent_a = make_agent(history_uid="history-A")
        agent_a.observe_character_events(SESSION_A_TEXT, "oke")
        # Backend restart: throw agents away, reload purely from disk.
        reloaded = load_character_state(CONF)
        self.assertEqual(len(reloaded.future_intentions), 1)
        agent_c = make_agent(history_uid="history-C")
        rows = agent_c.list_future_intentions()
        self.assertEqual(len(rows), 1)
        self.assertIn(SESSION_A_TEXT, rows[0]["text"])

    def test_C_context_block_carries_intention_without_transcript(self):
        agent_a = make_agent(history_uid="history-A")
        agent_a.observe_character_events(SESSION_A_TEXT, "oke")
        agent_b = make_agent(history_uid="history-B")
        block = build_future_intention_context(
            load_character_state(CONF).future_intentions, tz=JKT
        )
        self.assertIn(SESSION_A_TEXT, block)
        # Proof of mechanism: B's transcript has no A content.
        memory_text = " ".join(
            str(m.get("content", "")) for m in (agent_b._memory or [])
        )
        self.assertNotIn(SESSION_A_TEXT, memory_text)

    def test_D_past_due_is_due_not_upcoming(self):
        record_future_intention(CONF, SESSION_A_TEXT, now=NOW_A, tz=JKT)
        rows = load_character_state(CONF).future_intentions
        self.assertEqual(due_intentions(rows, now=NOW_A), [])
        later = NOW_A + timedelta(days=2)
        due = due_intentions(rows, now=later)
        self.assertEqual(len(due), 1)
        self.assertIn(SESSION_A_TEXT, due[0]["text"])
        # Consume-on-dispatch retires exactly once; second call is quiet.
        agent = make_agent(history_uid="history-B")
        self.assertEqual(agent.consume_due_future_intentions(moment=later), 1)
        self.assertEqual(agent.consume_due_future_intentions(moment=later), 0)

    def test_E_timezone_tomorrow_0700(self):
        a = record_future_intention(CONF, SESSION_A_TEXT, now=NOW_A, tz=JKT)
        b = record_future_intention("p14z", SESSION_A_TEXT, now=NOW_A, tz="UTC")
        self.assertIsNotNone(a["due_at"])
        self.assertIsNotNone(b["due_at"])
        self.assertEqual(
            abs(
                (
                    datetime.fromisoformat(a["due_at"])
                    - datetime.fromisoformat(b["due_at"])
                ).total_seconds()
            ),
            7 * 3600,
        )
        block = build_future_intention_context([a], tz=JKT)
        self.assertIn("07:00", block)
        self.assertIn("WIB", block)

    def test_F_no_fabrication(self):
        record_future_intention(CONF, SESSION_A_TEXT, now=NOW_A, tz=JKT)
        block = build_future_intention_context(
            load_character_state(CONF).future_intentions, tz=JKT
        )
        for invented in ("Matematika", "pacar baru", "08:00", "dokter"):
            self.assertNotIn(invented, block)
        self.assertEqual(build_future_intention_context([], tz=JKT), "")

    def test_G_multiple_intentions_due_soonest_first(self):
        record_future_intention(
            CONF, "lusa jam 8 aku mau nanya hasil lab", now=NOW_A, tz=JKT
        )
        record_future_intention(CONF, SESSION_A_TEXT, now=NOW_A, tz=JKT)
        rows = pending_intentions(load_character_state(CONF).future_intentions)
        self.assertEqual(len(rows), 2)
        # Besok (Oct 06) sorts before lusa (Oct 07).
        self.assertIn("Besok jam 07.00", rows[0]["text"])
        self.assertIn("lusa", rows[1]["text"])
        block = build_future_intention_context(rows, tz=JKT)
        self.assertIn("Besok jam 07.00 aku mau nanya lagi sesuatu.", block)
        self.assertIn("lusa jam 8 aku mau nanya hasil lab", block)

    def test_H_stated_date_grounds_kemarin_query(self):
        record_future_intention(CONF, SESSION_A_TEXT, now=NOW_A, tz=JKT)
        block = build_future_intention_context(
            load_character_state(CONF).future_intentions, tz=JKT
        )
        # NOW_A is Oct 05 12:00 Jakarta.
        self.assertIn("stated Oct 05", block)
        self.assertIn("due Oct 06 07:00", block)

    def test_H_fail_soft_without_created_at(self):
        block = build_future_intention_context(
            [{"text": SESSION_A_TEXT, "status": "pending"}], tz=JKT
        )
        self.assertIn(SESSION_A_TEXT, block)
        self.assertNotIn("stated Oct", block)

    def test_I_dated_survives_undated_flood(self):
        from src.open_llm_vtuber.future_intentions import (
            FutureIntention,
            add_future_intention,
        )

        rows = add_future_intention(
            [],
            detect_any_future_intention(SESSION_A_TEXT, now=NOW_A, tz=JKT),
        )
        for i in range(25):
            rows = add_future_intention(
                rows,
                FutureIntention(
                    text=f"aku mau cemilan nomor {i} xyz",
                    due_at=None,
                    created_at="2026-10-05T05:00:00+00:00",
                    kind="plan",
                    tz=JKT,
                ),
            )
        pending = pending_intentions(rows)
        self.assertLessEqual(len(pending), 20)
        self.assertTrue(any(SESSION_A_TEXT in row["text"] for row in pending))


if __name__ == "__main__":
    unittest.main()
