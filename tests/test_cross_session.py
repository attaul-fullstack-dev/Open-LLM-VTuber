"""Cross-session conversation continuity — deterministic tests.

Session summaries are the already-persisted rolling summaries of *other*
histories, retrieved read-only at history load and rendered per turn with
dynamic age tags. Fake LLM + fixed timestamps + tmp dirs; no network.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timezone

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.conversation_summary import (
    PREV_SESSION_MAX_CHARS,
    PREV_SESSION_MAX_COUNT,
    SUMMARY_SYSTEM_PROMPT,
    build_previous_session_context,
)
from src.open_llm_vtuber.character_state import build_character_memory_context
from src.open_llm_vtuber.agent.relationship_context import build_relationship_context
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_metadata,
    store_message,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.world_state import load_and_reconcile_world_state

JAKARTA = "Asia/Jakarta"
T0 = datetime(2026, 10, 1, 0, 0, tzinfo=timezone.utc)  # Oct 1 00:00 UTC.


class _FakeLLM:
    model = "xsession-test"
    max_tokens = 100

    def __init__(self):
        self.summary_calls = 0

    async def chat_completion(self, messages, system=None, tools=None):
        if system == SUMMARY_SYSTEM_PROMPT:
            self.summary_calls += 1
            yield "User and Lilith discussed Temporal Awareness. Done."
            return
        yield "oke."


class _FakeLive2D:
    def extract_emotion(self, _text):
        return []


def make_agent(conf_uid, history_uid, tz=None, llm=None):
    agent = BasicMemoryAgent(
        llm=llm or _FakeLLM(),
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


def write_history(conf_uid, uid, stamp, summary=None, updated_at=None):
    """Deterministic history file with fixed absolute timestamps."""
    entry = {"role": "metadata", "timestamp": stamp, "title": ""}
    if summary is not None:
        entry["conversation_summary"] = summary
    if updated_at is not None:
        entry["summary_updated_at"] = updated_at
    os.makedirs(os.path.join("chat_history", conf_uid), exist_ok=True)
    path = os.path.join("chat_history", conf_uid, f"{uid}.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(
            [
                entry,
                {"role": "human", "timestamp": stamp, "content": "halo sesi"},
                {
                    "role": "ai",
                    "timestamp": stamp,
                    "content": "halo juga rahasia-transkrip-unik",
                },
            ],
            handle,
        )


class CrossSessionBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        self.conf_uid = "xchar"

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()


class SessionSummaryPersistTest(CrossSessionBase, unittest.IsolatedAsyncioTestCase):
    async def test_1_session_a_summary_is_persisted(self):
        history_a = create_new_history(self.conf_uid)
        store_message(self.conf_uid, history_a, "human", "bug temporal")
        store_message(self.conf_uid, history_a, "ai", "sudah diperbaiki")
        store_message(self.conf_uid, history_a, "human", "lanjut")
        store_message(self.conf_uid, history_a, "ai", "siap")
        agent = make_agent(self.conf_uid, history_a)
        ok, _ = await agent.compact_conversation()
        self.assertTrue(ok)
        metadata = get_metadata(self.conf_uid, history_a)
        self.assertIn("Temporal Awareness", metadata.get("conversation_summary", ""))
        self.assertTrue(metadata.get("summary_updated_at"))


class SessionSummaryRetrieveTest(CrossSessionBase):
    def test_2_session_b_retrieves_session_a_summary(self):
        write_history(
            self.conf_uid,
            "sess-a",
            "2026-09-30T10:00:00+00:00",
            summary="User and Lilith discussed Temporal Awareness.",
            updated_at="2026-09-30T11:00:00+00:00",
        )
        write_history(self.conf_uid, "sess-b", "2026-10-01T00:00:00+00:00")
        agent = make_agent(self.conf_uid, "sess-b", tz=JAKARTA)
        cached = agent._prev_session_summaries
        self.assertEqual(len(cached), 1)
        self.assertEqual(
            cached[0]["text"], "User and Lilith discussed Temporal Awareness."
        )
        self.assertEqual(cached[0]["at"], "2026-09-30T11:00:00+00:00")

    def test_3_restart_does_not_lose_summary(self):
        write_history(
            self.conf_uid,
            "sess-a",
            "2026-09-30T10:00:00+00:00",
            summary="Temporal Awareness selesai, lanjut Cross-Session.",
            updated_at="2026-09-30T11:00:00+00:00",
        )
        write_history(self.conf_uid, "sess-b", "2026-10-01T00:00:00+00:00")
        # Fresh agent = post-restart process.
        agent = make_agent(self.conf_uid, "sess-b", tz=JAKARTA)
        prompt = agent._relationship_system_prompt("base")
        self.assertIn("Temporal Awareness selesai", prompt)

    def test_4_no_previous_session_means_normal_behavior(self):
        write_history(self.conf_uid, "only", "2026-10-01T00:00:00+00:00")
        agent = make_agent(self.conf_uid, "only", tz=JAKARTA)
        self.assertEqual(agent._prev_session_summaries, [])
        prompt = agent._relationship_system_prompt("base")
        self.assertNotIn("RELEVANT PREVIOUS SESSION", prompt)
        self.assertIn("Current date:", prompt)

    def test_5_empty_and_corrupt_summaries_fail_soft(self):
        write_history(
            self.conf_uid, "empty", "2026-09-28T10:00:00+00:00", summary="   "
        )
        write_history(self.conf_uid, "nosum", "2026-09-29T10:00:00+00:00")
        bad_path = os.path.join("chat_history", self.conf_uid, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as handle:
            handle.write("{not json")
        write_history(self.conf_uid, "current", "2026-10-01T00:00:00+00:00")
        agent = make_agent(self.conf_uid, "current", tz=JAKARTA)
        self.assertEqual(agent._prev_session_summaries, [])
        prompt = agent._relationship_system_prompt("base")
        self.assertNotIn("RELEVANT PREVIOUS SESSION", prompt)

    def test_6_multiple_sessions_bounded_newest_first(self):
        for day in (25, 26, 27, 28):
            write_history(
                self.conf_uid,
                f"sess-{day}",
                f"2026-09-{day}T10:00:00+00:00",
                summary=f"Summary of Sep {day}.",
                updated_at=f"2026-09-{day}T11:00:00+00:00",
            )
        write_history(self.conf_uid, "current", "2026-10-01T00:00:00+00:00")
        agent = make_agent(self.conf_uid, "current", tz=JAKARTA)
        cached = agent._prev_session_summaries
        self.assertEqual(len(cached), PREV_SESSION_MAX_COUNT)
        self.assertEqual(cached[0]["text"], "Summary of Sep 28.")
        self.assertEqual(cached[1]["text"], "Summary of Sep 27.")
        prompt = agent._relationship_system_prompt("base")
        self.assertNotIn("Summary of Sep 26", prompt)
        self.assertNotIn("Summary of Sep 25", prompt)

    def test_6b_long_summary_truncated(self):
        write_history(
            self.conf_uid,
            "long",
            "2026-09-30T10:00:00+00:00",
            summary="x" * (PREV_SESSION_MAX_CHARS + 200),
            updated_at="2026-09-30T11:00:00+00:00",
        )
        write_history(self.conf_uid, "current", "2026-10-01T00:00:00+00:00")
        agent = make_agent(self.conf_uid, "current", tz=JAKARTA)
        self.assertEqual(
            len(agent._prev_session_summaries[0]["text"]), PREV_SESSION_MAX_CHARS
        )

    def test_7_current_session_never_treated_as_previous(self):
        write_history(
            self.conf_uid,
            "current",
            "2026-10-01T00:00:00+00:00",
            summary="Current session own summary.",
            updated_at="2026-10-01T00:05:00+00:00",
        )
        agent = make_agent(self.conf_uid, "current", tz=JAKARTA)
        self.assertEqual(agent._prev_session_summaries, [])

    def test_8_summary_injected_exactly_once_and_stable(self):
        write_history(
            self.conf_uid,
            "sess-a",
            "2026-09-30T10:00:00+00:00",
            summary="Unik summary kalimat.",
            updated_at="2026-09-30T11:00:00+00:00",
        )
        write_history(self.conf_uid, "sess-b", "2026-10-01T00:00:00+00:00")
        agent = make_agent(self.conf_uid, "sess-b", tz=JAKARTA)
        first = agent._relationship_system_prompt("base")
        second = agent._relationship_system_prompt("base")
        self.assertEqual(first.count("Unik summary kalimat."), 1)
        self.assertEqual(first, second)

    def test_9_full_transcript_is_not_injected(self):
        write_history(
            self.conf_uid,
            "sess-a",
            "2026-09-30T10:00:00+00:00",
            summary="Ringkasan singkat saja.",
            updated_at="2026-09-30T11:00:00+00:00",
        )
        write_history(self.conf_uid, "sess-b", "2026-10-01T00:00:00+00:00")
        agent = make_agent(self.conf_uid, "sess-b", tz=JAKARTA)
        prompt = agent._relationship_system_prompt("base")
        self.assertIn("Ringkasan singkat saja.", prompt)
        self.assertNotIn("rahasia-transkrip-unik", prompt)

    def test_10_temporal_metadata_survives(self):
        write_history(
            self.conf_uid,
            "sess-a",
            "2026-09-29T10:00:00+00:00",
            summary="Temporal Awareness selesai.",
            updated_at="2026-09-29T11:00:00+00:00",
        )
        write_history(self.conf_uid, "sess-b", "2026-10-01T00:00:00+00:00")
        agent = make_agent(self.conf_uid, "sess-b", tz=JAKARTA)
        prompt = agent._relationship_system_prompt("base")
        # Sep 29 11:00 UTC = 18:00 Jakarta Sep 29 -> "2 days ago (Sep 29)".
        self.assertIn("[2 days ago | Sep 29]", prompt)

    def test_10b_missing_updated_at_falls_back_to_session_stamp(self):
        write_history(
            self.conf_uid,
            "sess-a",
            "2026-09-30T10:00:00+00:00",
            summary="Tanpa stamp update.",
        )
        write_history(self.conf_uid, "sess-b", "2026-10-01T00:00:00+00:00")
        agent = make_agent(self.conf_uid, "sess-b", tz=JAKARTA)
        self.assertEqual(len(agent._prev_session_summaries), 1)
        prompt = agent._relationship_system_prompt("base")
        self.assertIn("[Yesterday | Sep 30]", prompt)

    def test_11_previous_context_reaches_context_builder(self):
        write_history(
            self.conf_uid,
            "sess-a",
            "2026-09-30T10:00:00+00:00",
            summary="User and Lilith were working on Temporal Awareness.",
            updated_at="2026-09-30T11:00:00+00:00",
        )
        write_history(self.conf_uid, "sess-b", "2026-10-01T00:00:00+00:00")
        agent = make_agent(self.conf_uid, "sess-b", tz=JAKARTA)
        prompt = agent._relationship_system_prompt("base")
        self.assertIn("RELEVANT PREVIOUS SESSION:", prompt)
        self.assertIn("User and Lilith were working on Temporal Awareness.", prompt)
        self.assertIn("not the current one", prompt)


class UntouchedSystemsTest(CrossSessionBase):
    def test_12_memory_behavior_unchanged(self):
        state = make_agent(
            self.conf_uid, create_new_history(self.conf_uid)
        )._character_state
        state.memories.append(
            {
                "text": "user suka ramen",
                "added_at": "2026-09-30T10:00:00+00:00",
                "explicit": True,
            }
        )
        context = build_character_memory_context(state, tz=JAKARTA)
        self.assertIn("user suka ramen", context)
        self.assertIn("Yesterday", context)

    def test_13_life_state_behavior_unchanged(self):
        snapshot = load_and_reconcile_world_state(self.conf_uid, tz=JAKARTA)
        self.assertTrue(snapshot.location)
        self.assertTrue(snapshot.activity)
        self.assertIsNotNone(snapshot.last_update_at)

    def test_14_relationship_behavior_unchanged(self):
        context = build_relationship_context(
            "close", updated_at="2026-09-30T10:00:00+00:00", tz=JAKARTA
        )
        self.assertIn("close", context)
        self.assertIn("yesterday", context)


class PreviousBlockFormatTest(unittest.TestCase):
    def test_singular_plural_labels(self):
        one = build_previous_session_context(
            [{"age_tag": "[Today | Oct 1]", "text": "Satu."}]
        )
        self.assertIn("RELEVANT PREVIOUS SESSION:", one)
        self.assertNotIn("SESSIONS", one)
        two = build_previous_session_context(
            [
                {"age_tag": "[Today | Oct 1]", "text": "Satu."},
                {"age_tag": "[Yesterday | Sep 30]", "text": "Dua."},
            ]
        )
        self.assertIn("RELEVANT PREVIOUS SESSIONS:", two)

    def test_empty_items_yield_empty_block(self):
        self.assertEqual(build_previous_session_context([]), "")
        self.assertEqual(
            build_previous_session_context([{"age_tag": "", "text": "   "}]), ""
        )


if __name__ == "__main__":
    unittest.main()
