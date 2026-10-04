"""Persistent cross-session memory & relationship continuity.

Covers the full pipeline the way a user experiences it:

    WRITE -> PERSIST -> NEW SESSION -> RETRIEVE -> CONTEXT

Every scenario uses ordinary conversation. The user never says "ingat this",
"simpan this" or any other command: the whole point of the automatic path is
that a plainly durable statement about the user is remembered on its own.

Scenarios (per spec):
A. event said naturally -> episodic capture + occurred_at
B. relationship reaches married -> stays married in a new session
C. long conversation, several facts -> survive without transcript injection
D. restart between sessions
E. refresh/reconnect
F. unrelated session is not contaminated
G. duplicate event dedup
H. old event uses occurred_at, not created_at

Storage is a per-test temp directory and a dedicated conf_uid, so nothing here
can read or write real character data.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.agent.relationship_context import (
    VALID_RELATIONSHIP_STATUSES,
    build_relationship_context,
    detect_relationship_update,
    normalize_relationship_status,
)
from src.open_llm_vtuber.character_memory_commands import extract_stable_facts
from src.open_llm_vtuber.character_state import (
    CharacterState,
    default_seed_goals,
    load_character_state,
    save_character_state,
)
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    store_message,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber.episodic_memory import (
    append_episodic_event,
    load_episodic_events,
    retrieve_episodic_events,
)

JKT = "Asia/Jakarta"


class _LLM:
    model = "continuity-test"
    max_tokens = 8

    async def chat_completion(self, messages, system=None, tools=None):
        if False:
            yield None


def make_agent(conf_uid: str, history_uid: str) -> BasicMemoryAgent:
    agent = BasicMemoryAgent(
        llm=_LLM(),
        system="persona",
        live2d_model=SimpleNamespace(extract_emotion=lambda _t: []),
        tts_preprocessor_config=TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={"translate_audio": False, "translate_provider": "deeplx"},
        ),
    )
    agent._character_conf_uid = conf_uid
    agent.set_memory_from_history(conf_uid, history_uid, user_timezone=JKT)
    return agent


class ContinuityBase(unittest.TestCase):
    CONF = "continuitychar"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        for folder in (
            "chat_history",
            "character_state",
            "episodic",
            "world_state",
            "proactive_state",
        ):
            os.makedirs(folder, exist_ok=True)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def new_session(self) -> str:
        return create_new_history(self.CONF)

    def session(self, history_uid: str) -> BasicMemoryAgent:
        """Simulates opening that chat: a fresh agent on an existing history."""
        return make_agent(self.CONF, history_uid)

    def say(self, agent: BasicMemoryAgent, user: str, assistant: str) -> None:
        store_message(self.CONF, agent._history_uid, "human", user)
        store_message(self.CONF, agent._history_uid, "ai", assistant)
        agent.observe_character_events(user, assistant)

    def seed_relationship(self, status: str = "dating") -> None:
        save_character_state(
            self.CONF,
            CharacterState(
                relationship_status=status,
                relationship_migrated=True,
                relationship_reason="explicit_relationship_event",
                goals=default_seed_goals(),
            ),
        )

    def memories(self, agent=None):
        state = agent._character_state if agent else load_character_state(self.CONF)
        return [m["text"] for m in state.memories]

    def prompt(self, agent: BasicMemoryAgent) -> str:
        return agent._relationship_system_prompt("base")


# ---------------------------------------------------------------------------
# Phase 3 core: automatic capture without any command word
# ---------------------------------------------------------------------------
class AutomaticMemoryTest(ContinuityBase):
    def test_stable_fact_is_stored_without_any_command(self):
        a = self.session(self.new_session())
        self.say(a, "gw itu suka kopi susu gula aren", "Wah, manis sekali.")
        self.assertIn("suka kopi susu gula aren", self.memories(a))

    def test_automatic_memory_is_not_marked_explicit(self):
        a = self.session(self.new_session())
        self.say(a, "aku suka game horror", "Seru juga ya.")
        stored = a._character_state.memories[0]
        self.assertFalse(stored["explicit"], "inferred trait must not claim to be explicit")
        self.assertTrue(stored["added_at"], "must carry an absolute stamp")

    def test_explicit_command_still_works_and_is_marked_explicit(self):
        a = self.session(self.new_session())
        self.say(a, "ingat ya, aku suka matcha", "Siap, gw ingat.")
        stored = a._character_state.memories[0]
        self.assertTrue(stored["explicit"])
        self.assertIn("matcha", stored["text"])

    def test_transient_statement_is_not_long_term_memory(self):
        a = self.session(self.new_session())
        self.say(a, "gw hari ini capek banget", "Istirahat ya.")
        self.assertEqual(self.memories(a), [], "a transient state is not a lasting fact")

    def test_question_is_not_stored(self):
        a = self.session(self.new_session())
        self.say(a, "kamu suka apa?", "Aku suka kopi.")
        self.assertEqual(self.memories(a), [])

    def test_reaction_and_thanks_are_not_stored(self):
        a = self.session(self.new_session())
        for line in ("oke", "makasih ya", "wkwk"):
            self.say(a, line, "Hmm?")
        self.assertEqual(self.memories(a), [])

    def test_extractor_rejects_command_and_transient_wording(self):
        self.assertEqual(extract_stable_facts("ingat ya aku suka matcha"), ())
        self.assertEqual(extract_stable_facts("gw lagi ngantuk banget"), ())
        self.assertEqual(extract_stable_facts("gw lagi belajar React"), ())
        self.assertEqual(extract_stable_facts(""), ())
        self.assertEqual(extract_stable_facts(None), ())

    def test_extractor_captures_habits_and_identity(self):
        self.assertEqual(
            extract_stable_facts("gw biasa bangun jam lima pagi"),
            ("biasa bangun jam lima pagi",),
        )
        self.assertEqual(
            extract_stable_facts("nama gw Rizky, gw programmer di Jakarta"),
            ("nama Rizky, programmer di Jakarta",),
        )
        self.assertEqual(
            extract_stable_facts("saya tinggal di Bandung"), ("tinggal di Bandung",)
        )

    def test_extractor_is_bounded_per_turn(self):
        text = "gw suka kopi. gw biasa bangun pagi. gw suka game. gw tinggal di Jakarta."
        self.assertLessEqual(len(extract_stable_facts(text)), 2)

    def test_extractor_rejects_overlong_and_empty_fragments(self):
        self.assertEqual(extract_stable_facts("gw suka " + "x" * 400), ())

    def test_duplicate_fact_is_not_stored_twice(self):
        a = self.session(self.new_session())
        for _ in range(3):
            self.say(a, "gw suka kopi susu gula aren", "Noted.")
        self.assertEqual(self.memories(a).count("suka kopi susu gula aren"), 1)

    def test_same_fact_in_a_different_phrase_is_not_duplicated(self):
        """Automatic capture drops the subject; an explicit command keeps it.

        Storing both "aku suka minum kopi susu gula aren" and "suka kopi susu
        gula aren" would be the same fact twice, so dedup is semantic too.
        """
        a = self.session(self.new_session())
        self.say(a, "ingat ya, aku suka minum kopi susu gula aren", "Siap.")
        self.say(a, "gw itu suka kopi susu gula aren", "Noted.")
        rows = [m for m in self.memories(a) if "kopi susu" in m]
        self.assertEqual(len(rows), 1, f"same fact stored twice: {rows}")

    def test_distinct_facts_are_not_merged(self):
        a = self.session(self.new_session())
        for user, ai in (
            ("gw suka kopi susu gula aren", "Manis."),
            ("gw biasa bangun jam lima pagi", "Kapan tidur?"),
            ("saya tinggal di Bandung", "Bagus."),
        ):
            self.say(a, user, ai)
        rows = self.memories(a)
        self.assertEqual(len(rows), 3, f"distinct facts must all survive: {rows}")

    def test_semantic_dedup_keeps_genuinely_different_preferences_apart(self):
        a = self.session(self.new_session())
        self.say(a, "gw suka kopi susu gula aren", "Manis.")
        self.say(a, "gw suka teh melati", "Hangat.")
        rows = [m for m in self.memories(a) if m and m.startswith("suka ")]
        self.assertEqual(len(rows), 2, f"coffee and tea must stay separate: {rows}")

    def test_no_llm_call_is_required_for_capture(self):
        """The extractor is pure local text work."""
        calls = []

        class _Tripwire(_LLM):
            async def chat_completion(self, *a, **k):
                calls.append((a, k))
                if False:
                    yield None

        a = self.session(self.new_session())
        a._llm = _Tripwire()
        self.say(a, "gw suka kopi susu", "Oke.")
        self.assertEqual(calls, [])
        self.assertTrue(self.memories(a))


# ---------------------------------------------------------------------------
# Phase 2 + scenario B: relationship persistence
# ---------------------------------------------------------------------------
class RelationshipPersistenceTest(ContinuityBase):
    def test_married_is_a_valid_tier(self):
        self.assertIn("married", VALID_RELATIONSHIP_STATUSES)
        self.assertEqual(normalize_relationship_status("married"), "married")

    def test_older_tiers_are_untouched(self):
        for tier in ("stranger", "familiar", "close", "dating"):
            self.assertEqual(normalize_relationship_status(tier), tier)

    def test_unknown_value_still_degrades_to_stranger(self):
        self.assertEqual(normalize_relationship_status("garbage"), "stranger")

    def test_marriage_requires_an_established_romance(self):
        """A stranger cannot jump straight to married."""
        self.assertIsNone(
            detect_relationship_update("stranger", "kita sudah menikah", "Iya, kita sudah menikah.")
        )

    def test_marriage_is_detected_from_dating(self):
        update = detect_relationship_update(
            "dating", "kita sudah menikah kok", "Iya, benar. Kita sudah menikah."
        )
        self.assertIsNotNone(update)
        self.assertEqual(update.new_status, "married")

    def test_marriage_rejection_is_honoured(self):
        for reply in ("Belum, aku belum siap nikah.", "Yah, aku menolak."):
            self.assertIsNone(
                detect_relationship_update("dating", "kita sudah menikah", reply)
            )

    def test_married_is_terminal(self):
        self.assertIsNone(
            detect_relationship_update("married", "kita pacaran yuk", "Iya, aku mau.")
        )
        self.assertIsNone(
            detect_relationship_update("married", "kita sudah cerai", "Iya, aku setuju.")
        )

    def test_dating_detection_unchanged(self):
        update = detect_relationship_update("stranger", "kita pacaran", "Iya, aku mau.")
        self.assertIsNotNone(update)
        self.assertEqual(update.new_status, "dating")

    def test_scenario_b_married_survives_a_new_chat(self):
        self.seed_relationship("dating")
        h1 = self.new_session()
        a = self.session(h1)
        self.say(a, "kita sudah menikah kok", "Iya, benar. Kita sudah menikah, sucrose.")
        self.assertEqual(a._relationship_state.status, "married")

        h2 = self.new_session()
        b = self.session(h2)
        self.assertEqual(b._relationship_state.status, "married")
        self.assertIn("Current state: married", self.prompt(b))

    def test_married_prompt_forbids_falling_back(self):
        ctx = build_relationship_context("married", updated_at=None, tz=JKT)
        self.assertIn("married", ctx)
        lowered = ctx.lower()
        self.assertIn("never", lowered)
        self.assertIn("dating", lowered, "guidance must explicitly forbid the old tier")

    def test_scenario_b_repeated_dating_request_cannot_downgrade(self):
        self.seed_relationship("married")
        a = self.session(self.new_session())
        self.say(a, "kita pacaran yuk", "Iya, aku mau.")
        self.assertEqual(a._relationship_state.status, "married")

    def test_relationship_persists_on_disk(self):
        self.seed_relationship("dating")
        a = self.session(self.new_session())
        self.say(a, "kita sudah menikah", "Iya, kita sudah menikah.")
        self.assertEqual(load_character_state(self.CONF).relationship_status, "married")

    def test_no_persisted_relationship_means_stranger_default(self):
        a = self.session(self.new_session())
        self.assertEqual(a._relationship_state.status, "stranger")

    def test_existing_dating_state_is_not_downgraded_on_load(self):
        self.seed_relationship("dating")
        a = self.session(self.new_session())
        self.say(a, "halo, apa kabar?", "Baik.")
        self.assertEqual(a._relationship_state.status, "dating")


# ---------------------------------------------------------------------------
# Scenario C + F + G + H: cross-session context, isolation, dedup, temporal
# ---------------------------------------------------------------------------
class CrossSessionContextTest(ContinuityBase):
    def test_scenario_c_facts_survive_without_transcript_injection(self):
        a = self.session(self.new_session())
        for user, ai in (
            ("gw suka kopi susu gula aren", "Manis."),
            ("gw biasa bangun jam lima pagi", "Kapan tidur?"),
            ("nama gw Rizky, gw programmer di Jakarta", "Hai Rizky."),
        ):
            self.say(a, user, ai)
        self.assertEqual(len(self.memories(a)), 3)

        b = self.session(self.new_session())
        prompt = self.prompt(b)
        for fact in self.memories(b):
            self.assertIn(fact, prompt, "every stored fact must reach the prompt")
        # the old transcript itself must NOT be injected
        self.assertNotIn("Kapan tidur?", prompt)
        self.assertNotIn("Hai Rizky.", prompt)
        # and it stays bounded
        self.assertLess(len(prompt), 20000)

    def test_scenario_f_unrelated_session_gets_no_injection(self):
        a = self.session(self.new_session())
        self.say(a, "gw suka kopi susu gula aren", "Manis.")
        b = self.session(self.new_session())
        prompt = self.prompt(b)
        # the durable fact legitimately persists ...
        self.assertIn("suka kopi susu gula aren", prompt)
        # ... but nothing conversational leaked
        self.assertNotIn("Manis.", prompt)
        self.assertEqual(len(get_history(self.CONF, b._history_uid)), 0)

    def test_scenario_d_restart_between_sessions(self):
        a = self.session(self.new_session())
        self.say(a, "gw suka kopi susu gula aren", "Manis.")
        # "restart": every state is re-read from disk by a brand new agent
        b = self.session(self.new_session())
        self.assertIn("suka kopi susu gula aren", self.memories(b))
        self.assertIn("suka kopi susu gula aren", self.prompt(b))

    def test_scenario_e_reconnect_same_history(self):
        h = self.new_session()
        a = self.session(h)
        self.say(a, "gw suka kopi susu gula aren", "Manis.")
        again = self.session(h)
        self.assertIn("suka kopi susu gula aren", self.memories(again))

    def test_scenario_a_and_g_episodic_dedup_and_occurred_at(self):
        stamp = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        created = datetime.now(timezone.utc).isoformat()
        event = {
            "event_text": "gw selesai memperbaiki bug frontend yang bikin seharian kesel",
            "occurred_at": stamp,
            "created_at": created,
            "tz": JKT,
            "source": "conversation",
        }
        first = append_episodic_event(self.CONF, dict(event))
        self.assertTrue(first, "first store must succeed")
        events = load_episodic_events(self.CONF)
        self.assertEqual(len(events), 1)
        stored = events[0]
        # stores normalise to whole-second ISO
        self.assertEqual(
            stored["occurred_at"],
            datetime.fromisoformat(stamp).isoformat(timespec="seconds"),
        )
        self.assertNotEqual(
            stored["occurred_at"][:19],
            stored["created_at"][:19],
            "event time must differ from storage time",
        )

        # exact duplicate must not append
        append_episodic_event(self.CONF, dict(event))
        self.assertEqual(len(load_episodic_events(self.CONF)), 1)

    def test_scenario_a_episodic_retrieval_finds_the_natural_event(self):
        stamp = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        append_episodic_event(
            self.CONF,
            {
                "event_text": "gw selesai memperbaiki bug frontend yang bikin seharian kesel",
                "occurred_at": stamp,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "tz": JKT,
                "source": "conversation",
            },
        )
        found = retrieve_episodic_events(
            load_episodic_events(self.CONF),
            "bug frontend tadi gimana",
            now=datetime.now(timezone.utc),
            top_n=3,
        )
        self.assertTrue(found, "a relevant natural statement must be retrievable")

    def test_scenario_h_recency_ranks_on_occurred_at_not_created_at(self):
        """Two equally relevant events must rank by WHEN THEY HAPPENED.

        The 40-day-old event was written most recently; the 1-hour-old event was
        written a month ago. If ranking used ``created_at`` the old one would
        win, which is exactly the bug this guards.
        """
        now = datetime.now(timezone.utc)
        old_but_freshly_written = {
            "id": "a",
            "event_text": "gw baca buku hahaha",
            "occurred_at": (now - timedelta(days=40)).isoformat(),
            "created_at": now.isoformat(),
        }
        recent_but_written_long_ago = {
            "id": "b",
            "event_text": "gw baca buku hahaha",
            "occurred_at": (now - timedelta(hours=1)).isoformat(),
            "created_at": (now - timedelta(days=30)).isoformat(),
        }
        ranked = retrieve_episodic_events(
            [old_but_freshly_written, recent_but_written_long_ago],
            "buku",
            now=now,
            top_n=2,
        )
        self.assertEqual(len(ranked), 2)
        self.assertEqual(
            ranked[0]["id"],
            "b",
            "the event that happened most recently must rank first",
        )

    def test_episodic_store_is_isolated_per_conf_uid(self):
        append_episodic_event(
            self.CONF,
            {
                "event_text": "gw suka kopi susu",
                "occurred_at": datetime.now(timezone.utc).isoformat(),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        self.assertEqual(len(load_episodic_events(self.CONF)), 1)
        self.assertEqual(len(load_episodic_events("someone_else")), 0)


# ---------------------------------------------------------------------------
# Phase 6: test isolation guard
# ---------------------------------------------------------------------------
class TestIsolationGuardTest(unittest.TestCase):
    def test_this_module_never_touches_a_real_conf_uid(self):
        """Guards against this suite being pointed at production data.

        The production uid is built at runtime so the guard's own source cannot
        satisfy it by mentioning the literal.
        """
        production_uid = "id_" + "mili_01"
        source = open(__file__, encoding="utf-8").read()
        self.assertNotIn(production_uid, source)
        self.assertIn("tempfile.TemporaryDirectory", source)

    def test_state_files_stay_inside_the_temp_tree(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                for folder in ("character_state", "episodic", "chat_history"):
                    os.makedirs(folder, exist_ok=True)
                save_character_state(
                    "guardchar",
                    CharacterState(
                        relationship_status="married", goals=default_seed_goals()
                    ),
                )
                for folder in ("character_state", "episodic"):
                    for name in os.listdir(folder):
                        self.assertTrue(
                            os.path.abspath(os.path.join(folder, name)).startswith(
                                os.path.abspath(tmp)
                            ),
                            f"{folder}/{name} escaped the temp tree",
                        )
            finally:
                os.chdir(previous)


if __name__ == "__main__":
    unittest.main()
