"""Episodic continuity integration — G1/G2/G3 + temporal contract.

Covers the 13 required guarantees:
  1  occurred_at older than created_at ranks by occurred_at
  2  occurred_at missing/invalid falls back to created_at
  3  the clean user query is the retrieval query
  4  metadata["episodic_query"] is honoured
  5  missing metadata degrades to a safe fallback
  6  a relevant OLD event outranks an irrelevant NEW one
  7  dynamic age labels change with `now`
  8  "semalam" resolves to the correct absolute occurred_at
  9  "4 hari lalu" resolves to the correct absolute occurred_at
 10  restart never rewrites occurred_at
 11  the episodic context block stays bounded
 12  retrieval costs zero extra LLM calls
 13  telemetry never leaks event content

Deterministic and LLM-free throughout.
"""

import inspect
import json
import os
import pathlib
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber import episodic_memory as em
from src.open_llm_vtuber.episodic_memory import (
    EPISODIC_MAX_TOKENS,
    EPISODIC_TOP_N,
    append_episodic_event,
    load_episodic_events,
    render_episodic_context,
    resolve_occurred_at,
    retrieve_episodic_events,
)

UTC = timezone.utc
JKT = "Asia/Jakarta"
NOW = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)  # 19:00 JKT


def _logger_statements(source: str):
    """Return each logger.<level>( ... ) statement, paren-balanced."""
    out = []
    index = 0
    while True:
        found = source.find("logger.", index)
        if found < 0:
            return out
        start = source.find("(", found)
        if start < 0:
            return out
        depth = 0
        for position in range(start, len(source)):
            if source[position] == "(":
                depth += 1
            elif source[position] == ")":
                depth -= 1
                if depth == 0:
                    out.append(source[found : position + 1])
                    index = position
                    break
        else:
            return out


def ev(text, occurred_hours_ago, created_at=None, event_id="e1", session="s"):
    occurred = (NOW - timedelta(hours=occurred_hours_ago)).isoformat()
    return {
        "id": event_id,
        "event_text": text,
        "occurred_at": occurred,
        "session_uid": session,
        "source": "conversation",
        "created_at": created_at or NOW.isoformat(),
        "tz": JKT,
    }


class G2OccurredAtWinsTest(unittest.TestCase):
    """1 + 2: event time is the temporal source of truth."""

    def test_1_occurred_at_older_than_created_at_ranks_by_occurred_at(self):
        # Same query overlap, opposite creation order. Ranking must follow
        # WHEN IT HAPPENED, not when it was written.
        old = ev(
            "Gw belajar frontend",
            occurred_hours_ago=96,
            created_at=NOW.isoformat(),
            event_id="old",
        )
        new = ev(
            "Gw belajar frontend",
            occurred_hours_ago=2,
            created_at=NOW.isoformat(),
            event_id="new",
        )
        selected = retrieve_episodic_events([old, new], "belajar frontend", now=NOW)
        self.assertEqual(len(selected), 2)
        self.assertEqual(
            selected[0]["id"], "new", "more recent event time must rank first"
        )
        # created_at is identical, so any difference can only come from
        # occurred_at.
        self.assertEqual(old["created_at"], new["created_at"])

    def test_1b_storage_order_does_not_change_the_ranking(self):
        old = ev("belajar frontend", 96, event_id="old")
        new = ev("belajar frontend", 2, event_id="new")
        forward = retrieve_episodic_events([old, new], "belajar frontend", now=NOW)
        backward = retrieve_episodic_events([new, old], "belajar frontend", now=NOW)
        self.assertEqual([e["id"] for e in forward], [e["id"] for e in backward])
        self.assertEqual(forward[0]["id"], "new")

    def test_1c_tie_break_also_uses_event_time(self):
        # Equal relevance AND equal occurred_at: fallback to created_at, and
        # the sort must stay deterministic either way.
        a = ev("frontend", 10, created_at=NOW.isoformat(), event_id="a")
        b = ev(
            "frontend",
            10,
            created_at=(NOW - timedelta(days=3)).isoformat(),
            event_id="b",
        )
        first = [e["id"] for e in retrieve_episodic_events([a, b], "frontend", now=NOW)]
        second = [
            e["id"] for e in retrieve_episodic_events([b, a], "frontend", now=NOW)
        ]
        self.assertEqual(first, second)

    def test_2_invalid_occurred_at_falls_back_to_created_at(self):
        bad = {
            "id": "bad",
            "event_text": "Gw belajar frontend",
            "occurred_at": "not-a-timestamp",
            "session_uid": "s",
            "source": "conversation",
            "created_at": (NOW - timedelta(hours=1)).isoformat(),
            "tz": JKT,
        }
        missing = dict(bad, id="missing", occurred_at=None)
        for row in (bad, missing):
            with self.subTest(row=row["id"]):
                selected = retrieve_episodic_events([row], "belajar frontend", now=NOW)
                self.assertEqual(
                    len(selected), 1, "created_at fallback keeps it retrievable"
                )

    def test_2b_created_at_is_never_the_primary_representation(self):
        source = inspect.getsource(em._score_event)
        # occurred_at must be consulted first, created_at only as fallback.
        self.assertLess(
            source.index('event.get("occurred_at"'),
            source.index('event.get("created_at"'),
        )
        self.assertIn("or _parse_iso_or_none", source)

    def test_2c_future_occurred_at_does_not_boost_recency(self):
        future = ev("belajar frontend", -5, event_id="future")  # 5h in the future
        past = ev("belajar frontend", 5, event_id="past")
        selected = retrieve_episodic_events([future, past], "belajar frontend", now=NOW)
        self.assertEqual(selected[0]["id"], "past")


class G1QueryTest(unittest.TestCase):
    """3 + 4 + 5: the retrieval query is the clean user turn."""

    def _agent(self):
        from src.open_llm_vtuber.config_manager.utils import read_yaml
        import src.open_llm_vtuber.agent.agents.basic_memory_agent as m

        repo = pathlib.Path(em.__file__).resolve().parents[2]
        conf = read_yaml(str(repo / "conf.yaml"))["character_config"]
        settings = conf["agent_config"]["agent_settings"]["basic_memory_agent"]

        class Noop:
            async def chat_completion(self, *a, **k):
                return
                yield ""

            async def handle_abort(self, *a, **k):
                return

        return m.BasicMemoryAgent(
            llm=Noop(),
            system=conf["persona_prompt"],
            live2d_model=conf["live2d_model_name"],
            faster_first_response=True,
            segment_method=settings.get("segment_method", "pysbd"),
            use_mcpp=False,
            tool_prompts={},
            mcp_prompt_string="",
        )

    def test_3_clean_user_query_is_used_for_retrieval(self):
        agent = self._agent()
        agent._episodic_query = "aku kemarin bilang mau belajar frontend"
        agent._memory = [
            {"role": "user", "content": "pertanyaan完全 berbeda sama sekali"},
        ]
        self.assertEqual(
            agent._episodic_query_text(), "aku kemarin bilang mau belajar frontend"
        )

    def test_3b_whole_memory_is_never_used_as_the_query(self):
        agent = self._agent()
        agent._episodic_query = "mau belajar frontend"
        agent._memory = [
            {"role": "user", "content": "bahasanya sangat panjang " * 50},
            {"role": "assistant", "content": "jawaban panjang " * 50},
            {"role": "user", "content": "OKE"},
        ]
        query = agent._episodic_query_text()
        self.assertEqual(query, "mau belajar frontend")
        self.assertNotIn("bahasanya sangat panjang", query)

    def test_4_metadata_episodic_query_is_honoured(self):
        from src.open_llm_vtuber.agent.input_types import BatchInput

        agent = self._agent()
        agent._to_messages(
            BatchInput(texts=[], metadata={"episodic_query": "teks user bersih"})
        )
        self.assertEqual(agent._episodic_query, "teks user bersih")
        self.assertIsNone(agent._episodic_selection_cache)

    def test_4b_single_conversation_sets_the_metadata(self):
        from src.open_llm_vtuber.conversations import single_conversation as sc

        source = inspect.getsource(sc)
        self.assertIn('turn_metadata["episodic_query"]', source)
        # the query must be the clean user text, captured before the search
        # block is appended to the LLM input
        query_line = source.index('turn_metadata["episodic_query"]')
        search_line = source.index("execute_router_search")
        self.assertLess(query_line, search_line)

    def test_5_missing_metadata_falls_back_safely(self):
        agent = self._agent()
        agent._episodic_query = ""
        agent._memory = [
            {"role": "assistant", "content": "halo"},
            {"role": "user", "content": "pesan user terakhir"},
        ]
        self.assertEqual(agent._episodic_query_text(), "pesan user terakhir")

    def test_5b_no_user_text_at_all_is_empty_not_a_crash(self):
        agent = self._agent()
        agent._episodic_query = ""
        agent._memory = [{"role": "assistant", "content": "halo"}]
        self.assertEqual(agent._episodic_query_text(), "")
        self.assertEqual(agent._episodic_selection(), [])

    def test_5c_selection_is_shared_by_prompt_and_decision_paths(self):
        agent = self._agent()
        agent._character_conf_uid = "share"
        agent._user_timezone = JKT
        agent._episodic_query = "belajar frontend"
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                append_episodic_event(
                    "share",
                    {
                        "event_text": "Gw belajar frontend semalam",
                        "occurred_at": (NOW - timedelta(hours=26)).isoformat(),
                        "session_uid": "s",
                        "source": "conversation",
                        "tz": JKT,
                    },
                )
                import src.open_llm_vtuber.agent.agents.basic_memory_agent as agent_mod

                calls = {"n": 0}
                original = em.retrieve_episodic_events

                def counting(events, query, **kwargs):
                    calls["n"] += 1
                    return original(events, query, **kwargs)

                em.retrieve_episodic_events = counting
                agent_mod.retrieve_episodic_events = counting  # bound at import
                try:
                    first = agent._episodic_selection()
                    second = agent._episodic_selection()
                    prompt_block = agent._episodic_context_for_prompt()
                finally:
                    em.retrieve_episodic_events = original
                    agent_mod.retrieve_episodic_events = original
                self.assertEqual(first, second, "same turn must reuse the selection")
                self.assertEqual(calls["n"], 1, "one retrieval pass per turn")
                self.assertTrue(prompt_block)
            finally:
                os.chdir(previous)


class RelevanceFirstTest(unittest.TestCase):
    """6: relevance beats recency."""

    def test_6_relevant_old_event_beats_irrelevant_new_events(self):
        events = [
            ev("Aku Yesterday belajar frontend", 26, event_id="relevant_old"),
            ev("Gw makan nasi goreng", 2, event_id="irrelevant_new_1"),
            ev("Gw bermain game", 3, event_id="irrelevant_new_2"),
        ]
        selected = retrieve_episodic_events(
            events, "aku kemarin bilang mau belajar frontend", now=NOW
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["id"], "relevant_old")
        for newer in ("irrelevant_new_1", "irrelevant_new_2"):
            self.assertNotIn(newer, [e["id"] for e in selected])

    def test_6b_recency_boost_never_replaces_relevance(self):
        source = inspect.getsource(em._score_event)
        self.assertIn("if overlap == 0", source)
        self.assertIn("return 0.0", source)
        self.assertIn("score = float(overlap)", source)

    def test_6c_irrelevant_only_returns_nothing(self):
        events = [ev("Gw makan nasi goreng", 1), ev("Gw main game", 2)]
        self.assertEqual(
            retrieve_episodic_events(events, "belajar frontend", now=NOW), []
        )

    def test_6d_top_n_is_respected(self):
        events = [
            ev(f"frontend Learning {i}", i + 1, event_id=f"e{i}") for i in range(10)
        ]
        selected = retrieve_episodic_events(events, "frontend learning", now=NOW)
        self.assertLessEqual(len(selected), EPISODIC_TOP_N)


class DynamicAgeTest(unittest.TestCase):
    """7: age labels are rendered, never persisted."""

    def test_7_age_label_changes_with_now(self):
        event = ev("Gw belajar frontend", 26, event_id="x")
        early = render_episodic_context([event], now=NOW, tz=JKT)
        later = render_episodic_context([event], now=NOW + timedelta(days=4), tz=JKT)
        self.assertNotEqual(early, later, "labels must be computed at render time")
        self.assertIn("Yesterday", early)
        self.assertIn("5 days ago", later)
        self.assertEqual(early, render_episodic_context([event], now=NOW, tz=JKT))

    def test_7b_render_is_deterministic_for_a_fixed_now(self):
        event = ev("Gw belajar frontend", 3)
        a = render_episodic_context([event], now=NOW, tz=JKT)
        b = render_episodic_context([event], now=NOW, tz=JKT)
        self.assertEqual(a, b)

    def test_7c_no_relative_label_is_persisted_as_fact(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                append_episodic_event(
                    "c",
                    {
                        "event_text": "Gw belajar frontend",
                        "occurred_at": (NOW - timedelta(hours=26)).isoformat(),
                        "session_uid": "s",
                        "source": "conversation",
                        "tz": JKT,
                    },
                )
                raw = pathlib.Path("episodic", "c.json").read_text(encoding="utf-8")
                for relative in ("today", "yesterday", "2 days ago", "ago"):
                    self.assertNotIn(f'"{relative}', raw)
                stored = load_episodic_events("c")[0]
                self.assertTrue(stored["occurred_at"].endswith("+00:00"))
            finally:
                os.chdir(previous)


class TemporalParsingTest(unittest.TestCase):
    """8 + 9 + 10: relative text resolves to absolute, restart-safe."""

    def test_8_semalam_resolves_to_yesterday_local_start(self):
        from zoneinfo import ZoneInfo

        moment = datetime(2026, 10, 3, 15, 0, tzinfo=UTC)  # 22:00 JKT
        resolved = resolve_occurred_at("Gw belajar frontend semalam", moment, JKT)
        self.assertIsNotNone(resolved)
        local = datetime.fromisoformat(resolved).astimezone(ZoneInfo(JKT))
        self.assertEqual((local.year, local.month, local.day), (2026, 10, 2))
        self.assertEqual(local.hour, 0)

    def test_9_n_hari_lalu_resolves_to_that_day(self):
        from zoneinfo import ZoneInfo

        moment = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
        zone = ZoneInfo(JKT)
        for text, expected_day in (
            ("Gw belajar frontend 4 hari lalu", 29),
            ("Gw belajar frontend 2 hari lalu", 1),
        ):
            with self.subTest(text=text):
                resolved = resolve_occurred_at(text, moment, JKT)
                self.assertIsNotNone(resolved)
                local = datetime.fromisoformat(resolved).astimezone(zone)
                self.assertEqual(local.day, expected_day)
                self.assertEqual(local.hour, 0)

    def test_9b_vague_time_yields_none_not_an_invented_stamp(self):
        moment = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
        self.assertIsNone(resolve_occurred_at("Gw belajar frontend", moment, JKT))
        self.assertIsNone(resolve_occurred_at("", moment, JKT))

    def test_9c_absolute_text_is_preserved(self):
        moment = datetime(2026, 10, 3, 12, 0, tzinfo=UTC)
        resolved = resolve_occurred_at(
            "Gw belajar frontend 2 Oktober 2026 jam 14:00", moment, JKT
        )
        self.assertIsNotNone(resolved)

    def test_10_restart_never_rewrites_occurred_at(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                stored_here = (NOW - timedelta(days=4, hours=2)).isoformat()
                append_episodic_event(
                    "c",
                    {
                        "event_text": "Gw belajar frontend",
                        "occurred_at": stored_here,
                        "session_uid": "s",
                        "source": "conversation",
                        "tz": JKT,
                    },
                )
                # simulate any number of "restarts"
                for _ in range(3):
                    reloaded = load_episodic_events("c")
                    self.assertEqual(reloaded[0]["occurred_at"], stored_here)
                self.assertEqual(
                    json.loads(
                        pathlib.Path("episodic", "c.json").read_text(encoding="utf-8")
                    )[0]["occurred_at"],
                    stored_here,
                )
            finally:
                os.chdir(previous)

    def test_10b_invalid_stored_stamp_degrades_to_none_never_to_now(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                pathlib.Path("episodic").mkdir(parents=True, exist_ok=True)
                pathlib.Path("episodic", "c.json").write_text(
                    json.dumps([dict(ev("x", 1), occurred_at="rubbish")]),
                    encoding="utf-8",
                )
                row = load_episodic_events("c")[0]
                self.assertIsNone(row["occurred_at"])
            finally:
                os.chdir(previous)


class BudgetAndPrivacyTest(unittest.TestCase):
    """11 + 12 + 13."""

    def test_11_episodic_block_is_bounded(self):
        events = [
            ev(
                f"frontend learning event number {i} " + "detail " * 20,
                i + 1,
                event_id=f"e{i}",
            )
            for i in range(30)
        ]
        selected = retrieve_episodic_events(events, "frontend learning", now=NOW)
        block = render_episodic_context(selected, now=NOW, tz=JKT)
        self.assertLessEqual(len(selected), EPISODIC_TOP_N)
        from src.open_llm_vtuber.agent.context_window import estimate_tokens

        self.assertLessEqual(estimate_tokens(block), EPISODIC_MAX_TOKENS + 40)

    def test_11b_whole_store_never_reaches_the_prompt(self):
        events = [ev(f"frontend {i}", i + 1, event_id=f"e{i}") for i in range(50)]
        selected = retrieve_episodic_events(events, "frontend", now=NOW)
        block = render_episodic_context(selected, now=NOW, tz=JKT)
        self.assertLessEqual(len(selected), EPISODIC_TOP_N)
        self.assertLess(len(block), sum(len(e["event_text"]) for e in events))

    def test_12_retrieval_makes_no_llm_call(self):
        for fn in (
            em.retrieve_episodic_events,
            em.render_episodic_context,
            em.resolve_occurred_at,
        ):
            with self.subTest(fn=fn.__name__):
                source = inspect.getsource(fn)
                for banned in (
                    "chat_completion",
                    "requests",
                    "httpx",
                    "openai",
                    "async def",
                ):
                    self.assertNotIn(banned, source)

    def test_12b_retrieval_is_deterministic(self):
        events = [ev(f"frontend {i}", i, event_id=f"e{i}") for i in range(6)]
        runs = [
            [e["id"] for e in retrieve_episodic_events(events, "frontend", now=NOW)]
            for _ in range(4)
        ]
        self.assertEqual(len(set(map(tuple, runs))), 1)

    def test_13_telemetry_never_logs_event_content(self):
        import src.open_llm_vtuber.agent.agents.basic_memory_agent as m

        source = inspect.getsource(m.BasicMemoryAgent._episodic_selection)
        log_calls = _logger_statements(source)
        joined = "\n".join(log_calls)
        self.assertTrue(joined.strip(), "telemetry statements must exist")
        self.assertIn("events=", joined)
        self.assertIn("selected=", joined)
        self.assertIn("elapsed_ms=", joined)
        # Only counts/lengths may be logged. Passing the query VALUE or any
        # event field is forbidden; `query_chars`/`len(...)` are safe.
        for banned in (
            "event_text",
            "selected[",
            "events[",
            ".content",
            "authorization",
            "password",
            "token",
            "join(",
        ):
            self.assertNotIn(banned, joined, f"telemetry must not log {banned}")
        # the INFO statement may only carry counts and timing
        info_blocks = [b for b in log_calls if "events=" in b and "selected=" in b]
        self.assertTrue(info_blocks, "an INFO telemetry statement is required")
        for statement in info_blocks:
            for allowed in ("len(events)", "len(selected)", "elapsed_ms"):
                self.assertIn(allowed, statement)
            self.assertIn("events={}", statement)
            self.assertIn("selected={}", statement)

    def test_13b_debug_branch_also_logs_counts_only(self):
        import src.open_llm_vtuber.agent.agents.basic_memory_agent as m

        source = inspect.getsource(m.BasicMemoryAgent._episodic_selection)
        for statement in _logger_statements(source):
            for banned in ("event_text", "content", "token"):
                self.assertNotIn(banned, statement)

    def test_13c_noisy_log_only_when_something_was_selected(self):
        import src.open_llm_vtuber.agent.agents.basic_memory_agent as m

        source = inspect.getsource(m.BasicMemoryAgent._episodic_selection)
        self.assertIn("if selected:", source)
        self.assertIn("logger.info", source)


class SchemaBoundaryTest(unittest.TestCase):
    """The four memory kinds must not be mixed."""

    def test_event_schema_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            previous = os.getcwd()
            os.chdir(tmp)
            try:
                append_episodic_event(
                    "c",
                    {
                        "event_text": "t",
                        "occurred_at": NOW.isoformat(),
                        "session_uid": "s",
                        "source": "conversation",
                        "tz": JKT,
                    },
                )
                stored = load_episodic_events("c")[0]
                self.assertEqual(
                    sorted(stored),
                    [
                        "created_at",
                        "event_text",
                        "id",
                        "occurred_at",
                        "session_uid",
                        "source",
                        "tz",
                    ],
                )
            finally:
                os.chdir(previous)

    def test_no_new_store_or_retrieval_system(self):
        module = pathlib.Path(em.__file__).read_text(encoding="utf-8")
        self.assertNotIn("sqlite", module.lower())
        self.assertNotIn("embedding", module.lower())
        self.assertNotIn("faiss", module.lower())
        self.assertEqual(module.count("def retrieve_episodic_events"), 1)

    def test_episodic_block_is_separate_from_long_term_memory(self):
        import src.open_llm_vtuber.agent.agents.basic_memory_agent as m

        source = inspect.getsource(m.BasicMemoryAgent._relationship_system_prompt)
        self.assertIn("_episodic_context_for_prompt", source)
        self.assertIn("build_character_memory_context", source)


if __name__ == "__main__":
    unittest.main()
