"""Episodic search + stable pagination (audit finding C).

The overview only carries the latest N episodic events, so an event that is
old by event time can still be recalled by the retrieval layer while being
invisible in the Memory Manager. These tests pin the fix: a read-only
newest-first search with a durable keyset cursor, over a store with more than
50 synthetic events, so an old event is findable and paging never skips or
duplicates a row even when events are appended or deleted between pages.

All fixtures are synthetic and live in a temporary working directory; no
production or personal data is touched.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.open_llm_vtuber import episodic_memory as em
from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.chat_history_manager import create_new_history
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig

CONF = "episodic-search"
BASE = datetime(2026, 1, 1, tzinfo=timezone.utc)


def seed(text, day_offset):
    return em.append_episodic_event(
        CONF,
        {
            "event_text": text,
            "occurred_at": (BASE + timedelta(days=day_offset)).isoformat(),
            "session_uid": "search-session",
            "source": "conversation",
        },
    )


def seed_many(count):
    ids = []
    for i in range(count):
        stored = seed(f"JOURNAL catatan nomor {i} topik unik{i}", i)
        assert stored is not None
        ids.append(stored["id"])
    return ids


def make_agent():
    agent = BasicMemoryAgent(
        llm=SimpleNamespace(model="search", max_tokens=1, chat_completion=None),
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
    agent.set_memory_from_history(CONF, history_uid)
    return agent


class SearchBase(unittest.TestCase):
    def setUp(self):
        self._old = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()


class SearchFindabilityTest(SearchBase):
    def test_old_event_outside_the_overview_window_is_findable(self):
        agent = make_agent()
        ids = seed_many(60)
        # The overview shows only the newest 50; the oldest events are hidden.
        overview = agent.list_memory_overview(episodic_limit=50)
        visible = {row["id"] for row in overview["episodic"]["events"]}
        self.assertEqual(overview["episodic"]["total"], 60)
        self.assertNotIn(ids[0], visible)
        # The search reaches it.
        found = agent.search_episodic_memory("nomor 0 topik unik0")
        self.assertEqual(found["total"], 1)
        self.assertEqual(found["events"][0]["id"], ids[0])
        self.assertEqual(found["status"], "ok")

    def test_search_preserves_durable_ids_across_calls(self):
        agent = make_agent()
        seed_many(55)
        first = agent.search_episodic_memory("topik unik7")
        second = agent.search_episodic_memory("topik unik7")
        self.assertTrue(first["events"][0]["id"])
        self.assertEqual(first["events"][0]["id"], second["events"][0]["id"])

    def test_no_matches_reports_empty_not_error(self):
        agent = make_agent()
        seed_many(5)
        found = agent.search_episodic_memory("tidak ada kata seperti ini")
        self.assertEqual(found["events"], [])
        self.assertEqual(found["total"], 0)
        self.assertEqual(found["status"], "empty")

    def test_empty_query_browses_newest_first(self):
        agent = make_agent()
        ids = seed_many(5)
        found = agent.search_episodic_memory("", limit=3)
        self.assertEqual(found["total"], 5)
        self.assertEqual([e["id"] for e in found["events"]], list(reversed(ids[-3:])))
        self.assertTrue(found["has_more"])

    def test_ordering_is_by_event_time_not_storage_order(self):
        agent = make_agent()
        # Stored oldest-first below, but the newest experience is appended last.
        seed("AKTIVITAS kegiatan lama", 1)
        seed("AKTIVITAS kegiatan baru", 40)
        seed("AKTIVITAS kegiatan tengah", 20)
        found = agent.search_episodic_memory("AKTIVITAS")
        self.assertEqual(
            [e["event_text"] for e in found["events"]],
            [
                "AKTIVITAS kegiatan baru",
                "AKTIVITAS kegiatan tengah",
                "AKTIVITAS kegiatan lama",
            ],
        )


class PaginationTest(SearchBase):
    def test_paging_covers_everything_without_duplicates(self):
        agent = make_agent()
        ids = list(reversed(seed_many(60)))  # expected newest-first
        seen = []
        cursor = None
        pages = 0
        while True:
            page = agent.search_episodic_memory("", limit=7, cursor=cursor)
            seen.extend(e["id"] for e in page["events"])
            pages += 1
            self.assertEqual(page["total"], 60)
            if not page["has_more"]:
                self.assertIsNone(page["next_cursor"])
                break
            cursor = page["next_cursor"]
            self.assertLess(pages, 20)
        self.assertEqual(seen, ids)
        self.assertEqual(len(seen), len(set(seen)))

    def test_new_event_between_pages_neither_duplicates_nor_skips(self):
        agent = make_agent()
        ids = list(reversed(seed_many(30)))  # newest-first
        page1 = agent.search_episodic_memory("", limit=10)
        self.assertEqual([e["id"] for e in page1["events"]], ids[:10])
        # A sibling append lands newer than the cursor (sorts before page 1).
        seed("JOURNAL catatan nomor 999 topik unik999", 500)
        page2 = agent.search_episodic_memory("", limit=10, cursor=page1["next_cursor"])
        page2_ids = [e["id"] for e in page2["events"]]
        self.assertEqual(page2_ids, ids[10:20])
        self.assertFalse(set(page2_ids) & set(ids[:10]))
        # total reflects the query-independent match count at call time
        self.assertEqual(page2["total"], 31)

    def test_deleted_event_between_pages_is_absent_and_never_duplicated(self):
        agent = make_agent()
        ids = list(reversed(seed_many(30)))
        page1 = agent.search_episodic_memory("", limit=10)
        # Delete one event that would have been on page 2.
        self.assertTrue(agent.delete_episodic_event(ids[12]))
        page2 = agent.search_episodic_memory("", limit=10, cursor=page1["next_cursor"])
        page2_ids = [e["id"] for e in page2["events"]]
        self.assertNotIn(ids[12], page2_ids)
        self.assertFalse(set(page2_ids) & set(ids[:10]))
        self.assertEqual(page2["total"], 29)
        self.assertEqual(len(page2_ids), len(set(page2_ids)))
        # The page is backfilled from the next row so it still fills up to
        # `limit`; the removed row simply never appears.
        expected = [i for i in ids[10:20] if i != ids[12]]
        self.assertEqual(page2_ids[: len(expected)], expected)
        self.assertEqual(page2_ids[len(expected)], ids[20])


class SearchFailureStatusTest(SearchBase):
    def test_corrupt_store_reports_error_not_empty(self):
        agent = make_agent()
        os.makedirs("episodic", exist_ok=True)
        with open(f"episodic/{CONF}.json", "w", encoding="utf-8") as handle:
            handle.write("{not valid json")
        found = agent.search_episodic_memory("apa saja")
        self.assertEqual(found["status"], "error")
        self.assertEqual(found["events"], [])

    def test_missing_store_is_empty(self):
        agent = make_agent()
        found = agent.search_episodic_memory("apa saja")
        self.assertEqual(found["status"], "empty")
        self.assertEqual(found["total"], 0)


class SearchHandlerTest(SearchBase):
    def setUp(self):
        super().setUp()
        from src.open_llm_vtuber.websocket_handler import WebSocketHandler

        self.agent = make_agent()
        history_uid = create_new_history(CONF)
        self.handler = WebSocketHandler(default_context_cache=None)
        self.handler.client_contexts["client"] = SimpleNamespace(
            history_uid=history_uid,
            agent_engine=self.agent,
            character_config=SimpleNamespace(conf_uid=CONF),
            user_timezone="Asia/Jakarta",
        )

    def _socket(self):
        import json as _json

        class _FakeWS:
            def __init__(self):
                self.messages = []

            async def send_text(self, payload):
                self.messages.append(_json.loads(payload))

        return _FakeWS()

    def test_search_round_trip_and_append_flag(self):
        import asyncio as _asyncio

        seed_many(12)
        ws = self._socket()
        _asyncio.run(
            self.handler._handle_search_episodic_events(
                ws, "client", {"query": "", "limit": 5}
            )
        )
        reply = ws.messages[-1]
        self.assertEqual(reply["type"], "episodic-search-results")
        self.assertEqual(reply["query"], "")
        self.assertFalse(reply["append"])
        self.assertEqual(reply["total"], 12)
        self.assertEqual(len(reply["events"]), 5)
        self.assertTrue(reply["has_more"])
        cursor = reply["next_cursor"]
        self.assertTrue(cursor)

        ws2 = self._socket()
        _asyncio.run(
            self.handler._handle_search_episodic_events(
                ws2, "client", {"query": "", "limit": 5, "cursor": cursor}
            )
        )
        reply2 = ws2.messages[-1]
        self.assertTrue(reply2["append"])
        self.assertFalse(
            {e["id"] for e in reply["events"]} & {e["id"] for e in reply2["events"]}
        )

    def test_search_query_round_trip(self):
        import asyncio as _asyncio

        seed_many(8)
        ws = self._socket()
        _asyncio.run(
            self.handler._handle_search_episodic_events(
                ws, "client", {"query": "nomor 3 topik unik3"}
            )
        )
        reply = ws.messages[-1]
        self.assertEqual(reply["total"], 1)
        self.assertIn("unik3", reply["events"][0]["event_text"])


if __name__ == "__main__":
    unittest.main()
