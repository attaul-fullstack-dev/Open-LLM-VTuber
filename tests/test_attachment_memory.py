"""Attachment memory — storage, honesty, persistence, retrieval, deletion.

Covers: metadata/summary storage, failed attachments never producing false
memories, real-disk persistence across simulated refresh/restart (including
one genuine cross-process check), cross-session retrieval, deletion with a
copy-source audit, content-hash dedup, storage-failure safety, agent capture
wiring, the single-conversation scheduling hook, and thin WS handlers.
No network, no real LLM; the describe call is served by a canned fake.
"""

import asyncio
import json
import os
import subprocess
import sys
import tempfile
import unittest
from datetime import datetime, timezone
from types import SimpleNamespace

from src.open_llm_vtuber.attachment_memory import (
    append_attachment_memory,
    build_attachment_summary_prompt,
    clear_attachment_memories,
    content_hash_for,
    delete_attachment_memories_for_session,
    delete_attachment_memory,
    describe_and_store_attachments,
    find_attachment_memory_copies,
    load_attachment_memories,
    parse_attachment_summary,
    purge_attachment_memory,
    render_attachment_context,
    retrieve_attachment_memories,
    save_attachment_memories,
)
from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    delete_history,
    get_history,
    store_message,
    update_summary_metadata,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig
from src.open_llm_vtuber import episodic_memory as em
from src.open_llm_vtuber.episodic_memory import (
    delete_episodic_events_by_ids,
    load_episodic_events,
)
from src.open_llm_vtuber.character_state import (
    add_character_memory,
    load_character_state,
)
from src.open_llm_vtuber.world_state import get_world_state_path
from src.open_llm_vtuber.agent.input_types import (
    BatchInput,
    TextData,
    TextSource,
)

CONF = "attachmenttest"
DATA_URL = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
DATA_URL_2 = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////2wBDAf//////////////////////////////////////////////////////////////////////////////////////wAARCAABAAEDASIAAhEBAxEB/8QAFQABAQAAAAAAAAAAAAAAAAAAAAv/xAAUEAEAAAAAAAAAAAAAAAAAAAAA/8QAFQEBAQAAAAAAAAAAAAAAAAAAAAX/xAAUEQEAAAAAAAAAAAAAAAAAAAAA/9oACAEBAAE/AP/EABQRAQAAAAAAAAAAAAAAAAAAAMD/2gAIAQMBAT8AH//Z"


def record(
    name="kucing.png",
    data=DATA_URL,
    status="processed",
    summary="Seekor kucing oranye tidur di sofa.",
    session="sess-a",
    request="req-1",
):
    return {
        "filename": name,
        "mime_type": "image/png",
        "source": "upload",
        "content_hash": content_hash_for(data),
        "received_at": "2026-10-09T10:00:00+00:00",
        "session_uid": session,
        "request_id": request,
        "status": status,
        "summary": summary,
        "summary_model": "test-model",
        "tz": "Asia/Jakarta",
    }


def image_item(name="kucing.png", data=DATA_URL, source="upload"):
    return {
        "name": name,
        "mime_type": "image/png",
        "source": source,
        "data": data,
    }


class DescribeFakeLLM:
    """Canned describe-call provider: records calls, yields fixed chunks."""

    model = "describe-fake"
    max_tokens = None

    def __init__(self, chunks=None, error=None):
        self.calls = []
        self.chunks = list(chunks or [])
        self.error = error

    async def chat_completion(self, messages, system=None, tools=None):
        self.calls.append({"messages": messages, "system": system})
        if self.error is not None:
            raise self.error
        for chunk in self.chunks:
            yield chunk


def describe_response(*summaries):
    items = [
        {"index": i, "visible": True, "summary": text}
        for i, text in enumerate(summaries)
    ]
    return [json.dumps({"items": items})]


class TmpCwdTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()


class StorageTest(TmpCwdTestCase):
    def test_append_load_roundtrip_keeps_metadata(self):
        stored = append_attachment_memory(CONF, record())
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertTrue(stored["id"])
        self.assertTrue(stored["created_at"])
        loaded = load_attachment_memories(CONF)
        self.assertEqual(len(loaded), 1)
        row = loaded[0]
        self.assertEqual(row["filename"], "kucing.png")
        self.assertEqual(row["mime_type"], "image/png")
        self.assertEqual(row["source"], "upload")
        self.assertEqual(row["session_uid"], "sess-a")
        self.assertEqual(row["request_id"], "req-1")
        self.assertEqual(row["status"], "processed")
        self.assertEqual(row["summary"], "Seekor kucing oranye tidur di sofa.")
        self.assertEqual(row["content_hash"], content_hash_for(DATA_URL))
        self.assertTrue(row["received_at"])
        # Raw image bytes are never persisted.
        raw = open(f"attachment_memory/{CONF}.json", encoding="utf-8").read()
        self.assertNotIn("iVBORw0KGgo", raw)

    def test_failed_record_has_empty_summary(self):
        stored = append_attachment_memory(
            CONF, record(status="failed", summary="should be dropped")
        )
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertEqual(stored["status"], "failed")
        self.assertEqual(stored["summary"], "")
        self.assertEqual(load_attachment_memories(CONF)[0]["summary"], "")

    def test_processed_without_summary_rejected(self):
        self.assertIsNone(
            append_attachment_memory(CONF, record(status="processed", summary="  "))
        )
        self.assertEqual(load_attachment_memories(CONF), [])

    def test_rejects_bad_status_hash_and_mime(self):
        self.assertIsNone(append_attachment_memory(CONF, record(status="maybe")))
        bad = record()
        bad["content_hash"] = "zzz"
        self.assertIsNone(append_attachment_memory(CONF, bad))
        bad = record()
        bad["mime_type"] = "text/plain"
        self.assertIsNone(append_attachment_memory(CONF, bad))
        self.assertEqual(load_attachment_memories(CONF), [])

    def test_restart_persistence(self):
        append_attachment_memory(CONF, record())
        append_attachment_memory(CONF, record(name="motor.jpg", data=DATA_URL_2))
        # Fresh load = post-restart read path.
        self.assertEqual(len(load_attachment_memories(CONF)), 2)

    def test_corrupt_and_missing(self):
        self.assertEqual(load_attachment_memories("nope"), [])
        os.makedirs("attachment_memory", exist_ok=True)
        with open("attachment_memory/c1.json", "w", encoding="utf-8") as handle:
            handle.write("{not json")
        self.assertEqual(load_attachment_memories("c1"), [])

    def test_store_path_is_directory_fails_soft(self):
        os.makedirs("attachment_memory", exist_ok=True)
        os.makedirs("attachment_memory/c1.json", exist_ok=True)
        self.assertFalse(save_attachment_memories("c1", [record()]))
        self.assertEqual(load_attachment_memories("c1"), [])

    def test_save_failure_on_append_returns_none(self):
        os.makedirs("attachment_memory", exist_ok=True)
        os.makedirs("attachment_memory/c1.json", exist_ok=True)
        self.assertIsNone(append_attachment_memory("c1", record()))


class DedupTest(TmpCwdTestCase):
    def test_same_bytes_not_stored_twice(self):
        first = append_attachment_memory(CONF, record())
        self.assertIsNotNone(first)
        # Same bytes under a different filename is still the same image.
        self.assertIsNone(append_attachment_memory(CONF, record(name="renamed.png")))
        self.assertEqual(len(load_attachment_memories(CONF)), 1)

    def test_failed_upgraded_by_successful_retry(self):
        failed = append_attachment_memory(CONF, record(status="failed", summary=""))
        self.assertIsNotNone(failed)
        stored = append_attachment_memory(CONF, record())
        self.assertIsNotNone(stored)
        assert stored is not None
        rows = load_attachment_memories(CONF)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "processed")
        self.assertNotEqual(rows[0]["id"], failed["id"])

    def test_different_bytes_both_stored(self):
        append_attachment_memory(CONF, record())
        append_attachment_memory(
            CONF, record(name="motor.jpg", data=DATA_URL_2, summary="Motor merah.")
        )
        self.assertEqual(len(load_attachment_memories(CONF)), 2)


class PromptParseTest(unittest.TestCase):
    def test_prompt_carries_all_images_in_order(self):
        items = [image_item("a.png"), image_item("b.png", data=DATA_URL_2)]
        system, messages = build_attachment_summary_prompt(
            items, "lihat ini", datetime(2026, 10, 9, 10, 0, tzinfo=timezone.utc), None
        )
        self.assertIn("JSON", system)
        content = messages[0]["content"]
        images = [c for c in content if c.get("type") == "image_url"]
        self.assertEqual(len(images), 2)
        self.assertEqual(images[0]["image_url"]["url"], DATA_URL)
        self.assertEqual(images[1]["image_url"]["url"], DATA_URL_2)
        self.assertIn("[0] a.png", content[0]["text"])

    def test_parse_valid_batch(self):
        items, reason = parse_attachment_summary(
            json.dumps(
                {
                    "items": [
                        {"index": 0, "visible": True, "summary": "Kucing."},
                        {"index": 1, "visible": False, "summary": None},
                    ]
                }
            ),
            2,
        )
        self.assertIsNone(reason)
        self.assertTrue(items[0]["visible"])
        self.assertEqual(items[0]["summary"], "Kucing.")
        self.assertFalse(items[1]["visible"])

    def test_parse_degrades_bad_entries_to_invisible(self):
        items, reason = parse_attachment_summary(
            json.dumps({"items": [{"index": 0}, "nope"]}), 3
        )
        self.assertIsNone(reason)
        self.assertEqual(len(items), 3)
        self.assertTrue(all(item["visible"] is False for item in items))

    def test_parse_garbage_returns_reason(self):
        items, reason = parse_attachment_summary("bukan json sama sekali", 1)
        self.assertEqual(items, [])
        self.assertEqual(reason, "invalid_json")
        items, reason = parse_attachment_summary("", 1)
        self.assertEqual(reason, "empty_response")


class DescribeTest(TmpCwdTestCase):
    def test_describe_stores_processed_per_item(self):
        llm = DescribeFakeLLM(describe_response("Kucing oranye.", "Motor merah."))
        stored = asyncio.run(
            describe_and_store_attachments(
                llm.chat_completion,
                CONF,
                [image_item("a.png"), image_item("b.png", data=DATA_URL_2)],
                "lihat dua gambar ini",
                "sess-a",
                "req-9",
                datetime(2026, 10, 9, 10, 0, tzinfo=timezone.utc),
                "Asia/Jakarta",
                summary_model="describe-fake",
            )
        )
        self.assertEqual(len(llm.calls), 1)
        self.assertEqual(len(stored), 2)
        self.assertTrue(all(r["status"] == "processed" for r in stored))
        self.assertEqual(stored[0]["summary"], "Kucing oranye.")
        self.assertEqual(stored[1]["summary"], "Motor merah.")
        self.assertEqual(stored[0]["session_uid"], "sess-a")
        self.assertEqual(stored[0]["request_id"], "req-9")
        self.assertEqual(stored[0]["summary_model"], "describe-fake")
        self.assertEqual(len(load_attachment_memories(CONF)), 2)

    def test_invisible_item_becomes_failed_without_summary(self):
        llm = DescribeFakeLLM(
            [
                json.dumps(
                    {
                        "items": [
                            {"index": 0, "visible": True, "summary": "Kucing."},
                            {"index": 1, "visible": False, "summary": None},
                        ]
                    }
                )
            ]
        )
        stored = asyncio.run(
            describe_and_store_attachments(
                llm.chat_completion,
                CONF,
                [image_item("a.png"), image_item("b.png", data=DATA_URL_2)],
                "lihat ini",
                "sess-a",
                "req-9",
                datetime(2026, 10, 9, 10, 0, tzinfo=timezone.utc),
            )
        )
        by_name = {r["filename"]: r for r in stored}
        self.assertEqual(by_name["a.png"]["status"], "processed")
        self.assertEqual(by_name["b.png"]["status"], "failed")
        self.assertEqual(by_name["b.png"]["summary"], "")

    def test_llm_error_stores_failed_not_fabricated(self):
        llm = DescribeFakeLLM(error=RuntimeError("provider down"))
        stored = asyncio.run(
            describe_and_store_attachments(
                llm.chat_completion,
                CONF,
                [image_item("a.png")],
                "lihat ini",
                "sess-a",
                "req-9",
                datetime(2026, 10, 9, 10, 0, tzinfo=timezone.utc),
            )
        )
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["status"], "failed")
        self.assertEqual(stored[0]["summary"], "")
        # Honest failure note: the file existed, nothing was read.
        self.assertEqual(stored[0]["filename"], "a.png")

    def test_malformed_response_stores_failed(self):
        llm = DescribeFakeLLM(["definitely not json {{{"])
        stored = asyncio.run(
            describe_and_store_attachments(
                llm.chat_completion,
                CONF,
                [image_item("a.png")],
                "lihat ini",
                "sess-a",
                "req-9",
                datetime(2026, 10, 9, 10, 0, tzinfo=timezone.utc),
            )
        )
        self.assertEqual(len(stored), 1)
        self.assertEqual(stored[0]["status"], "failed")
        self.assertEqual(stored[0]["summary"], "")

    def test_empty_items_stores_nothing(self):
        llm = DescribeFakeLLM(describe_response("x"))
        stored = asyncio.run(
            describe_and_store_attachments(
                llm.chat_completion,
                CONF,
                [],
                "tanpa gambar",
                "sess-a",
                "req-9",
                datetime(2026, 10, 9, 10, 0, tzinfo=timezone.utc),
            )
        )
        self.assertEqual(stored, [])
        self.assertEqual(llm.calls, [])


class RetrievalTest(TmpCwdTestCase):
    def _seed(self):
        append_attachment_memory(CONF, record())
        append_attachment_memory(
            CONF,
            record(
                name="motor.jpg",
                data=DATA_URL_2,
                summary="Motor merah diparkir di gang.",
                session="sess-b",
            ),
        )
        append_attachment_memory(
            CONF, record(status="failed", summary="", session="sess-c")
        )
        return load_attachment_memories(CONF)

    def test_retrieves_relevant_processed_only(self):
        records = self._seed()
        found = retrieve_attachment_memories(records, "kucing oranye di sofa")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["filename"], "kucing.png")

    def test_failed_records_never_retrieved(self):
        records = self._seed()
        # The failed record shares the filename but has no summary.
        found = retrieve_attachment_memories(records, "kucing.png")
        self.assertTrue(all(r["status"] == "processed" for r in found))
        self.assertTrue(all(r["summary"] for r in found))

    def test_empty_query_or_store_returns_none(self):
        self.assertEqual(retrieve_attachment_memories(self._seed(), "   "), [])
        self.assertEqual(retrieve_attachment_memories([], "kucing"), [])
        self.assertEqual(retrieve_attachment_memories(self._seed(), "topik asing"), [])

    def test_cross_session_retrieval(self):
        self._seed()
        # A brand-new session asks about an older session's attachment.
        found = retrieve_attachment_memories(
            load_attachment_memories(CONF), "motor merah gang"
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["session_uid"], "sess-b")

    def test_render_budget_header_and_honesty(self):
        records = self._seed()
        # Retrieval returns up to TOP_N; render must stay within budget.
        text = render_attachment_context(records, max_tokens=300)
        self.assertIn("attachment memories", text.lower())
        self.assertIn("kucing.png", text)
        self.assertIn("original", text.lower())
        from src.open_llm_vtuber.agent.context_window import estimate_tokens

        self.assertLessEqual(estimate_tokens(text), 300)
        # Failed records are never rendered even if passed directly.
        failed_only = [r for r in records if r["status"] == "failed"]
        self.assertEqual(render_attachment_context(failed_only), "")
        self.assertEqual(render_attachment_context([]), "")


class DeleteConsistencyTest(TmpCwdTestCase):
    def _seed_two_sessions(self):
        append_attachment_memory(CONF, record(session="sess-a"))
        append_attachment_memory(
            CONF,
            record(name="b.png", data=DATA_URL_2, summary="B.", session="sess-a"),
        )
        append_attachment_memory(
            CONF,
            record(
                name="c.png",
                data=DATA_URL + "x",
                summary="C.",
                session="sess-b",
            ),
        )

    def test_delete_single_and_missing(self):
        stored = append_attachment_memory(CONF, record())
        assert stored is not None
        self.assertTrue(delete_attachment_memory(CONF, stored["id"]))
        self.assertFalse(delete_attachment_memory(CONF, stored["id"]))
        self.assertFalse(delete_attachment_memory(CONF, ""))
        self.assertEqual(load_attachment_memories(CONF), [])

    def test_clear_returns_count(self):
        self._seed_two_sessions()
        self.assertEqual(clear_attachment_memories(CONF), 3)
        self.assertEqual(clear_attachment_memories(CONF), 0)

    def test_session_cascade(self):
        self._seed_two_sessions()
        self.assertEqual(delete_attachment_memories_for_session(CONF, "sess-a"), 2)
        remaining = load_attachment_memories(CONF)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["session_uid"], "sess-b")
        self.assertEqual(delete_attachment_memories_for_session(CONF, "nope"), 0)
        self.assertEqual(delete_attachment_memories_for_session(CONF, ""), 0)

    def test_copy_audit_finds_all_sources(self):
        stored = append_attachment_memory(CONF, record())
        assert stored is not None
        # Episodic copy quoting the filename.
        em.append_episodic_event(
            CONF,
            {
                "event_text": "User showed a photo named kucing.png yesterday.",
                "session_uid": "sess-a",
            },
        )
        # Transcript copy.
        history_uid = create_new_history(CONF)
        store_message(CONF, history_uid, "human", "lihat kucing.png yang aku kirim")
        # Rolling-summary copy.
        update_summary_metadata(
            CONF,
            history_uid,
            expected_summarized_through=0,
            conversation_summary="Ringkasan: membahas kucing.png dan motor.",
            summarized_through=1,
            summary_updated_at="2026-10-09T11:00:00+00:00",
        )
        copies = find_attachment_memory_copies(CONF, record_id=stored["id"])
        self.assertIsNotNone(copies["attachment_memory"])
        self.assertEqual(len(copies["episodic"]), 1)
        self.assertEqual(copies["transcripts"], [history_uid])
        self.assertEqual(copies["summaries"], [history_uid])

    def test_deletion_boundary_is_honest(self):
        """Deleting the record + its history still leaves episodic copies.

        Episodic memory has no delete API by design, so the audit must keep
        reporting them instead of claiming a full wipe.
        """
        stored = append_attachment_memory(CONF, record())
        assert stored is not None
        em.append_episodic_event(
            CONF,
            {
                "event_text": "User showed a photo named kucing.png yesterday.",
                "session_uid": "sess-a",
            },
        )
        history_uid = create_new_history(CONF)
        store_message(CONF, history_uid, "human", "lihat kucing.png")
        self.assertTrue(delete_attachment_memory(CONF, stored["id"]))
        self.assertTrue(delete_history(CONF, history_uid))
        self.assertEqual(delete_attachment_memories_for_session(CONF, history_uid), 0)
        copies = find_attachment_memory_copies(CONF, filename="kucing.png")
        self.assertIsNone(copies["attachment_memory"])
        self.assertEqual(copies["transcripts"], [])
        self.assertEqual(copies["summaries"], [])
        # The episodic duplicate survives: deletion is scoped, not total.
        self.assertEqual(len(copies["episodic"]), 1)


class FakeAgentLLM(DescribeFakeLLM):
    model = "agent-test-model"


class FakeLive2D:
    def extract_emotion(self, _text):
        return []


def make_agent(conf_uid, history_uid, llm):
    agent = BasicMemoryAgent(
        llm=llm,
        system="persona Mili tetap aktif",
        live2d_model=FakeLive2D(),
        tts_preprocessor_config=TTSPreprocessorConfig(
            remove_special_char=True,
            translator_config={
                "translate_audio": False,
                "translate_provider": "deeplx",
            },
        ),
        context_window_override=2200,
    )
    agent.set_memory_from_history(conf_uid, history_uid)
    return agent


class AgentIntegrationTest(TmpCwdTestCase):
    def test_agent_capture_stores_record(self):
        history_uid = create_new_history(CONF)
        llm = FakeAgentLLM(describe_response("Kucing oranye tidur."))
        agent = make_agent(CONF, history_uid, llm)
        asyncio.run(
            agent.capture_attachment_memory(
                [image_item()], "kucingku lucu ya", history_uid, "req-7"
            )
        )
        rows = load_attachment_memories(CONF)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["status"], "processed")
        self.assertEqual(rows[0]["summary"], "Kucing oranye tidur.")
        self.assertEqual(rows[0]["request_id"], "req-7")
        self.assertEqual(rows[0]["summary_model"], "agent-test-model")
        # The describe call re-sent the image bytes (never stored).
        sent = llm.calls[0]["messages"][0]["content"]
        self.assertTrue(
            any(
                part.get("type") == "image_url" and part["image_url"]["url"] == DATA_URL
                for part in sent
            )
        )

    def test_agent_capture_skips_without_images_or_ids(self):
        history_uid = create_new_history(CONF)
        llm = FakeAgentLLM(describe_response("x"))
        agent = make_agent(CONF, history_uid, llm)
        asyncio.run(agent.capture_attachment_memory([], "halo", history_uid, "r"))
        self.assertEqual(llm.calls, [])
        self.assertEqual(load_attachment_memories(CONF), [])

    def test_agent_capture_never_raises_without_chat_fn(self):
        history_uid = create_new_history(CONF)
        agent = make_agent(CONF, history_uid, FakeAgentLLM([]))
        agent._llm = SimpleNamespace()
        asyncio.run(
            agent.capture_attachment_memory([image_item()], "lihat", history_uid, "r")
        )
        self.assertEqual(load_attachment_memories(CONF), [])

    def test_agent_selection_and_prompt_injection(self):
        history_uid = create_new_history(CONF)
        llm = FakeAgentLLM(describe_response("Kucing oranye tidur di sofa."))
        agent = make_agent(CONF, history_uid, llm)
        asyncio.run(
            agent.capture_attachment_memory(
                [image_item()], "kucingku", history_uid, "r"
            )
        )
        # New turn in the same store (refresh/new-session equivalent).
        agent._episodic_query = "kucing oranye sofa"
        selected = agent._attachment_selection()
        self.assertEqual(len(selected), 1)
        block = agent._attachment_context_for_prompt()
        self.assertIn("kucing.png", block)
        self.assertIn("Kucing oranye tidur di sofa.", block)
        system = agent._relationship_system_prompt("persona dasar")
        self.assertIn("Kucing oranye tidur di sofa.", system)

    def test_agent_selection_empty_without_match(self):
        history_uid = create_new_history(CONF)
        llm = FakeAgentLLM(describe_response("Kucing oranye."))
        agent = make_agent(CONF, history_uid, llm)
        asyncio.run(
            agent.capture_attachment_memory([image_item()], "kucing", history_uid, "r")
        )
        agent._episodic_query = "resep rendang ayam"
        self.assertEqual(agent._attachment_selection(), [])
        self.assertEqual(agent._attachment_context_for_prompt(), "")

    def test_agent_remove_and_clear_wrappers(self):
        history_uid = create_new_history(CONF)
        agent = make_agent(CONF, history_uid, FakeAgentLLM([]))
        stored = append_attachment_memory(CONF, record())
        assert stored is not None
        self.assertTrue(agent.remove_attachment_memory(stored["id"]))
        self.assertFalse(agent.remove_attachment_memory(stored["id"]))
        append_attachment_memory(CONF, record())
        self.assertEqual(agent.clear_attachment_memories(), 1)
        self.assertEqual(agent.list_attachment_memories(), [])


class RestartSimulationTest(TmpCwdTestCase):
    def test_cross_process_persistence(self):
        """Genuine restart: a NEW OS process reads the store file."""
        append_attachment_memory(CONF, record())
        append_attachment_memory(
            CONF, record(name="b.png", data=DATA_URL_2, summary="Motor.")
        )
        store_path = os.path.abspath(f"attachment_memory/{CONF}.json")
        repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        script = (
            "import json, os;"
            "os.chdir(%r);"
            % os.getcwd()
            + "import sys; sys.path.insert(0, %r);" % repo_src
            + "from src.open_llm_vtuber.attachment_memory import "
            "load_attachment_memories;"
            "rows = load_attachment_memories(%r);" % CONF + "print(json.dumps(["
            "(r['filename'], r['status'], r['summary']) for r in rows]))"
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        rows = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][0], "kucing.png")
        self.assertEqual(rows[0][1], "processed")
        self.assertIn("kucing", rows[0][2].lower())
        self.assertTrue(os.path.exists(store_path))


# ---------------------------------------------------------------------------
# Scheduling hook: the real single-conversation pipeline must fire the agent
# capture exactly once per image turn, with sanitized images + turn refs.
# ---------------------------------------------------------------------------

from src.open_llm_vtuber.agent.output_types import (  # noqa: E402
    Actions,
    DisplayText,
    SentenceOutput,
)
from src.open_llm_vtuber.conversations import (  # noqa: E402
    conversation_utils as cu_mod,
)
from src.open_llm_vtuber.conversations.single_conversation import (  # noqa: E402
    process_single_conversation,
)
from src.open_llm_vtuber import websocket_handler as wh_mod  # noqa: E402
from unittest.mock import AsyncMock, patch  # noqa: E402


def sentence(text):
    return SentenceOutput(
        display_text=DisplayText(text=text, name="Mili", avatar=""),
        tts_text=text,
        actions=Actions(),
    )


class HookFakeAgent:
    def __init__(self):
        self._memory = []
        self.capture = AsyncMock()

    async def chat(self, input_data):
        yield sentence("Oke, aku lihat gambarnya.")


async def _capture_shim(images, user_text, history_uid, request_id):
    raise AssertionError("must be replaced per test")


class SchedulingHookTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _context(self, history_uid, agent):
        return SimpleNamespace(
            history_uid=history_uid,
            user_timezone="Asia/Jakarta",
            voice_output_enabled=False,
            asr_engine=None,
            live2d_model=None,
            tts_engine=None,
            translate_engine=None,
            character_config=SimpleNamespace(
                conf_uid=CONF,
                character_name="Mili",
                avatar="",
                human_name="Human",
            ),
            agent_engine=agent,
        )

    async def _run_turn(self, context, text, images):
        sent = []

        async def _websocket_send(payload):
            sent.append(payload)

        with patch.object(cu_mod, "PLAYBACK_COMPLETE_TIMEOUT_S", 0.05):
            result = await process_single_conversation(
                context=context,
                websocket_send=_websocket_send,
                client_uid="hook-client",
                user_input=text,
                images=images,
                session_emoji="😊",
                metadata={},
            )
        return result, sent

    async def test_image_turn_schedules_capture_once(self):
        history_uid = create_new_history(CONF)
        agent = HookFakeAgent()
        agent.capture_attachment_memory = AsyncMock()
        context = self._context(history_uid, agent)
        images = [
            {
                "source": "upload",
                "data": DATA_URL,
                "mime_type": "image/png",
                "name": "kucing.png",
                "size": 70,
            }
        ]
        result, _ = await self._run_turn(context, "lihat kucingku", images)
        self.assertIn("aku lihat gambarnya", result)
        # Fire-and-forget task may still be in flight: yield once.
        await asyncio.sleep(0.05)
        agent.capture_attachment_memory.assert_awaited_once()
        args = agent.capture_attachment_memory.await_args.args
        self.assertEqual(args[0], images)
        self.assertEqual(args[1], "lihat kucingku")
        self.assertEqual(args[2], history_uid)
        self.assertTrue(args[3])

    async def test_text_only_turn_never_schedules_capture(self):
        history_uid = create_new_history(CONF)
        agent = HookFakeAgent()
        agent.capture_attachment_memory = AsyncMock()
        context = self._context(history_uid, agent)
        await self._run_turn(context, "halo mili", None)
        await asyncio.sleep(0.05)
        agent.capture_attachment_memory.assert_not_awaited()


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send_text(self, payload):
        self.sent.append(json.loads(payload))


def make_ws_handler(agent, history_uid="hist-1"):
    handler = wh_mod.WebSocketHandler.__new__(wh_mod.WebSocketHandler)
    context = SimpleNamespace(
        history_uid=history_uid,
        user_timezone="Asia/Jakarta",
        character_config=SimpleNamespace(conf_uid=CONF),
        agent_engine=agent,
    )
    handler.client_contexts = {"c1": context}
    handler._proactive_states = {}
    handler._cancel_proactive_timer = AsyncMock()
    handler._pause_proactive_for_maintenance = AsyncMock()
    handler._resume_proactive_after_maintenance = AsyncMock()
    handler._update_user_timezone = lambda context, data: None
    handler._activate_proactive_for_history = AsyncMock()
    return handler


class WsHandlerTest(TmpCwdTestCase):
    def test_fetch_lists_records(self):
        append_attachment_memory(CONF, record())
        agent = SimpleNamespace(
            list_attachment_memories=lambda: load_attachment_memories(CONF)
        )
        handler = make_ws_handler(agent)
        socket = FakeSocket()
        asyncio.run(handler._handle_fetch_attachment_memories(socket, "c1", {}))
        self.assertEqual(socket.sent[0]["type"], "attachment-memories")
        self.assertEqual(len(socket.sent[0]["memories"]), 1)
        self.assertEqual(socket.sent[0]["memories"][0]["filename"], "kucing.png")

    def test_delete_by_id(self):
        stored = append_attachment_memory(CONF, record())
        assert stored is not None
        agent = SimpleNamespace(
            purge_attachment_memory=lambda rid: purge_attachment_memory(CONF, rid)
        )
        handler = make_ws_handler(agent)
        socket = FakeSocket()
        asyncio.run(
            handler._handle_delete_attachment_memory(
                socket, "c1", {"record_id": stored["id"]}
            )
        )
        self.assertEqual(socket.sent[0]["type"], "attachment-memory-deleted")
        self.assertTrue(socket.sent[0]["success"])
        self.assertTrue(socket.sent[0]["purge"]["attachment_removed"])
        self.assertEqual(load_attachment_memories(CONF), [])

    def test_clear_all(self):
        append_attachment_memory(CONF, record())
        agent = SimpleNamespace(
            clear_attachment_memories=lambda: clear_attachment_memories(CONF)
        )
        handler = make_ws_handler(agent)
        socket = FakeSocket()
        asyncio.run(handler._handle_clear_attachment_memories(socket, "c1", {}))
        self.assertEqual(socket.sent[0]["type"], "attachment-memories-cleared")
        self.assertEqual(socket.sent[0]["removed"], 1)

    def test_delete_history_cascades_session_records(self):
        history_uid = create_new_history(CONF)
        append_attachment_memory(CONF, record(session=history_uid))
        append_attachment_memory(
            CONF,
            record(name="b.png", data=DATA_URL_2, summary="B.", session="other"),
        )
        agent = SimpleNamespace(
            set_memory_from_history=lambda **kwargs: None,
        )
        handler = make_ws_handler(agent, history_uid=history_uid)
        socket = FakeSocket()
        asyncio.run(
            handler._handle_delete_history(socket, "c1", {"history_uid": history_uid})
        )
        self.assertEqual(socket.sent[0]["type"], "history-deleted")
        self.assertTrue(socket.sent[0]["success"])
        remaining = load_attachment_memories(CONF)
        self.assertEqual(len(remaining), 1)
        self.assertEqual(remaining[0]["session_uid"], "other")


def seed_episodic(text, session="sess-a"):
    stored = em.append_episodic_event(
        CONF, {"event_text": text, "session_uid": session}
    )
    assert stored is not None
    return stored


class EpisodicSelectiveDeleteTest(TmpCwdTestCase):
    def test_delete_by_ids_removes_only_matched(self):
        first = seed_episodic("User showed kucing.png yesterday.")
        second = seed_episodic("User fixed a WebSocket bug.")
        result = delete_episodic_events_by_ids(CONF, [first["id"]])
        self.assertEqual(result["removed"], [first["id"]])
        self.assertEqual(result["missing"], [])
        self.assertTrue(result["saved"])
        remaining = load_episodic_events(CONF)
        self.assertEqual([e["id"] for e in remaining], [second["id"]])

    def test_delete_missing_and_empty_is_safe(self):
        seed_episodic("User fixed a WebSocket bug.")
        result = delete_episodic_events_by_ids(CONF, ["no-such-id"])
        self.assertEqual(result["removed"], [])
        self.assertEqual(result["missing"], ["no-such-id"])
        self.assertTrue(result["saved"])
        self.assertEqual(len(load_episodic_events(CONF)), 1)
        # No-op on a fresh store performs no write at all.
        noop = delete_episodic_events_by_ids("fresh-conf", [])
        self.assertTrue(noop["saved"])
        self.assertFalse(os.path.exists(os.path.join("episodic", "fresh-conf.json")))

    def test_delete_all_keeps_valid_store(self):
        first = seed_episodic("Event one.")
        second = seed_episodic("Event two.")
        result = delete_episodic_events_by_ids(CONF, [first["id"], second["id"]])
        self.assertEqual(len(result["removed"]), 2)
        self.assertEqual(load_episodic_events(CONF), [])


class PurgeTest(TmpCwdTestCase):
    def _seed_with_copies(self):
        stored = append_attachment_memory(CONF, record())
        assert stored is not None
        quoting = seed_episodic("User showed a photo named kucing.png yesterday.")
        unrelated = seed_episodic("User fixed a WebSocket bug.")
        history_uid = create_new_history(CONF)
        store_message(CONF, history_uid, "human", "lihat kucing.png yang aku kirim")
        update_summary_metadata(
            CONF,
            history_uid,
            expected_summarized_through=0,
            conversation_summary="Ringkasan: membahas kucing.png dan motor.",
            summarized_through=1,
            summary_updated_at="2026-10-09T11:00:00+00:00",
        )
        return stored, quoting, unrelated, history_uid

    def test_purge_removes_record_and_episodic_copies(self):
        stored, quoting, unrelated, history_uid = self._seed_with_copies()
        status = purge_attachment_memory(CONF, stored["id"])
        self.assertTrue(status["found"])
        self.assertTrue(status["attachment_removed"])
        self.assertEqual(status["episodic_removed"], [quoting["id"]])
        self.assertEqual(status["episodic_failed"], [])
        self.assertEqual(status["transcripts_remaining"], [history_uid])
        self.assertEqual(status["summaries_remaining"], [history_uid])
        # Transcripts and summaries are reported, never rewritten.
        self.assertFalse(status["complete"])
        self.assertEqual(load_attachment_memories(CONF), [])
        remaining = load_episodic_events(CONF)
        self.assertEqual([e["id"] for e in remaining], [unrelated["id"]])
        rows = get_history(CONF, history_uid)
        self.assertTrue(any("kucing.png" in m["content"] for m in rows))

    def test_purge_clean_when_no_copies(self):
        stored = append_attachment_memory(CONF, record())
        assert stored is not None
        status = purge_attachment_memory(CONF, stored["id"])
        self.assertTrue(status["found"])
        self.assertTrue(status["attachment_removed"])
        self.assertEqual(status["episodic_removed"], [])
        self.assertEqual(status["transcripts_remaining"], [])
        self.assertEqual(status["summaries_remaining"], [])
        self.assertEqual(status["character_memories_remaining"], [])
        self.assertFalse(status["world_state_match"])
        self.assertTrue(status["complete"])

    def test_purge_unknown_id_touches_nothing(self):
        stored = append_attachment_memory(CONF, record())
        assert stored is not None
        event = seed_episodic("User showed kucing.png.")
        status = purge_attachment_memory(CONF, "no-such-id")
        self.assertFalse(status["found"])
        self.assertFalse(status["attachment_removed"])
        self.assertFalse(status["complete"])
        self.assertEqual(len(load_attachment_memories(CONF)), 1)
        self.assertEqual(len(load_episodic_events(CONF)), 1)
        self.assertEqual(event["event_text"], "User showed kucing.png.")

    def test_purge_partial_failure_reports_failed_source(self):
        stored = append_attachment_memory(CONF, record())
        assert stored is not None
        quoting = seed_episodic("User showed kucing.png.")
        with patch(
            "src.open_llm_vtuber.episodic_memory.save_episodic_events",
            return_value=False,
        ):
            status = purge_attachment_memory(CONF, stored["id"])
        # The record still deletes; the failed source is named honestly.
        self.assertTrue(status["attachment_removed"])
        self.assertEqual(status["episodic_failed"], [quoting["id"]])
        self.assertEqual(status["episodic_removed"], [])
        self.assertFalse(status["complete"])
        self.assertEqual(load_attachment_memories(CONF), [])
        # The episodic file is byte-identical: the copy truly remains.
        self.assertEqual(len(load_episodic_events(CONF)), 1)

    def test_character_and_world_copies_reported_not_removed(self):
        stored = append_attachment_memory(CONF, record())
        assert stored is not None
        add_character_memory(CONF, "Foto kucingku tersimpan sebagai kucing.png")
        world_path = get_world_state_path(CONF)
        os.makedirs(os.path.dirname(world_path), exist_ok=True)
        with open(world_path, "w", encoding="utf-8") as handle:
            handle.write('{"note": "user shared kucing.png today"}')
        copies = find_attachment_memory_copies(CONF, record_id=stored["id"])
        self.assertEqual(len(copies["character_memories"]), 1)
        self.assertTrue(copies["world_state_match"])
        status = purge_attachment_memory(CONF, stored["id"])
        self.assertTrue(status["attachment_removed"])
        self.assertEqual(len(status["character_memories_remaining"]), 1)
        self.assertTrue(status["world_state_match"])
        self.assertFalse(status["complete"])
        # Scanner is read-only for these sources: facts and world survive.
        facts = [m.get("text", "") for m in load_character_state(CONF).memories]
        self.assertTrue(any("kucing.png" in fact for fact in facts))
        self.assertIn("kucing.png", open(world_path, encoding="utf-8").read())

    def test_agent_purge_wrapper(self):
        history_uid = create_new_history(CONF)
        agent = make_agent(CONF, history_uid, FakeAgentLLM([]))
        stored = append_attachment_memory(CONF, record())
        assert stored is not None
        quoting = seed_episodic("User showed kucing.png.")
        status = agent.purge_attachment_memory(stored["id"])
        self.assertTrue(status["complete"] or status["attachment_removed"])
        self.assertEqual(status["episodic_removed"], [quoting["id"]])
        self.assertEqual(load_attachment_memories(CONF), [])
        empty = agent.purge_attachment_memory("")
        self.assertFalse(empty["found"])


class RetrievalAfterDeleteTest(TmpCwdTestCase):
    def _agent_with_memory(self):
        history_uid = create_new_history(CONF)
        llm = FakeAgentLLM(describe_response("Kucing oranye tidur di sofa."))
        agent = make_agent(CONF, history_uid, llm)
        asyncio.run(
            agent.capture_attachment_memory(
                [image_item()], "kucingku", history_uid, "r"
            )
        )
        seed_episodic("User showed a photo named kucing.png yesterday.")
        seed_episodic("User fixed a WebSocket bug.")
        return agent

    def test_selections_empty_after_purge(self):
        agent = self._agent_with_memory()
        agent._episodic_query = "kucing oranye sofa"
        self.assertEqual(len(agent._attachment_selection()), 1)
        self.assertTrue(agent._attachment_context_for_prompt())
        self.assertEqual(len(agent._episodic_selection()), 1)
        rows = load_attachment_memories(CONF)
        self.assertEqual(len(rows), 1)
        status = agent.purge_attachment_memory(rows[0]["id"])
        self.assertTrue(status["attachment_removed"])
        # New turn resets both per-turn caches (mirrors _to_messages).
        agent._attachment_selection_cache = None
        agent._episodic_selection_cache = None
        self.assertEqual(agent._attachment_selection(), [])
        self.assertEqual(agent._attachment_context_for_prompt(), "")
        self.assertEqual(agent._episodic_selection(), [])
        system = agent._relationship_system_prompt("persona dasar")
        self.assertNotIn("Kucing oranye tidur di sofa.", system)
        self.assertNotIn("kucing.png", system)

    def test_unrelated_memories_survive_purge(self):
        agent = self._agent_with_memory()
        rows = load_attachment_memories(CONF)
        agent.purge_attachment_memory(rows[0]["id"])
        agent._episodic_query = "websocket bug diperbaiki"
        remaining = agent._episodic_selection()
        self.assertEqual(len(remaining), 1)
        self.assertIn("WebSocket bug", remaining[0]["event_text"])

    def test_scoped_remove_leaves_episodic_copy_retrievable(self):
        agent = self._agent_with_memory()
        rows = load_attachment_memories(CONF)
        self.assertTrue(agent.remove_attachment_memory(rows[0]["id"]))
        self.assertEqual(agent._attachment_selection(), [])
        # Scoped removal is honest: the episodic duplicate still answers.
        agent._episodic_query = "kucing sofa"
        found = agent._episodic_selection()
        self.assertEqual(len(found), 1)
        self.assertIn("kucing.png", found[0]["event_text"])

    def test_new_turn_resets_attachment_selection_cache(self):
        """Stale attachment recall must never leak into the next turn."""
        history_uid = create_new_history(CONF)
        agent = make_agent(CONF, history_uid, FakeAgentLLM([]))
        agent._attachment_selection_cache = ("halo", [{"id": "stale"}])
        agent._to_messages(
            BatchInput(
                texts=[
                    TextData(
                        source=TextSource.INPUT,
                        content="halo",
                        from_name="Human",
                    )
                ],
                metadata={"episodic_query": "halo"},
            )
        )
        self.assertIsNone(agent._attachment_selection_cache)
        self.assertIsNone(agent._episodic_selection_cache)

    def test_restart_after_purge(self):
        agent = self._agent_with_memory()
        rows = load_attachment_memories(CONF)
        agent.purge_attachment_memory(rows[0]["id"])
        repo_src = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        script = (
            "import json, os;"
            "os.chdir(%r);"
            % os.getcwd()
            + "import sys; sys.path.insert(0, %r);" % repo_src
            + "from src.open_llm_vtuber.attachment_memory import "
            "load_attachment_memories;"
            "from src.open_llm_vtuber import episodic_memory as em;"
            "print(json.dumps({"
            "'attachments': len(load_attachment_memories(%r)),"
            % CONF
            + "'episodic_kucing': sum("
            "1 for e in em.load_episodic_events(%r) "
            % CONF
            + "if 'kucing.png' in e.get('event_text', '')),"
            "'episodic_total': len(em.load_episodic_events(%r))}))" % CONF
        )
        proc = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        state = json.loads(proc.stdout.strip().splitlines()[-1])
        self.assertEqual(state["attachments"], 0)
        self.assertEqual(state["episodic_kucing"], 0)
        self.assertEqual(state["episodic_total"], 1)


class DeletionSideEffectTest(TmpCwdTestCase):
    def _seed(self):
        history_a = create_new_history(CONF)
        history_b = create_new_history(CONF)
        append_attachment_memory(CONF, record(session=history_a))
        append_attachment_memory(
            CONF,
            record(name="b.png", data=DATA_URL_2, summary="B.", session=history_b),
        )
        event = seed_episodic("User showed kucing.png.")
        store_message(CONF, history_a, "human", "lihat kucing.png")
        store_message(CONF, history_b, "human", "halo sesi B")
        return history_a, history_b, event

    def test_delete_history_cascade_side_effect_free(self):
        history_a, history_b, event = self._seed()
        agent = SimpleNamespace(set_memory_from_history=lambda **kwargs: None)
        handler = make_ws_handler(agent, history_uid=history_a)
        socket = FakeSocket()
        asyncio.run(
            handler._handle_delete_history(socket, "c1", {"history_uid": history_a})
        )
        self.assertTrue(socket.sent[0]["success"])
        rows = load_attachment_memories(CONF)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["session_uid"], history_b)
        # Cascade never touches episodic rows or other transcripts.
        self.assertEqual(len(load_episodic_events(CONF)), 1)
        self.assertEqual(load_episodic_events(CONF)[0]["id"], event["id"])
        rows_b = get_history(CONF, history_b)
        self.assertEqual(
            [(m["role"], m["content"]) for m in rows_b], [("human", "halo sesi B")]
        )

    def test_clear_all_side_effect_free(self):
        history_a, _, event = self._seed()
        removed = clear_attachment_memories(CONF)
        self.assertEqual(removed, 2)
        self.assertEqual(load_attachment_memories(CONF), [])
        self.assertEqual(len(load_episodic_events(CONF)), 1)
        self.assertEqual(load_episodic_events(CONF)[0]["id"], event["id"])
        rows = get_history(CONF, history_a)
        self.assertTrue(any("kucing.png" in m["content"] for m in rows))


if __name__ == "__main__":
    unittest.main()
