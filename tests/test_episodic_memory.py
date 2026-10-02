"""Episodic memory foundation — deterministic tests (fake clock/LLM).

Covers: heuristic gate, extraction parsing, temporal resolution,
storage roundtrip/dedup/corruption, retrieval/recency/budget,
cross-session flow, and failure safety. No network, no real LLM.
"""

import json
import os
import tempfile
import unittest
import unittest.mock
from datetime import datetime, timezone

from src.open_llm_vtuber import episodic_memory as em
from src.open_llm_vtuber.episodic_memory import (
    append_episodic_event,
    build_extraction_prompt,
    is_episodic_candidate,
    load_episodic_events,
    parse_extraction_result,
    render_episodic_context,
    resolve_occurred_at,
    retrieve_episodic_events,
)

JAKARTA = "Asia/Jakarta"
# 2026-10-02T05:00:00Z == 12:00 Oct 2 Jakarta.
T0 = datetime(2026, 10, 2, 5, 0, tzinfo=timezone.utc)


def event(text, occurred=None, created="2026-10-02T05:00:00+00:00", session="sess-a"):
    return {
        "id": "x",
        "event_text": text,
        "occurred_at": occurred,
        "session_uid": session,
        "source": "conversation",
        "created_at": created,
        "tz": JAKARTA,
    }


class HeuristicTest(unittest.TestCase):
    def test_small_talk_skipped(self):
        for text in ["wkwk iya", "iya", "oke", "haha", "ok sip", "oh"]:
            self.assertFalse(is_episodic_candidate(text), text)

    def test_short_text_skipped(self):
        self.assertFalse(is_episodic_candidate("Gw tadi makan."))

    def test_pure_question_skipped(self):
        self.assertFalse(is_episodic_candidate("Kapan kereta terakhir berangkat?"))

    def test_future_intent_skipped(self):
        for text in [
            "Besok gw mau beli monitor baru yang besar.",
            "Nanti gw akan pergi ke toko untuk membeli keyboard.",
            "Rencana gw mau belajar gitar tahun depan.",
        ]:
            self.assertFalse(is_episodic_candidate(text), text)

    def test_real_past_experience_candidate(self):
        for text in [
            "Gw tadi habisin 3 jam benerin bug WebSocket yang menyebalkan.",
            "Kemarin gw habis memperbaiki bug websocket selama 3 jam penuh.",
            "Gw sudah menyelesaikan website portofolio gw kemarin sore.",
            "Gw tadi habisin 3 jam benerin websocket",
        ]:
            self.assertTrue(is_episodic_candidate(text), text)


class ExtractionParseTest(unittest.TestCase):
    def test_valid_event(self):
        raw = (
            '{"event_text": "User fixed a bug.", '
            '"occurred_at": "2026-10-01T00:00:00+00:00", "confidence": 0.9}'
        )
        parsed = parse_extraction_result(raw)
        self.assertIsNotNone(parsed)
        assert parsed is not None
        self.assertEqual(parsed["event_text"], "User fixed a bug.")
        self.assertAlmostEqual(parsed["confidence"], 0.9)

    def test_null_event(self):
        self.assertIsNone(parse_extraction_result('{"event": null}'))
        self.assertIsNone(
            parse_extraction_result('{"event_text": "", "confidence": 0.9}')
        )

    def test_low_confidence_discarded(self):
        raw = '{"event_text": "Maybe something.", "confidence": 0.4}'
        self.assertIsNone(parse_extraction_result(raw))

    def test_malformed_json_discarded(self):
        self.assertIsNone(parse_extraction_result("not json at all"))
        self.assertIsNone(parse_extraction_result('{"event_text": unclosed'))
        self.assertIsNone(parse_extraction_result(""))
        self.assertIsNone(parse_extraction_result(None))

    def test_no_invented_timestamp_shape(self):
        # occurred_at must be a string or null, never a fabricated object.
        raw = '{"event_text": "X happened.", "occurred_at": {"day": "someday"}, "confidence": 0.9}'
        self.assertIsNone(parse_extraction_result(raw))

    def test_n_days_ago_gate(self):
        self.assertTrue(
            is_episodic_candidate("2 hari lalu gw deploy backend ke VPS baru.")
        )
        self.assertTrue(is_episodic_candidate("Gw deploy backend 3 days ago sore."))

    def test_prompt_mentions_json_and_no_invention(self):
        system, user = build_extraction_prompt("Gw tadi lari.", T0, JAKARTA)
        self.assertIn("JSON", system)
        self.assertIn("Gw tadi lari.", user)


class TemporalTest(unittest.TestCase):
    def test_yesterday_jakarta(self):
        # Spec case: request 05:00Z Oct 2 == 12:00 Oct 2 Jakarta.
        # "Kemarin" => Oct 1 Jakarta => 2026-09-30T17:00:00+00:00.
        result = resolve_occurred_at(
            "Kemarin gw habisin 3 jam benerin WebSocket.", T0, JAKARTA
        )
        self.assertEqual(result, "2026-09-30T17:00:00+00:00")

    def test_today(self):
        result = resolve_occurred_at("Gw hari ini memperbaiki bug.", T0, JAKARTA)
        # Oct 2 00:00+07 == Oct 1 17:00Z.
        self.assertEqual(result, "2026-10-01T17:00:00+00:00")

    def test_explicit_absolute_date(self):
        # Explicit date wins over the deictic in the same turn.
        result = resolve_occurred_at(
            "Gw deploy tanggal 28 September kemarin.", T0, JAKARTA
        )
        self.assertEqual(result, "2026-09-27T17:00:00+00:00")

    def test_explicit_month_name_only(self):
        result = resolve_occurred_at("Rilis 15 September sudah lewat.", T0, JAKARTA)
        self.assertEqual(result, "2026-09-14T17:00:00+00:00")

    def test_unknown_time_is_null(self):
        self.assertIsNone(
            resolve_occurred_at("Minggu lalu gw kayaknya ngoding sesuatu.", T0, JAKARTA)
        )
        self.assertIsNone(resolve_occurred_at("", T0, JAKARTA))
        self.assertIsNone(resolve_occurred_at("   ", T0, JAKARTA))

    def test_different_timezone(self):
        # 05:00Z Oct 2 == 01:00 Oct 2 New York (EDT, UTC-4).
        result = resolve_occurred_at(
            "Kemarin gw begadang ngoding.", T0, "America/New_York"
        )
        # Oct 1 00:00-04:00 == Oct 1 04:00Z.
        self.assertEqual(result, "2026-10-01T04:00:00+00:00")

    def test_n_days_ago_resolution(self):
        cases = [
            ("1 hari lalu gw deploy backend.", "2026-09-30T17:00:00+00:00"),
            ("2 hari lalu gw deploy backend.", "2026-09-29T17:00:00+00:00"),
            ("7 hari lalu gw deploy backend.", "2026-09-24T17:00:00+00:00"),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(resolve_occurred_at(text, T0, JAKARTA), expected)

    def test_n_days_ago_different_timezone(self):
        result = resolve_occurred_at(
            "2 hari lalu gw deploy backend.", T0, "America/New_York"
        )
        self.assertEqual(result, "2026-09-30T04:00:00+00:00")

    def test_n_days_ago_malformed(self):
        self.assertIsNone(resolve_occurred_at("0 hari lalu gw deploy.", T0, JAKARTA))
        self.assertIsNone(
            resolve_occurred_at("Hari lalu gw deploy backend.", T0, JAKARTA)
        )

    def test_absolute_date_beats_yesterday(self):
        # Explicit date must win over the deictic in the same turn.
        result = resolve_occurred_at(
            "tanggal 28 September kemarin gw deploy backend.", T0, JAKARTA
        )
        self.assertEqual(result, "2026-09-27T17:00:00+00:00")

    def test_explicit_date_only(self):
        result = resolve_occurred_at("Rilis 15 September sudah lewat.", T0, JAKARTA)
        self.assertEqual(result, "2026-09-14T17:00:00+00:00")

    def test_yesterday_only(self):
        result = resolve_occurred_at("kemarin gw deploy backend.", T0, JAKARTA)
        self.assertEqual(result, "2026-09-30T17:00:00+00:00")

    def test_today_only(self):
        result = resolve_occurred_at("hari ini gw deploy backend.", T0, JAKARTA)
        self.assertEqual(result, "2026-10-01T17:00:00+00:00")


class StorageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_append_load_roundtrip(self):
        stored = append_episodic_event(
            "c1",
            {
                "event_text": "User fixed a WebSocket bug.",
                "occurred_at": "2026-09-30T17:00:00+00:00",
                "session_uid": "sess-a",
                "source": "conversation",
                "tz": JAKARTA,
            },
        )
        self.assertIsNotNone(stored)
        assert stored is not None
        self.assertTrue(stored["id"])
        self.assertTrue(stored["created_at"])
        loaded = load_episodic_events("c1")
        self.assertEqual(len(loaded), 1)
        self.assertEqual(loaded[0]["event_text"], "User fixed a WebSocket bug.")

    def test_restart_persistence(self):
        append_episodic_event("c1", {"event_text": "User ran a marathon."})
        # Fresh load = post-restart read path.
        self.assertEqual(len(load_episodic_events("c1")), 1)

    def test_corrupt_and_missing(self):
        self.assertEqual(load_episodic_events("nope"), [])
        os.makedirs("episodic", exist_ok=True)
        with open("episodic/c1.json", "w", encoding="utf-8") as handle:
            handle.write("{not json")
        self.assertEqual(load_episodic_events("c1"), [])
        with open("episodic/c1.json", "w", encoding="utf-8") as handle:
            handle.write('{"not": "a list"}')
        self.assertEqual(load_episodic_events("c1"), [])
        self.assertIsNone(append_episodic_event("c1", {"event_text": "   "}))

    def test_invalid_occurred_at_rejected(self):
        self.assertIsNone(
            append_episodic_event(
                "c1",
                {"event_text": "User fixed a bug.", "occurred_at": "someday"},
            )
        )
        self.assertIsNone(
            append_episodic_event(
                "c1",
                {"event_text": "User fixed a bug.", "occurred_at": {"day": "x"}},
            )
        )
        self.assertEqual(load_episodic_events("c1"), [])

    def test_corrupt_occurred_at_degrades_on_load(self):
        append_episodic_event(
            "c1",
            {
                "event_text": "User fixed a bug.",
                "occurred_at": "2026-09-30T17:00:00+00:00",
            },
        )
        path = os.path.join("episodic", "c1.json")
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        data[0]["occurred_at"] = "not-a-time"
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle)
        loaded = load_episodic_events("c1")
        self.assertEqual(len(loaded), 1)
        self.assertIsNone(loaded[0]["occurred_at"])
        self.assertEqual(loaded[0]["event_text"], "User fixed a bug.")

    def test_dedup(self):
        append_episodic_event("c1", {"event_text": "User fixed a bug."})
        self.assertIsNone(
            append_episodic_event("c1", {"event_text": "user fixed a bug"})
        )
        self.assertIsNone(
            append_episodic_event("c1", {"event_text": "user  fixed   a bug!"})
        )
        self.assertEqual(len(load_episodic_events("c1")), 1)
        self.assertTrue(
            append_episodic_event("c1", {"event_text": "User cooked rendang."})
            is not None
        )

    def test_soft_cap(self):
        topics = [
            "User fixed a websocket bug today.",
            "User cooked rendang for dinner.",
            "User deployed the backend to a VPS.",
            "User ran a marathon in the morning.",
            "User bought a mechanical keyboard.",
            "User watched a documentary tonight.",
            "User watered the balcony plants.",
        ]
        with unittest.mock.patch.object(em, "EPISODIC_MAX_EVENTS", 5):
            for topic in topics:
                append_episodic_event("c1", {"event_text": topic})
            self.assertEqual(len(load_episodic_events("c1")), 5)


class RetrievalTest(unittest.TestCase):
    def setUp(self):
        self._events = [
            event(
                "User spent about 3 hours fixing a WebSocket bug.",
                "2026-09-30T17:00:00+00:00",
                "2026-10-01T01:00:00+00:00",
            ),
            event(
                "User cooked rendang for family dinner.",
                "2026-09-28T17:00:00+00:00",
                "2026-09-29T01:00:00+00:00",
            ),
            event(
                "User deployed the Mili backend to a VPS.",
                "2026-09-25T17:00:00+00:00",
                "2026-09-26T01:00:00+00:00",
            ),
        ]

    def test_relevant_event(self):
        found = retrieve_episodic_events(
            self._events, "Lu masih inget waktu gw lama benerin bug?", now=T0
        )
        self.assertTrue(found)
        self.assertIn("WebSocket", found[0]["event_text"])

    def test_irrelevant_event(self):
        found = retrieve_episodic_events(self._events, "Apa cuaca hari ini?", now=T0)
        self.assertEqual(found, [])

    def test_recency_boost_orders_ties(self):
        tied = [
            event("User fixed a websocket bug.", None, "2026-09-20T01:00:00+00:00"),
            event(
                "User fixed a websocket bug today.", None, "2026-10-02T01:00:00+00:00"
            ),
        ]
        found = retrieve_episodic_events(tied, "websocket bug", now=T0)
        self.assertEqual(len(found), 2)
        self.assertIn("today", found[0]["event_text"])

    def test_top_n(self):
        many = [
            event(
                f"User fixed websocket bug number {i}.",
                None,
                "2026-10-02T01:00:00+00:00",
            )
            for i in range(10)
        ]
        found = retrieve_episodic_events(many, "websocket bug", now=T0, top_n=3)
        self.assertEqual(len(found), 3)

    def test_token_budget(self):
        many = [
            event(
                f"User fixed websocket bug number {i} yesterday.",
                None,
                "2026-10-02T01:00:00+00:00",
            )
            for i in range(10)
        ]
        rendered = render_episodic_context(many, now=T0, tz=JAKARTA, max_tokens=40)
        self.assertIn("Relevant episodic experiences", rendered)
        self.assertLessEqual(len(rendered.split()), 60)

    def test_empty_result(self):
        self.assertEqual(retrieve_episodic_events([], "bug", now=T0), [])
        self.assertEqual(retrieve_episodic_events(self._events, "", now=T0), [])
        self.assertEqual(render_episodic_context([], now=T0, tz=JAKARTA), "")

    def test_age_render_uses_occurred(self):
        rendered = render_episodic_context([self._events[0]], now=T0, tz=JAKARTA)
        # occurred Sep 30 17:00Z == Oct 1 00:00 Jakarta; now Oct 2 12:00
        # Jakarta => yesterday.
        self.assertIn("Yesterday", rendered)


class CrossSessionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_session_a_to_session_b(self):
        # Session A stores, summary-independent.
        stored = append_episodic_event(
            "c1",
            {
                "event_text": "User spent about 3 hours fixing a WebSocket bug.",
                "occurred_at": "2026-09-30T17:00:00+00:00",
                "session_uid": "sess-A",
                "source": "conversation",
                "tz": JAKARTA,
            },
        )
        self.assertIsNotNone(stored)
        # Session B retrieves without any summary machinery.
        loaded = load_episodic_events("c1")
        found = retrieve_episodic_events(
            loaded,
            "Lu masih inget waktu gw lama banget benerin bug itu?",
            now=T0,
        )
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["session_uid"], "sess-A")
        rendered = render_episodic_context(found, now=T0, tz=JAKARTA)
        self.assertIn("WebSocket", rendered)


class FailureTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    async def test_extraction_error_returns_none(self):
        async def run():
            async def boom(messages, system):
                raise RuntimeError("provider down")
                yield ""

            return await em.extract_and_store_episodic(
                boom,
                "c1",
                "Gw tadi habisin 3 jam benerin bug WebSocket yang besar.",
                "sess-a",
                T0,
                JAKARTA,
            )

        self.assertIsNone(await run())

    async def test_storage_error_returns_none(self):
        async def run():
            async def fake_llm(messages, system):
                yield (
                    '{"event_text": "User fixed a bug.", '
                    '"occurred_at": null, "confidence": 0.95}'
                )

            with unittest.mock.patch.object(
                em, "save_episodic_events", return_value=False
            ):
                return await em.extract_and_store_episodic(
                    fake_llm,
                    "c1",
                    "Gw tadi habisin 3 jam benerin bug WebSocket yang besar.",
                    "sess-a",
                    T0,
                    JAKARTA,
                )

        self.assertIsNone(await run())

    def test_retrieval_error_returns_empty(self):
        with unittest.mock.patch.object(
            em, "_token_set", side_effect=RuntimeError("boom")
        ):
            result = retrieve_episodic_events(
                [event("User fixed a bug.")], "bug", now=T0
            )
            self.assertEqual(result, [])

    async def test_full_pipeline_success(self):
        async def run():
            async def fake_llm(messages, system):
                yield (
                    '{"event_text": "User spent 3 hours fixing a bug.", '
                    '"occurred_at": null, "confidence": 0.9}'
                )

            return await em.extract_and_store_episodic(
                fake_llm,
                "c1",
                "Gw tadi habisin 3 jam benerin bug WebSocket yang besar.",
                "sess-a",
                T0,
                JAKARTA,
            )

        stored = await run()
        self.assertIsNotNone(stored)
        assert stored is not None
        # Heuristic fallback resolves "tadi" to request local day.
        self.assertEqual(stored["occurred_at"], "2026-10-01T17:00:00+00:00")
        self.assertEqual(stored["session_uid"], "sess-a")


if __name__ == "__main__":
    unittest.main()
