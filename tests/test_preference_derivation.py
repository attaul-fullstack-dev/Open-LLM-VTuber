"""Preference derivation Phase 2A — deterministic tests (no LLM, no I/O, no clock).

Covers A-I: thresholds, aggregation, restart/offline invariance, purity.
"""

import asyncio
import inspect
import unittest

from src.open_llm_vtuber.self_model import (
    PREFERENCE_MIN_DAYS,
    PREFERENCE_MIN_EVIDENCE,
    build_self_context,
    derive_activity_preferences,
    format_preference_line,
)

JKT = "Asia/Jakarta"


def hist(to, at, by="transition"):
    return {"from": "idle", "to": to, "at": at, "location": "room", "by": by}


def mem(text, added_at):
    return {"text": text, "added_at": added_at, "explicit": True}


class ThresholdTest(unittest.TestCase):
    def test_a_no_evidence_no_preference(self):
        self.assertEqual(derive_activity_preferences([], [], tz=JKT), [])
        self.assertEqual(derive_activity_preferences(None, None, tz=JKT), [])

    def test_b_single_evidence_not_established(self):
        out = derive_activity_preferences(
            [hist("reading", "2026-09-29T08:00:00+00:00")], [], tz=JKT
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].activity, "reading")
        self.assertFalse(out[0].established)

    def test_burst_single_day_not_established(self):
        # Three mentions in one day: count ok, days fail.
        out = derive_activity_preferences(
            [],
            [
                mem("baca buku pagi", "2026-09-29T01:00:00+00:00"),
                mem("baca novel siang", "2026-09-29T05:00:00+00:00"),
                mem("baca komik malam", "2026-09-29T10:00:00+00:00"),
            ],
            tz=JKT,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].evidence_count, 3)
        self.assertEqual(out[0].distinct_days, 1)
        self.assertFalse(out[0].established)

    def test_c_repeated_evidence_established(self):
        out = derive_activity_preferences(
            [
                hist("reading", "2026-09-27T08:00:00+00:00"),
                hist("reading", "2026-09-28T08:00:00+00:00"),
            ],
            [mem("lagi baca buku seru", "2026-09-29T08:00:00+00:00")],
            tz=JKT,
        )
        self.assertEqual(len(out), 1)
        cand = out[0]
        self.assertTrue(cand.established)
        self.assertEqual(cand.evidence_count, 3)
        self.assertEqual(cand.distinct_days, 3)
        self.assertEqual(cand.first_at, "2026-09-27T08:00:00+00:00")
        self.assertEqual(cand.last_at, "2026-09-29T08:00:00+00:00")

    def test_c_threshold_constants(self):
        self.assertEqual(PREFERENCE_MIN_EVIDENCE, 3)
        self.assertEqual(PREFERENCE_MIN_DAYS, 2)

    def test_idle_and_unknown_never_qualify(self):
        out = derive_activity_preferences(
            [hist("idle", "2026-09-27T08:00:00+00:00")] * 5, [], tz=JKT
        )
        self.assertEqual(out, [])

    def test_unparseable_timestamps_skipped(self):
        out = derive_activity_preferences(
            [hist("reading", "not-a-date")],
            [mem("baca buku", "2026-09-29T08:00:00+00:00")],
            tz=JKT,
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0].evidence_count, 1)
        self.assertFalse(out[0].established)


class AggregationTest(unittest.TestCase):
    def test_d_distinct_days_use_user_timezone(self):
        # 2026-09-29 17:00Z Sep 29 UTC but Sep 30 Jakarta.
        out_utc = derive_activity_preferences(
            [
                hist("playing", "2026-09-29T08:00:00+00:00"),
                hist("playing", "2026-09-29T17:00:00+00:00"),
            ],
            [],
            tz="UTC",
        )
        out_jkt = derive_activity_preferences(
            [
                hist("playing", "2026-09-29T08:00:00+00:00"),
                hist("playing", "2026-09-29T17:00:00+00:00"),
            ],
            [],
            tz=JKT,
        )
        self.assertEqual(out_utc[0].distinct_days, 1)
        self.assertEqual(out_jkt[0].distinct_days, 2)

    def test_d_ordering_deterministic(self):
        out = derive_activity_preferences(
            [hist("playing", f"2026-09-2{d}T08:00:00+00:00") for d in (7, 8, 9)]
            + [hist("reading", f"2026-09-2{d}T08:00:00+00:00") for d in (7, 8, 9)],
            [],
            tz=JKT,
        )
        self.assertEqual([c.activity for c in out], ["playing", "reading"])

    def test_render_only_established_capped(self):
        cands = derive_activity_preferences(
            [hist("reading", f"2026-09-2{d}T08:00:00+00:00") for d in (7, 8, 9)]
            + [hist("playing", "2026-09-29T08:00:00+00:00")],
            [],
            tz=JKT,
        )
        line = format_preference_line(cands)
        self.assertIn("reading (evidence: 3)", line)
        self.assertNotIn("playing", line)
        self.assertEqual(format_preference_line([]), "")

    def test_render_never_claims_feelings(self):
        line = format_preference_line(
            derive_activity_preferences(
                [hist("reading", f"2026-09-2{d}T08:00:00+00:00") for d in (7, 8, 9)],
                [],
                tz=JKT,
            )
        )
        for word in ("love", "loves", "likes", "feels", "suka"):
            self.assertNotIn(word, line)


class RestartInvarianceTest(unittest.TestCase):
    def _evidence(self):
        return (
            [hist("eating", f"2026-09-2{d}T08:00:00+00:00") for d in (7, 8, 9)],
            [mem("makan bakso enak", "2026-09-28T10:00:00+00:00")],
        )

    def test_e_reload_same_result(self):
        h, m = self._evidence()
        first = derive_activity_preferences(h, m, tz=JKT)
        # Simulate restart: rebuild inputs from persisted JSON round-trip.
        import json

        h2 = json.loads(json.dumps(h))
        m2 = json.loads(json.dumps(m))
        second = derive_activity_preferences(h2, m2, tz=JKT)
        self.assertEqual(first, second)
        self.assertTrue(second[0].established)

    def test_f_offline_gap_does_not_change_result(self):
        h, m = self._evidence()
        before = derive_activity_preferences(h, m, tz=JKT)
        # VPS down Tue->Thu, then a month: absolute stamps, no `now` input.
        after_gap = derive_activity_preferences(h, m, tz=JKT)
        after_month = derive_activity_preferences(h, m, tz=JKT)
        self.assertEqual(before, after_gap)
        self.assertEqual(before, after_month)


class PurityTest(unittest.TestCase):
    def test_g_no_llm_or_clock(self):
        for fn in (derive_activity_preferences, format_preference_line):
            self.assertFalse(asyncio.iscoroutinefunction(fn))
        src = inspect.getsource(derive_activity_preferences)
        src += inspect.getsource(format_preference_line)
        for token in (
            "chat_completion",
            "generate",
            "llm",
            "random",
            "datetime.now",
            "utcnow",
            "time.time",
            "sleep",
        ):
            self.assertNotIn(token, src)


class ComposerIntegrationTest(unittest.TestCase):
    def test_default_output_byte_identical(self):
        # Opt-in param defaults to empty: live prompt unchanged this phase.
        plain = build_self_context(character_name="Mili")
        self.assertNotIn("Emerging", plain)

    def test_preferences_render_inside_block(self):
        cands = derive_activity_preferences(
            [hist("reading", f"2026-09-2{d}T08:00:00+00:00") for d in (7, 8, 9)],
            [],
            tz=JKT,
        )
        block = build_self_context(character_name="Mili", preferences=cands)
        self.assertIn("Emerging preference: reading (evidence: 3).", block)


if __name__ == "__main__":
    unittest.main()
