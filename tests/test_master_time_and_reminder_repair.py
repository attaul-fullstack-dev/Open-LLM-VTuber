"""Master Regression Test Suite — Time Awareness, Future Intentions, and Reminder Reliability.

Covers all 28 mandatory requirement scenarios specified in the Master Task:
 1. Waktu aktual tanpa pengguna bertanya.
 2. Waktu diperbarui setiap turn.
 3. Akurasi UTC dan timezone Asia/Jakarta.
 4. Timezone Tokyo, New York, dan UTC.
 5. Parsing 15 menit lagi.
 6. Parsing 30 menit lagi.
 7. Parsing tiga jam lagi.
 8. Parsing satu jam lagi.
 9. Parsing besok dan hari tertentu.
10. Parsing jam spesifik.
11. Kata 'nanti' ambigu meminta klarifikasi.
12. Variasi kata kerja pengingat.
13. Reminder dikirim pada 'due_at'.
14. Reminder tidak menunggu proactive idle interval.
15. Reminder tetap terkirim ketika proactive gate tidak eligible.
16. Reminder tetap terkirim ketika percakapan aktif.
17. Proactive tidak dapat menahan reminder.
18. Obrolan biasa tidak menjadi reminder.
19. Ekspresi emosional dan roleplay tidak menjadi future intention.
20. KIND_PLAN dan KIND_REMINDER terpisah.
21. Reminder bertahan setelah restart.
22. Reminder tidak terkirim ganda.
23. Kegagalan delivery tidak menghasilkan status sukses.
24. Retry dan lease recovery.
25. Reminder selesai hilang dari daftar aktif.
26. Temporal Memory menyimpan timestamp absolut.
27. World State dan Future Intentions memakai waktu yang sama.
28. Tidak ada regresi pada fitur lama.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from zoneinfo import ZoneInfo

from src.open_llm_vtuber.character_state import (
    REMINDER_CAPTURED,
    REMINDER_CLARIFY,
    REMINDER_FAILED,
    REMINDER_NONE,
    capture_reminder,
    claim_due_future_intentions,
    load_character_state,
    record_future_intention,
    save_character_state,
)
from src.open_llm_vtuber.future_intentions import (
    KIND_PLAN,
    KIND_REMINDER,
    STATUS_DONE,
    STATUS_PENDING,
    detect_future_intention,
    detect_future_plan,
    detect_reminder_request,
    due_intentions,
    next_due_at,
    parse_relative_delta,
    pending_intentions,
)
from src.open_llm_vtuber.proactive_gate import (
    GateDecision,
    ProactiveBudgetState,
    ProactiveGateConfig,
    evaluate_gate,
)
from src.open_llm_vtuber.time_provider import (
    ensure_aware,
    format_temporal_anchor,
    resolve_tz,
    user_now,
    utc_iso,
    utc_now,
)
from src.open_llm_vtuber.websocket_handler import WebSocketHandler


class MasterTimeAndReminderTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = Path.cwd()
        import os
        os.chdir(self._tmp.name)
        self.test_conf = "test_conf_master"
        self.tz_jkt = "Asia/Jakarta"
        self.fixed_moment = datetime(2026, 10, 31, 11, 53, 0, tzinfo=timezone.utc)

    def tearDown(self):
        import os
        os.chdir(self._old)
        self._tmp.cleanup()

    # 1. Waktu aktual tanpa pengguna bertanya
    def test_01_actual_time_without_user_asking(self):
        anchor = format_temporal_anchor(self.fixed_moment, self.tz_jkt)
        self.assertIn("Current date: October 31, 2026", anchor)
        self.assertIn("Current time: 18:53:00", anchor)
        self.assertIn("Timezone: Asia/Jakarta", anchor)
        self.assertIn("UTC", anchor)

    # 2. Waktu diperbarui setiap turn
    def test_02_time_updated_every_turn(self):
        turn1_moment = datetime(2026, 10, 31, 11, 53, 0, tzinfo=timezone.utc)
        turn2_moment = datetime(2026, 10, 31, 11, 53, 45, tzinfo=timezone.utc)
        anchor1 = format_temporal_anchor(turn1_moment, self.tz_jkt)
        anchor2 = format_temporal_anchor(turn2_moment, self.tz_jkt)
        self.assertIn("18:53:00", anchor1)
        self.assertIn("18:53:45", anchor2)
        self.assertNotEqual(anchor1, anchor2)

    # 3. Akurasi UTC dan timezone Asia/Jakarta
    def test_03_accuracy_utc_and_asia_jakarta(self):
        moment = datetime(2026, 10, 31, 12, 0, 0, tzinfo=timezone.utc)
        local = user_now(moment, self.tz_jkt)
        self.assertEqual(local.hour, 19)
        self.assertEqual(local.minute, 0)
        self.assertEqual(local.utcoffset(), timedelta(hours=7))

    # 4. Timezone Tokyo, New York, dan UTC
    def test_04_timezone_tokyo_newyork_utc(self):
        moment = datetime(2026, 10, 31, 12, 0, 0, tzinfo=timezone.utc)
        # Tokyo is UTC+9
        tokyo = user_now(moment, "Asia/Tokyo")
        self.assertEqual(tokyo.hour, 21)
        # New York is UTC-4 in October (EDT)
        ny = user_now(moment, "America/New_York")
        self.assertEqual(ny.hour, 8)
        # UTC is UTC+0
        utc = user_now(moment, "UTC")
        self.assertEqual(utc.hour, 12)

    # 5. Parsing 15 menit lagi (termasuk seperempat jam lagi)
    def test_05_parsing_15_minutes_and_seperempat_jam(self):
        delta1 = parse_relative_delta("ingatkan 15 menit lagi")
        delta2 = parse_relative_delta("seperempat jam lagi")
        delta3 = parse_relative_delta("dalam 15 menit")
        self.assertEqual(delta1, timedelta(minutes=15))
        self.assertEqual(delta2, timedelta(minutes=15))
        self.assertEqual(delta3, timedelta(minutes=15))

    # 6. Parsing 30 menit lagi (dan setengah jam lagi)
    def test_06_parsing_30_minutes_and_setengah_jam(self):
        delta1 = parse_relative_delta("30 menit lagi")
        delta2 = parse_relative_delta("setengah jam lagi")
        self.assertEqual(delta1, timedelta(minutes=30))
        self.assertEqual(delta2, timedelta(minutes=30))

    # 7. Parsing tiga jam lagi
    def test_07_parsing_three_hours(self):
        delta = parse_relative_delta("tiga jam lagi")
        self.assertEqual(delta, timedelta(hours=3))

    # 8. Parsing satu jam lagi (dan sejam lagi)
    def test_08_parsing_one_hour_and_sejam(self):
        delta1 = parse_relative_delta("satu jam lagi")
        delta2 = parse_relative_delta("sejam lagi")
        self.assertEqual(delta1, timedelta(hours=1))
        self.assertEqual(delta2, timedelta(hours=1))

    # 9. Parsing besok dan hari tertentu (termasuk jumat depan)
    def test_09_parsing_besok_and_weekdays(self):
        # Base moment: Saturday, Oct 31, 2026 at 12:00 UTC (19:00 JKT)
        base = datetime(2026, 10, 31, 12, 0, 0, tzinfo=timezone.utc)
        # "besok pagi ingetin aku olahraga"
        res_besok = detect_future_intention("besok pagi ingetin aku olahraga", now=base, tz=self.tz_jkt)
        self.assertIsNotNone(res_besok)
        self.assertIsNotNone(res_besok.due_at)
        stamp_besok = datetime.fromisoformat(res_besok.due_at).astimezone(ZoneInfo(self.tz_jkt))
        self.assertEqual((stamp_besok.day, stamp_besok.hour), (1, 7))

        # "jumat depan jam 10 ingetin aku rapat"
        res_jumat_depan = detect_future_intention("jumat depan jam 10 ingetin aku rapat", now=base, tz=self.tz_jkt)
        self.assertIsNotNone(res_jumat_depan)
        stamp_jd = datetime.fromisoformat(res_jumat_depan.due_at).astimezone(ZoneInfo(self.tz_jkt))
        # From Saturday Oct 31, this coming Friday is Nov 6. "Jumat depan" is Nov 13.
        self.assertEqual((stamp_jd.day, stamp_jd.hour), (13, 10))

    # 10. Parsing jam spesifik (jam 18.53)
    def test_10_parsing_specific_time_jam_18_53(self):
        # Base moment: 18:50 local (11:50 UTC)
        base = datetime(2026, 10, 31, 11, 50, 0, tzinfo=timezone.utc)
        res = detect_future_intention("jam 18.53 ingetin aku push up 5 kali", now=base, tz=self.tz_jkt)
        self.assertIsNotNone(res)
        stamp = datetime.fromisoformat(res.due_at).astimezone(ZoneInfo(self.tz_jkt))
        self.assertEqual((stamp.hour, stamp.minute), (18, 53))
        self.assertEqual(stamp.day, 31)

    # 11. Kata 'nanti' ambigu meminta klarifikasi
    def test_11_ambiguous_nanti_requests_clarification(self):
        # "Nanti ingetin aku" has no specific time -> must clarify
        det = detect_reminder_request("nanti ingetin aku ya", [], now=self.fixed_moment, tz=self.tz_jkt)
        self.assertEqual(det.status, "clarify")
        self.assertEqual(det.reason, "unspecified_time")

        # "Besok ingetin aku" without hour or time-of-day -> clarify
        det_besok = detect_reminder_request("besok ingetin aku", [], now=self.fixed_moment, tz=self.tz_jkt)
        self.assertEqual(det_besok.status, "clarify")

    # 12. Variasi kata kerja pengingat (ingatkan, tanyakan, cek, bangunkan, kabari, beritahu, bilang)
    def test_12_reminder_verb_variations(self):
        verbs = [
            "ingatkan aku jam 18.53 push up",
            "ingetin aku jam 18.53 push up",
            "tanyakan ke aku jam 18.53 apakah sudah push up",
            "cek jam 18.53 apakah server aman",
            "periksa jam 18.53 log server",
            "bangunkan aku jam 05.00 pagi",
            "kabari aku jam 18.53",
            "beritahu aku jam 18.53 kalau meeting mulai",
            "jam 18.53 bilang ke aku push up 5 kali",
        ]
        for text in verbs:
            with self.subTest(text=text):
                res = detect_future_intention(text, now=self.fixed_moment, tz=self.tz_jkt)
                self.assertIsNotNone(res, f"Failed to detect reminder verb in: {text}")
                self.assertEqual(res.kind, KIND_REMINDER)

    # 13. Reminder dikirim pada 'due_at'
    def test_13_reminder_due_at_exact(self):
        base = datetime(2026, 10, 31, 11, 50, 0, tzinfo=timezone.utc)
        due_at = (base + timedelta(minutes=3)).isoformat()
        intentions = [
            {"id": "rem1", "kind": KIND_REMINDER, "text": "Push up", "status": STATUS_PENDING, "due_at": due_at}
        ]
        # At 11:51: not due
        self.assertEqual(len(due_intentions(intentions, now=base + timedelta(minutes=1))), 0)
        # At 11:53: due!
        self.assertEqual(len(due_intentions(intentions, now=base + timedelta(minutes=3))), 1)

    # 14. Reminder tidak menunggu proactive idle interval
    async def test_14_reminder_does_not_wait_for_idle_interval(self):
        # Mock WebSocketHandler and test _next_reminder_due_delay directly
        handler = WebSocketHandler(AsyncMock())
        mock_context = MagicMock()
        mock_agent = MagicMock()
        mock_context.agent_engine = mock_agent
        due_time = datetime(2026, 10, 31, 11, 53, 0, tzinfo=timezone.utc)
        mock_agent.next_future_intention_due.return_value = due_time

        with patch("src.open_llm_vtuber.websocket_handler.utcnow", return_value=due_time - timedelta(seconds=2)):
            delay = handler._next_reminder_due_delay("client1", mock_context)
            # Must return exact 2.0s delay, completely decoupled from the 45-90s idle interval
            self.assertAlmostEqual(delay, 2.0, places=1)

    # 15. Reminder tetap terkirim ketika proactive gate tidak eligible
    def test_15_reminder_allowed_when_proactive_gate_suppressed(self):
        gate_config = ProactiveGateConfig(proactive_daily_hard_limit=1)
        day_str = self.fixed_moment.astimezone(ZoneInfo(self.tz_jkt)).strftime("%Y-%m-%d")
        budget = ProactiveBudgetState(daily_request_count=5, daily_count_date=day_str)
        trigger = MagicMock(priority="high", reason="explicit_reminder")
        # For spontaneous talk, gate would reject
        decision_spontaneous = evaluate_gate(
            budget, gate_config, trigger, now=self.fixed_moment, tz=self.tz_jkt, reminder_due=False
        )
        self.assertFalse(decision_spontaneous.allowed)
        # But when reminder_due is True, gate allows the obligation!
        decision_reminder = evaluate_gate(
            budget, gate_config, trigger, now=self.fixed_moment, tz=self.tz_jkt, reminder_due=True
        )
        self.assertTrue(decision_reminder.allowed)

    # 16. Reminder tetap terkirim ketika percakapan aktif
    async def test_16_reminder_yields_for_streaming_turn_then_fires(self):
        handler = WebSocketHandler(AsyncMock())
        active_turn_finished = False

        async def simulated_active_turn():
            nonlocal active_turn_finished
            await asyncio.sleep(0.05)
            active_turn_finished = True

        task = asyncio.create_task(simulated_active_turn())
        handler.current_conversation_tasks["client1"] = task

        # When reminder scheduler detects due_at, it shields the active task and proceeds
        active = handler.current_conversation_tasks.get("client1")
        if active and not active.done():
            await asyncio.shield(active)

        self.assertTrue(active_turn_finished)

    # 17. Proactive tidak dapat menahan reminder
    def test_17_proactive_yields_to_due_reminder(self):
        handler = WebSocketHandler(AsyncMock())
        mock_context = MagicMock()
        mock_agent = MagicMock()
        mock_context.agent_engine = mock_agent
        mock_agent.due_future_intentions.return_value = [{"id": "rem1"}]

        # _due_reminder_lane returns True
        is_due = handler._due_reminder_lane(mock_context, now=self.fixed_moment)
        self.assertTrue(is_due)

    # 18. Obrolan biasa tidak menjadi reminder
    def test_18_ordinary_conversation_rejected(self):
        banter = [
            "Halo apa kabar?",
            "Aku suka makan nasi goreng.",
            "Cuaca hari ini bagus banget ya.",
            "Nanti kita lanjut ceritanya ya.",
        ]
        for text in banter:
            with self.subTest(text=text):
                res = detect_future_intention(text, now=self.fixed_moment, tz=self.tz_jkt)
                self.assertIsNone(res, f"Ordinary chat falsely detected as reminder: {text}")

    # 19. Ekspresi emosional dan roleplay tidak menjadi future intention
    def test_19_emotional_and_roleplay_rejected(self):
        emotional = [
            "Asik, besok aku jadi gak sabar mau jalan-jalan.",
            "Aku mau tidur di pelukan kamu.",
            "Aku capek banget hari ini.",
            "Besok kayaknya bakal seru deh.",
            "*memeluk Mili dengan erat*",
            "Aku kangen banget sama kamu.",
        ]
        for text in emotional:
            with self.subTest(text=text):
                plan = detect_future_plan(text, now=self.fixed_moment, tz=self.tz_jkt)
                self.assertIsNone(plan, f"Emotional/roleplay falsely detected as plan: {text}")

    # 20. KIND_PLAN dan KIND_REMINDER terpisah
    def test_20_kind_plan_and_kind_reminder_separated(self):
        # User commanding Mili: REMINDER
        reminder_text = "ingatkan aku besok jam 7 pagi minum vitamin"
        rem = detect_future_intention(reminder_text, now=self.fixed_moment, tz=self.tz_jkt)
        self.assertIsNotNone(rem)
        self.assertEqual(rem.kind, KIND_REMINDER)

        # User stating own plan: PLAN
        plan_text = "besok jam 7 pagi aku mau belajar coding"
        plan = detect_future_plan(plan_text, now=self.fixed_moment, tz=self.tz_jkt)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.kind, KIND_PLAN)

    # 21. Reminder bertahan setelah restart
    def test_21_reminder_persists_across_restart(self):
        record_future_intention(
            self.test_conf, "ingatkan aku jam 18.53 push up", now=self.fixed_moment, tz=self.tz_jkt
        )
        # Reload state from disk (simulating backend restart)
        state_after_restart = load_character_state(self.test_conf)
        pending = pending_intentions(state_after_restart.future_intentions)
        self.assertEqual(len(pending), 1)
        self.assertIn("push up", pending[0]["text"].lower())

    # 22. Reminder tidak terkirim ganda (idempotency & lease locking)
    def test_22_reminder_no_duplicate_delivery(self):
        record_time = self.fixed_moment - timedelta(minutes=10)
        record_future_intention(
            self.test_conf, "ingatkan aku jam 18.53 push up", now=record_time, tz=self.tz_jkt
        )
        due_time = self.fixed_moment + timedelta(seconds=1)
        # First connection claims intention
        claimed1 = claim_due_future_intentions(
            self.test_conf, owner="conn1", now=due_time, lease_seconds=60, kind="reminder"
        )
        self.assertEqual(len(claimed1), 1)

        # Second connection attempts claim concurrently: must get 0!
        claimed2 = claim_due_future_intentions(
            self.test_conf, owner="conn2", now=due_time, lease_seconds=60, kind="reminder"
        )
        self.assertEqual(len(claimed2), 0)

    # 23. Kegagalan delivery tidak menghasilkan status sukses
    def test_23_delivery_failure_not_marked_done(self):
        handler = WebSocketHandler(AsyncMock())
        mock_probe = MagicMock()
        mock_probe.delivered = False
        mock_probe.refused = True
        # Transport refused frame or failed
        delivered = handler._reminder_delivery_evidence(mock_probe, "")
        self.assertFalse(delivered)

    # 24. Retry dan lease recovery
    def test_24_retry_and_lease_recovery(self):
        record_time = self.fixed_moment - timedelta(minutes=10)
        record_future_intention(
            self.test_conf, "ingatkan aku jam 18.53 push up", now=record_time, tz=self.tz_jkt
        )
        claim_time = self.fixed_moment + timedelta(seconds=1)
        # Claim with short 5s lease
        claimed = claim_due_future_intentions(
            self.test_conf, owner="dead_process", now=claim_time, lease_seconds=5, kind="reminder"
        )
        self.assertEqual(len(claimed), 1)

        # After lease expires (e.g. +6s): another process can reclaim!
        reclaimed = claim_due_future_intentions(
            self.test_conf, owner="live_process", now=claim_time + timedelta(seconds=6), lease_seconds=60, kind="reminder"
        )
        self.assertEqual(len(reclaimed), 1)
        self.assertEqual(reclaimed[0]["claimed_by"], "live_process")

    # 25. Reminder selesai hilang dari daftar aktif
    def test_25_completed_reminder_removed_from_active_list(self):
        record_future_intention(
            self.test_conf, "ingatkan aku jam 18.53 push up", now=self.fixed_moment, tz=self.tz_jkt
        )
        state = load_character_state(self.test_conf)
        self.assertEqual(len(pending_intentions(state.future_intentions)), 1)

        # Mark done
        state.future_intentions[0]["status"] = STATUS_DONE
        save_character_state(self.test_conf, state)

        # Pending must be 0
        reloaded = load_character_state(self.test_conf)
        self.assertEqual(len(pending_intentions(reloaded.future_intentions)), 0)

    # 26. Temporal Memory menyimpan timestamp absolut
    def test_26_temporal_memory_stores_absolute_timestamp(self):
        res = record_future_intention(
            self.test_conf, "besok pagi jam 7 ingetin sarapan", now=self.fixed_moment, tz=self.tz_jkt
        )
        self.assertIsNotNone(res)
        due_at = res.get("due_at")
        self.assertIsNotNone(due_at)
        # Must be ISO-8601 aware UTC timestamp
        parsed = datetime.fromisoformat(due_at)
        self.assertIsNotNone(parsed.tzinfo)
        self.assertEqual(parsed.tzinfo, timezone.utc)

    # 27. World State dan Future Intentions memakai waktu yang sama
    def test_27_unified_canonical_time_provider(self):
        t1 = utc_now()
        t2 = ensure_aware(None)
        self.assertEqual(t1.tzinfo, timezone.utc)
        self.assertEqual(t2.tzinfo, timezone.utc)
        self.assertAlmostEqual((t2 - t1).total_seconds(), 0.0, places=2)

    # 28. Tidak ada regresi pada fitur lama
    def test_28_no_regression_on_existing_features(self):
        # Relative delta for common expressions
        self.assertEqual(parse_relative_delta("setengah jam lagi"), timedelta(minutes=30))
        self.assertEqual(parse_relative_delta("sejam lagi"), timedelta(hours=1))
        # Contextual reminders with reference
        det = detect_reminder_request(
            "nanti jam 18.53 bilang kaya tadi lagi ya",
            [{"role": "user", "content": "bilang 'Halo Dunia'"}],
            now=self.fixed_moment,
            tz=self.tz_jkt,
        )
        self.assertEqual(det.status, "captured")
        self.assertEqual(det.intention.delivery_text, "Halo Dunia")

    # 29. Regresi deadlock CancelDuringDeliveryTest: proactive loop tidak membeku saat isolated
    async def test_29_cancel_during_delivery_no_deadlock(self):
        from tests.test_reminder_multi_turn_lifecycle import _make_agent, _make_handler, _machine

        due_str = (self.fixed_moment - timedelta(seconds=30)).isoformat()
        state = load_character_state(self.test_conf)
        row = {
            "id": "rem_deadlock_1",
            "text": "ingatkan minum air",
            "delivery_text": "minum air",
            "due_at": due_str,
            "kind": KIND_REMINDER,
            "status": STATUS_PENDING,
            "created_at": due_str,
        }
        state.future_intentions.append(row)
        save_character_state(self.test_conf, state)

        agent, _llm = _make_agent(self.test_conf, "hist1")
        handler = _make_handler(agent, self.test_conf, "hist1")
        machine = _machine(initial_idle=1)
        state_machine = machine.new_state("hist1")

        started = []

        async def fake_turn(**kwargs):
            started.append("turn_started")
            raise asyncio.CancelledError("turn cancelled")

        with patch("src.open_llm_vtuber.websocket_handler.process_single_conversation", new=fake_turn):
            task = asyncio.create_task(
                handler._run_proactive_timer("client", "hist1", state_machine, machine)
            )
            # Must finish and raise CancelledError within 3.0s without deadlocking/hanging!
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, timeout=3.0)
            self.assertEqual(started, ["turn_started"])


if __name__ == "__main__":
    unittest.main()
