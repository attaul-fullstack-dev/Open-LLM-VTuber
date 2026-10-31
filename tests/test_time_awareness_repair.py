"""Regression test suite for Mili Time Awareness & Reminder Reliability Repair.

Covers the 12 mandatory scenarios:
1. Actual time acquired dynamically from canonical time provider without user asking.
2. UTC vs user timezone conversion handled accurately across multiple timezones.
3. Relative reminders ("30 menit lagi", "tiga jam lagi", "1 jam lagi") calculate exact due timestamps.
4. Specific hour & day anchors ("Nanti jam 11.51 ingetin aku", "Hari Jumat tanyakan...") parsed correctly,
   bare "nanti" triggers clarification.
5. Reminders survive backend restart (persisted in character_state JSON).
6. Duplicate delivery prevention (delivery journal and lease locking).
7. Delivered reminder disappears from active list (pending_intentions).
8. Failed reminder does not claim false success (retains pending status on failure/release).
9. Distinguishes explicit reminder, future intention, and ordinary/banter/emotional statements (rejection).
10. Proactive behavior does not run without relevant intention/reason.
11. Temporal memory stores absolute timestamp and timezone, rendered dynamically.
12. Life State (world_state) and core systems read unified canonical time.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from src.open_llm_vtuber.character_state import (
    claim_due_future_intentions,
    complete_claimed_future_intentions,
    complete_future_intention,
    load_character_state,
    record_future_intention,
    release_claimed_future_intentions,
    save_character_state,
)
from src.open_llm_vtuber.future_intentions import (
    KIND_PLAN,
    KIND_REMINDER,
    STATUS_DONE,
    STATUS_PENDING,
    detect_any_future_intention,
    detect_future_intention,
    detect_future_plan,
    detect_reminder_request,
    due_intentions,
    parse_relative_delta,
    pending_intentions,
)
from src.open_llm_vtuber.reminder_delivery import (
    delivered_ids,
    get_delivery_journal_path,
    record_delivery,
)
from src.open_llm_vtuber.time_provider import (
    ensure_aware,
    format_temporal_anchor,
    local_hour,
    memory_age_label,
    parse_iso,
    relative_day_parts,
    resolve_tz,
    user_local_datetime,
    user_now,
    utc_iso,
    utc_now,
    utcnow,
)
import src.open_llm_vtuber.world_state as ws


class TestScenario1GlobalTimeAwareness(unittest.TestCase):
    """Scenario 1: Actual time acquired without user asking."""

    def test_system_prompt_temporal_anchor_generation(self):
        moment = datetime(2026, 10, 31, 4, 30, 0, tzinfo=timezone.utc)
        tz_jkt = "Asia/Jakarta"
        anchor = format_temporal_anchor(moment=moment, tz=tz_jkt)
        self.assertIn("Current date: October 31, 2026 (Saturday)", anchor)
        self.assertIn("Current time: 11:30", anchor)
        self.assertIn("Timezone: Asia/Jakarta", anchor)

    def test_canonical_time_provider_pure_clock(self):
        now_utc = utc_now()
        self.assertIsNotNone(now_utc.tzinfo)
        self.assertEqual(now_utc.tzinfo, timezone.utc)
        now_jkt = user_now(tz="Asia/Jakarta")
        self.assertIsNotNone(now_jkt.tzinfo)
        # Jakarta is UTC+7
        delta = now_jkt.utcoffset()
        self.assertEqual(delta, timedelta(hours=7))


class TestScenario2TimezoneDifferences(unittest.TestCase):
    """Scenario 2: UTC vs user timezone differences handled accurately."""

    def test_multi_timezone_conversion(self):
        # 2026-10-31 03:00 UTC
        base_utc = datetime(2026, 10, 31, 3, 0, 0, tzinfo=timezone.utc)
        # Jakarta is UTC+7 -> 10:00 AM
        jkt_time = user_local_datetime(base_utc, "Asia/Jakarta")
        self.assertEqual(jkt_time.hour, 10)
        self.assertEqual(local_hour(base_utc, "Asia/Jakarta"), 10)

        # Tokyo is UTC+9 -> 12:00 PM (Noon)
        tokyo_time = user_local_datetime(base_utc, "Asia/Tokyo")
        self.assertEqual(tokyo_time.hour, 12)
        self.assertEqual(local_hour(base_utc, "Asia/Tokyo"), 12)

        # New York is UTC-4 (EDT in late Oct) -> 23:00 previous day
        ny_time = user_local_datetime(base_utc, "America/New_York")
        self.assertEqual(ny_time.hour, 23)
        self.assertEqual(ny_time.day, 30)

        # Unknown timezone falls back safely to UTC without exception
        fallback_time = user_local_datetime(base_utc, "Invalid/Nonexistent_Timezone")
        self.assertEqual(fallback_time.hour, 3)

    def test_iso_storage_canonical_utc(self):
        local_dt = datetime(2026, 10, 31, 10, 0, 0, tzinfo=ZoneInfo("Asia/Jakarta"))
        iso_str = utc_iso(local_dt)
        self.assertEqual(iso_str, "2026-10-31T03:00:00+00:00")
        parsed = parse_iso(iso_str)
        self.assertEqual(parsed.tzinfo, timezone.utc)
        self.assertEqual(parsed.hour, 3)


class TestScenario3RelativeReminders(unittest.TestCase):
    """Scenario 3: Relative reminders calculate exact due timestamps."""

    def test_indonesian_relative_expressions(self):
        now_fixed = datetime(2026, 10, 31, 10, 0, 0, tzinfo=timezone.utc)
        tz = "Asia/Jakarta"

        cases = [
            ("30 menit lagi ingetin aku minum obat", timedelta(minutes=30)),
            ("tiga jam lagi tolong ingetin jemput adik", timedelta(hours=3)),
            ("1 jam lagi ingetin cek oven ya", timedelta(hours=1)),
            ("sejam lagi tolong ingetin kirim email", timedelta(hours=1)),
            ("setengah jam lagi bangunin aku ya", timedelta(minutes=30)),
            ("dalam 15 menit tolong ingetin telepon bos", timedelta(minutes=15)),
            ("2 hari lagi kabari aku soal progress", timedelta(days=2)),
            ("sepuluh menit lagi tanyain apakah tugas sudah beres", timedelta(minutes=10)),
        ]
        for text, expected_delta in cases:
            with self.subTest(text=text):
                found = detect_future_intention(text, now=now_fixed, tz=tz)
                self.assertIsNotNone(found, f"Failed to detect reminder in: {text}")
                self.assertEqual(found.kind, KIND_REMINDER)
                self.assertIsNotNone(found.due_at)
                expected_due = (now_fixed + expected_delta).isoformat(timespec="seconds")
                self.assertEqual(found.due_at, expected_due)

    def test_parse_relative_delta_pure_helper(self):
        self.assertEqual(parse_relative_delta("dalam 45 menit lagi"), timedelta(minutes=45))
        self.assertEqual(parse_relative_delta("dua jam lagi"), timedelta(hours=2))
        self.assertEqual(parse_relative_delta("seminggu lagi"), timedelta(days=7))
        self.assertEqual(parse_relative_delta("sehari lagi"), timedelta(days=1))
        self.assertIsNone(parse_relative_delta("nanti sore"))


class TestScenario4SpecificHourAndBareNanti(unittest.TestCase):
    """Scenario 4: Specific hour reminders & day anchors parsed correctly, bare 'nanti' asks clarification."""

    def test_specific_clock_reminder(self):
        # 2026-10-31 Saturday 04:00 UTC = 11:00 Jakarta
        now_fixed = datetime(2026, 10, 31, 4, 0, 0, tzinfo=timezone.utc)
        tz = "Asia/Jakarta"
        # User asks: "Nanti jam 11.51 ingetin aku minum obat"
        found = detect_future_intention("Nanti jam 11.51 ingetin aku minum obat", now=now_fixed, tz=tz)
        self.assertIsNotNone(found)
        self.assertEqual(found.kind, KIND_REMINDER)
        self.assertIsNotNone(found.due_at)

        due_dt = parse_iso(found.due_at)
        due_jkt = due_dt.astimezone(ZoneInfo(tz))
        self.assertEqual(due_jkt.hour, 11)
        self.assertEqual(due_jkt.minute, 51)
        self.assertEqual(due_jkt.day, 31)

    def test_weekday_reminder(self):
        # Saturday 2026-10-31
        now_fixed = datetime(2026, 10, 31, 4, 0, 0, tzinfo=timezone.utc)
        tz = "Asia/Jakarta"
        # Next Friday is 2026-11-06
        found = detect_future_intention("Hari Jumat tanyakan apakah tugas sudah beres", now=now_fixed, tz=tz)
        self.assertIsNotNone(found)
        self.assertEqual(found.kind, KIND_REMINDER)
        self.assertIsNotNone(found.due_at)

        due_dt = parse_iso(found.due_at)
        due_jkt = due_dt.astimezone(ZoneInfo(tz))
        self.assertEqual(due_jkt.weekday(), 4)  # 4 = Friday
        self.assertEqual(due_jkt.day, 6)
        self.assertEqual(due_jkt.month, 11)

    def test_bare_nanti_requires_clarification(self):
        now_fixed = datetime(2026, 10, 31, 4, 0, 0, tzinfo=timezone.utc)
        tz = "Asia/Jakarta"
        # "nanti ingetin aku ya" without clock or interval must NOT be hardcoded to +3h!
        found = detect_future_intention("nanti ingetin aku ya", now=now_fixed, tz=tz)
        self.assertIsNotNone(found)
        self.assertEqual(found.kind, KIND_REMINDER)
        self.assertIsNone(found.due_at, "Bare 'nanti' must not guess +3h, due_at must be None")


class TestScenario5BackendRestartPersistence(unittest.TestCase):
    """Scenario 5: Reminders survive backend restart via character_state persistence."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_cwd = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs("character_state", exist_ok=True)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def test_persistence_across_process_restart(self):
        conf = "mili_test_restart"
        now_fixed = datetime(2026, 10, 31, 4, 0, 0, tzinfo=timezone.utc)
        tz = "Asia/Jakarta"
        text = "besok jam 9 ingetin aku kirim laporan"

        # Step 1: Detect and record a reminder
        stored = record_future_intention(conf, text, now=now_fixed, tz=tz)
        self.assertIsNotNone(stored)

        # Step 2: Verify it is in state file on disk
        state_file = f"character_state/{conf}.json"
        self.assertTrue(os.path.exists(state_file))
        with open(state_file, "r", encoding="utf-8") as f:
            disk_data = json.load(f)
        self.assertEqual(len(disk_data.get("future_intentions", [])), 1)

        # Step 3: Simulate backend restart: fresh process reads file
        reloaded_state = load_character_state(conf)
        self.assertEqual(len(reloaded_state.future_intentions), 1)
        persisted = reloaded_state.future_intentions[0]
        self.assertEqual(persisted["text"], stored["text"])
        self.assertEqual(persisted["due_at"], stored["due_at"])
        self.assertEqual(persisted["status"], STATUS_PENDING)
        self.assertEqual(persisted["kind"], KIND_REMINDER)


class TestScenario6DuplicateDeliveryPrevention(unittest.TestCase):
    """Scenario 6: No duplicate deliveries via delivery journal and lease lock."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_cwd = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs("character_state", exist_ok=True)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def test_journal_and_lease_prevent_duplicates(self):
        conf = "mili_test_dedup"
        moment = datetime(2026, 10, 31, 10, 0, 0, tzinfo=timezone.utc)
        stored = record_future_intention(conf, "ingetin minum obat jam 9", now=moment, tz="Asia/Jakarta")
        self.assertIsNotNone(stored)

        state = load_character_state(conf)
        item_id = state.future_intentions[0]["id"]
        # Make it due
        state.future_intentions[0]["due_at"] = (moment - timedelta(minutes=5)).isoformat()
        save_character_state(conf, state)

        # 1. First worker claims the due reminder
        claimed = claim_due_future_intentions(conf, owner="worker-1", now=moment)
        self.assertEqual(len(claimed), 1)
        self.assertEqual(claimed[0]["id"], item_id)

        # 2. Concurrent worker cannot claim the same reminder (lease locked)
        claimed_concurrent = claim_due_future_intentions(conf, owner="worker-2", now=moment)
        self.assertEqual(len(claimed_concurrent), 0)

        # 3. Mark delivery in persistent journal
        self.assertNotIn(item_id, delivered_ids(conf))
        self.assertTrue(record_delivery(conf, [item_id], owner="worker-1", now=moment))
        self.assertIn(item_id, delivered_ids(conf))

        # 4. Even if lease expires or releases, delivery journal blocks re-delivery
        release_claimed_future_intentions(conf, owner="worker-1", intention_ids=[item_id])
        claimed_after_release = claim_due_future_intentions(conf, owner="worker-2", now=moment + timedelta(minutes=15))
        self.assertEqual(len(claimed_after_release), 0, "Delivery journal must prevent re-claiming already delivered reminder")


class TestScenario7DeliveredDisappearsFromActiveList(unittest.TestCase):
    """Scenario 7: Delivered reminder disappears from active list (pending_intentions)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_cwd = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs("character_state", exist_ok=True)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def test_completed_reminder_disappears(self):
        conf = "mili_test_lifecycle"
        now_fixed = datetime(2026, 10, 31, 4, 0, 0, tzinfo=timezone.utc)
        tz = "Asia/Jakarta"

        stored = record_future_intention(conf, "1 jam lagi ingetin aku minum obat", now=now_fixed, tz=tz)
        self.assertIsNotNone(stored)

        state = load_character_state(conf)
        self.assertEqual(len(pending_intentions(state.future_intentions)), 1)
        item_id = state.future_intentions[0]["id"]

        # Mark completed upon successful delivery
        completed = complete_future_intention(conf, item_id)
        self.assertTrue(completed)

        # Active list is now empty!
        reloaded = load_character_state(conf)
        active = pending_intentions(reloaded.future_intentions)
        self.assertEqual(len(active), 0)
        self.assertEqual(reloaded.future_intentions[0]["status"], STATUS_DONE)


class TestScenario8FailedReminderDoesNotClaimSuccess(unittest.TestCase):
    """Scenario 8: Failed reminder does not claim false success, stays pending."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old_cwd = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs("character_state", exist_ok=True)

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def test_failed_delivery_releases_lease_and_retains_pending(self):
        conf = "mili_test_failure"
        now_fixed = datetime(2026, 10, 31, 4, 0, 0, tzinfo=timezone.utc)
        tz = "Asia/Jakarta"

        stored = record_future_intention(conf, "30 menit lagi ingetin cek server", now=now_fixed, tz=tz)
        self.assertIsNotNone(stored)

        state = load_character_state(conf)
        item_id = state.future_intentions[0]["id"]
        # Make due
        state.future_intentions[0]["due_at"] = (now_fixed - timedelta(minutes=1)).isoformat()
        save_character_state(conf, state)

        # Acquire lease for delivery attempt
        claimed = claim_due_future_intentions(conf, owner="worker-a", now=now_fixed)
        self.assertEqual(len(claimed), 1)

        # Delivery attempt fails (e.g. client disconnect, LLM error)
        # Release lease without writing delivery receipt or calling complete
        release_claimed_future_intentions(conf, owner="worker-a", intention_ids=[item_id])

        # Intention MUST still be pending and not marked delivered
        self.assertNotIn(item_id, delivered_ids(conf))
        reloaded = load_character_state(conf)
        active = pending_intentions(reloaded.future_intentions)
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["status"], STATUS_PENDING)


class TestScenario9FutureIntentionsSeparationAndFalsePositiveRejection(unittest.TestCase):
    """Scenario 9: Differentiate explicit reminder, future intention, and ordinary/banter chat."""

    def test_casual_banter_and_emotion_rejected(self):
        now_fixed = datetime(2026, 10, 31, 4, 0, 0, tzinfo=timezone.utc)
        tz = "Asia/Jakarta"

        false_positive_cases = [
            "asik, besok aku jadi gak sabar 🥵",
            "nanti malam aku pengen peluk kamu",
            "besok aku capek banget kayaknya",
            "besok aku kangen banget sama kamu",
            "nanti minta jatah ya sayang",
            "besok aku pusing mikirin kerjaan",
            "besok aku mager banget",
            "besok aku mau terbang ke bulan wkwk",
            "canda besok aku mau jadi astronaut",
            "tiap hari aku selalu bangun pagi",
            "kalau besok hujan mungkin aku gak pergi",
            "besok kamu ada acara apa?",
            "hari ini aku libur, mau santai aja",
        ]
        for text in false_positive_cases:
            with self.subTest(text=text):
                plan = detect_future_plan(text, now=now_fixed, tz=tz)
                self.assertIsNone(plan, f"Casual/banter text should NOT be detected as plan: {text}")
                reminder = detect_future_intention(text, now=now_fixed, tz=tz)
                self.assertIsNone(reminder, f"Casual/banter text should NOT be detected as reminder: {text}")

    def test_distinction_reminder_vs_plan_vs_casual(self):
        now_fixed = datetime(2026, 10, 31, 4, 0, 0, tzinfo=timezone.utc)
        tz = "Asia/Jakarta"

        # Explicit reminder
        res_rem = detect_any_future_intention("besok ingetin aku bayar tagihan", now=now_fixed, tz=tz)
        self.assertIsNotNone(res_rem)
        self.assertEqual(res_rem.kind, KIND_REMINDER)

        # Future plan (first-person plan with action)
        res_plan = detect_any_future_intention("besok jam 9 aku ada ujian kalkulus", now=now_fixed, tz=tz)
        self.assertIsNotNone(res_plan)
        self.assertEqual(res_plan.kind, KIND_PLAN)

        # Casual chat
        res_casual = detect_any_future_intention("cuaca hari ini cerah banget ya", now=now_fixed, tz=tz)
        self.assertIsNone(res_casual)


class TestScenario10ProactiveBehaviorRequiresRelevantIntention(unittest.TestCase):
    """Scenario 10: Proactive reminder behavior only triggers when an intention is due."""

    def test_due_intentions_filtering(self):
        now_fixed = datetime(2026, 10, 31, 10, 0, 0, tzinfo=timezone.utc)

        intentions = [
            # Future intention not yet due
            {
                "id": "plan-future",
                "text": "besok ada ujian",
                "due_at": "2026-11-01T07:00:00+00:00",
                "status": STATUS_PENDING,
            },
            # Completed intention
            {
                "id": "plan-done",
                "text": "sudah lewat",
                "due_at": "2026-10-31T09:00:00+00:00",
                "status": STATUS_DONE,
            },
            # Intention due right now
            {
                "id": "plan-due",
                "text": "ingetin minum obat",
                "due_at": "2026-10-31T09:59:00+00:00",
                "status": STATUS_PENDING,
            },
        ]

        due = due_intentions(intentions, moment=now_fixed)
        self.assertEqual(len(due), 1)
        self.assertEqual(due[0]["id"], "plan-due")


class TestScenario11TemporalMemoryTimestampAndIntegrity(unittest.TestCase):
    """Scenario 11: Memory stores absolute ISO timestamp and renders truthful age label."""

    def test_memory_age_label_and_non_destructive_rendering(self):
        moment = datetime(2026, 10, 31, 12, 0, 0, tzinfo=timezone.utc)
        tz = "Asia/Jakarta"

        # Stored timestamp is 2026-10-31 03:00 UTC (10:00 AM Jakarta today)
        raw_stamp_today = "2026-10-31T03:00:00+00:00"
        label_today = memory_age_label(raw_stamp_today, moment=moment, tz=tz)
        self.assertIn("Today", label_today)

        # Stored timestamp is 2026-10-30 (Yesterday in Jakarta)
        raw_stamp_yesterday = "2026-10-30T03:00:00+00:00"
        label_yesterday = memory_age_label(raw_stamp_yesterday, moment=moment, tz=tz)
        self.assertIn("Yesterday", label_yesterday)

        # Stored timestamp is 5 days ago
        raw_stamp_past = "2026-10-26T03:00:00+00:00"
        label_past = memory_age_label(raw_stamp_past, moment=moment, tz=tz)
        self.assertIn("5 days ago", label_past)

        # Raw timestamp remains unaltered
        self.assertEqual(raw_stamp_past, "2026-10-26T03:00:00+00:00")


class TestScenario12UnifiedTimeLifeStateAndCore(unittest.TestCase):
    """Scenario 12: Life State (world_state) and core system read consistent canonical time."""

    def test_world_state_delegates_to_time_provider(self):
        # Both modules provide compatible API backed by time_provider
        now_ws = ws.utcnow()
        now_tp = utc_now()
        self.assertEqual(now_ws.tzinfo, timezone.utc)
        self.assertEqual(now_tp.tzinfo, timezone.utc)
        self.assertAlmostEqual(now_ws.timestamp(), now_tp.timestamp(), delta=1.0)

        # user_local_datetime consistency
        moment = datetime(2026, 10, 31, 5, 0, 0, tzinfo=timezone.utc)
        self.assertEqual(
            ws.user_local_datetime(moment, "Asia/Jakarta"),
            user_local_datetime(moment, "Asia/Jakarta"),
        )
        self.assertEqual(
            ws.local_hour(moment, "Asia/Jakarta"),
            local_hour(moment, "Asia/Jakarta"),
        )
        self.assertEqual(
            ws.format_temporal_anchor(moment, "Asia/Jakarta"),
            format_temporal_anchor(moment, "Asia/Jakarta"),
        )


if __name__ == "__main__":
    unittest.main()
