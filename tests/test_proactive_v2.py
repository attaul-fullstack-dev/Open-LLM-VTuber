"""Proactive V2 — deterministic gate, budget, persistence, tool-loop cap.

Read-only guarantees under test:
* the gate never calls a model (it is a pure function);
* budget checks happen before any dispatch;
* user-initiated chat never consumes the proactive budget;
* state survives reconnect and restart;
* the tool interaction loop cannot exceed its hard cap.
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber import proactive_gate as pg
from src.open_llm_vtuber.agent.agents.basic_memory_agent import TOOL_LOOP_MAX_ITERATIONS
from src.open_llm_vtuber.proactive_gate import (
    DEFAULT_DAILY_HARD_LIMIT,
    GATE_BACKOFF,
    local_day_key,
    local_hour_key,
    GATE_DAILY_LIMIT,
    GATE_DORMANT,
    GATE_HOURLY_BUDGET,
    GATE_IDLE_BUDGET,
    GATE_MIN_GAP,
    GATE_OK,
    GATE_QUIET_HOURS,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    ProactiveBudgetState,
    ProactiveGateConfig,
    LEGACY_DAILY_HARD_LIMIT,
    TriggerReason,
    classify_trigger,
    current_gap_seconds,
    evaluate_gate,
    is_quiet_hours,
    load_proactive_state,
    normalize_daily_hard_limit,
    record_proactive_answered,
    record_proactive_dispatch,
    record_suppressed,
    resolve_user_tz,
    save_proactive_state,
)

UTC = timezone.utc
JKT = "Asia/Jakarta"
# 2026-10-03 04:00Z == 11:00 JKT (outside quiet hours)
NOON_JKT = datetime(2026, 10, 3, 4, 0, tzinfo=UTC)
CFG = ProactiveGateConfig()
HIGH = TriggerReason(PRIORITY_HIGH, "unfinished_topic", "x")
MED = TriggerReason(PRIORITY_MEDIUM, "relevant_memory", "x")
LOW = TriggerReason(PRIORITY_LOW, "generic_idle", "x")


class DailyBudgetTest(unittest.TestCase):
    def test_59_allowed_60_allowed_61_blocked(self):
        state = ProactiveBudgetState()
        for index in range(59):
            record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
            # clear the gap/backoff so only the daily ceiling is under test
            state.last_proactive_at = None
            state.backoff_until = None
            state.dormant_until = None
            state.consecutive_unanswered = 0
            state.hourly_count_hour = None
        self.assertEqual(state.daily_request_count, 59)
        decision = evaluate_gate(state, CFG, MED, now=NOON_JKT, tz=JKT)
        self.assertTrue(decision.allowed, decision.reason)
        self.assertEqual(decision.daily_remaining, 1)

        record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
        self.assertEqual(state.daily_request_count, 60)
        blocked = evaluate_gate(state, CFG, MED, now=NOON_JKT, tz=JKT)
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.reason, GATE_DAILY_LIMIT)
        self.assertEqual(blocked.daily_remaining, 0)

    def test_daily_limit_is_60_not_61(self):
        self.assertEqual(CFG.proactive_daily_hard_limit, 60)

    def test_counter_resets_at_user_local_midnight(self):
        state = ProactiveBudgetState()
        for _ in range(60):
            record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
            state.last_proactive_at = None
            state.backoff_until = None
            state.dormant_until = None
            state.consecutive_unanswered = 0
        self.assertFalse(evaluate_gate(state, CFG, MED, now=NOON_JKT, tz=JKT).allowed)

        # 2026-10-04 01:00Z == 2026-10-04 08:00 JKT -> new user-local day AND
        # outside quiet hours, so only the daily ceiling is under test.
        next_local_midnight = datetime(2026, 10, 4, 1, 0, tzinfo=UTC)
        fresh = evaluate_gate(state, CFG, MED, now=next_local_midnight, tz=JKT)
        self.assertEqual(state.daily_request_count, 0)
        self.assertTrue(fresh.allowed, fresh.reason)

    def test_midnight_is_evaluated_in_user_timezone_not_utc(self):
        # 2026-10-03 18:00Z is 2026-10-04 01:00 in JKT (new user-local day)
        # but still 2026-10-03 in UTC.
        moment = datetime(2026, 10, 3, 18, 0, tzinfo=UTC)
        in_jkt = ProactiveBudgetState(
            daily_request_count=60, daily_count_date="2026-10-03"
        )
        pg.roll_counters(in_jkt, moment, resolve_user_tz(JKT))
        self.assertEqual(in_jkt.daily_request_count, 0, "must reset on JKT midnight")

        in_utc = ProactiveBudgetState(
            daily_request_count=60, daily_count_date="2026-10-03"
        )
        pg.roll_counters(in_utc, moment, resolve_user_tz("UTC"))
        self.assertEqual(
            in_utc.daily_request_count, 60, "must NOT reset before UTC midnight"
        )

    def test_reconnect_does_not_reset_the_counter(self):
        state = ProactiveBudgetState()
        record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
        again = ProactiveBudgetState.from_dict(state.to_dict())  # reconnect reload
        self.assertEqual(again.daily_request_count, 1)
        self.assertEqual(again.last_proactive_at, state.last_proactive_at)

    def test_user_chat_does_not_consume_proactive_budget(self):
        state = ProactiveBudgetState()
        record_proactive_answered(state)  # what a user reply does
        self.assertEqual(state.daily_request_count, 0)
        self.assertEqual(state.hourly_meaningful_count, 0)


class MinimumGapTest(unittest.TestCase):
    def test_gap_is_300_seconds(self):
        self.assertEqual(CFG.minimum_proactive_gap_seconds, 300)

    def test_under_gap_blocked_at_gap_allowed(self):
        state = ProactiveBudgetState(last_proactive_at=pg._iso(NOON_JKT))
        state.backoff_until = None
        under = evaluate_gate(
            state, CFG, MED, now=NOON_JKT + timedelta(seconds=299), tz=JKT
        )
        self.assertFalse(under.allowed)
        self.assertEqual(under.reason, GATE_MIN_GAP)
        at = evaluate_gate(
            state, CFG, MED, now=NOON_JKT + timedelta(seconds=300), tz=JKT
        )
        self.assertTrue(at.allowed, at.reason)


class UnansweredDormantTest(unittest.TestCase):
    def test_threshold_is_three(self):
        self.assertEqual(CFG.maximum_unanswered_consecutive, 3)

    def test_first_second_and_third_ignore_then_dormant(self):
        state = ProactiveBudgetState()
        record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
        self.assertEqual(state.consecutive_unanswered, 1)
        self.assertIsNone(state.dormant_until)
        record_proactive_dispatch(
            state, CFG, MED, now=NOON_JKT + timedelta(hours=2), tz=JKT
        )
        self.assertEqual(state.consecutive_unanswered, 2)
        self.assertIsNone(state.dormant_until)
        record_proactive_dispatch(
            state, CFG, MED, now=NOON_JKT + timedelta(hours=4), tz=JKT
        )
        self.assertEqual(state.consecutive_unanswered, 3)
        self.assertIsNotNone(state.dormant_until)

    def test_dormant_suppresses_until_window_expires(self):
        state = ProactiveBudgetState()
        record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
        record_proactive_dispatch(
            state, CFG, MED, now=NOON_JKT + timedelta(hours=2), tz=JKT
        )
        record_proactive_dispatch(
            state, CFG, MED, now=NOON_JKT + timedelta(hours=4), tz=JKT
        )
        inside = evaluate_gate(
            state, CFG, MED, now=NOON_JKT + timedelta(hours=5), tz=JKT
        )
        self.assertFalse(inside.allowed)
        self.assertIn(inside.reason, (GATE_DORMANT, GATE_BACKOFF))
        after = evaluate_gate(
            state, CFG, MED, now=NOON_JKT + timedelta(hours=8), tz=JKT
        )
        self.assertTrue(after.allowed, after.reason)

    def test_user_reply_clears_unanswered_and_dormant(self):
        state = ProactiveBudgetState()
        record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
        record_proactive_dispatch(
            state, CFG, MED, now=NOON_JKT + timedelta(hours=2), tz=JKT
        )
        record_proactive_answered(state)
        self.assertEqual(state.consecutive_unanswered, 0)
        self.assertIsNone(state.dormant_until)
        self.assertIsNone(state.backoff_until)

    def test_suppressed_turn_costs_no_budget_but_widens_gap(self):
        state = ProactiveBudgetState()
        record_suppressed(state, CFG, now=NOON_JKT, tz=JKT)
        self.assertEqual(state.daily_request_count, 0)
        self.assertEqual(state.consecutive_unanswered, 1)
        self.assertIsNotNone(state.backoff_until)


class BackoffTest(unittest.TestCase):
    def test_backoff_doubles_and_is_capped(self):
        state = ProactiveBudgetState()
        self.assertEqual(current_gap_seconds(state, CFG), 300.0)
        state.consecutive_unanswered = 1
        self.assertEqual(current_gap_seconds(state, CFG), 300.0)
        state.consecutive_unanswered = 2
        self.assertEqual(current_gap_seconds(state, CFG), 600.0)
        state.consecutive_unanswered = 3
        self.assertEqual(current_gap_seconds(state, CFG), 1200.0)
        state.consecutive_unanswered = 4
        self.assertEqual(current_gap_seconds(state, CFG), 2400.0)
        state.consecutive_unanswered = 5
        self.assertEqual(current_gap_seconds(state, CFG), 4800.0)
        state.consecutive_unanswered = 6
        self.assertEqual(current_gap_seconds(state, CFG), 9600.0)
        for ignored in range(7, 40):
            self.assertLessEqual(
                current_gap_seconds(
                    ProactiveBudgetState(consecutive_unanswered=ignored), CFG
                ),
                10800.0,
            )
        self.assertEqual(
            current_gap_seconds(ProactiveBudgetState(consecutive_unanswered=99), CFG),
            10800.0,
        )

    def test_backoff_is_not_a_two_value_switch(self):
        gaps = {
            current_gap_seconds(ProactiveBudgetState(consecutive_unanswered=i), CFG)
            for i in range(0, 8)
        }
        self.assertGreater(len(gaps), 2)

    def test_ignored_threshold_is_two(self):
        self.assertEqual(CFG.ignored_threshold_before_backoff, 2)


class QuietHoursTest(unittest.TestCase):
    def test_start_boundary_is_quiet_end_boundary_is_not(self):
        zone = resolve_user_tz(JKT)
        # 23:00 JKT == 16:00Z
        self.assertTrue(
            is_quiet_hours(datetime(2026, 10, 3, 16, 0, tzinfo=UTC), zone, CFG)
        )
        # 07:00 JKT == 00:00Z -> first non-quiet hour
        self.assertFalse(
            is_quiet_hours(datetime(2026, 10, 4, 0, 0, tzinfo=UTC), zone, CFG)
        )
        # 22:59 JKT == 15:59Z -> one minute BEFORE the window: not quiet
        self.assertFalse(
            is_quiet_hours(datetime(2026, 10, 3, 15, 59, tzinfo=UTC), zone, CFG)
        )
        # 06:59 JKT quiet
        self.assertTrue(
            is_quiet_hours(datetime(2026, 10, 3, 23, 59, tzinfo=UTC), zone, CFG)
        )

    def test_crosses_midnight(self):
        zone = resolve_user_tz(JKT)
        for hour in (23, 0, 1, 2, 3, 4, 5, 6):
            moment = datetime(2026, 10, 3, 16, 0, tzinfo=UTC) + timedelta(
                hours=hour - 23
            )
            self.assertTrue(
                is_quiet_hours(moment, zone, CFG), f"hour {hour} should be quiet"
            )

    def test_uses_user_timezone_not_server_timezone(self):
        moment = datetime(2026, 10, 3, 20, 0, tzinfo=UTC)  # 23:00 JKT, 20:00 UTC
        self.assertTrue(is_quiet_hours(moment, resolve_user_tz(JKT), CFG))
        self.assertFalse(is_quiet_hours(moment, resolve_user_tz("UTC"), CFG))

    def test_quiet_hours_blocks_the_gate(self):
        state = ProactiveBudgetState()
        blocked = evaluate_gate(
            state, CFG, MED, now=datetime(2026, 10, 3, 17, 0, tzinfo=UTC), tz=JKT
        )
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.reason, GATE_QUIET_HOURS)

    def test_life_state_is_not_blocked_by_quiet_hours(self):
        # Quiet hours only gate proactive cognition; world_state stays pure and
        # keeps reconciling. Verified structurally: no gate symbol is imported
        # into world_state.
        import pathlib

        world = pathlib.Path(
            pathlib.Path(pg.__file__).parent / "world_state.py"
        ).read_text(encoding="utf-8")
        self.assertNotIn("proactive_gate", world)
        self.assertNotIn("quiet_hours", world)


class PriorityTest(unittest.TestCase):
    def test_high_priority_signals(self):
        self.assertEqual(
            classify_trigger(user_question_pending=True).priority, PRIORITY_HIGH
        )
        self.assertEqual(
            classify_trigger(unfinished_topic=True).priority, PRIORITY_HIGH
        )
        self.assertEqual(
            classify_trigger(explicit_reminder=True).priority, PRIORITY_HIGH
        )

    def test_medium_priority_signals(self):
        self.assertEqual(
            classify_trigger(relationship_event=True).priority, PRIORITY_MEDIUM
        )
        self.assertEqual(
            classify_trigger(meaningful_life_event=True).priority, PRIORITY_MEDIUM
        )
        self.assertEqual(
            classify_trigger(
                has_useful_memory=True, memory_relevance_score=0.9
            ).priority,
            PRIORITY_MEDIUM,
        )

    def test_low_when_nothing_meaningful(self):
        reason = classify_trigger()
        self.assertEqual(reason.priority, PRIORITY_LOW)
        self.assertFalse(reason.is_meaningful)

    def test_weak_memory_score_stays_low(self):
        self.assertEqual(
            classify_trigger(
                has_useful_memory=True, memory_relevance_score=0.2
            ).priority,
            PRIORITY_LOW,
        )

    def test_low_idle_never_calls_the_llm_without_a_budget(self):
        zero_idle = ProactiveGateConfig(
            idle_trigger_budget_per_hour=0, idle_trigger_budget_per_day=0
        )
        state = ProactiveBudgetState()
        decision = evaluate_gate(state, zero_idle, LOW, now=NOON_JKT, tz=JKT)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, GATE_IDLE_BUDGET)

    def test_idle_budgets_are_fifteen_per_hour(self):
        self.assertEqual(CFG.idle_trigger_budget_per_hour, 15)
        self.assertEqual(CFG.idle_trigger_budget_per_day, 0)

    def test_meaningful_trigger_still_allowed_with_idle_budget_zero(self):
        state = ProactiveBudgetState()
        self.assertTrue(evaluate_gate(state, CFG, MED, now=NOON_JKT, tz=JKT).allowed)
        self.assertTrue(evaluate_gate(state, CFG, HIGH, now=NOON_JKT, tz=JKT).allowed)

    def test_classification_needs_no_model(self):
        import inspect

        source = inspect.getsource(classify_trigger)
        for banned in ("chat_completion", "requests", "http", "llm"):
            self.assertNotIn(banned, source)


class HourlyBudgetTest(unittest.TestCase):
    def test_three_per_hour_then_defer(self):
        self.assertEqual(CFG.meaningful_trigger_budget_per_hour, 3)
        state = ProactiveBudgetState()
        record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
        state.last_proactive_at = None
        state.backoff_until = None
        state.dormant_until = None
        record_proactive_dispatch(
            state, CFG, MED, now=NOON_JKT + timedelta(minutes=20), tz=JKT
        )
        state.last_proactive_at = None
        state.backoff_until = None
        state.dormant_until = None
        record_proactive_dispatch(
            state, CFG, MED, now=NOON_JKT + timedelta(minutes=40), tz=JKT
        )
        self.assertEqual(state.hourly_meaningful_count, 3)
        deferred = evaluate_gate(
            state, CFG, HIGH, now=NOON_JKT + timedelta(minutes=50), tz=JKT
        )
        self.assertFalse(deferred.allowed)
        self.assertEqual(deferred.reason, GATE_HOURLY_BUDGET)
        # no HIGH bypass was invented: it waits for the next hourly window
        # (a fresh state isolates the hourly counter from the gap/backoff and
        # DORMANT gates, which are covered by their own tests)
        later = ProactiveBudgetState(
            hourly_meaningful_count=3,
            hourly_idle_count=0,
            # No hour recorded yet -> roll_counters resets on the next hour.
            hourly_count_hour=None,
        )
        next_hour = evaluate_gate(
            later, CFG, HIGH, now=NOON_JKT + timedelta(hours=1, minutes=5), tz=JKT
        )
        self.assertTrue(next_hour.allowed, next_hour.reason)

    def test_no_invented_high_bypass(self):
        state = ProactiveBudgetState(
            hourly_meaningful_count=3,
            hourly_idle_count=0,
            hourly_count_hour=pg.local_hour_key(NOON_JKT, resolve_user_tz(JKT)),
        )
        decision = evaluate_gate(state, CFG, HIGH, now=NOON_JKT, tz=JKT)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, GATE_HOURLY_BUDGET)


class PersistenceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._prev = os.getcwd()
        os.chdir(self._tmp.name)

    def tearDown(self):
        os.chdir(self._prev)
        self._tmp.cleanup()

    def test_round_trip_survives_reload(self):
        state = ProactiveBudgetState()
        record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
        record_proactive_dispatch(
            state, CFG, MED, now=NOON_JKT + timedelta(hours=2), tz=JKT
        )
        self.assertTrue(save_proactive_state("mili", state))
        reloaded = load_proactive_state("mili")
        self.assertEqual(reloaded.daily_request_count, state.daily_request_count)
        self.assertEqual(reloaded.consecutive_unanswered, state.consecutive_unanswered)
        self.assertEqual(reloaded.last_proactive_at, state.last_proactive_at)
        self.assertEqual(reloaded.backoff_until, state.backoff_until)
        self.assertEqual(reloaded.dormant_until, state.dormant_until)
        self.assertEqual(reloaded.daily_count_date, state.daily_count_date)
        self.assertEqual(
            reloaded.hourly_meaningful_count, state.hourly_meaningful_count
        )
        self.assertEqual(reloaded.hourly_idle_count, state.hourly_idle_count)
        self.assertEqual(reloaded.hourly_count_hour, state.hourly_count_hour)

    def test_restart_honours_saved_budget_and_gap(self):
        state = ProactiveBudgetState()
        for _ in range(60):
            record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
            state.last_proactive_at = None
            state.backoff_until = None
            state.dormant_until = None
        save_proactive_state("mili", state)
        after_restart = load_proactive_state("mili")
        decision = evaluate_gate(after_restart, CFG, MED, now=NOON_JKT, tz=JKT)
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, GATE_DAILY_LIMIT)

    def test_restart_does_not_grant_a_fresh_budget(self):
        state = ProactiveBudgetState()
        record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
        save_proactive_state("mili", state)
        # simulate a brand new process reading the file for the first time
        fresh_process_state = load_proactive_state("mili")
        self.assertEqual(fresh_process_state.daily_request_count, 1)

    def test_corrupt_state_fails_safe(self):
        os.makedirs(pg.STATE_DIR, exist_ok=True)
        with open(pg.state_path("mili"), "w", encoding="utf-8") as handle:
            handle.write("{not json")
        state = load_proactive_state("mili")
        self.assertIsInstance(state, ProactiveBudgetState)
        self.assertEqual(state.daily_request_count, 0)
        self.assertIsNone(state.last_proactive_at)

    def test_missing_state_is_zeroed(self):
        state = load_proactive_state("never-existed")
        self.assertEqual(state.daily_request_count, 0)
        self.assertEqual(state.consecutive_unanswered, 0)

    def test_only_absolute_timestamps_persisted(self):
        state = ProactiveBudgetState()
        record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
        raw = json.dumps(state.to_dict())
        for relative in ("today", "tomorrow", "yesterday", "ago", "just now"):
            self.assertNotIn(relative, raw)
        self.assertIn("+00:00", raw)

    def test_persisted_keys_are_the_agreed_minimum(self):
        state = ProactiveBudgetState()
        record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
        keys = set(state.to_dict())
        for required in (
            "last_proactive_at",
            "consecutive_unanswered",
            "backoff_until",
            "dormant_until",
            "daily_request_count",
            "daily_count_date",
            "hourly_meaningful_count",
            "hourly_idle_count",
            "hourly_count_hour",
        ):
            self.assertIn(required, keys)

    def test_save_failure_never_raises(self):
        self.assertFalse(save_proactive_state("", ProactiveBudgetState()))


class GateOrderingTest(unittest.TestCase):
    def test_connection_check_comes_first(self):
        state = ProactiveBudgetState()
        decision = evaluate_gate(
            state, CFG, MED, now=NOON_JKT, tz=JKT, connection_valid=False
        )
        self.assertFalse(decision.allowed)

    def test_in_flight_lock_blocks(self):
        state = ProactiveBudgetState()
        decision = evaluate_gate(
            state, CFG, MED, now=NOON_JKT, tz=JKT, generation_in_progress=True
        )
        self.assertFalse(decision.allowed)

    def test_gate_is_pure_and_llm_free(self):
        import inspect

        source = inspect.getsource(evaluate_gate)
        for banned in ("chat_completion", "requests", "http", "subprocess"):
            self.assertNotIn(banned, source)

    def test_allowed_decision_shape(self):
        decision = evaluate_gate(ProactiveBudgetState(), CFG, MED, now=NOON_JKT, tz=JKT)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.reason, GATE_OK)
        self.assertEqual(decision.priority, PRIORITY_MEDIUM)


class ToolLoopCapTest(unittest.TestCase):
    def test_cap_is_four(self):
        self.assertEqual(TOOL_LOOP_MAX_ITERATIONS, 4)

    def test_both_loops_are_capped(self):
        import pathlib

        source = pathlib.Path(
            pathlib.Path(pg.__file__).parent
            / "agent"
            / "agents"
            / "basic_memory_agent.py"
        ).read_text(encoding="utf-8")
        # every tool loop must consult the cap before calling the provider
        self.assertEqual(
            source.count("if tool_iterations >= TOOL_LOOP_MAX_ITERATIONS:"), 2
        )
        self.assertEqual(source.count("tool_iterations = 0"), 2)
        self.assertEqual(source.count("tool_iterations += 1"), 2)

    def test_cap_is_enforced_before_each_provider_call(self):
        import inspect

        from src.open_llm_vtuber.agent.agents.basic_memory_agent import (
            BasicMemoryAgent,
        )

        for loop_name in (
            "_openai_tool_interaction_loop",
            "_claude_tool_interaction_loop",
        ):
            with self.subTest(loop=loop_name):
                source = inspect.getsource(getattr(BasicMemoryAgent, loop_name))
                self.assertIn("TOOL_LOOP_MAX_ITERATIONS", source)
                self.assertLess(
                    source.index("TOOL_LOOP_MAX_ITERATIONS"),
                    source.index("self._llm.chat_completion"),
                )

    def test_gate_budget_is_independent_of_tool_budget(self):
        # The tool cap is a per-turn safety guard; the proactive budget is a
        # per-day accounting. They must not share a counter.
        state = ProactiveBudgetState()
        record_proactive_dispatch(state, CFG, MED, now=NOON_JKT, tz=JKT)
        self.assertEqual(state.daily_request_count, 1)
        self.assertEqual(TOOL_LOOP_MAX_ITERATIONS, 4)


class DailyHardLimitConsistencyTest(unittest.TestCase):
    """The daily ceiling must be 60 on every path, including old configs."""

    def test_fresh_default_config_is_60(self):
        self.assertEqual(ProactiveGateConfig().proactive_daily_hard_limit, 60)
        self.assertEqual(DEFAULT_DAILY_HARD_LIMIT, 60)

    def test_legacy_35_config_migrates_to_60(self):
        # 35 was the default shipped with the first Proactive V2 build and was
        # never a user choice, so an old config file must not restore it.
        self.assertEqual(LEGACY_DAILY_HARD_LIMIT, 35)
        self.assertEqual(
            ProactiveGateConfig(
                proactive_daily_hard_limit=35
            ).proactive_daily_hard_limit,
            60,
        )
        self.assertEqual(
            ProactiveGateConfig.from_dict(
                {"proactive_daily_hard_limit": 35}
            ).proactive_daily_hard_limit,
            60,
        )
        self.assertEqual(normalize_daily_hard_limit(35), 60)
        self.assertEqual(normalize_daily_hard_limit("35"), 60)

    def test_explicit_user_override_is_preserved(self):
        # Any other value is a deliberate choice and must survive.
        for value in (0, 5, 25, 100, 500):
            with self.subTest(value=value):
                self.assertEqual(
                    ProactiveGateConfig(
                        proactive_daily_hard_limit=value
                    ).proactive_daily_hard_limit,
                    value,
                )
                self.assertEqual(normalize_daily_hard_limit(value), value)

    def test_unparsable_value_falls_back_to_60(self):
        self.assertEqual(
            ProactiveGateConfig(
                proactive_daily_hard_limit="abc"
            ).proactive_daily_hard_limit,
            60,
        )
        self.assertEqual(normalize_daily_hard_limit(None), 60)

    def test_runtime_wiring_path_normalises(self):
        # The real construction path in the websocket handler.
        import inspect

        from src.open_llm_vtuber.websocket_handler import WebSocketHandler

        source = inspect.getsource(WebSocketHandler._proactive_config)
        self.assertIn("ProactiveGateConfig(", source)
        self.assertIn(
            "proactive_daily_hard_limit=settings.proactive_daily_hard_limit", source
        )

    def test_conf_yaml_declares_60(self):
        from src.open_llm_vtuber.config_manager.utils import read_yaml

        settings = read_yaml("conf.yaml")["character_config"]["agent_config"][
            "agent_settings"
        ]["basic_memory_agent"]
        self.assertEqual(settings["proactive_daily_hard_limit"], 60)

    def test_pydantic_default_is_60(self):
        from src.open_llm_vtuber.config_manager.agent import BasicMemoryAgentConfig

        self.assertEqual(
            BasicMemoryAgentConfig.model_fields["proactive_daily_hard_limit"].default,
            60,
        )

    def _state_at(self, count):
        zone = resolve_user_tz(JKT)
        return ProactiveBudgetState(
            daily_request_count=count,
            daily_count_date=local_day_key(NOON_JKT, zone),
            hourly_meaningful_count=0,
            hourly_idle_count=0,
            hourly_count_hour=local_hour_key(NOON_JKT, zone),
        )

    def test_hard_limit_boundary_59_60_61(self):
        for count, expected in ((59, True), (60, False), (61, False)):
            with self.subTest(count=count):
                decision = evaluate_gate(
                    self._state_at(count), CFG, HIGH, now=NOON_JKT, tz=JKT
                )
                self.assertEqual(decision.allowed, expected)
                if not expected:
                    self.assertEqual(decision.reason, GATE_DAILY_LIMIT)

    def test_legacy_config_no_longer_caps_at_35(self):
        legacy = ProactiveGateConfig(proactive_daily_hard_limit=35)
        self.assertEqual(legacy.proactive_daily_hard_limit, 60)
        # count=35 was the LAST allowed slot under the legacy ceiling; after
        # migration it is still allowed.
        at_35 = evaluate_gate(self._state_at(35), legacy, HIGH, now=NOON_JKT, tz=JKT)
        self.assertTrue(at_35.allowed, at_35.reason)
        self.assertEqual(at_35.daily_remaining, 25)
        # the effective ceiling is now the new one
        at_60 = evaluate_gate(self._state_at(60), legacy, HIGH, now=NOON_JKT, tz=JKT)
        self.assertFalse(at_60.allowed)
        self.assertEqual(at_60.reason, GATE_DAILY_LIMIT)

    def test_user_driven_chat_never_consumes_the_budget(self):
        state = ProactiveBudgetState()
        record_proactive_answered(state)
        self.assertEqual(state.daily_request_count, 0)
        decision = evaluate_gate(state, CFG, HIGH, now=NOON_JKT, tz=JKT)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.daily_remaining, 60)

    def test_tuned_gate_values_are_locked(self):
        expected = {
            "minimum_proactive_gap_seconds": 300,
            "idle_trigger_budget_per_hour": 15,
            "idle_trigger_budget_per_day": 0,
            "quiet_hours_start_hour": 23,
            "quiet_hours_end_hour": 7,
            "maximum_unanswered_consecutive": 3,
            "ignored_threshold_before_backoff": 2,
            "backoff_multiplier": 2.0,
            "max_backoff_seconds": 10800,
            "meaningful_trigger_budget_per_hour": 3,
            "behavior_on_budget_exhausted": "degrade_to_silent",
        }
        for key, value in expected.items():
            with self.subTest(key=key):
                self.assertEqual(getattr(CFG, key), value)


class ConfigContractTest(unittest.TestCase):
    def test_final_tuned_values(self):
        self.assertEqual(CFG.minimum_proactive_gap_seconds, 300)
        self.assertEqual(CFG.proactive_daily_hard_limit, 60)
        self.assertEqual(CFG.maximum_unanswered_consecutive, 3)
        self.assertEqual(CFG.ignored_threshold_before_backoff, 2)
        self.assertEqual(CFG.backoff_multiplier, 2.0)
        self.assertEqual(CFG.max_backoff_seconds, 10800)
        self.assertEqual(CFG.quiet_hours_start_hour, 23)
        self.assertEqual(CFG.quiet_hours_end_hour, 7)
        self.assertEqual(CFG.meaningful_trigger_budget_per_hour, 3)
        self.assertEqual(CFG.idle_trigger_budget_per_hour, 15)
        self.assertEqual(CFG.idle_trigger_budget_per_day, 0)
        self.assertEqual(CFG.behavior_on_budget_exhausted, "degrade_to_silent")

    def test_from_dict_is_tolerant(self):
        cfg = ProactiveGateConfig.from_dict(
            {"proactive_daily_hard_limit": "7", "backoff_multiplier": "bad", "x": 1}
        )
        self.assertEqual(cfg.proactive_daily_hard_limit, 7)
        self.assertEqual(cfg.backoff_multiplier, 2.0)
        self.assertEqual(cfg.minimum_proactive_gap_seconds, 300)

    def test_from_dict_of_garbage_returns_defaults(self):
        self.assertEqual(ProactiveGateConfig.from_dict("nope").to_dict(), CFG.to_dict())

    def test_conf_yaml_matches_the_tuned_values(self):
        from src.open_llm_vtuber.config_manager.utils import read_yaml

        settings = read_yaml("conf.yaml")["character_config"]["agent_config"][
            "agent_settings"
        ]["basic_memory_agent"]
        self.assertEqual(settings["proactive_daily_hard_limit"], 60)
        self.assertEqual(settings["minimum_proactive_gap_seconds"], 300)
        self.assertEqual(settings["maximum_unanswered_consecutive"], 3)
        self.assertEqual(settings["max_backoff_seconds"], 10800)
        self.assertEqual(settings["meaningful_trigger_budget_per_hour"], 3)
        self.assertEqual(settings["idle_trigger_budget_per_hour"], 15)
        self.assertEqual(settings["idle_trigger_budget_per_day"], 0)
        self.assertEqual(settings["quiet_hours_start_hour"], 23)
        self.assertEqual(settings["quiet_hours_end_hour"], 7)
        self.assertEqual(settings["behavior_on_budget_exhausted"], "degrade_to_silent")

    def test_pydantic_defaults_match_the_tuned_gate(self):
        from src.open_llm_vtuber.config_manager.agent import BasicMemoryAgentConfig

        fields = BasicMemoryAgentConfig.model_fields
        self.assertEqual(fields["minimum_proactive_gap_seconds"].default, 300)
        self.assertEqual(fields["maximum_unanswered_consecutive"].default, 3)
        self.assertEqual(fields["max_backoff_seconds"].default, 10800)
        self.assertEqual(fields["meaningful_trigger_budget_per_hour"].default, 3)
        self.assertEqual(fields["idle_trigger_budget_per_hour"].default, 15)
        self.assertEqual(fields["idle_trigger_budget_per_day"].default, 0)


class ProactiveTuningContractTest(unittest.TestCase):
    """One contract for the approved activity-frequency tuning.

    These checks use the real gate and real persisted-state accounting with a
    fixed clock, so they prove behavior rather than merely repeating numbers.
    """

    def test_idle_trigger_gets_fifteen_hourly_calls_but_not_sixteen(self):
        state = ProactiveBudgetState()
        for index in range(15):
            moment = NOON_JKT + timedelta(minutes=2 * index)
            decision = evaluate_gate(state, CFG, LOW, now=moment, tz=JKT)
            self.assertTrue(decision.allowed, (index, decision.reason))
            self.assertEqual(decision.reason, GATE_OK)
            record_proactive_dispatch(state, CFG, LOW, now=moment, tz=JKT)
            self.assertEqual(state.hourly_idle_count, index + 1)
            self.assertEqual(state.hourly_meaningful_count, 0)
            # Isolate the idle counter from the gap/backoff/dormant and
            # unanswered gates, which have their own dedicated tests.
            state.last_proactive_at = None
            state.backoff_until = None
            state.dormant_until = None
            state.consecutive_unanswered = 0
        sixteenth = evaluate_gate(
            state, CFG, LOW, now=NOON_JKT + timedelta(minutes=30), tz=JKT
        )
        self.assertFalse(sixteenth.allowed)
        self.assertEqual(sixteenth.reason, GATE_IDLE_BUDGET)

    def test_idle_trigger_cannot_bypass_the_minimum_gap(self):
        state = ProactiveBudgetState(last_proactive_at=pg._iso(NOON_JKT))
        state.backoff_until = None
        too_soon = evaluate_gate(
            state, CFG, LOW, now=NOON_JKT + timedelta(seconds=299), tz=JKT
        )
        self.assertFalse(too_soon.allowed)
        self.assertEqual(too_soon.reason, GATE_MIN_GAP)
        on_time = evaluate_gate(
            state, CFG, LOW, now=NOON_JKT + timedelta(seconds=300), tz=JKT
        )
        self.assertTrue(on_time.allowed, on_time.reason)

    def test_idle_hourly_counter_resets_next_hour(self):
        state = ProactiveBudgetState()
        for index in range(15):
            record_proactive_dispatch(
                state, CFG, LOW, now=NOON_JKT + timedelta(minutes=2 * index), tz=JKT
            )
        self.assertEqual(state.hourly_idle_count, 15)
        # Isolate the counter reset from gap/backoff/dormant pressure, which
        # have their own dedicated tests.
        state.last_proactive_at = None
        state.backoff_until = None
        state.dormant_until = None
        state.consecutive_unanswered = 0
        following = evaluate_gate(
            state, CFG, LOW, now=NOON_JKT + timedelta(hours=1, minutes=5), tz=JKT
        )
        self.assertEqual(state.hourly_idle_count, 0)
        self.assertTrue(following.allowed, following.reason)

    def test_quiet_hours_still_block_idle_triggers(self):
        # 16:30Z is 23:30 in Jakarta, inside the 23:00-07:00 window.
        quiet_moment = datetime(2026, 10, 3, 16, 30, tzinfo=UTC)
        decision = evaluate_gate(
            ProactiveBudgetState(), CFG, LOW, now=quiet_moment, tz=JKT
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, GATE_QUIET_HOURS)

    def test_daily_hard_limit_still_blocks_after_sixty(self):
        zone = resolve_user_tz(JKT)
        state = ProactiveBudgetState(
            daily_request_count=60,
            daily_count_date=local_day_key(NOON_JKT, zone),
            hourly_meaningful_count=0,
            hourly_idle_count=0,
            hourly_count_hour=local_hour_key(NOON_JKT, zone),
        )
        blocked = evaluate_gate(state, CFG, HIGH, now=NOON_JKT, tz=JKT)
        self.assertFalse(blocked.allowed)
        self.assertEqual(blocked.reason, GATE_DAILY_LIMIT)
        self.assertEqual(blocked.daily_remaining, 0)

    def test_user_driven_chat_still_does_not_consume_budget(self):
        state = ProactiveBudgetState()
        record_proactive_answered(state)
        self.assertEqual(state.daily_request_count, 0)
        self.assertEqual(state.hourly_meaningful_count, 0)
        self.assertEqual(state.hourly_idle_count, 0)
        decision = evaluate_gate(state, CFG, HIGH, now=NOON_JKT, tz=JKT)
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.daily_remaining, 60)

    def test_tool_loop_cap_is_unchanged(self):
        self.assertEqual(TOOL_LOOP_MAX_ITERATIONS, 4)


if __name__ == "__main__":
    unittest.main()
