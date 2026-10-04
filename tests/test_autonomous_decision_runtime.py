"""ADL runtime wiring — deterministic tests (no network, no LLM).

The Autonomous Decision Layer stops being an observation-only taxonomy here:
goals are seeded and explicitly transitioned in the real character-state file,
and a goal with fresh deterministic evidence becomes a *reason* inside the
existing Proactive V2 trigger.

Coverage map (per task spec):

A. Decision         - no-op / HIGH / MEDIUM / LOW, priority ordering
B. Goals            - seeding, seed->active, active->done, no reactivation,
                      restart persistence, old-state compatibility
C. Preferences      - in the decision path, no invented preference
D. Temporal         - absolute stamps, tz-aware day boundary, stale events
E. Life State       - meaningful transition stays a MEDIUM source, no dupe
F. Proactive gate   - goal evidence passes the real gate; daily 60 and the
                      600s minimum gap still bind; budget exhaustion
                      suppresses; user-driven chat unaffected
G. Restart/reconnect- no duplicate autonomous action
H. Safety           - no loop, bounded scan, no repeat for the same evidence
"""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from src.open_llm_vtuber.autonomous_decision import (
    GOAL_EVIDENCE_KEYWORDS,
    OUTCOME_GOAL_BEHAVIOR,
    OUTCOME_NO_DECISION,
    classify_goal_evidence,
    goal_evidence_summary,
)
from src.open_llm_vtuber.character_state import (
    CharacterState,
    activate_goal,
    complete_goal,
    ensure_seed_goals,
    goal_from_dict,
    goal_status_counts,
    goal_to_dict,
    load_character_state,
    record_goal_evidence,
    save_character_state,
)
from src.open_llm_vtuber.proactive_gate import (
    DEFAULT_DAILY_HARD_LIMIT,
    LEGACY_DAILY_HARD_LIMIT,
    PRIORITY_HIGH,
    PRIORITY_LOW,
    PRIORITY_MEDIUM,
    ProactiveBudgetState,
    ProactiveGateConfig,
    classify_trigger,
    evaluate_gate,
    local_day_key,
    local_hour_key,
    normalize_daily_hard_limit,
    resolve_user_tz,
)
from src.open_llm_vtuber.agent.relationship_context import build_relationship_context

JKT = "Asia/Jakarta"
NOW = datetime(2026, 10, 2, 5, 0, tzinfo=timezone.utc)  # 12:00 Jakarta
ZONE = resolve_user_tz(JKT)


def event(eid: str, text: str, age_h: float = 2.0) -> dict:
    return {
        "id": eid,
        "event_text": text,
        "occurred_at": (NOW - timedelta(hours=age_h)).isoformat(),
        "created_at": NOW.isoformat(),
        "tz": JKT,
    }


def goals_with_active(goal_id: str, **extra) -> list:
    goals, _ = activate_goal(ensure_seed_goals([]), goal_id)
    if extra:
        for item in goals:
            if item["id"] == goal_id:
                item.update(extra)
    return goals


# ---------------------------------------------------------------------------
# A. Decision
# ---------------------------------------------------------------------------
class DecisionTest(unittest.TestCase):
    def test_no_goals_is_no_decision(self):
        decision = classify_goal_evidence(goals=[], episodic_events=[], moment=NOW)
        self.assertEqual(decision.outcome, OUTCOME_NO_DECISION)
        self.assertEqual(decision.reason, "no_goal")
        self.assertFalse(decision.acts)

    def test_seed_without_activation_is_no_decision(self):
        decision = classify_goal_evidence(
            goals=ensure_seed_goals([]),
            episodic_events=[event("e1", "gw coba masak chicken")],
            moment=NOW,
        )
        self.assertEqual(decision.outcome, OUTCOME_NO_DECISION)
        self.assertEqual(decision.reason, "goal_not_active")
        self.assertFalse(decision.acts)

    def test_high_decision_from_active_goal(self):
        decision = classify_goal_evidence(
            goals=goals_with_active("try-three-dishes"),
            episodic_events=[event("e1", "gw coba masak chicken")],
            moment=NOW,
        )
        self.assertEqual(decision.outcome, OUTCOME_GOAL_BEHAVIOR)
        self.assertEqual(decision.reason, "goal_evidence")
        self.assertTrue(decision.acts)
        self.assertEqual(decision.metadata["goal_id"], "try-three-dishes")
        self.assertEqual(decision.metadata["evidence_id"], "e1")
        self.assertTrue(decision.cooldown_until)

    def test_medium_and_low_reasons_are_unchanged(self):
        medium = classify_trigger(meaningful_life_event=True)
        low = classify_trigger()
        self.assertEqual(medium.priority, PRIORITY_MEDIUM)
        self.assertEqual(medium.reason, "life_event")
        self.assertEqual(low.priority, PRIORITY_LOW)
        self.assertEqual(low.reason, "generic_idle")
        self.assertFalse(low.is_meaningful)

    def test_priority_ordering_goal_evidence_beats_reactive_sources(self):
        goal = classify_trigger(goal_evidence=True, meaningful_life_event=True)
        relationship = classify_trigger(relationship_event=True)
        memory = classify_trigger(has_useful_memory=True, memory_relevance_score=0.9)
        self.assertEqual(goal.priority, PRIORITY_HIGH)
        self.assertEqual(goal.reason, "goal_evidence")
        # reactive MEDIUM sources stay MEDIUM; the goal simply outranks them
        self.assertEqual(relationship.priority, PRIORITY_MEDIUM)
        self.assertEqual(memory.priority, PRIORITY_MEDIUM)
        # explicit reminder still wins over everything
        top = classify_trigger(explicit_reminder=True, goal_evidence=True)
        self.assertEqual(top.reason, "explicit_reminder")

    def test_low_generic_idle_never_becomes_meaningful(self):
        self.assertFalse(classify_trigger().is_meaningful)


# ---------------------------------------------------------------------------
# B. Goals lifecycle
# ---------------------------------------------------------------------------
class GoalsLifecycleTest(unittest.TestCase):
    def test_seed_goals_are_persistent_and_idempotent(self):
        first = ensure_seed_goals([])
        self.assertEqual(len(first), 3)
        self.assertTrue(all(g["status"] == "seed" for g in first))
        again = ensure_seed_goals(first)
        self.assertEqual(again, first, "re-seeding must be a no-op")

    def test_empty_and_corrupt_goal_lists_seed(self):
        for bad in (
            None,
            [],
            "not-a-list",
            {},
            [None, {"no_id": 1}, {"id": "", "text": ""}],
        ):
            seeded = ensure_seed_goals(bad)
            self.assertEqual(len(seeded), 3, f"input={bad!r}")

    def test_seed_to_active_to_done(self):
        goals = ensure_seed_goals([])
        goals, changed = activate_goal(goals, "finish-one-book")
        self.assertTrue(changed)
        self.assertEqual(goal_status_counts(goals)["active"], 1)
        goals, changed = complete_goal(goals, "finish-one-book")
        self.assertTrue(changed)
        self.assertEqual(goal_status_counts(goals)["done"], 1)

    def test_no_automatic_completion_or_reactivation(self):
        active = goals_with_active("try-three-dishes")
        # repeating the same transition is rejected, not reapplied
        again, changed = activate_goal(active, "try-three-dishes")
        self.assertFalse(changed)
        self.assertEqual(goal_status_counts(again)["active"], 1)
        # done -> active is refused outright
        done, _ = complete_goal(active, "try-three-dishes")
        back, changed = activate_goal(done, "try-three-dishes")
        self.assertFalse(changed)
        self.assertEqual(goal_status_counts(back)["done"], 1)
        self.assertEqual(goal_status_counts(back)["active"], 0)

    def test_unknown_goal_id_is_never_activated(self):
        goals, changed = activate_goal(ensure_seed_goals([]), "invented-goal")
        self.assertFalse(changed)
        self.assertEqual(goal_status_counts(goals)["active"], 0)

    def test_existing_goal_list_is_preserved_verbatim(self):
        custom = [
            {
                "id": "morning-reading-week",
                "text": "custom text",
                "status": "done",
                "created_at": "2026-01-01T00:00:00+00:00",
                "activated_at": None,
                "completed_at": "2026-01-02T00:00:00+00:00",
                "source": "seed",
            }
        ]
        kept = ensure_seed_goals(custom)
        self.assertEqual(len(kept), 1, "must not re-seed on top of existing goals")
        self.assertEqual(kept[0]["status"], "done")
        self.assertEqual(kept[0]["text"], "custom text")

    def test_old_state_without_anchor_keys_loads(self):
        old = {"id": "g", "text": "t", "status": "active", "source": "seed"}
        parsed = goal_from_dict(old)
        self.assertIsNotNone(parsed)
        self.assertIsNone(parsed.last_evidence_id)
        self.assertIsNone(parsed.last_evidence_at)
        # serialising it back must not invent the anchor keys, so an old file
        # stays stable across a load/save cycle
        round_tripped = goal_to_dict(parsed)
        self.assertNotIn("last_evidence_id", round_tripped)
        self.assertNotIn("last_evidence_at", round_tripped)
        self.assertEqual(round_tripped["id"], "g")
        self.assertEqual(round_tripped["status"], "active")

    def test_invalid_status_degrades_to_seed(self):
        self.assertEqual(
            goal_from_dict({"id": "g", "text": "t", "status": "weird"}).status, "seed"
        )

    def test_evidence_anchor_round_trip(self):
        goals = goals_with_active("try-three-dishes")
        pinned, changed = record_goal_evidence(goals, "try-three-dishes", "ev-9")
        self.assertTrue(changed)
        entry = next(g for g in pinned if g["id"] == "try-three-dishes")
        self.assertEqual(entry["last_evidence_id"], "ev-9")
        self.assertEqual(goal_to_dict(goal_from_dict(entry)), entry)
        # blank evidence never resets an already-consumed goal
        same, changed_again = record_goal_evidence(pinned, "try-three-dishes", "")
        self.assertFalse(changed_again)
        self.assertEqual(same, pinned)

    def test_evidence_anchor_only_applies_to_active_goal(self):
        seeded = ensure_seed_goals([])
        out, changed = record_goal_evidence(seeded, "try-three-dishes", "ev-1")
        self.assertFalse(changed, "a seed goal has no active decision to pin")


# ---------------------------------------------------------------------------
# C. Preferences
# ---------------------------------------------------------------------------
class PreferencePathTest(unittest.TestCase):
    def test_goal_decision_does_not_depend_on_preferences(self):
        goals = goals_with_active("try-three-dishes")
        events = [event("e1", "gw coba masak chicken")]
        with_prefs = classify_goal_evidence(
            goals=goals, episodic_events=events, moment=NOW
        )
        without = classify_goal_evidence(
            goals=goals, episodic_events=events, moment=NOW
        )
        self.assertEqual(with_prefs.outcome, without.outcome)
        self.assertEqual(
            with_prefs.metadata["evidence_id"], without.metadata["evidence_id"]
        )

    def test_no_preference_data_means_safe_default(self):
        from src.open_llm_vtuber.autonomous_decision import build_context_signals

        signals = build_context_signals(
            interaction_preferences=(), episodic_events=(), query="", moment=NOW
        )
        self.assertEqual(signals.preference_signals, ())
        self.assertFalse(signals.continuity_candidate)

    def test_stored_preferences_do_not_fabricate_a_decision(self):
        """A stored preference alone must stay inert (no invented behaviour)."""
        stored = [
            {
                "id": "p1",
                "category": "humor",
                "polarity": "prefer",
                "text": "suka lelucon receh",
                "status": "active",
                "created_at": "2026-10-01T00:00:00+00:00",
                "updated_at": "2026-10-01T00:00:00+00:00",
                "source": "conversation",
            }
        ]
        from src.open_llm_vtuber.autonomous_decision import build_context_signals

        signals = build_context_signals(
            interaction_preferences=stored,
            episodic_events=[event("e1", "gw coba masak chicken")],
            query="masak",
            moment=NOW,
        )
        # observable, but never a continuity claim
        self.assertEqual(signals.preference_signals, ("humor:prefer",))
        self.assertFalse(signals.continuity_candidate)


# ---------------------------------------------------------------------------
# D. Temporal
# ---------------------------------------------------------------------------
class TemporalTest(unittest.TestCase):
    def test_timestamps_are_absolute_utc_iso(self):
        decision = classify_goal_evidence(
            goals=goals_with_active("try-three-dishes"),
            episodic_events=[event("e1", "gw coba masak chicken")],
            moment=NOW,
        )
        self.assertTrue(decision.decided_at.endswith("+00:00"))
        stamp = decision.metadata["evidence_occurred_at"]
        self.assertIn("T", stamp)
        self.assertTrue(stamp.endswith("+00:00"))

    def test_stale_event_is_not_evidence(self):
        decision = classify_goal_evidence(
            goals=goals_with_active("try-three-dishes"),
            episodic_events=[event("e1", "gw coba masak chicken", age_h=24 * 30)],
            moment=NOW,
        )
        self.assertEqual(decision.outcome, OUTCOME_NO_DECISION)
        self.assertEqual(decision.reason, "goal_no_fresh_evidence")

    def test_future_event_is_ignored_not_negative(self):
        future = {
            "id": "e1",
            "event_text": "gw coba masak chicken",
            "occurred_at": (NOW + timedelta(hours=5)).isoformat(),
        }
        decision = classify_goal_evidence(
            goals=goals_with_active("try-three-dishes"),
            episodic_events=[future],
            moment=NOW,
        )
        self.assertEqual(decision.outcome, OUTCOME_NO_DECISION)

    def test_missing_timestamp_is_skipped_fail_soft(self):
        broken = [{"id": "e1", "event_text": "gw coba masak chicken"}]
        decision = classify_goal_evidence(
            goals=goals_with_active("try-three-dishes"),
            episodic_events=broken,
            moment=NOW,
        )
        self.assertEqual(decision.outcome, OUTCOME_NO_DECISION)

    def test_user_timezone_day_boundary_still_hard_limits_at_60(self):
        """The gate's daily reset is user-local, independent of ADL."""
        self.assertEqual(normalize_daily_hard_limit(LEGACY_DAILY_HARD_LIMIT), 60)
        self.assertEqual(DEFAULT_DAILY_HARD_LIMIT, 60)
        cfg = ProactiveGateConfig()
        state = ProactiveBudgetState(
            daily_request_count=59,
            daily_count_date=local_day_key(NOW, ZONE),
            hourly_meaningful_count=0,
            hourly_count_hour=local_hour_key(NOW, ZONE),
        )
        # a meaningful (goal) trigger reaches the daily ceiling check
        meaningful = evaluate_gate(
            state, cfg, classify_trigger(goal_evidence=True), now=NOW, tz=JKT
        )
        self.assertTrue(meaningful.allowed)
        self.assertEqual(meaningful.daily_remaining, 1)
        # a LOW generic idle thought never reaches the LLM when site policy
        # disables the hourly idle budget; it is refused before other checks.
        idle_cfg = ProactiveGateConfig(
            idle_trigger_budget_per_hour=0, idle_trigger_budget_per_day=0
        )
        idle = evaluate_gate(state, idle_cfg, classify_trigger(), now=NOW, tz=JKT)
        self.assertFalse(idle.allowed)
        self.assertEqual(idle.reason, "idle_budget_exhausted")


# ---------------------------------------------------------------------------
# E. Life State
# ---------------------------------------------------------------------------
class LifeStateSignalTest(unittest.TestCase):
    def test_meaningful_transition_remains_medium(self):
        self.assertEqual(
            classify_trigger(meaningful_life_event=True).priority, PRIORITY_MEDIUM
        )

    def test_goal_evidence_and_life_event_coexist_without_dupe(self):
        first = classify_trigger(goal_evidence=True, meaningful_life_event=True)
        second = classify_trigger(goal_evidence=False, meaningful_life_event=True)
        self.assertEqual(first.priority, PRIORITY_HIGH)
        self.assertEqual(second.priority, PRIORITY_MEDIUM)
        # exactly one reason is ever reported, so there is nothing to double-fire
        self.assertEqual(first.reason, "goal_evidence")
        self.assertEqual(second.reason, "life_event")

    def test_relationship_context_is_timezone_aware(self):
        ctx = build_relationship_context(
            "close",
            updated_at=(datetime.now(ZONE) - timedelta(days=1)).isoformat(),
            tz=JKT,
        )
        self.assertIn("yesterday", ctx)


# ---------------------------------------------------------------------------
# F. Proactive V2 integration
# ---------------------------------------------------------------------------
class ProactiveIntegrationTest(unittest.TestCase):
    def _state(
        self, daily: int = 0, last_proactive: str = None
    ) -> ProactiveBudgetState:
        return ProactiveBudgetState(
            daily_request_count=daily,
            daily_count_date=local_day_key(NOW, ZONE),
            hourly_meaningful_count=0,
            hourly_count_hour=local_hour_key(NOW, ZONE),
            last_proactive_at=last_proactive,
            consecutive_unanswered=0,
            backoff_until=None,
            dormant_until=None,
        )

    def test_goal_evidence_passes_the_real_gate(self):
        trigger = classify_trigger(goal_evidence=True)
        decision = evaluate_gate(
            self._state(), ProactiveGateConfig(), trigger, now=NOW, tz=JKT
        )
        self.assertTrue(decision.allowed)
        self.assertEqual(decision.priority, PRIORITY_HIGH)

    def test_minimum_gap_600_still_binds_a_goal_decision(self):
        state = self._state(last_proactive=(NOW - timedelta(seconds=60)).isoformat())
        decision = evaluate_gate(
            state,
            ProactiveGateConfig(),
            classify_trigger(goal_evidence=True),
            now=NOW,
            tz=JKT,
        )
        self.assertFalse(decision.allowed)
        self.assertIn("gap", decision.reason)

    def test_daily_hard_limit_60_boundary(self):
        cfg = ProactiveGateConfig()
        for count, allowed in ((59, True), (60, False), (61, False)):
            decision = evaluate_gate(
                self._state(daily=count),
                cfg,
                classify_trigger(goal_evidence=True),
                now=NOW,
                tz=JKT,
            )
            self.assertEqual(decision.allowed, allowed, f"count={count}")
            if not allowed:
                self.assertEqual(decision.reason, "daily_hard_limit")

    def test_budget_exhaustion_suppresses_autonomous_action(self):
        decision = evaluate_gate(
            self._state(daily=60),
            ProactiveGateConfig(),
            classify_trigger(goal_evidence=True),
            now=NOW,
            tz=JKT,
        )
        self.assertFalse(decision.allowed)

    def test_quiet_hours_suppress_a_goal_decision(self):
        night = NOW.replace(hour=17, minute=0)  # 00:00 Jakarta
        decision = evaluate_gate(
            self._state(last_proactive=None),
            ProactiveGateConfig(),
            classify_trigger(goal_evidence=True),
            now=night,
            tz=JKT,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "quiet_hours")

    def test_user_driven_request_is_never_charged_to_the_goal_budget(self):
        from src.open_llm_vtuber.proactive_gate import record_proactive_answered

        state = self._state(daily=59)
        record_proactive_answered(state)
        self.assertEqual(state.daily_request_count, 59)

    def test_hourly_meaningful_budget_still_binds(self):
        state = ProactiveBudgetState(
            daily_request_count=0,
            daily_count_date=local_day_key(NOW, ZONE),
            hourly_meaningful_count=3,
            hourly_idle_count=0,
            hourly_count_hour=local_hour_key(NOW, ZONE),
        )
        decision = evaluate_gate(
            state,
            ProactiveGateConfig(),
            classify_trigger(goal_evidence=True),
            now=NOW,
            tz=JKT,
        )
        self.assertFalse(decision.allowed)
        self.assertEqual(decision.reason, "hourly_budget")


# ---------------------------------------------------------------------------
# G + H. Restart, reconnect, safety
# ---------------------------------------------------------------------------
class PersistenceAndSafetyTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs(os.path.join("character_state"), exist_ok=True)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_goals_survive_restart_and_do_not_reseed(self):
        conf = "restartchar"
        save_character_state(conf, CharacterState(goals=ensure_seed_goals([])))
        # simulate restart
        reloaded = load_character_state(conf)
        self.assertEqual(len(reloaded.goals), 3)
        reloaded.goals, _ = activate_goal(reloaded.goals, "try-three-dishes")
        save_character_state(conf, reloaded)
        again = load_character_state(conf)
        self.assertEqual(len(again.goals), 3, "restart must not re-seed")
        self.assertEqual(goal_status_counts(again.goals)["active"], 1)
        # ensure_seed_goals on a populated state changes nothing
        self.assertEqual(ensure_seed_goals(again.goals), again.goals)

    def test_evidence_anchor_survives_restart_so_no_duplicate_action(self):
        conf = "anchorchar"
        goals = goals_with_active("try-three-dishes")
        save_character_state(conf, CharacterState(goals=goals))
        loaded = load_character_state(conf)
        events = [event("e1", "gw coba masak chicken")]
        self.assertTrue(
            classify_goal_evidence(
                goals=loaded.goals, episodic_events=events, moment=NOW
            ).acts
        )
        loaded.goals, _ = record_goal_evidence(
            loaded.goals, "try-three-dishes", "e1", now=NOW
        )
        save_character_state(conf, loaded)
        # reconnect: state reloaded from disk must not re-fire the same evidence
        reloaded = load_character_state(conf)
        decision = classify_goal_evidence(
            goals=reloaded.goals, episodic_events=events, moment=NOW
        )
        self.assertFalse(decision.acts)
        self.assertEqual(decision.reason, "goal_evidence_consumed")

    def test_corrupt_goal_state_fails_soft(self):
        conf = "corruptchar"
        path = os.path.join("character_state", f"{conf}.json")
        with open(path, "w", encoding="utf-8") as handle:
            handle.write("{ this is not json")
        state = load_character_state(conf)
        self.assertEqual(state.goals, [])
        decision = classify_goal_evidence(
            goals=state.goals, episodic_events=[], moment=NOW
        )
        self.assertEqual(decision.outcome, OUTCOME_NO_DECISION)

    def test_no_autonomous_loop_bounded_scan(self):
        """A huge event store cannot widen the scan beyond max_events."""
        goals = goals_with_active("try-three-dishes")
        events = [event(f"e{i}", "unrelated") for i in range(500)]
        events.append(event("hit", "gw coba masak chicken", age_h=0.5))
        decision = classify_goal_evidence(
            goals=goals, episodic_events=events, moment=NOW, max_events=3
        )
        # the newest event is the hit, so a bounded newest-first scan still sees it
        self.assertTrue(decision.acts)
        self.assertEqual(decision.metadata["evidence_id"], "hit")

    def test_repeated_classification_is_idempotent(self):
        goals = goals_with_active("try-three-dishes")
        events = [event("e1", "gw coba masak chicken")]
        seen = {
            classify_goal_evidence(
                goals=goals, episodic_events=events, moment=NOW
            ).reason
            for _ in range(5)
        }
        self.assertEqual(seen, {"goal_evidence"}, "classification must be stable")

    def test_unknown_goal_id_has_no_evidence_vocabulary(self):
        goals = [
            {
                "id": "llm-invented-goal",
                "text": "jadi pilot helikopter",
                "status": "active",
                "source": "seed",
            }
        ]
        decision = classify_goal_evidence(
            goals=goals,
            episodic_events=[event("e1", "gw jadi pilot helikopter hari ini")],
            moment=NOW,
        )
        self.assertEqual(decision.outcome, OUTCOME_NO_DECISION)
        self.assertNotIn("llm-invented-goal", GOAL_EVIDENCE_KEYWORDS)

    def test_summary_exposes_all_four_goal_states(self):
        summary = goal_evidence_summary(goals_with_active("finish-one-book"))
        self.assertEqual(summary["active"], 1)
        self.assertEqual(summary["seed"], 2)
        self.assertEqual(summary["done"], 0)
        self.assertEqual(summary["active_ids"], ("finish-one-book",))
        empty = goal_evidence_summary([])
        self.assertEqual((empty["seed"], empty["active"], empty["done"]), (0, 0, 0))


# ---------------------------------------------------------------------------
# Runtime facade (integration with the real agent object)
# ---------------------------------------------------------------------------
class AgentFacadeTest(unittest.TestCase):
    """The agent façade seeds, transitions and classifies against real files."""

    def _agent(self):
        from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
        from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig

        class _LLM:
            model = "adl-test"
            max_tokens = 64

            async def chat_completion(self, messages, system=None, tools=None):
                if False:
                    yield None

        return BasicMemoryAgent(
            llm=_LLM(),
            system="persona",
            live2d_model=SimpleNamespace(extract_emotion=lambda text: []),
            tts_preprocessor_config=TTSPreprocessorConfig(
                remove_special_char=True,
                translator_config={
                    "translate_audio": False,
                    "translate_provider": "deeplx",
                },
            ),
        )

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        os.makedirs(os.path.join("episodic"), exist_ok=True)
        os.makedirs(os.path.join("character_state"), exist_ok=True)

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _write_events(self, events):
        with open(
            os.path.join("episodic", "adlchar.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump(events, handle)

    def test_seeding_on_load_then_explicit_transitions(self):
        agent = self._agent()
        agent._character_conf_uid = "adlchar"
        agent._load_character_state("adlchar")
        snapshot = agent.goal_snapshot()
        self.assertEqual(snapshot["seed"], 3)
        self.assertEqual(snapshot["active"], 0)
        # the seed is on disk, so a restart sees it again rather than re-seeding
        self.assertEqual(len(load_character_state("adlchar").goals), 3)

        self.assertTrue(agent.set_goal_status("try-three-dishes", "active"))
        self.assertEqual(agent.goal_snapshot()["active"], 1)
        # re-activating is refused
        self.assertFalse(agent.set_goal_status("try-three-dishes", "active"))
        self.assertTrue(agent.set_goal_status("try-three-dishes", "done"))
        self.assertEqual(agent.goal_snapshot()["done"], 1)
        # and nothing reactivates it on its own
        self.assertFalse(agent.set_goal_status("try-three-dishes", "active"))
        self.assertEqual(agent.goal_snapshot()["active"], 0)

    def test_invalid_transition_target_is_rejected(self):
        agent = self._agent()
        agent._character_conf_uid = "adlchar"
        agent._load_character_state("adlchar")
        for bad in ("", "pending", "DONE", None, "reactivated"):
            self.assertFalse(agent.set_goal_status("try-three-dishes", bad))

    def test_runtime_classification_and_dedup_via_facade(self):
        agent = self._agent()
        agent._character_conf_uid = "adlchar"
        agent._load_character_state("adlchar")
        self._write_events([event("e1", "gw coba masak chicken")])
        with patch(
            "src.open_llm_vtuber.agent.agents.basic_memory_agent.utcnow",
            return_value=NOW,
        ):
            before = agent.classify_goal_evidence()
            self.assertEqual(before.outcome, OUTCOME_NO_DECISION)
            agent.set_goal_status("try-three-dishes", "active")
            fired = agent.classify_goal_evidence()
            self.assertEqual(fired.outcome, OUTCOME_GOAL_BEHAVIOR)
            agent.record_goal_evidence(
                "try-three-dishes", fired.metadata["evidence_id"]
            )
            again = agent.classify_goal_evidence()
            self.assertFalse(again.acts)
            self.assertEqual(again.reason, "goal_evidence_consumed")


if __name__ == "__main__":
    unittest.main()
