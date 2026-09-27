"""Stage 7 — Simulated Life deterministic tests (fake clock, no waiting).

Covers spec items A-S: initial state, transitions, energy bounds/recovery/
consumption, location consistency, +5m/+2h/+9h/+2d reconciliation, offline
gaps, idempotence, persistence roundtrip/corrupt/fail-soft, isolation,
restart, emotion isolation, zero-LLM-calls, and no-scheduler guarantees.
"""

import asyncio
import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone

from src.open_llm_vtuber import world_state
from src.open_llm_vtuber.world_state import (
    HISTORY_CAP,
    MAX_ENERGY,
    MIN_ENERGY,
    VALID_ACTIVITIES,
    VALID_LOCATIONS,
    WorldState,
    build_world_state_context,
    default_state,
    derive_mood,
    derive_time_context,
    load_and_reconcile_world_state,
    load_world_state,
    reconcile,
    save_world_state,
    transition,
)

UTC = timezone.utc


def noon(day=26):
    return datetime(2026, 9, day, 12, 0, 0, tzinfo=UTC)


def night(day=26):
    return datetime(2026, 9, day, 23, 0, 0, tzinfo=UTC)


def state_at(activity, energy=80, location="room", at=None, history=None):
    moment = at or noon()
    stamp = moment.isoformat(timespec="seconds")
    return WorldState(
        location=location,
        activity=activity,
        energy=energy,
        mood=derive_mood(activity, energy),
        time_context=derive_time_context(moment),
        activity_started_at=stamp,
        last_update_at=stamp,
        recent_activity_history=list(history or []),
    )


class InitialStateTests(unittest.TestCase):
    """A. Initial state."""

    def test_defaults_are_valid_and_bounded(self):
        state = default_state(noon())
        self.assertIn(state.location, VALID_LOCATIONS)
        self.assertIn(state.activity, VALID_ACTIVITIES)
        self.assertGreaterEqual(state.energy, MIN_ENERGY)
        self.assertLessEqual(state.energy, MAX_ENERGY)
        self.assertEqual(state.recent_activity_history, [])
        self.assertIsNotNone(state.activity_started_at)
        self.assertIsNotNone(state.last_update_at)
        self.assertEqual(state.time_context, "afternoon")


class TransitionTests(unittest.TestCase):
    """B. Basic activity transitions (pure core, no I/O)."""

    def test_sleeping_wakes_after_8h(self):
        start = night() - timedelta(hours=9)
        state = state_at("sleeping", energy=20, at=start)
        updated, changed = transition(state, 9 * 3600, night())
        self.assertEqual(updated.activity, "idle")
        self.assertTrue(changed)

    def test_eating_ends_after_45m(self):
        start = noon() - timedelta(hours=1)
        state = state_at("eating", at=start)
        updated, _ = transition(state, 3600, noon())
        self.assertEqual(updated.activity, "idle")

    def test_playing_ends_after_2h(self):
        start = noon() - timedelta(hours=3)
        state = state_at("playing", at=start)
        updated, _ = transition(state, 3 * 3600, noon())
        self.assertEqual(updated.activity, "idle")

    def test_reading_ends_after_3h(self):
        start = noon() - timedelta(hours=4)
        state = state_at("reading", at=start)
        updated, _ = transition(state, 4 * 3600, noon())
        self.assertEqual(updated.activity, "idle")

    def test_resting_ends_after_2h(self):
        start = noon() - timedelta(hours=3)
        state = state_at("resting", energy=50, at=start)
        updated, _ = transition(state, 3 * 3600, noon())
        self.assertEqual(updated.activity, "idle")

    def test_history_appended_and_capped(self):
        history = [
            {"from": "idle", "to": "reading", "at": "t", "location": "room"}
            for _ in range(HISTORY_CAP)
        ]
        start = noon() - timedelta(hours=4)
        state = state_at("reading", at=start, history=history)
        updated, changed = transition(state, 4 * 3600, noon())
        self.assertTrue(changed)
        self.assertEqual(updated.activity, "idle")
        self.assertLessEqual(len(updated.recent_activity_history), HISTORY_CAP)
        self.assertEqual(updated.recent_activity_history[-1]["to"], "idle")
        self.assertEqual(updated.recent_activity_history[-1]["from"], "reading")


class EnergyTests(unittest.TestCase):
    """C/D/E. Bounds, recovery, consumption."""

    def test_energy_never_exceeds_bounds(self):
        low = state_at("playing", energy=2, at=noon() - timedelta(hours=100))
        updated, _ = transition(low, 100 * 3600, noon())
        self.assertGreaterEqual(updated.energy, MIN_ENERGY)

        rested = state_at("sleeping", energy=99, at=noon() - timedelta(hours=100))
        updated, _ = transition(rested, 100 * 3600, noon())
        self.assertLessEqual(updated.energy, MAX_ENERGY)

    def test_sleeping_and_resting_recover_energy(self):
        woke = state_at("sleeping", energy=10, at=noon() - timedelta(hours=2))
        updated, _ = transition(woke, 2 * 3600, noon())
        self.assertGreater(updated.energy, 10)

        rested = state_at("resting", energy=10, at=noon() - timedelta(hours=2))
        updated, _ = transition(rested, 2 * 3600, noon())
        self.assertGreater(updated.energy, 10)

    def test_active_pastimes_consume_energy(self):
        for activity in ("playing", "reading", "idle"):
            before = state_at(activity, energy=80, at=noon() - timedelta(hours=2))
            after, _ = transition(before, 2 * 3600, noon())
            self.assertLess(after.energy, 80, activity)
        # Playing drains faster than reading.
        play = transition(
            state_at("playing", energy=80, at=noon() - timedelta(hours=1)),
            3600,
            noon(),
        )[0].energy
        read = transition(
            state_at("reading", energy=80, at=noon() - timedelta(hours=1)),
            3600,
            noon(),
        )[0].energy
        self.assertLess(play, read)


class LocationConsistencyTests(unittest.TestCase):
    """F. Location/activity consistency."""

    def test_sleeping_reading_resting_stay_in_room(self):
        for activity in ("sleeping", "resting", "reading"):
            state = state_at(activity, location="outside", at=noon())
            updated, _ = transition(state, 60, noon())
            self.assertEqual(updated.location, "room", activity)

    def test_eating_happens_in_kitchen(self):
        state = state_at("eating", location="room", at=noon())
        updated, _ = transition(state, 60, noon())
        self.assertEqual(updated.location, "kitchen")

    def test_playing_outside_by_day_inside_at_night(self):
        day = state_at("playing", location="room", at=noon() - timedelta(minutes=1))
        updated, _ = transition(day, 60, noon())
        self.assertEqual(updated.location, "outside")

        evening = night() - timedelta(minutes=1)
        late = state_at("playing", location="outside", at=evening)
        updated, _ = transition(late, 60, night())
        self.assertEqual(updated.location, "room")

    def test_idle_keeps_location(self):
        state = state_at("idle", location="outside", at=noon())
        updated, _ = transition(state, 60, noon())
        self.assertEqual(updated.location, "outside")


class ReconciliationTests(unittest.TestCase):
    """G/H/I/J. Fake-clock reconciliation spans."""

    def test_plus_5_minutes(self):
        base = noon()
        state = state_at("reading", energy=80, at=base)
        updated, changed = reconcile(state, base + timedelta(minutes=5))
        # +5min of reading drifts 80 -> 79.83 -> rounds back to 80: no
        # material change, so the baseline (and last_update_at) is kept
        # for future decay instead of being consumed by the timestamp.
        self.assertFalse(changed)
        self.assertEqual(updated.activity, "reading")
        self.assertEqual(updated.energy, 80)
        self.assertEqual(updated.last_update_at, state.last_update_at)
        self.assertGreaterEqual(updated.energy, MIN_ENERGY)

    def test_plus_2_hours(self):
        base = noon()
        state = state_at("playing", energy=80, at=base)
        updated, changed = reconcile(state, base + timedelta(hours=2))
        self.assertTrue(changed)
        self.assertEqual(updated.activity, "idle")
        self.assertLess(updated.energy, 80)

    def test_plus_9_hours_sleep(self):
        base = night()
        state = state_at("sleeping", energy=15, at=base)
        updated, changed = reconcile(state, base + timedelta(hours=9))
        self.assertTrue(changed)
        self.assertEqual(updated.activity, "idle")
        self.assertGreater(updated.energy, 15)

    def test_plus_2_days_offline_single_step(self):
        base = noon(day=24)
        state = state_at("sleeping", energy=10, at=base)
        updated, changed = reconcile(state, noon(day=26))
        self.assertTrue(changed)
        # Offline gap collapses into one step: awake, restored, one record.
        self.assertEqual(updated.activity, "idle")
        self.assertEqual(updated.energy, MAX_ENERGY)
        self.assertEqual(len(updated.recent_activity_history), 1)

    def test_backwards_clock_is_noop(self):
        state = state_at("idle", at=noon())
        updated, changed = reconcile(state, noon() - timedelta(hours=1))
        self.assertFalse(changed)
        self.assertEqual(updated.last_update_at, state.last_update_at)


class IdempotenceTests(unittest.TestCase):
    """K. Same `now` twice: no duplicates, no drift."""

    def test_double_reconcile_is_noop(self):
        base = noon()
        state = state_at("reading", energy=80, at=base)
        first, changed_first = reconcile(state, base + timedelta(hours=1))
        second, changed_second = reconcile(first, base + timedelta(hours=1))
        self.assertTrue(changed_first)
        self.assertFalse(changed_second)
        self.assertEqual(first.last_update_at, second.last_update_at)
        self.assertEqual(first.recent_activity_history, second.recent_activity_history)
        self.assertEqual(first.energy, second.energy)


class PersistenceTests(unittest.TestCase):
    """L/M/N. Roundtrip, corrupt file, save failure."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base_dir = os.path.join(self._tmp.name, "world_state")

    def tearDown(self):
        self._tmp.cleanup()

    def test_roundtrip(self):
        original = state_at("reading", energy=72, at=noon())
        self.assertTrue(save_world_state("mili", original, self.base_dir))
        loaded = load_world_state("mili", noon(), self.base_dir)
        self.assertEqual(loaded.location, original.location)
        self.assertEqual(loaded.activity, original.activity)
        self.assertEqual(loaded.energy, original.energy)
        self.assertEqual(loaded.mood, original.mood)
        self.assertEqual(
            loaded.recent_activity_history, original.recent_activity_history
        )

    def test_corrupt_json_yields_safe_default(self):
        os.makedirs(self.base_dir, exist_ok=True)
        with open(os.path.join(self.base_dir, "mili.json"), "w") as file:
            file.write("{not valid json!!!")
        loaded = load_world_state("mili", noon(), self.base_dir)
        self.assertIn(loaded.activity, VALID_ACTIVITIES)
        self.assertGreaterEqual(loaded.energy, MIN_ENERGY)
        self.assertLessEqual(loaded.energy, MAX_ENERGY)

    def test_save_failure_is_fail_soft(self):
        # A file where the directory should be forces save failure.
        blocker = os.path.join(self._tmp.name, "blocker")
        with open(blocker, "w") as file:
            file.write("x")
        self.assertFalse(save_world_state("mili", state_at("idle", at=noon()), blocker))
        # load_and_reconcile never raises and still returns usable state.
        state = load_and_reconcile_world_state("mili", noon(), blocker)
        self.assertIn(state.activity, VALID_ACTIVITIES)


class IsolationTests(unittest.TestCase):
    """O. Character/config isolation."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base_dir = os.path.join(self._tmp.name, "world_state")

    def tearDown(self):
        self._tmp.cleanup()

    def test_characters_do_not_leak(self):
        save_world_state(
            "mili-a", state_at("sleeping", energy=10, at=noon()), self.base_dir
        )
        save_world_state(
            "mili-b", state_at("playing", energy=90, at=noon()), self.base_dir
        )
        a = load_world_state("mili-a", noon(), self.base_dir)
        b = load_world_state("mili-b", noon(), self.base_dir)
        self.assertEqual(a.activity, "sleeping")
        self.assertEqual(b.activity, "playing")

    def test_restart_offline_reconciliation(self):
        """P. Persist, fresh load, reconcile the gap (simulated restart)."""
        start = noon(day=24)
        save_world_state(
            "mili", state_at("sleeping", energy=12, at=start), self.base_dir
        )
        # Fresh "process": load from disk two days later and reconcile.
        reloaded = load_and_reconcile_world_state("mili", noon(day=26), self.base_dir)
        self.assertEqual(reloaded.activity, "idle")
        self.assertGreater(reloaded.energy, 12)


class EmotionIsolationTests(unittest.IsolatedAsyncioTestCase):
    """Q. World mood never touches Actions.expressions / Actions.emotions."""

    def setUp(self):
        self._old_cwd = os.getcwd()
        self._tmp = tempfile.TemporaryDirectory()
        os.chdir(self._tmp.name)
        self.conf_uid = "mili-world-q"

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._tmp.cleanup()

    def test_world_module_has_no_emotion_writes(self):
        import ast
        import inspect

        tree = ast.parse(inspect.getsource(world_state))
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        # No code may reference the emotion action channel (docstring
        # constraints excluded: AST sees code, not prose).
        self.assertNotIn("Actions", names)
        self.assertNotIn("expressions", attrs)
        self.assertNotIn("emotions", attrs)
        self.assertNotIn("extract_emotion", names)
        self.assertNotIn("extract_emotion", attrs)

    async def test_chat_turn_actions_come_only_from_markers(self):
        from src.open_llm_vtuber.agent.agents.basic_memory_agent import (
            BasicMemoryAgent,
        )
        from src.open_llm_vtuber.agent.input_types import (
            BatchInput,
            TextData,
            TextSource,
        )
        from src.open_llm_vtuber.chat_history_manager import create_new_history
        from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig

        class _CountingLLM:
            model = "world-q-test"
            max_tokens = 100

            def __init__(self):
                self.calls = 0

            async def chat_completion(self, messages, system=None, tools=None):
                self.calls += 1
                yield "[joy] Halo."

        class _MarkerLive2D:
            def extract_emotion(self, text):
                return [3] if "[joy]" in text else []

            def extract_emotion_keys(self, text):
                return ["joy"] if "[joy]" in text else []

            def remove_emotion_keywords(self, text):
                return text.replace("[joy]", "").strip()

        # Force a mood far from joy: sleepy/exhausted world. Seed at the
        # real current time (not a fixed clock) so the sleeping activity
        # cannot expire between seeding and the agent's lazy reconcile.
        from src.open_llm_vtuber.world_state import utcnow

        save_world_state(self.conf_uid, state_at("sleeping", energy=5, at=utcnow()))
        history_uid = create_new_history(self.conf_uid)
        llm = _CountingLLM()
        agent = BasicMemoryAgent(
            llm=llm,
            system="persona Mili",
            live2d_model=_MarkerLive2D(),
            tts_preprocessor_config=TTSPreprocessorConfig(
                remove_special_char=True,
                translator_config={
                    "translate_audio": False,
                    "translate_provider": "deeplx",
                },
            ),
            context_window_override=2000,
        )
        agent.set_memory_from_history(self.conf_uid, history_uid)

        prompt = agent._relationship_system_prompt(agent._system)
        self.assertIn("[Mili World State]", prompt)
        self.assertIn("mood=sleepy", prompt)

        seen = []
        async for item in agent.chat(
            BatchInput(texts=[TextData(source=TextSource.INPUT, content="hai")])
        ):
            seen.append(item)
        outputs = [item for item in seen if hasattr(item, "actions")]
        self.assertTrue(outputs)
        for output in outputs:
            self.assertEqual(list(output.actions.expressions or []), [3])
            self.assertEqual(list(output.actions.emotions or []), ["joy"])
            self.assertNotIn("[joy]", output.display_text.text)


class ZeroLLMCallTests(unittest.TestCase):
    """R. Transition path makes zero LLM/provider calls."""

    def test_reconcile_and_prompt_build_call_no_llm(self):
        from src.open_llm_vtuber.agent.agents.basic_memory_agent import (
            BasicMemoryAgent,
        )
        from src.open_llm_vtuber.chat_history_manager import create_new_history
        from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig

        old_cwd = os.getcwd()
        tmp = tempfile.TemporaryDirectory()
        os.chdir(tmp.name)
        try:

            class _SilentLLM:
                model = "world-r-test"
                max_tokens = 100

                def __init__(self):
                    self.calls = 0

                async def chat_completion(self, messages, system=None, tools=None):
                    self.calls += 1
                    yield "diam."

            class _QuietLive2D:
                def extract_emotion(self, _text):
                    return []

            history_uid = create_new_history("mili-world-r")
            llm = _SilentLLM()
            agent = BasicMemoryAgent(
                llm=llm,
                system="persona Mili",
                live2d_model=_QuietLive2D(),
                tts_preprocessor_config=TTSPreprocessorConfig(
                    remove_special_char=True,
                    translator_config={
                        "translate_audio": False,
                        "translate_provider": "deeplx",
                    },
                ),
                context_window_override=2000,
            )
            agent.set_memory_from_history("mili-world-r", history_uid)
            agent._relationship_system_prompt(agent._system)
            load_and_reconcile_world_state("mili-world-r", noon() + timedelta(hours=3))
            self.assertEqual(llm.calls, 0)
        finally:
            os.chdir(old_cwd)
            tmp.cleanup()


class NoSchedulerTests(unittest.TestCase):
    """S. No second scheduler/task is created by the world module."""

    def test_no_scheduler_primitives_in_world_state(self):
        import ast
        import inspect

        # AST sees code, not docstring prose (which documents the
        # no-scheduler constraint itself).
        tree = ast.parse(inspect.getsource(world_state))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(a.name.split(".")[0] for a in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        self.assertNotIn("asyncio", imports)
        self.assertNotIn("sched", imports)
        self.assertEqual([n for n in ast.walk(tree) if isinstance(n, ast.Await)], [])
        names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
        attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        for token in (
            "create_task",
            "sleep",
            "setInterval",
            "setTimeout",
            "Timer",
            "Thread",
            "BackgroundScheduler",
            "daemon",
        ):
            self.assertNotIn(token, names, token)
            self.assertNotIn(token, attrs, token)

    def test_no_tasks_spawned_on_reconcile(self):
        async def _count():
            import asyncio

            before = len(asyncio.all_tasks())
            load_and_reconcile_world_state("mili-world-s", noon())
            await asyncio.sleep(0)
            return before, len(asyncio.all_tasks())

        before, after = asyncio.run(_count())
        self.assertEqual(before, after)


class PromptFormatTests(unittest.TestCase):
    def test_compact_single_line_format(self):
        state = state_at("reading", energy=72, at=noon())
        state.mood = "calm"
        line = build_world_state_context(state)
        self.assertTrue(line.startswith("[Mili World State]"))
        for key in ("location=", "activity=", "energy=", "mood=", "time_context="):
            self.assertIn(key, line)
        # Compact: header + exactly one state line.
        self.assertEqual(len(line.strip().splitlines()), 2)


class UserTimezoneTests(unittest.TestCase):
    """Phase 2: user-local hour drives time rules; storage stays UTC."""

    TZ = "Asia/Jakarta"  # UTC+7, no DST

    def utc(self, hour, minute=0):
        return datetime(2026, 9, 26, hour, minute, tzinfo=timezone.utc)

    def test_utc7_boundaries(self):
        # local = UTC + 7
        cases = [
            (22, 0, "morning"),  # 05:00 local
            (4, 0, "afternoon"),  # 11:00 local
            (10, 0, "evening"),  # 17:00 local
            (15, 0, "night"),  # 22:00 local
        ]
        for utc_hour, minute, expected in cases:
            with self.subTest(utc_hour=utc_hour):
                self.assertEqual(
                    derive_time_context(self.utc(utc_hour, minute), self.TZ),
                    expected,
                )

    def test_utc7_edges(self):
        edges = [
            ((21, 59), "night"),  # 04:59 local
            ((22, 0), "morning"),  # 05:00 local
            ((3, 59), "morning"),  # 10:59 local
            ((4, 0), "afternoon"),  # 11:00 local
            ((9, 59), "afternoon"),  # 16:59 local
            ((10, 0), "evening"),  # 17:00 local
            ((14, 59), "evening"),  # 21:59 local
            ((15, 0), "night"),  # 22:00 local
        ]
        for (hour, minute), expected in edges:
            with self.subTest(utc=f"{hour}:{minute:02d}"):
                self.assertEqual(
                    derive_time_context(self.utc(hour, minute), self.TZ),
                    expected,
                )

    def test_observed_case_2240_wib_is_night(self):
        # 22:40 WIB == 15:40 UTC must read night, not afternoon.
        moment = datetime(2026, 9, 26, 15, 40, tzinfo=timezone.utc)
        self.assertEqual(derive_time_context(moment, self.TZ), "night")
        self.assertEqual(derive_time_context(moment, None), "afternoon")

    def test_live_report_boundaries(self):
        # Exact cases from the live bug report (UTC -> Asia/Jakarta).
        cases = [
            ((15, 16), "night"),  # 22:16 WIB
            ((16, 16), "night"),  # 23:16 WIB
            ((21, 0), "night"),  # 04:00 WIB
            ((22, 0), "morning"),  # 05:00 WIB
        ]
        for (hour, minute), expected in cases:
            with self.subTest(utc=f"{hour}:{minute:02d}"):
                self.assertEqual(
                    derive_time_context(
                        datetime(2026, 9, 26, hour, minute, tzinfo=timezone.utc),
                        self.TZ,
                    ),
                    expected,
                )

    def test_other_timezone_not_hardcoded(self):
        # America/New_York is UTC-4 in September: 15:40 UTC -> 11:40 local.
        moment = datetime(2026, 9, 26, 15, 40, tzinfo=timezone.utc)
        self.assertEqual(derive_time_context(moment, "America/New_York"), "afternoon")
        # 02:00 UTC -> 22:00 local previous day -> night.
        early = datetime(2026, 9, 26, 2, 0, tzinfo=timezone.utc)
        self.assertEqual(derive_time_context(early, "America/New_York"), "night")
        self.assertEqual(derive_time_context(early, None), "night")

    def test_invalid_timezone_falls_back_to_utc(self):
        moment = datetime(2026, 9, 26, 15, 40, tzinfo=timezone.utc)
        self.assertEqual(derive_time_context(moment, "Not/AZone"), "afternoon")
        self.assertEqual(derive_time_context(moment, ""), "afternoon")

    def test_reconcile_uses_local_hour_for_night_rule(self):
        # 15:40 UTC idle low energy: UTC says afternoon (stay idle),
        # Jakarta says 22:40 night -> sleeping.
        base = datetime(2026, 9, 26, 15, 30, tzinfo=timezone.utc)
        now = datetime(2026, 9, 26, 15, 40, tzinfo=timezone.utc)
        idle = WorldState(
            location="room",
            activity="idle",
            energy=20,
            mood="tired",
            time_context="afternoon",
            activity_started_at=base.isoformat(),
            last_update_at=base.isoformat(),
            recent_activity_history=[],
        )
        as_utc, _ = reconcile(idle, now, None)
        self.assertEqual(as_utc.activity, "idle")
        as_local, changed = reconcile(idle, now, self.TZ)
        self.assertTrue(changed)
        self.assertEqual(as_local.activity, "sleeping")
        self.assertEqual(as_local.time_context, "night")

    def test_persisted_timestamps_stay_utc(self):
        import tempfile

        base_dir = tempfile.mkdtemp()
        now = datetime(2026, 9, 26, 15, 40, tzinfo=timezone.utc)
        state = load_and_reconcile_world_state("tz-char", now, base_dir, self.TZ)
        self.assertEqual(state.time_context, "night")
        with open(os.path.join(base_dir, "tz-char.json")) as f:
            raw = json.load(f)
        for key in ("activity_started_at", "last_update_at"):
            self.assertTrue(raw[key].endswith("+00:00"), raw[key])


class FetchReconcileEnergyTests(unittest.TestCase):
    """fetch/load path must reconcile energy from elapsed wall-clock time."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.base_dir = os.path.join(self._tmp.name, "world_state")

    def tearDown(self):
        self._tmp.cleanup()

    def _seed(self, activity, energy, at):
        state = WorldState(
            location="room",
            activity=activity,
            energy=energy,
            mood=derive_mood(activity, energy),
            time_context="afternoon",
            activity_started_at=at.isoformat(),
            last_update_at=at.isoformat(),
            recent_activity_history=[],
        )
        self.assertTrue(save_world_state("mili", state, self.base_dir))
        return state

    def test_elapsed_36min_idle_drops_energy(self):
        # Live case: last_update 15:40 UTC idle 80, now 16:16 UTC.
        base = datetime(2026, 9, 26, 15, 40, tzinfo=timezone.utc)
        self._seed("idle", 80, base)
        updated = load_and_reconcile_world_state(
            "mili", base + timedelta(minutes=36), self.base_dir
        )
        # idle -1/hour * 0.6h = -0.6 -> 79.4 -> 79
        self.assertEqual(updated.energy, 79)
        self.assertEqual(updated.activity, "idle")
        # Persisted baseline advanced to the material change.
        reloaded = load_world_state("mili", base + timedelta(minutes=36), self.base_dir)
        self.assertEqual(reloaded.energy, 79)

    def test_frequent_refresh_no_longer_freezes_energy(self):
        # Regression: refreshes every 2 minutes for 2h of reading must
        # decay (linear ideal 76), not stick at 80 forever. Each material
        # persist rounds to int, so pathological polling may drift ~±2
        # around the ideal; sparse reconciles stay exact (see test above).
        base = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
        self._seed("reading", 80, base)
        now = base
        for _ in range(60):
            now += timedelta(minutes=2)
            load_and_reconcile_world_state("mili", now, self.base_dir)
        final = load_world_state("mili", now, self.base_dir)
        self.assertLess(final.energy, 80)
        self.assertGreaterEqual(final.energy, 72)
        self.assertLessEqual(final.energy, 76)

    def test_noop_refresh_keeps_baseline_for_future_decay(self):
        base = datetime(2026, 9, 26, 12, 0, tzinfo=timezone.utc)
        self._seed("reading", 80, base)
        first = load_and_reconcile_world_state(
            "mili", base + timedelta(minutes=5), self.base_dir
        )
        self.assertEqual(first.energy, 80)
        # Baseline preserved: a later +65min reconcile sees the full span.
        later = load_and_reconcile_world_state(
            "mili", base + timedelta(minutes=65), self.base_dir
        )
        self.assertEqual(later.energy, 78)


if __name__ == "__main__":
    unittest.main()
