"""PERSIST-6042 regression: explicit save must persist, honestly.

Covers the proven root cause:
- explicit "simpan ... sebagai ..." / "pastikan ... tersimpan" requests
  previously matched NO capture detector, so nothing was ever written while
  the model claimed "sudah aku simpan";
- temporal substring collisions ("akan" in "digunakan", "telah" in
  "setelah") misclassified turns in the episodic gate;
- existing lanes (4819 future intention, 10:30 episodic event, stable
  memories, previous-session summaries) must keep working.
"""

import os
import tempfile
import unittest

from src.open_llm_vtuber.character_memory_commands import parse_memory_command
from src.open_llm_vtuber.character_state import (
    add_character_memory,
    build_character_memory_context,
    load_character_state,
)
from src.open_llm_vtuber.episodic_memory import (
    append_episodic_event,
    is_episodic_candidate,
    load_episodic_events,
    retrieve_episodic_events,
)
from src.open_llm_vtuber.future_intentions import (
    build_future_intention_context,
    detect_any_future_intention,
    pending_intentions,
)
from src.open_llm_vtuber.memory_receipt import (
    build_memory_receipt_block,
    process_memory_command,
)

FIX_MESSAGE = (
    "Simpan fakta unik berikut sebagai informasi permanen: "
    "kode PERSIST-FIX-001. Ini hanya untuk eksperimen."
)
AUDIT_MESSAGE = (
    "Untuk eksperimen persistence ini, simpan fakta unik berikut sebagai "
    "informasi yang harus tetap tersedia setelah server/backend restart: "
    "kode PERSIST-6042. Kode ini hanya dibuat untuk eksperimen ini dan "
    "belum pernah digunakan sebelumnya."
)
CONFIRM_MESSAGE = (
    "Pastikan PERSIST-FIX-001 benar-benar tersimpan secara persisten dan "
    "bukan hanya tersedia selama sesi chat ini."
)


class ExplicitSaveGrammarTest(unittest.TestCase):
    def test_sebagai_save_initial(self):
        result = parse_memory_command(FIX_MESSAGE)
        self.assertEqual(result.action, "remember")
        self.assertIn("PERSIST-FIX-001", result.payload or "")

    def test_sebagai_save_mid_sentence(self):
        result = parse_memory_command(AUDIT_MESSAGE)
        self.assertEqual(result.action, "remember")
        self.assertIn("PERSIST-6042", result.payload or "")

    def test_pastikan_tersimpan(self):
        result = parse_memory_command(CONFIRM_MESSAGE)
        self.assertEqual(result.action, "remember")
        self.assertIn("PERSIST-FIX-001", result.payload or "")

    def test_save_negatives_stay_none(self):
        for text in (
            "Aku simpan uang sebagai dana darurat.",
            "Apa kode eksperimen yang baru saja kusuruh simpan? Jangan menebak.",
            "Apa kode yang kusimpan? Jangan menebak.",
            "Simpan file ini.",
            "Catatannya ada di meja.",
            "Jangan lupa makan.",
            "Pastikan ini tersimpan.",
            "Mil, besok jam 14.23 aku mau nanya kamu tentang angka 4819.",
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_memory_command(text).action, "none")

    def test_old_positives_preserved(self):
        for text in (
            "Ingat ya, gw suka kopi.",
            "Simpan ini, gw suka kopi.",
            "Tolong ingat ini: kodeku 777.",
            "Catat di ingatan kamu, gw suka kopi.",
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_memory_command(text).action, "remember")


class ExplicitSavePersistenceTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        self.conf_uid = "persist-fix-uid"

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _save_turn(self, text):
        """Simulate pre-turn receipt + post-turn observer for one turn."""
        receipt = process_memory_command(
            text,
            remember_fn=(
                lambda payload: add_character_memory(
                    self.conf_uid, payload, explicit=True
                )
                is not None
            ),
            forget_fn=lambda payload: False,
        )
        self.assertIsNotNone(receipt)
        self.assertTrue(receipt.stored)
        # Post-turn observer runs the same write again (dedup must hold).
        again = add_character_memory(self.conf_uid, receipt.payload, explicit=True)
        self.assertIsNotNone(again)
        return receipt

    def test_explicit_save_creates_persistent_record(self):
        receipt = self._save_turn(FIX_MESSAGE)
        # Record is on disk, not just transcript: fresh load finds it.
        fresh = load_character_state(self.conf_uid)
        texts = [m.get("text", "") for m in fresh.memories]
        self.assertTrue(any("PERSIST-FIX-001" in t for t in texts))
        self.assertTrue(
            any(
                m.get("explicit")
                for m in fresh.memories
                if "PERSIST-FIX" in m.get("text", "")
            )
        )
        _ = receipt

    def test_no_duplicate_rows_same_turn(self):
        self._save_turn(FIX_MESSAGE)
        hits = [
            m
            for m in load_character_state(self.conf_uid).memories
            if "PERSIST-FIX-001" in m.get("text", "")
        ]
        self.assertEqual(len(hits), 1)

    def test_same_session_recall_renders_code(self):
        self._save_turn(FIX_MESSAGE)
        block = build_character_memory_context(load_character_state(self.conf_uid))
        self.assertIn("PERSIST-FIX-001", block)

    def test_cross_session_recall_after_reload(self):
        self._save_turn(FIX_MESSAGE)
        # Genuinely new session: fresh state object, no shared references.
        other_session_state = load_character_state(self.conf_uid)
        block = build_character_memory_context(other_session_state)
        self.assertIn("PERSIST-FIX-001", block)

    def test_old_state_without_new_fields_loads(self):
        # Legacy-shaped state (no kind/explicit extras) must still load.
        legacy = load_character_state(self.conf_uid)
        legacy.memories.append({"text": "punya kucing bernama Mochi"})
        from src.open_llm_vtuber.character_state import save_character_state

        self.assertTrue(save_character_state(self.conf_uid, legacy))
        reloaded = load_character_state(self.conf_uid)
        self.assertTrue(any("Mochi" in m.get("text", "") for m in reloaded.memories))
        block = build_character_memory_context(reloaded)
        self.assertIn("Mochi", block)


class MemoryReceiptHonestyTest(unittest.TestCase):
    def test_stored_receipt_permits_confirmation(self):
        receipt = process_memory_command(
            FIX_MESSAGE,
            remember_fn=lambda payload: True,
            forget_fn=lambda payload: False,
        )
        self.assertTrue(receipt.stored)
        block = build_memory_receipt_block(receipt)
        self.assertIn("HAS been written", block)
        self.assertIn("PERSIST-FIX-001", block)

    def test_failed_write_forbids_saved_claim(self):
        def _boom(payload):
            raise IOError("simulated disk failure")

        receipt = process_memory_command(
            FIX_MESSAGE, remember_fn=_boom, forget_fn=lambda payload: False
        )
        self.assertIsNotNone(receipt)
        self.assertFalse(receipt.stored)
        block = build_memory_receipt_block(receipt)
        # Backend semantic, not brittle wording: the block must state the
        # write did NOT happen and forbid claiming otherwise.
        self.assertIn("NOT saved", block)
        self.assertIn("Do NOT claim it was saved", block)
        self.assertNotIn("HAS been written", block)

    def test_no_command_no_block(self):
        receipt = process_memory_command(
            "halo, apa kabar hari ini?",
            remember_fn=lambda payload: True,
            forget_fn=lambda payload: False,
        )
        self.assertIsNone(receipt)
        self.assertEqual(build_memory_receipt_block(receipt), "")
        self.assertEqual(build_memory_receipt_block(None), "")

    def test_forget_receipt_paths(self):
        removed = process_memory_command(
            "lupakan kode lama itu",
            remember_fn=lambda payload: True,
            forget_fn=lambda payload: True,
        )
        self.assertEqual(removed.action, "forget")
        self.assertIn("HAS been removed", build_memory_receipt_block(removed))
        missing = process_memory_command(
            "lupakan kode lama itu",
            remember_fn=lambda payload: True,
            forget_fn=lambda payload: False,
        )
        self.assertFalse(missing.stored)
        self.assertIn("Do NOT claim", build_memory_receipt_block(missing))


class TemporalSubstringRegressionTest(unittest.TestCase):
    def test_substring_collisions_rejected(self):
        for text in (
            "Setelah makan siang kami pulang dan beristirahat dengan tenang.",
            "Kode ini belum pernah digunakan sebelumnya dan akan disimpan.",
            "Aku menggunakan aplikasi itu setiap hari untuk bekerja.",
            "Besok jam 14.23 aku ada ujian penting dan harus belajar.",
        ):
            with self.subTest(text=text):
                self.assertFalse(is_episodic_candidate(text))

    def test_valid_temporal_cases_preserved(self):
        for text in (
            "Semalam aku begadang mengerjakan tugas sampai pagi buta sekali.",
            "Aku sudah selesai memperbaiki bug itu 4 hari lalu dengan cepat.",
            "Aku menghabiskan 3 jam memperbaiki bug frontend pada 7 Oktober "
            "2026 pukul 10:30 WIB.",
            "Aku baru saja menghabiskan 3 jam memperbaiki bug frontend.",
            "Tadi aku sudah makan siang bersama teman-teman di kantin.",
            "I just finished writing the report for last week meeting today.",
        ):
            with self.subTest(text=text):
                self.assertTrue(is_episodic_candidate(text))


class ExistingLanesRegressionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        self.conf_uid = "lanes-regression-uid"

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def test_4819_future_intention_still_detected(self):
        found = detect_any_future_intention(
            "Mil, besok jam 14.23 aku mau nanya kamu tentang angka 4819. "
            "Ini juga rencana yang belum aku tanyakan sekarang."
        )
        self.assertIsNotNone(found)
        self.assertEqual(found.kind, "plan")
        # ... and it must NOT leak into the episodic lane.
        self.assertFalse(
            is_episodic_candidate(
                "Mil, besok jam 14.23 aku mau nanya kamu tentang angka 4819. "
                "Ini juga rencana yang belum aku tanyakan sekarang."
            )
        )

    def test_4819_intention_recallable(self):
        from src.open_llm_vtuber.character_state import record_future_intention

        stored = record_future_intention(
            self.conf_uid,
            "Mil, besok jam 14.23 aku mau nanya kamu tentang angka 4819.",
        )
        self.assertIsNotNone(stored)
        state = load_character_state(self.conf_uid)
        self.assertTrue(
            any(
                "4819" in row.get("text", "")
                for row in pending_intentions(state.future_intentions)
            )
        )
        self.assertIn("4819", build_future_intention_context(state.future_intentions))

    def test_october_bug_episodic_event_retrievable(self):
        stored = append_episodic_event(
            self.conf_uid,
            {
                "event_text": "Aku menghabiskan 3 jam memperbaiki bug "
                "frontend pada 7 Oktober 2026 pukul 10:30 WIB.",
                "occurred_at": "2026-10-07T03:30:00+00:00",
                "session_uid": "2026-10-07_05-56-06_test",
                "source": "conversation",
            },
        )
        self.assertIsNotNone(stored)
        # Cross-session: fresh load from disk, then retrieve.
        events = load_episodic_events(self.conf_uid)
        hits = retrieve_episodic_events(events, "bug frontend 10:30 Oktober")
        self.assertTrue(hits)
        self.assertIn("10:30", hits[0].get("event_text", ""))

    def test_stable_memories_still_work(self):
        from src.open_llm_vtuber.character_memory_commands import (
            extract_stable_facts,
        )

        facts = extract_stable_facts("Aku suka kopi susu gula aren.")
        self.assertTrue(facts)
        for fact in facts:
            self.assertIsNotNone(
                add_character_memory(self.conf_uid, fact, explicit=False)
            )
        block = build_character_memory_context(load_character_state(self.conf_uid))
        self.assertIn("kopi susu", block)


class AgentReceiptWiringTest(unittest.TestCase):
    """Pre-turn receipt through the agent's own write path (no mocks)."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._old = os.getcwd()
        os.chdir(self._tmp.name)
        self.conf_uid = "receipt-wiring-uid"

    def tearDown(self):
        os.chdir(self._old)
        self._tmp.cleanup()

    def _agent(self):
        from src.open_llm_vtuber.agent.agents.basic_memory_agent import (
            BasicMemoryAgent,
        )

        agent = object.__new__(BasicMemoryAgent)
        agent._character_conf_uid = self.conf_uid
        agent._character_state = load_character_state(self.conf_uid)
        agent._memory_receipt_block = ""
        return agent

    def test_pre_turn_write_verified_before_response(self):
        agent = self._agent()
        agent._refresh_memory_receipt(FIX_MESSAGE)
        self.assertIn("HAS been written", agent._memory_receipt_block)
        self.assertIn("PERSIST-FIX-001", agent._memory_receipt_block)
        # The fact is already in the agent's in-memory state, so the
        # response generated right after sees it in the memory block.
        block = build_character_memory_context(agent._character_state)
        self.assertIn("PERSIST-FIX-001", block)

    def test_ordinary_turn_leaves_no_receipt(self):
        agent = self._agent()
        agent._refresh_memory_receipt("halo, apa kabar hari ini?")
        self.assertEqual(agent._memory_receipt_block, "")

    def test_failed_pre_turn_write_forbids_claim(self):
        agent = self._agent()
        agent.add_character_memory = lambda *a, **k: False
        agent._refresh_memory_receipt(FIX_MESSAGE)
        self.assertIn("Do NOT claim it was saved", agent._memory_receipt_block)
        self.assertNotIn("HAS been written", agent._memory_receipt_block)

    def test_last_user_text_multipart_safe(self):
        from src.open_llm_vtuber.agent.agents.basic_memory_agent import (
            BasicMemoryAgent,
        )

        self.assertEqual(
            BasicMemoryAgent._last_user_text(
                [
                    {"role": "system", "content": "x"},
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "simpan ini sebagai data"},
                            {"type": "image_url", "image_url": {"url": "u"}},
                        ],
                    },
                ]
            ),
            "simpan ini sebagai data",
        )
        self.assertEqual(BasicMemoryAgent._last_user_text([]), "")
        self.assertEqual(
            BasicMemoryAgent._last_user_text([{"role": "assistant", "content": "hi"}]),
            "",
        )


if __name__ == "__main__":
    unittest.main()
