import os
import tempfile
import unittest

from src.open_llm_vtuber.agent.agents.basic_memory_agent import BasicMemoryAgent
from src.open_llm_vtuber.character_memory_commands import parse_memory_command
from src.open_llm_vtuber.character_state import (
    load_character_state,
    set_character_relationship,
)
from src.open_llm_vtuber.chat_history_manager import (
    create_new_history,
    get_history,
    store_message,
)


REMEMBER_POSITIVES = (
    "Ingat ya, gw suka kopi.",
    "Ingat yah, gw suka kopi.",
    "Ingat yaa, gw suka kopi.",
    "Ingat yaaa, gw suka kopi.",
    "Ingat yak, gw suka kopi.",
    "Inget ya, gw suka kopi.",
    "Inget yah, gw suka kopi.",
    "Inget dong, gw suka kopi.",
    "Tolong ingat, gw suka kopi.",
    "Tolong inget yah, gw suka kopi.",
    "Tolong diingat, gw suka kopi.",
    "Jangan lupa kalau gw suka kopi.",
    "Jangan lupa yah, gw suka kopi.",
    "Catat, gw suka kopi.",
    "Catet ya, gw suka kopi.",
    "Catat di ingatan kamu, gw suka kopi.",
    "Simpan ini, gw suka kopi.",
    "Simpen ya, gw suka kopi.",
    "Simpan ke ingatan, gw suka kopi.",
    "Masukkan ke ingatan, gw suka kopi.",
    "Masukin ke ingatan kamu, gw suka kopi.",
    "Mulai sekarang ingat kalau gw suka kopi.",
    "Ingat buat ke depannya, gw suka kopi.",
    "Eh, inget ya, gw suka kopi.",
    "Oh iya, inget yah, gw suka kopi.",
    "Btw, catat ya, gw suka kopi.",
)

REMEMBER_NEGATIVES = (
    "Ingat gak gw pernah bilang apa?",
    "Inget gak Silent Hill?",
    "Kamu masih inget aku?",
    "Gw jadi inget sesuatu.",
    "Aku inget dulu pernah ke sana.",
    "Baru inget besok Senin.",
    "Dia bilang jangan lupa makan.",
    "Jangan lupa makan.",
    "Jangan lupa tidur.",
    "Catatannya ada di meja.",
    "Simpan file ini.",
    "Ingat ya.",
    "Tolong inget.",
    "Remember?",
    "Lu inget Silent Hill gak?",
    "Filmnya judulnya Jangan Lupa.",
    "Karakter itu bilang 'ingat aku'.",
    "Jangan lupa cek server nanti.",
    "Jangan lupa balas aku.",
)

FORGET_POSITIVES = (
    "Lupakan kalau gw suka kopi.",
    "Lupain kalau gw suka kopi.",
    "Lupain soal gw suka kopi.",
    "Hapus dari ingatan, gw suka kopi.",
    "Hapus dari ingatan kamu, gw suka kopi.",
    "Hapus memory tentang gw suka kopi.",
    "Jangan ingat lagi kalau gw suka kopi.",
    "Jangan inget lagi soal gw suka kopi.",
)

FORGET_NEGATIVES = (
    "Gw lupa tadi ngomong apa.",
    "Kamu lupa ya?",
    "Aku lupa judulnya.",
    "Jangan lupa kalau gw suka kopi.",
    "Lupa gak sih?",
    "Kayaknya gw udah lupa.",
)


class CharacterMemoryCommandParserTests(unittest.TestCase):
    def test_remember_command_matrix(self):
        for text in REMEMBER_POSITIVES:
            with self.subTest(text=text):
                result = parse_memory_command(text)
                self.assertEqual(result.action, "remember")
                self.assertEqual(result.payload, "gw suka kopi")
                self.assertTrue(result.matched_trigger.startswith("remember_"))

    def test_remember_english_commands(self):
        for text in (
            "Remember, I like coffee.",
            "Remember that I like coffee.",
            "Remember this: I like coffee.",
            "Please remember, I like coffee.",
            "Please remember that I like coffee.",
        ):
            with self.subTest(text=text):
                result = parse_memory_command(text)
                self.assertEqual(result.action, "remember")
                self.assertEqual(result.payload, "I like coffee")

    def test_additional_remember_families_and_particles(self):
        cases = {
            "Ingat, gue suka kopi.": "gue suka kopi",
            "Ingat ini ya, aku suka hujan.": "aku suka hujan",
            "Inget baik-baik ya, saya sedang belajar HTML.": (
                "saya sedang belajar HTML"
            ),
            "Ingat satu hal, gua lebih suka jawaban singkat.": (
                "gua lebih suka jawaban singkat"
            ),
            "Tolong inget kalau ane suka ramen.": "ane suka ramen",
            "Jgn lupa, game favorit gw Silent Hill.": ("game favorit gw Silent Hill"),
            "Jangan lupain kalau gw gak suka dipanggil formal.": (
                "gw gak suka dipanggil formal"
            ),
            "Catet ini ya, panggil gw Irfan.": "panggil gw Irfan",
            "Catat bahwa user biasanya belajar malam.": ("user biasanya belajar malam"),
            "Simpen di ingatanmu, gw pengen belajar Inggris.": (
                "gw pengen belajar Inggris"
            ),
            "Simpan sebagai ingatan, gw suka film detektif.": ("gw suka film detektif"),
            "Masukin ini ke ingatan, gw suka kopi susu.": "gw suka kopi susu",
            "Inget untuk kedepannya, gw suka kopi.": "gw suka kopi",
            "Eh btw, catat dong, gw suka kopi.": "gw suka kopi",
            "Inget ya deh, gw suka kopi.": "gw suka kopi",
        }
        for text, payload in cases.items():
            with self.subTest(text=text):
                result = parse_memory_command(text)
                self.assertEqual(result.action, "remember")
                self.assertEqual(result.payload, payload)

    def test_remember_false_positive_matrix(self):
        for text in REMEMBER_NEGATIVES:
            with self.subTest(text=text):
                self.assertEqual(parse_memory_command(text).action, "none")

    def test_forget_command_matrix(self):
        for text in FORGET_POSITIVES:
            with self.subTest(text=text):
                result = parse_memory_command(text)
                self.assertEqual(result.action, "forget")
                self.assertEqual(result.payload, "gw suka kopi")
                self.assertTrue(result.matched_trigger.startswith("forget_"))

    def test_additional_forget_families(self):
        cases = {
            "Lupakan bahwa gw suka kopi.": "gw suka kopi",
            "Lupakan yang tentang gw suka kopi.": "gw suka kopi",
            "Lupain yang tadi tentang gw suka kopi.": "gw suka kopi",
            "Hapus dari ingatanmu, gw suka kopi.": "gw suka kopi",
            "Jangan ingat kalau gw suka kopi lagi.": "gw suka kopi",
            "Udah jangan inget soal gw suka kopi.": "gw suka kopi",
            "Forget that I like coffee.": "I like coffee",
            "Forget about me liking coffee.": "me liking coffee",
        }
        for text, payload in cases.items():
            with self.subTest(text=text):
                result = parse_memory_command(text)
                self.assertEqual(result.action, "forget")
                self.assertEqual(result.payload, payload)

    def test_forget_false_positive_matrix(self):
        for text in FORGET_NEGATIVES:
            with self.subTest(text=text):
                self.assertNotEqual(parse_memory_command(text).action, "forget")

    def test_normalization_and_command_separators(self):
        cases = (
            "  INGAT YAH   gw suka kopi  ",
            "Ingat ya: gw suka kopi.",
            "Ingat ya... gw suka kopi.",
            "Ingat ya--- gw suka kopi.",
            "Ingat ya!!! gw suka kopi.",
            "Ingat ya 😊, gw suka kopi.",
        )
        for text in cases:
            with self.subTest(text=text):
                result = parse_memory_command(text)
                self.assertEqual(result.action, "remember")
                self.assertEqual(result.payload.lower(), "gw suka kopi")

    def test_live_ingat_yah_regression(self):
        for text in (
            "ok, ingat yah, gw juga suka film tentang detektif",
            "ok, ingat ya, gw juga suka film tentang detektif",
            "oh iya, inget yah, gw juga suka film tentang detektif",
        ):
            with self.subTest(text=text):
                result = parse_memory_command(text)
                self.assertEqual(result.action, "remember")
                self.assertEqual(result.payload, "gw juga suka film tentang detektif")

    def test_empty_and_useless_payloads_are_rejected(self):
        for text in (
            "Ingat ya.",
            "Inget.",
            "Catat ini.",
            "Tolong ingat.",
            "Jangan lupa.",
            "Remember.",
            "Ingat ya!!!",
        ):
            with self.subTest(text=text):
                self.assertEqual(parse_memory_command(text).action, "none")

    def test_forget_has_priority_over_remember(self):
        result = parse_memory_command("Jangan ingat lagi kalau gw suka kopi.")
        self.assertEqual(result.action, "forget")
        self.assertEqual(result.payload, "gw suka kopi")


class CharacterMemoryCommandIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.old_cwd = os.getcwd()
        self.temporary_directory = tempfile.TemporaryDirectory()
        os.chdir(self.temporary_directory.name)
        self.conf_uid = "mili-memory-parser-v2"

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temporary_directory.cleanup()

    def make_agent(self):
        agent = object.__new__(BasicMemoryAgent)
        agent._character_conf_uid = self.conf_uid
        agent._character_state = load_character_state(self.conf_uid)
        return agent

    def test_duplicate_commands_reuse_existing_storage_dedup(self):
        agent = self.make_agent()
        self.assertTrue(
            agent._observe_character_memory_request("Ingat ya, gw suka kopi.")
        )
        self.assertTrue(
            agent._observe_character_memory_request("Inget yah, gw suka kopi.")
        )
        self.assertEqual(len(agent.list_character_memories()), 1)

    def test_remember_then_forget_uses_existing_character_memory_api(self):
        agent = self.make_agent()
        self.assertTrue(
            agent._observe_character_memory_request("Catet ya, gw suka kopi.")
        )
        self.assertEqual(agent.list_character_memories()[0]["text"], "gw suka kopi")
        self.assertTrue(
            agent._observe_character_memory_request("Lupain soal gw suka kopi.")
        )
        self.assertEqual(agent.list_character_memories(), [])

    def test_forget_reports_real_removal_not_phantom_success(self):
        """A forget that removes rows returns True; one that matches nothing
        returns False instead of a phantom success."""
        agent = self.make_agent()
        self.assertTrue(
            agent._observe_character_memory_request("Catet ya, gw suka kopi.")
        )
        # a real removal
        self.assertTrue(agent.remove_character_memory("gw suka kopi"))
        self.assertEqual(agent.list_character_memories(), [])
        # nothing left to remove: must not claim a removal happened
        self.assertFalse(agent.remove_character_memory("gw suka kopi"))
        # matching nothing at all must also report honestly
        self.assertFalse(agent.remove_character_memory("hal yang tidak ada"))

    def test_forget_returns_false_when_nothing_is_stored(self):
        agent = self.make_agent()
        self.assertFalse(agent.remove_character_memory("apapun"))

    def test_ui_exact_delete_keeps_substring_sibling(self):
        """Regression for the 2026-10-11 incident: the store held "nantii"
        and "nantii [User has also provided images"; deleting "nantii" via
        the UI removed BOTH rows (removed=2 in production logs) because the
        overlap rule treats one as a substring of the other.

        The UI sends the exact stored text, so exact mode must delete only
        that row, and the survivor must still be there after a reload.
        """
        from src.open_llm_vtuber.character_state import load_character_state

        agent = self.make_agent()
        self.assertTrue(agent.add_character_memory("nantii"))
        self.assertTrue(
            agent.add_character_memory("nantii [User has also provided images")
        )
        self.assertEqual(len(agent.list_character_memories()), 2)

        self.assertTrue(agent.remove_character_memory("nantii", exact=True))

        remaining = [row["text"] for row in agent.list_character_memories()]
        self.assertEqual(remaining, ["nantii [User has also provided images"])
        reloaded = load_character_state(self.conf_uid)
        self.assertEqual(
            [row["text"] for row in reloaded.memories],
            ["nantii [User has also provided images"],
        )

    def test_ui_exact_delete_of_unknown_text_removes_nothing(self):
        agent = self.make_agent()
        self.assertTrue(agent.add_character_memory("gw suka kopi"))
        self.assertFalse(agent.remove_character_memory("teh manis", exact=True))
        self.assertEqual(
            [row["text"] for row in agent.list_character_memories()],
            ["gw suka kopi"],
        )

    def test_empty_delete_is_refused_and_changes_nothing(self):
        """An empty identifier must never reach the store, run a clear-all,
        or wipe rows: it reports False and the file is untouched."""
        import os

        agent = self.make_agent()
        self.assertTrue(agent.add_character_memory("gw suka kopi"))
        path = os.path.join("character_state", f"{self.conf_uid}.json")
        with open(path, encoding="utf-8") as handle:
            before = handle.read()
        self.assertFalse(agent.remove_character_memory(""))
        self.assertFalse(agent.remove_character_memory("   "))
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), before)
        self.assertEqual(
            [row["text"] for row in agent.list_character_memories()],
            ["gw suka kopi"],
        )

    def test_chat_overlap_mode_still_finds_paraphrased_facts(self):
        """The spoken forget-command path keeps overlap matching: the exact
        mode above is only for the UI, which sends the stored text verbatim.
        """
        agent = self.make_agent()
        self.assertTrue(agent.add_character_memory("gw suka kopi"))
        self.assertTrue(agent.remove_character_memory("soal gw suka kopi"))
        self.assertEqual(agent.list_character_memories(), [])

    def test_memory_survives_new_chat_and_is_visible_through_api(self):
        create_new_history(self.conf_uid)
        agent = self.make_agent()
        agent._observe_character_memory_request("Ingat yah, gw suka kopi.")

        create_new_history(self.conf_uid)
        recreated_agent = self.make_agent()
        memories = recreated_agent.list_character_memories()
        self.assertEqual([item["text"] for item in memories], ["gw suka kopi"])
        self.assertTrue(memories[0]["explicit"])

    def test_parser_does_not_modify_relationship_or_history(self):
        history_uid = create_new_history(self.conf_uid)
        store_message(self.conf_uid, history_uid, "human", "pesan lama")
        transcript_before = get_history(self.conf_uid, history_uid)
        set_character_relationship(
            self.conf_uid,
            "dating",
            trigger="synthetic_test_event",
        )

        agent = self.make_agent()
        agent._observe_character_memory_request("Inget yah, gw suka kopi.")

        self.assertEqual(
            load_character_state(self.conf_uid).relationship_status, "dating"
        )
        self.assertEqual(get_history(self.conf_uid, history_uid), transcript_before)

    def test_parser_makes_no_llm_call(self):
        class FailIfCalledLLM:
            calls = 0

            async def chat_completion(self, *args, **kwargs):
                self.calls += 1
                raise AssertionError("memory parser must not call the LLM")

        llm = FailIfCalledLLM()
        agent = self.make_agent()
        agent._llm = llm
        self.assertTrue(
            agent._observe_character_memory_request("Ingat yah, gw suka kopi.")
        )
        self.assertEqual(llm.calls, 0)


class DeleteHandlerSafetyTest(unittest.TestCase):
    """The WS delete handler must remove exactly the requested row.

    Uses the real agent against an isolated store, behind the real handler
    object (no sockets, no network).
    """

    def setUp(self):
        self.old_cwd = os.getcwd()
        self.temporary_directory = tempfile.TemporaryDirectory()
        os.chdir(self.temporary_directory.name)
        self.conf_uid = "mili-memory-delete-handler"

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temporary_directory.cleanup()

    def _agent(self):
        agent = object.__new__(BasicMemoryAgent)
        agent._character_conf_uid = self.conf_uid
        agent._character_state = load_character_state(self.conf_uid)
        return agent

    def _handler(self, agent):
        from types import SimpleNamespace

        from src.open_llm_vtuber.websocket_handler import WebSocketHandler

        handler = WebSocketHandler.__new__(WebSocketHandler)
        handler._proactive_timer_tasks = {}
        handler._proactive_maintenance = set()
        handler.client_connections = {}
        handler.client_contexts = {
            "c1": SimpleNamespace(agent_engine=agent, history_uid=None)
        }
        return handler

    def _run_delete(self, handler, text):
        import asyncio
        import json as _json

        sent = []

        class _FakeWS:
            async def send_text(self, payload):
                sent.append(_json.loads(payload))

        asyncio.run(
            handler._handle_delete_character_memory(
                _FakeWS(), "c1", {"type": "delete-character-memory", "text": text}
            )
        )
        return sent

    def _texts(self, agent):
        return [row["text"] for row in agent.list_character_memories()]

    def test_handler_deletes_exactly_the_requested_row(self):
        from src.open_llm_vtuber.character_state import load_character_state

        agent = self._agent()
        self.assertTrue(agent.add_character_memory("nantii"))
        self.assertTrue(
            agent.add_character_memory("nantii [User has also provided images")
        )
        handler = self._handler(agent)

        replies = self._run_delete(handler, "nantii")

        self.assertEqual(len(replies), 1)
        self.assertTrue(replies[0]["success"])
        self.assertEqual(self._texts(agent), ["nantii [User has also provided images"])
        reloaded = load_character_state(self.conf_uid)
        self.assertEqual(
            [row["text"] for row in reloaded.memories],
            ["nantii [User has also provided images"],
        )

    def test_handler_empty_text_is_refused_without_touching_the_store(self):
        import os as _os
        from unittest import mock as _mock

        agent = self._agent()
        self.assertTrue(agent.add_character_memory("gw suka kopi"))
        handler = self._handler(agent)
        path = _os.path.join("character_state", f"{self.conf_uid}.json")
        with open(path, encoding="utf-8") as handle:
            before = handle.read()

        with _mock.patch.object(
            agent,
            "reset_character_memory",
            side_effect=AssertionError("delete must never reset"),
        ):
            replies = self._run_delete(handler, "")
            empty_replies = self._run_delete(handler, "   ")

        self.assertFalse(replies[0]["success"])
        self.assertFalse(empty_replies[0]["success"])
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(handle.read(), before)
        self.assertEqual(self._texts(agent), ["gw suka kopi"])


class ListFreshnessTest(unittest.TestCase):
    """The Memory Manager list must reflect the authoritative file.

    Regression for the "stored but not shown" bug: `list_character_memories`
    served the agent's in-memory snapshot (loaded once per history resume),
    so facts stored afterwards -- by a sibling connection or any path that
    does not rebind this agent -- were missing from the UI while Mili
    already recalled them.
    """

    def setUp(self):
        self.old_cwd = os.getcwd()
        self.temporary_directory = tempfile.TemporaryDirectory()
        os.chdir(self.temporary_directory.name)
        self.conf_uid = "mili-memory-list-fresh"

    def tearDown(self):
        os.chdir(self.old_cwd)
        self.temporary_directory.cleanup()

    def _agent(self):
        agent = object.__new__(BasicMemoryAgent)
        agent._character_conf_uid = self.conf_uid
        agent._character_state = load_character_state(self.conf_uid)
        return agent

    def test_list_sees_facts_stored_behind_the_snapshot(self):
        from src.open_llm_vtuber.character_state import add_character_memory

        agent = self._agent()
        self.assertEqual(agent.list_character_memories(), [])
        # another session writes while this agent holds its old snapshot
        self.assertIsNotNone(
            add_character_memory(self.conf_uid, "gw suka kopi", explicit=True)
        )
        self.assertIsNotNone(
            add_character_memory(self.conf_uid, "user ulang tahun Juni", explicit=True)
        )
        listed = [row["text"] for row in agent.list_character_memories()]
        self.assertEqual(listed, ["gw suka kopi", "user ulang tahun Juni"])

    def test_list_stays_consistent_for_a_new_session(self):
        """A fresh agent (reconnect / new history resume) sees the same rows
        the UI fetch serves: no filter may silently drop a stored fact."""
        from src.open_llm_vtuber.character_state import (
            load_character_state,
        )

        first = self._agent()
        self.assertTrue(first.add_character_memory("gw suka kopi"))
        self.assertTrue(first.add_character_memory("auto fact", explicit=False))

        second = self._agent()
        first_listed = [row["text"] for row in first.list_character_memories()]
        second_listed = [row["text"] for row in second.list_character_memories()]
        on_disk = [row["text"] for row in load_character_state(self.conf_uid).memories]
        self.assertEqual(first_listed, ["gw suka kopi", "auto fact"])
        self.assertEqual(second_listed, first_listed)
        self.assertEqual(on_disk, first_listed)

    def test_list_falls_back_to_snapshot_without_a_character(self):
        agent = self._agent()
        agent._character_conf_uid = ""
        agent._character_state = load_character_state(self.conf_uid)
        # no conf_uid: cannot re-read, must serve the snapshot, never crash
        self.assertEqual(agent.list_character_memories(), [])


if __name__ == "__main__":
    unittest.main()
