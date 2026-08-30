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


if __name__ == "__main__":
    unittest.main()
