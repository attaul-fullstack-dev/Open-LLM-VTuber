import unittest
from pathlib import Path

import yaml

from open_llm_vtuber.utils.tts_preprocessor import remove_special_characters


ROOT = Path(__file__).resolve().parents[1]


class MiliEmojiPersonaContractTests(unittest.TestCase):
    def _persona(self, path: Path) -> str:
        config = yaml.safe_load(path.read_text(encoding="utf-8"))
        return str(config["character_config"]["persona_prompt"])

    def test_preset_allows_optional_non_spam_emoji(self) -> None:
        persona = self._persona(ROOT / "characters" / "id_mili.yaml")
        lowered = persona.lower()
        self.assertIn("sesekali memakai emoji", lowered)
        self.assertIn("biasanya gunakan 0--1 emoji", lowered)
        self.assertIn("sebagian besar pesan", lowered)
        self.assertIn("jangan menempelkan emoji secara mekanis", lowered)
        self.assertIn("jangan memakai rangkaian emoji", lowered)

    def test_active_persona_matches_emoji_contract(self) -> None:
        persona = self._persona(ROOT / "conf.yaml")
        self.assertIn("sesekali memakai emoji", persona.lower())
        self.assertIn("Marker ekspresi Live2D", persona)

    def test_live2d_marker_contract_is_preserved(self) -> None:
        persona = self._persona(ROOT / "characters" / "id_mili.yaml")
        self.assertIn("Marker ekspresi Live2D", persona)
        self.assertIn("metadata teknis", persona)

    def test_tts_filter_removes_emoji_but_keeps_spoken_text(self) -> None:
        self.assertEqual(
            remove_special_characters("Ih, apaan sih 😒"),
            "Ih, apaan sih ",
        )


if __name__ == "__main__":
    unittest.main()
