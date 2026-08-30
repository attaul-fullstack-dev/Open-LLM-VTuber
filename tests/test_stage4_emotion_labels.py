import unittest
import os
import tempfile

from src.open_llm_vtuber.live2d_model import Live2dModel
from src.open_llm_vtuber.agent.output_types import Actions, SentenceOutput
from src.open_llm_vtuber.agent.transformers import (
    sentence_divider,
    actions_extractor,
    display_processor,
    tts_filter,
)
from src.open_llm_vtuber.config_manager import TTSPreprocessorConfig


def _model_with_map(emotion_map):
    """Build a Live2dModel from a minimal model_dict.json in a temp dir."""
    model_dict = [
        {
            "name": "mao_pro",
            "emotionMap": emotion_map,
        }
    ]
    tmp_dir = tempfile.mkdtemp()
    path = os.path.join(tmp_dir, "model_dict.json")
    import json as _json

    with open(path, "w", encoding="utf-8") as f:
        _json.dump(model_dict, f)
    return Live2dModel("mao_pro", model_dict_path=path)


class ExtractEmotionKeysTests(unittest.TestCase):
    def setUp(self):
        self.model = _model_with_map(
            {
                "neutral": 0,
                "anger": 2,
                "joy": 3,
                "smirk": 3,
                "sadness": 1,
                "surprise": 3,
            }
        )

    def test_extracts_label_keys_in_order(self):
        text = "Aku capek [sadness] banget, [joy] deh."
        self.assertEqual(self.model.extract_emotion_keys(text), ["sadness", "joy"])

    def test_legacy_extract_emotion_still_returns_indices(self):
        text = "[anger] jangan gitu."
        self.assertEqual(self.model.extract_emotion(text), [2])

    def test_no_match_returns_empty(self):
        self.assertEqual(self.model.extract_emotion_keys("no emotion here"), [])

    def test_case_insensitive(self):
        self.assertEqual(self.model.extract_emotion_keys("[JOY]"), ["joy"])


class ExtractHighIntensityEmotionKeysTests(unittest.TestCase):
    """Stage 5 high-intensity labels: anger_strong + embarrassed."""

    def setUp(self):
        self.model = _model_with_map(
            {
                "neutral": 0,
                "anger": 2,
                "anger_strong": 8,
                "embarrassed": 6,
                "joy": 3,
                "smirk": 3,
            }
        )

    def test_extracts_anger_strong_label(self):
        self.assertEqual(
            self.model.extract_emotion_keys("Dasar! [anger_strong] tidak percaya."),
            ["anger_strong"],
        )

    def test_extracts_embarrassed_label(self):
        self.assertEqual(
            self.model.extract_emotion_keys("Eh [embarrassed], kok diomongin gitu."),
            ["embarrassed"],
        )

    def test_ordinary_anger_stays_separate_from_anger_strong(self):
        self.assertEqual(self.model.extract_emotion_keys("[anger] jangan."), ["anger"])
        self.assertEqual(
            self.model.extract_emotion_keys("[anger_strong] jangan."),
            ["anger_strong"],
        )

    def test_tags_are_stripped_from_text(self):
        cleaned = self.model.remove_emotion_keywords("Eh [embarrassed] gimana sih.")
        self.assertNotIn("[embarrassed]", cleaned)
        self.assertNotIn("[anger_strong]", self.model.remove_emotion_keywords("Dasar [anger_strong]!"))

    def test_legacy_index_of_high_intensity_labels(self):
        self.assertEqual(self.model.extract_emotion("[embarrassed]"), [6])
        self.assertEqual(self.model.extract_emotion("[anger_strong]"), [8])


class ActionsEmotionsSerializationTests(unittest.TestCase):
    def test_emotions_field_round_trips(self):
        actions = Actions(expressions=[2], emotions=["anger"])
        as_dict = actions.to_dict()
        self.assertEqual(as_dict["expressions"], [2])
        self.assertEqual(as_dict["emotions"], ["anger"])

    def test_to_dict_omits_none_fields(self):
        actions = Actions()
        self.assertNotIn("emotions", actions.to_dict())
        self.assertNotIn("expressions", actions.to_dict())

    def test_emotions_via_extractor_path(self):
        # Mirror what actions_extractor does: labels come from extract_emotion_keys.
        model = _model_with_map({"neutral": 0, "anger": 2})
        actions = Actions()
        keys = model.extract_emotion_keys("Kamu [anger] ya?")
        if keys:
            actions.emotions = keys
        self.assertEqual(actions.to_dict().get("emotions"), ["anger"])
        self.assertIsNone(actions.expressions)


class EmotionTagCleanupPipelineTests(unittest.IsolatedAsyncioTestCase):
    """The downstream display/TTS/history text must be free of emotion tags."""

    def setUp(self):
        self.model = _model_with_map(
            {
                "neutral": 0,
                "anger": 2,
                "joy": 3,
                "smirk": 3,
                "sadness": 1,
                "surprise": 3,
            }
        )

    async def _run_pipeline(self, *tokens):
        """Run the real decorator chain (sentence_divider -> actions -> display
        -> tts) over a token stream, mirroring the BasicMemoryAgent wiring."""

        @tts_filter(
            TTSPreprocessorConfig(
                remove_special_char=True,
                translator_config={
                    "translate_audio": False,
                    "translate_provider": "deeplx",
                },
            )
        )
        @display_processor()
        @actions_extractor(self.model)
        @sentence_divider(valid_tags=["think"])
        async def source():
            for token in tokens:
                yield token

        outputs = [output async for output in source()]
        sentence_outputs = [
            output for output in outputs if isinstance(output, SentenceOutput)
        ]
        self.assertTrue(sentence_outputs, "pipeline produced no SentenceOutput")
        return sentence_outputs

    async def test_smirk_tag_still_yields_emotion_label(self):
        outputs = await self._run_pipeline("Masa gitu aja harus aku yang [smirk]")
        combined = "".join(o.tts_text for o in outputs)
        self.assertEqual(combined, "Masa gitu aja harus aku yang")
        self.assertTrue(
            any(o.actions.emotions == ["smirk"] for o in outputs),
            "emotion label should still be extracted for Stage 4",
        )

    async def test_visible_display_text_has_no_emotion_tag(self):
        outputs = await self._run_pipeline("Masa gitu aja [smirk]")
        for output in outputs:
            self.assertNotIn("[smirk]", output.display_text.text)
            self.assertNotIn("smirk", output.display_text.text)
        self.assertEqual(outputs[-1].display_text.text.strip(), "Masa gitu aja")

    async def test_tts_text_does_not_read_emotion_tag(self):
        outputs = await self._run_pipeline("Hujan turun [joy], seru deh.")
        for output in outputs:
            self.assertNotIn("[joy]", output.tts_text)
        self.assertNotIn("[joy]", "".join(o.tts_text for o in outputs))

    async def test_contextual_face_still_maps_to_squint_smile(self):
        outputs = await self._run_pipeline("Senang banget [smirk]")
        self.assertTrue(any(o.actions.emotions == ["smirk"] for o in outputs))
        # frontend maps smirk -> squint_smile (asserted in frontend tests); here
        # we only need the label to survive cleanup.
        self.assertEqual(outputs[-1].actions.emotions, ["smirk"])

    async def test_multiple_sentences_all_cleaned(self):
        outputs = await self._run_pipeline(
            "Aku capek [sadness].", "Terus akhirnya ketawa [joy].", "Ya udah."
        )
        all_text = "".join(o.display_text.text for o in outputs)
        self.assertNotIn("[", all_text)
        self.assertNotIn("]", all_text)

    async def test_text_without_emotion_tag_is_unchanged(self):
        outputs = await self._run_pipeline("Hujan itu siklus air.", "Capek ya hari ini.")
        joined = "".join(o.display_text.text for o in outputs)
        self.assertIn("Hujan itu siklus air.", joined)
        self.assertIn("Capek ya hari ini.", joined)

    async def test_marker_at_start_extracted_and_cleaned(self):
        # Stage 4 timing fix: the prompt now asks for the single turn-level
        # marker at the very START of the response, e.g. "[smirk] text...".
        outputs = await self._run_pipeline("[smirk] Masa gitu aja harus aku jelasin.")
        labelled = [o for o in outputs if o.actions.emotions]
        self.assertEqual(len(labelled), 1, "exactly one sentence carries the marker")
        self.assertEqual(labelled[0].actions.emotions, ["smirk"])
        for output in outputs:
            self.assertNotIn("[smirk]", output.display_text.text)
            self.assertNotIn("[smirk]", output.tts_text)
        self.assertNotIn("[", "".join(o.display_text.text for o in outputs))

    async def test_first_sentence_emotion_then_unmarked_sentences(self):
        # With the marker moved to the start, the emotion arrives on the FIRST
        # sentence and later sentences carry none — the frontend latch keeps
        # the face. Prove the label is not stuck on the last sentence anymore.
        outputs = await self._run_pipeline(
            "[joy] Akhirnya berhasil juga.", "Seru banget.", "Besok lanjut lagi."
        )
        labelled = [o for o in outputs if o.actions.emotions]
        self.assertEqual(len(labelled), 1)
        self.assertEqual(labelled[0].actions.emotions, ["joy"])
        self.assertLess(
            outputs.index(labelled[0]),
            len(outputs) - 1,
            "marker must be at the start, not the last sentence",
        )
        self.assertNotIn("[", "".join(o.display_text.text for o in outputs))


class EmotionTagCleanupProactiveTests(unittest.IsolatedAsyncioTestCase):
    """Proactive turns share the same decorated chain, so emotion tags are
    cleaned there too while the label still reaches the avatar."""

    def setUp(self):
        self._old_cwd = os.getcwd()
        self._temp = tempfile.TemporaryDirectory()
        os.chdir(self._temp.name)
        self.conf_uid = "mili-proactive"
        self.model = _model_with_map({"neutral": 0, "joy": 3})
        from src.open_llm_vtuber.chat_history_manager import (
            create_new_history,
            store_message,
        )

        self.history_uid = create_new_history(self.conf_uid)
        store_message(self.conf_uid, self.history_uid, "human", "Aku lagi belajar.")

    def tearDown(self):
        os.chdir(self._old_cwd)
        self._temp.cleanup()

    async def test_proactive_response_cleans_emotion_tag_but_keeps_label(self):
        from src.open_llm_vtuber.agent.agents.basic_memory_agent import (
            BasicMemoryAgent,
        )

        class _TagFakeLLM:
            model = "proactive-test"
            max_tokens = 100

            async def chat_completion(self, messages, system=None, tools=None):
                self.calls = getattr(self, "calls", []) + [messages]
                yield "Eh, kepikiran aja [joy] gitu loh."

        llm = _TagFakeLLM()
        agent = BasicMemoryAgent(
            llm=llm,
            system="persona Mili",
            live2d_model=self.model,
            tts_preprocessor_config=TTSPreprocessorConfig(
                remove_special_char=True,
                translator_config={
                    "translate_audio": False,
                    "translate_provider": "deeplx",
                },
            ),
        )
        agent.set_memory_from_history(self.conf_uid, self.history_uid)

        outputs = [o async for o in agent.chat_proactively()]
        sentence_outputs = [
            o for o in outputs if isinstance(o, SentenceOutput)
        ]
        self.assertTrue(sentence_outputs)
        self.assertTrue(
            any(o.actions.emotions == ["joy"] for o in sentence_outputs),
            "proactive should still carry the emotion label",
        )
        for output in sentence_outputs:
            self.assertNotIn("[joy]", output.display_text.text)
            self.assertNotIn("[joy]", output.tts_text)


class EmotionPromptBiasContractTests(unittest.TestCase):
    """Prompt-content contract: the LLM instruction file must keep explicit
    per-label choice rules so smirk is not a default tsundere habit. This is a
    content-guard against the confirmed bias that smirk got chosen for both a
    clearly sad context and a plain neutral/factual context."""

    PROMPT_PATH = os.path.join(
        "prompts", "utils", "live2d_expression_prompt.txt"
    )

    def setUp(self):
        with open(self.PROMPT_PATH, encoding="utf-8") as f:
            self.content = f.read()

    def test_prompt_keeps_marker_placement_rule(self):
        self.assertIn("at the very START", self.content)
        self.assertIn("exactly one marker", self.content)

    def test_prompt_says_neutral_is_the_default(self):
        # Neutral / no-marker must be the common case for factual replies.
        self.assertIn("[neutral]", self.content)
        self.assertIn("Default to NO marker", self.content)

    def test_prompt_guards_smirk_from_overuse(self):
        # smirk must be reserved for genuinely sly/teasing tone, not casual/sarcasm.
        self.assertIn("[smirk]", self.content)
        self.assertIn("smirk", self.content)
        self.assertIn("teasing", self.content)
        self.assertIn("only when your response is genuinely happy", self.content)  # joy rule

    def test_prompt_keeps_sadness_and_anger_valid(self):
        self.assertIn("[sadness]", self.content)
        self.assertIn("[anger]", self.content)
        self.assertIn("genuinely sad", self.content)
        self.assertIn("genuine irritation", self.content)

    def test_prompt_bases_choice_on_response_tone_not_user_words(self):
        self.assertIn("emotional tone", self.content)
        self.assertIn("not on the user's words", self.content)

    def test_prompt_guards_high_intensity_labels(self):
        # anger_strong reserved for genuinely intense anger; ordinary tsundere
        # irritation must stay [anger]. embarrassed only for real fluster/shy.
        self.assertIn("[anger_strong]", self.content)
        self.assertIn("[embarrassed]", self.content)
        self.assertIn("furious", self.content)
        self.assertIn("angry outburst", self.content)
        self.assertIn("flustered", self.content)
        self.assertIn("merely because the conversation is playful", self.content)

    def test_prompt_anger_scales_appropriately_for_ordinary_cases(self):
        # Ordinary annoyance / tsundere scolding must NOT become anger_strong.
        self.assertIn("[anger]", self.content)
        self.assertIn("mild or moderate anger", self.content)
        self.assertIn("tsundere pouting", self.content)
        self.assertIn("stay [anger]", self.content)

    def test_prompt_anger_strong_reaches_severe_context(self):
        # The threshold must not be so strict that an EXPLICITLY severe context
        # (deliberate destruction / betrayal of something important, no remorse)
        # never selects anger_strong. The contract must name that severe category.
        self.assertIn("deliberate destruction", self.content)
        self.assertIn("betrayal", self.content)
        self.assertIn("serious hurt/harm", self.content)
        self.assertIn("truly furious", self.content)

    def test_prompt_anger_strong_is_not_inferred_from_keyword_or_generic(self):
        # Neither a single angry keyword nor generic negative banter may force
        # anger_strong; it is reserved for genuinely intense, elevated anger.
        self.assertIn("single angry keyword", self.content)
        self.assertIn("generic negative", self.content)
        self.assertIn("intensity", self.content.lower())
        self.assertNotIn("barely", self.content)  # placeholder no-op guard

    def test_prompt_does_not_map_generic_labels_to_strong_states(self):
        # smirk / joy / anger must not be described as if they become the strong
        # states — that would recreate the smirk-overuse problem.
        self.assertIn("ordinary irritation", self.content.lower())
        self.assertIn("prefer no marker or [joy]", self.content)


if __name__ == "__main__":
    unittest.main()