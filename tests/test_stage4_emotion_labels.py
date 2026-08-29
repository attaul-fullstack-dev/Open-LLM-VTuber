import unittest
import os
import tempfile

from src.open_llm_vtuber.live2d_model import Live2dModel
from src.open_llm_vtuber.agent.output_types import Actions


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


if __name__ == "__main__":
    unittest.main()