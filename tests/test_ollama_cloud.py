import unittest
from unittest.mock import patch

from src.open_llm_vtuber.agent.stateless_llm.ollama_llm import OllamaLLM


class OllamaCloudTests(unittest.TestCase):
    @patch("src.open_llm_vtuber.agent.stateless_llm.ollama_llm.requests.post")
    def test_remote_endpoint_skips_local_preload_and_cleanup(self, post):
        llm = OllamaLLM(
            model="gemma4:31b-cloud",
            base_url="https://ollama.com/v1",
            llm_api_key="test-key",
            temperature=0.8,
            top_p=0.9,
            max_tokens=384,
        )

        self.assertFalse(llm.is_local)
        self.assertEqual(llm.top_p, 0.9)
        self.assertEqual(llm.max_tokens, 384)
        post.assert_not_called()

        llm.cleanup()
        post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
