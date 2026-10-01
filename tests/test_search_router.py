"""Phase 3 deterministic search routing — deterministic unit tests.

Covers: explicit search (A), freshness/current info (B), normal chat
(C), ambiguous personal search (D), explicit web reference (E),
boundaries (F), false-positive/negative regression pairs, orchestration
(router -> existing executor -> Phase 1 formats -> state), and per-turn
state mapping. No network, no LLM.
"""

import unittest

from src.open_llm_vtuber.mcpp.search_router import (
    NORMAL_CHAT,
    SEARCH_EXECUTED_NO_RESULTS,
    SEARCH_EXECUTED_SUCCESS,
    SEARCH_EXECUTION_ERROR,
    SEARCH_NOT_REQUESTED,
    SEARCH_REQUIRED,
    SEARCH_REQUIRED_NOT_EXECUTED,
    UNCERTAIN,
    build_router_search_block,
    classify_search_intent,
    execute_router_search,
    is_search_required,
    normalize_search_query,
    state_from_search_text,
)


class ExplicitSearchTest(unittest.TestCase):
    def test_a_explicit(self):
        for text in [
            "Cari informasi tentang MiMo-V2.6-Flash",
            "Cari game The NOexistenceN of you AND me",
            "Search Claude Opus 5.5",
            "Carikan informasi tentang OpenAI",
            "cari mimo v2.6 flash",
            "SEARCH: apa itu Rust?",
            "Find information about black holes",
            "Cek informasi tentang harga emas",
        ]:
            with self.subTest(text=text):
                self.assertEqual(classify_search_intent(text), SEARCH_REQUIRED)
                self.assertTrue(is_search_required(text))

    def test_a_numbered_list_prefix(self):
        # Phase 4: browser inputs arrive as numbered list items.
        for text in [
            "1. Cari game The NOexistenceN of you AND me",
            "2. Cari MiMo-V2.6-Flash",
            "1) Cari MiMo-V2.6-Flash",
            "- Cari MiMo-V2.6-Flash",
            "• Cari MiMo-V2.6-Flash",
            '"Cari MiMo-V2.6-Flash"',
            "Cari MiMo-V2.6-Flash",
        ]:
            with self.subTest(text=text):
                self.assertEqual(classify_search_intent(text), SEARCH_REQUIRED)
                self.assertTrue(is_search_required(text))


class FreshnessTest(unittest.TestCase):
    def test_b_current_info(self):
        for text in [
            "Apa versi terbaru React?",
            "Berita terbaru tentang OpenAI",
            "Harga emas sekarang",
            "Siapa CEO X saat ini?",
            "What is the latest news about AI?",
            "Berapa harga iPhone sekarang?",
            "Kapan versi terbaru Python rilis?",
        ]:
            with self.subTest(text=text):
                self.assertTrue(is_search_required(text), text)


class NormalChatTest(unittest.TestCase):
    def test_c_normal(self):
        for text in [
            "Aku capek hari ini",
            "Aku tadi main game",
            "Menurutmu aku makan apa?",
            "Halo, apa kabar?",
            "Aku lagi capek.",
            "Menurut kamu aku cocok pakai baju apa?",
        ]:
            with self.subTest(text=text):
                self.assertFalse(is_search_required(text), text)
                self.assertIn(classify_search_intent(text), (NORMAL_CHAT, UNCERTAIN))


class AmbiguousTest(unittest.TestCase):
    def test_d_personal_physical_stays_chat(self):
        for text in [
            "Aku cari charger tadi tapi nggak ketemu",
            "Aku 2 kali cari charger tadi tapi nggak ketemu",
            "Tadi aku cari charger",
            "Aku lagi cari dompet",
            "Tadi aku cari file itu",
            "Aku mau cari makan",
            "Aku sedang mencari ide",
            "Cari apa ya enaknya buat makan?",
            "aku cari game tadi",
        ]:
            with self.subTest(text=text):
                self.assertFalse(is_search_required(text), text)


class ExplicitWebTest(unittest.TestCase):
    def test_e_web_reference(self):
        for text in [
            "Cek di internet apakah X sudah dirilis",
            "Kasih aku sumber tentang X",
            "Cek apakah kabar itu benar",
            "Lihat apakah MiMo sudah dirilis",
            "Ada berita tentang gempa?",
            "Kasih link sumbernya dong",
        ]:
            with self.subTest(text=text):
                self.assertTrue(is_search_required(text), text)


class BoundaryTest(unittest.TestCase):
    def test_f_boundaries(self):
        self.assertFalse(is_search_required(""))
        self.assertFalse(is_search_required("   "))
        self.assertFalse(is_search_required(None))
        self.assertTrue(is_search_required("  CARI INFORMASI TENTANG X!!  "))
        self.assertTrue(is_search_required("cari info tentang x."))
        # Very short / very long inputs never crash and stay decided.
        self.assertFalse(is_search_required("ok"))
        long_chat = "aku capek sekali hari ini. " * 200
        self.assertFalse(is_search_required(long_chat))
        long_search = "Cari informasi tentang " + "x" * 2000
        self.assertTrue(is_search_required(long_search))


class FalsePositiveNegativeTest(unittest.TestCase):
    def test_charger_pair(self):
        self.assertFalse(is_search_required("aku cari charger tadi"))
        self.assertTrue(is_search_required("cari charger USB-C terbaik sekarang"))

    def test_game_pair(self):
        self.assertFalse(is_search_required("aku cari game tadi"))
        self.assertTrue(is_search_required("cari informasi tentang game X"))

    def test_food_pair(self):
        self.assertFalse(is_search_required("Aku mau cari makan"))
        self.assertTrue(is_search_required("cari info harga makan siang terbaru"))

    def test_opinion_with_freshness_stays_router_honest(self):
        # Opinion without web markers stays chat even with "hari ini".
        self.assertFalse(is_search_required("Aku capek hari ini"))
        # ...but an explicit source request still routes.
        self.assertTrue(
            is_search_required("Menurutmu berita terbaru apa yang penting?")
        )


class NormalizeTest(unittest.TestCase):
    def test_query_preserved(self):
        self.assertEqual(
            normalize_search_query("  Cari   informasi tentang X?  "),
            "Cari informasi tentang X?",
        )
        self.assertEqual(normalize_search_query(""), "")


class StateMappingTest(unittest.TestCase):
    def test_states(self):
        self.assertEqual(
            state_from_search_text(False, "Search results found:\nFound 3..."),
            SEARCH_EXECUTED_SUCCESS,
        )
        self.assertEqual(
            state_from_search_text(
                False, "Search completed successfully but returned no results..."
            ),
            SEARCH_EXECUTED_NO_RESULTS,
        )
        self.assertEqual(
            state_from_search_text(True, "whatever"),
            SEARCH_EXECUTION_ERROR,
        )
        self.assertEqual(
            state_from_search_text(False, "Search execution failed: X"),
            SEARCH_EXECUTION_ERROR,
        )
        # Unknown shape fails closed, never claims success.
        self.assertEqual(
            state_from_search_text(False, "???"),
            SEARCH_EXECUTION_ERROR,
        )

    def test_block_format(self):
        block = build_router_search_block("q", SEARCH_EXECUTED_SUCCESS, 10, "R")
        self.assertIn("[WEB SEARCH EXECUTED BY APPLICATION]", block)
        self.assertIn("query: q", block)
        self.assertIn("status: " + SEARCH_EXECUTED_SUCCESS, block)
        self.assertIn("result_count: 10", block)
        self.assertIn("Do not call the search tool again", block)

    def test_constants_sane(self):
        self.assertEqual(SEARCH_NOT_REQUESTED, "SEARCH_NOT_REQUESTED")
        self.assertEqual(SEARCH_REQUIRED_NOT_EXECUTED, "SEARCH_REQUIRED_NOT_EXECUTED")
        self.assertEqual(UNCERTAIN, "UNCERTAIN")


class FakeExecutor:
    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def run_single_tool(self, name, tool_id, args):
        self.calls.append((name, dict(args)))
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def ok_result(text):
    return (False, text, {}, [{"type": "text", "text": text}])


class OrchestrationTest(unittest.IsolatedAsyncioTestCase):
    async def test_router_executes_exactly_once_with_phase1_text(self):
        executed = FakeExecutor(
            [ok_result("Search results found:\nFound 10 search results:\n...")]
        )
        result = await execute_router_search(executed, "Cari MiMo-V2.6-Flash")
        self.assertEqual(result.state, SEARCH_EXECUTED_SUCCESS)
        self.assertEqual(result.result_count, 10)
        self.assertEqual(len(executed.calls), 1)
        name, args = executed.calls[0]
        self.assertEqual(name, "search")
        # Original user request preserved as the query.
        self.assertEqual(args["query"], "Cari MiMo-V2.6-Flash")
        self.assertIn("[WEB SEARCH EXECUTED BY APPLICATION]", result.block_text)
        self.assertIn("Found 10 search results", result.block_text)

    async def test_no_results_state(self):
        executed = FakeExecutor(
            [
                ok_result(
                    "Search completed successfully but returned no results"
                    " for query: x. You may try a differently phrased query."
                )
            ]
        )
        result = await execute_router_search(executed, "Cari X")
        self.assertEqual(result.state, SEARCH_EXECUTED_NO_RESULTS)
        self.assertIn("status: " + SEARCH_EXECUTED_NO_RESULTS, result.block_text)

    async def test_executor_failure_maps_to_error(self):
        executed = FakeExecutor([RuntimeError("down")])
        result = await execute_router_search(executed, "Cari X")
        self.assertEqual(result.state, SEARCH_EXECUTION_ERROR)
        self.assertEqual(result.block_text.count("Search execution failed"), 1)

    async def test_error_text_maps_to_error(self):
        executed = FakeExecutor(
            [ok_result("Search execution failed: RuntimeError. Do not treat...")]
        )
        result = await execute_router_search(executed, "Cari X")
        self.assertEqual(result.state, SEARCH_EXECUTION_ERROR)

    async def test_normal_chat_never_invokes_executor(self):
        # The pipeline only calls execute_router_search when
        # is_search_required() is true; assert the gate directly.
        for text in ["Aku capek hari ini", "aku cari charger tadi"]:
            self.assertFalse(is_search_required(text))
        # And a NORMAL classifier output never yields a block.
        self.assertIn(classify_search_intent("Aku capek"), (NORMAL_CHAT, UNCERTAIN))


if __name__ == "__main__":
    unittest.main()
