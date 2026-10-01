"""Phase 1 web-search reliability — deterministic tests (fake MCP, no network).

Covers: structured SUCCESS/NO_RESULTS/SEARCH_ERROR, exactly-one safe
retry on empty results, schema compatibility, non-search tools
untouched, GitHub/fetch flow untouched, injection compatibility.
"""

import unittest

from src.open_llm_vtuber.mcpp.search_reliability import (
    NO_RESULTS,
    SEARCH_ERROR,
    SUCCESS,
    classify_search_text,
    execute_search_with_reliability,
    format_search_text,
    rephrase_search_query,
)
from src.open_llm_vtuber.mcpp.tool_executor import ToolExecutor
from src.open_llm_vtuber.mcpp.tool_manager import ToolManager
from src.open_llm_vtuber.mcpp.types import FormattedTool

SUCCESS_TEXT_10 = (
    "Found 10 search results:\n\n1. Introducing Claude Opus 5.5\n"
    "   URL: https://www.anthropic.com/x\n   Summary: s\n"
)
SUCCESS_TEXT_1 = (
    "Found 1 search results:\n\n1. Only\n   URL: https://x.example/\n   Summary: s\n"
)
EMPTY_TEXT = (
    "No results were found for your search query. This could be due to "
    "DuckDuckGo's bot detection or the query returned no matches. Please "
    "try rephrasing your search or try again in a few minutes."
)


def result_dict(text):
    return {"metadata": {}, "content_items": [{"type": "text", "text": text}]}


class FakeMCPClient:
    """Scripted call_tool: list of texts or exceptions, records args."""

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    async def call_tool(self, server_name, tool_name, tool_args):
        self.calls.append(dict(tool_args))
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def make_executor(script, tools=("search", "fetch_content", "get_current_time")):
    client = FakeMCPClient(script)
    manager = ToolManager(
        initial_tools_dict={
            name: FormattedTool(input_schema={}, related_server="ddg-search")
            for name in tools
        }
    )
    return ToolExecutor(client, manager), client


class ClassifyTest(unittest.TestCase):
    def test_success_markers(self):
        self.assertEqual(classify_search_text(SUCCESS_TEXT_10), (SUCCESS, 10))
        self.assertEqual(classify_search_text(SUCCESS_TEXT_1), (SUCCESS, 1))

    def test_empty_marker(self):
        self.assertEqual(classify_search_text(EMPTY_TEXT), (NO_RESULTS, 0))
        self.assertEqual(classify_search_text(""), (NO_RESULTS, 0))
        self.assertEqual(classify_search_text(None), (NO_RESULTS, 0))

    def test_malformed_is_error(self):
        status, count = classify_search_text("??? unstructured blob ???")
        self.assertEqual(status, SEARCH_ERROR)
        self.assertIsNone(count)

    def test_rephrase_strips_quotes(self):
        out = rephrase_search_query('game "the nonexistence" you and me')
        self.assertNotIn('"', out)
        self.assertIn("nonexistence", out)
        self.assertEqual(rephrase_search_query("plain query"), "")
        self.assertEqual(rephrase_search_query(""), "")


class ReliabilityRunnerTest(unittest.IsolatedAsyncioTestCase):
    async def test_1_success_10_no_retry(self):
        async def call(args):
            return result_dict(SUCCESS_TEXT_10)

        calls = []

        async def counting(args):
            calls.append(args)
            return await call(args)

        final, outcome = await execute_search_with_reliability(
            counting, {"query": "Claude Opus 5.5", "max_results": 10}
        )
        self.assertEqual(outcome.status, SUCCESS)
        self.assertEqual(outcome.result_count, 10)
        self.assertEqual(outcome.attempts, 1)
        self.assertEqual(len(calls), 1)
        text = final["content_items"][0]["text"]
        self.assertTrue(text.startswith("Search results found:"))
        self.assertIn("Claude Opus 5.5", text)

    async def test_2_success_1_no_retry(self):
        final, outcome = await execute_search_with_reliability(
            lambda args: _coro(result_dict(SUCCESS_TEXT_1)), {"query": "x"}
        )
        self.assertEqual(outcome.status, SUCCESS)
        self.assertEqual(outcome.result_count, 1)
        self.assertEqual(outcome.attempts, 1)

    async def test_3_empty_then_empty_single_retry(self):
        seen = []

        async def call(args):
            seen.append(args.get("query"))
            return result_dict(EMPTY_TEXT)

        final, outcome = await execute_search_with_reliability(
            call, {"query": 'game "the nonexistence" you and me'}
        )
        self.assertEqual(outcome.status, NO_RESULTS)
        self.assertEqual(outcome.result_count, 0)
        self.assertEqual(outcome.attempts, 2)
        self.assertEqual(len(seen), 2)
        # Retry used the rephrased (dequoted) query.
        self.assertNotIn('"', seen[1])
        text = final["content_items"][0]["text"]
        self.assertIn("returned no results", text)
        self.assertNotIn("No results were found", text)

    async def test_4_empty_then_success(self):
        script = [result_dict(EMPTY_TEXT), result_dict(SUCCESS_TEXT_10)]

        async def call(args):
            return script.pop(0)

        final, outcome = await execute_search_with_reliability(
            call, {"query": '"quoted thing"'}
        )
        self.assertEqual(outcome.status, SUCCESS)
        self.assertEqual(outcome.result_count, 10)
        self.assertEqual(outcome.attempts, 2)

    async def test_5_mcp_exception_is_search_error_no_retry(self):
        calls = []

        async def call(args):
            calls.append(args)
            raise RuntimeError("connection reset by peer")

        final, outcome = await execute_search_with_reliability(
            call, {"query": "anything"}
        )
        self.assertEqual(outcome.status, SEARCH_ERROR)
        self.assertIsNone(outcome.result_count)
        self.assertEqual(len(calls), 1)
        text = final["content_items"][0]["text"]
        self.assertIn("Search execution failed:", text)
        self.assertNotIn("No results", text)
        self.assertNotIn("Traceback", text)

    async def test_6_malformed_response_is_search_error(self):
        final, outcome = await execute_search_with_reliability(
            lambda args: _coro(result_dict("??? unstructured blob ???")),
            {"query": "x"},
        )
        self.assertEqual(outcome.status, SEARCH_ERROR)

    async def test_6b_tool_reported_error_is_search_error(self):
        bad = {"metadata": {}, "content_items": [{"type": "error", "text": "boom"}]}
        final, outcome = await execute_search_with_reliability(
            lambda args: _coro(bad), {"query": "x"}
        )
        self.assertEqual(outcome.status, SEARCH_ERROR)
        self.assertEqual(outcome.attempts, 1)


async def _coro(value):
    return value


class ExecutorContractTest(unittest.IsolatedAsyncioTestCase):
    async def test_7_schema_compatible_search_call(self):
        executor, client = make_executor([result_dict(SUCCESS_TEXT_10)])
        is_error, text, metadata, items = await executor.run_single_tool(
            "search",
            "call-1",
            {"query": "Claude Opus 5.5", "max_results": 10, "region": ""},
        )
        self.assertFalse(is_error)
        self.assertIn("Search results found:", text)
        # Same input shape the LLM always sent still works.
        self.assertEqual(client.calls[0]["query"], "Claude Opus 5.5")
        self.assertEqual(client.calls[0]["max_results"], 10)

    async def test_8_non_search_tool_untouched(self):
        executor, client = make_executor([result_dict("12:00 Asia/Shanghai")])
        is_error, text, _, _ = await executor.run_single_tool(
            "get_current_time", "call-2", {"timezone": "Asia/Shanghai"}
        )
        self.assertFalse(is_error)
        self.assertEqual(text, "12:00 Asia/Shanghai")
        self.assertEqual(len(client.calls), 1)

    async def test_9_github_fetch_flow_unchanged(self):
        body = "# repo readme\n" + "x" * 100
        executor, client = make_executor([result_dict(body)])
        is_error, text, _, _ = await executor.run_single_tool(
            "fetch_content", "call-3", {"url": "https://github.com/x/y"}
        )
        self.assertFalse(is_error)
        self.assertEqual(text, body)
        # No retry machinery on the fetch path.
        self.assertEqual(len(client.calls), 1)

    async def test_10_injection_stays_compatible(self):
        from src.open_llm_vtuber.mcpp.types import (
            ToolCallObject,
            ToolCallFunctionObject,
        )

        executor, _ = make_executor([result_dict(SUCCESS_TEXT_1)])
        call = ToolCallObject(
            id="call-4",
            function=ToolCallFunctionObject(name="search", arguments='{"query": "x"}'),
        )
        seen = []
        async for update in executor.execute_tools([call], "OpenAI"):
            seen.append(update)
        final = seen[-1]
        self.assertEqual(final["type"], "final_tool_results")
        content = final["results"][0]["content"]
        self.assertIn("Search results found:", content)
        statuses = [u["status"] for u in seen if u.get("type") == "tool_call_status"]
        self.assertIn("completed", statuses)


class FormatTest(unittest.TestCase):
    def test_formats_carry_no_internals(self):
        self.assertIn("Search results found:", format_search_text(SUCCESS, "abc"))
        no = format_search_text(NO_RESULTS, EMPTY_TEXT, "q")
        self.assertIn("returned no results", no)
        self.assertIn("q", no)
        err = format_search_text(SEARCH_ERROR, "", "q", "RuntimeError: x")
        self.assertIn("Search execution failed:", err)
        self.assertNotIn("Traceback", err)


if __name__ == "__main__":
    unittest.main()
