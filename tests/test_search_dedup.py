"""Duplicate-search guard — deterministic tests (fake MCP counters).

Proves with real MCP call counts that a duplicate same-turn search
executes MCP exactly once, while different queries, next turns,
non-search tools, and Phase 1 retries are unaffected. No network.
"""

import unittest

from src.open_llm_vtuber.mcpp.tool_executor import ToolExecutor
from src.open_llm_vtuber.mcpp.tool_manager import ToolManager
from src.open_llm_vtuber.mcpp.types import (
    ToolCallFunctionObject,
    ToolCallObject,
)

SUCCESS_TEXT = (
    "Found 10 search results:\n\n1. MiMo\n   URL: https://x/\n   Summary: s\n"
)
EMPTY_TEXT = (
    "No results were found for your search query. This could be due to "
    "DuckDuckGo's bot detection or the query returned no matches. Please "
    "try rephrasing your search or try again in a few minutes."
)


def result_dict(text):
    return {"metadata": {}, "content_items": [{"type": "text", "text": text}]}


class FakeMCPClient:
    def __init__(self, script):
        self.script = list(script)
        self.search_calls = 0
        self.fetch_calls = 0

    async def call_tool(self, server_name, tool_name, tool_args):
        if tool_name == "search":
            self.search_calls += 1
        if tool_name == "fetch_content":
            self.fetch_calls += 1
        action = self.script.pop(0)
        if isinstance(action, Exception):
            raise action
        return action


def make_executor(script):
    from src.open_llm_vtuber.mcpp.types import FormattedTool

    client = FakeMCPClient(script)
    manager = ToolManager(
        initial_tools_dict={
            "search": FormattedTool(input_schema={}, related_server="ddg-search"),
            "fetch_content": FormattedTool(
                input_schema={}, related_server="ddg-search"
            ),
        }
    )
    return ToolExecutor(client, manager), client


def search_call(query, call_id="call-1"):
    return ToolCallObject(
        id=call_id,
        function=ToolCallFunctionObject(
            name="search", arguments=f'{{"query": "{query}"}}'
        ),
    )


async def collect(executor, calls):
    seen = []
    async for update in executor.execute_tools(calls, "OpenAI"):
        seen.append(update)
    return seen


class DedupGuardTest(unittest.IsolatedAsyncioTestCase):
    async def test_a_identical_search_one_mcp_call(self):
        executor, client = make_executor([result_dict(SUCCESS_TEXT)])
        # Router turn: direct run_single_tool (Phase 1 path, unguarded).
        is_error, text, _, _ = await executor.run_single_tool(
            "search", "router-1", {"query": "MiMo-V2.6-Flash"}
        )
        self.assertFalse(is_error)
        executor.note_router_search("MiMo-V2.6-Flash", text)
        self.assertEqual(client.search_calls, 1)
        # Model duplicate in the same turn.
        seen = await collect(executor, [search_call("MiMo-V2.6-Flash")])
        self.assertEqual(client.search_calls, 1)
        statuses = [u for u in seen if u.get("type") == "tool_call_status"]
        self.assertEqual(statuses[-1]["status"], "completed")
        self.assertIn("Found 10 search results", statuses[-1]["content"])
        final = seen[-1]
        self.assertEqual(final["type"], "final_tool_results")
        self.assertIn("Found 10 search results", final["results"][0]["content"])

    async def test_b_case_whitespace_normalized(self):
        executor, client = make_executor([result_dict(SUCCESS_TEXT)])
        await executor.run_single_tool(
            "search", "router-1", {"query": "MiMo-V2.6-Flash"}
        )
        executor.note_router_search("MiMo-V2.6-Flash", "cached")
        await collect(executor, [search_call("  mimo-v2.6-flash  ")])
        self.assertEqual(client.search_calls, 1)

    async def test_c_different_query_executes(self):
        executor, client = make_executor(
            [result_dict(SUCCESS_TEXT), result_dict(SUCCESS_TEXT)]
        )
        await executor.run_single_tool(
            "search", "router-1", {"query": "MiMo-V2.6-Flash"}
        )
        executor.note_router_search("MiMo-V2.6-Flash", "cached")
        await collect(executor, [search_call("Claude Opus 5.5")])
        self.assertEqual(client.search_calls, 2)

    async def test_d_search_then_fetch(self):
        executor, client = make_executor(
            [result_dict(SUCCESS_TEXT), result_dict("# readme")]
        )
        await executor.run_single_tool(
            "search", "router-1", {"query": "MiMo-V2.6-Flash"}
        )
        executor.note_router_search("MiMo-V2.6-Flash", "cached")
        fetch = ToolCallObject(
            id="call-f",
            function=ToolCallFunctionObject(
                name="fetch_content",
                arguments='{"url": "https://github.com/x/y"}',
            ),
        )
        await collect(executor, [fetch])
        self.assertEqual(client.search_calls, 1)
        self.assertEqual(client.fetch_calls, 1)

    async def test_e_next_turn_executes_again(self):
        executor, client = make_executor(
            [result_dict(SUCCESS_TEXT), result_dict(SUCCESS_TEXT)]
        )
        # Turn 1.
        await executor.run_single_tool(
            "search", "router-1", {"query": "MiMo-V2.6-Flash"}
        )
        executor.note_router_search("MiMo-V2.6-Flash", "cached")
        executor.clear_router_search()
        # Turn 2, same query via normal tool path.
        await collect(executor, [search_call("MiMo-V2.6-Flash")])
        self.assertEqual(client.search_calls, 2)

    async def test_f_phase1_retry_survives(self):
        executor, client = make_executor(
            [result_dict(EMPTY_TEXT), result_dict(SUCCESS_TEXT)]
        )
        # Router search with empty first attempt: Phase 1 retries
        # internally (same normalized query on retry is fine — the retry
        # path never consults the turn guard).
        is_error, text, _, _ = await executor.run_single_tool(
            "search", "router-1", {"query": '"quoted thing"'}
        )
        self.assertFalse(is_error)
        self.assertIn("Search results found:", text)
        self.assertEqual(client.search_calls, 2)

    async def test_f2_no_retry_bypass_via_guard(self):
        # After a successful router search, a model duplicate must NOT
        # trigger any new MCP call (hence no Phase 1 retry either).
        executor, client = make_executor(
            [result_dict(SUCCESS_TEXT), result_dict(EMPTY_TEXT)]
        )
        await executor.run_single_tool(
            "search", "router-1", {"query": "MiMo-V2.6-Flash"}
        )
        executor.note_router_search("MiMo-V2.6-Flash", "cached-prior")
        await collect(executor, [search_call("MiMo-V2.6-Flash")])
        self.assertEqual(client.search_calls, 1)

    def test_g_cleanup_idempotent(self):
        executor, _ = make_executor([])
        executor.note_router_search("Q", "cached")
        self.assertEqual(
            executor._router_cached_result("search", {"query": "q"}), "cached"
        )
        executor.clear_router_search()
        self.assertIsNone(executor._router_cached_result("search", {"query": "q"}))
        # Clearing twice / clearing empty never raises.
        executor.clear_router_search()

    def test_h_guard_ignores_non_search_and_bad_input(self):
        executor, _ = make_executor([])
        executor.note_router_search("Q", "cached")
        self.assertIsNone(executor._router_cached_result("fetch_content", {"url": "Q"}))
        self.assertIsNone(executor._router_cached_result("search", None))
        self.assertIsNone(executor._router_cached_result("search", {"nq": 1}))
        self.assertIsNone(executor._router_cached_result("search", {"query": ""}))
        self.assertIsNone(executor._router_cached_result("search", {"query": "other"}))


if __name__ == "__main__":
    unittest.main()
