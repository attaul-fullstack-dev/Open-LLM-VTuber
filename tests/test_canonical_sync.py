"""Canonical-final lifecycle signals (live-vs-history sync).

The persisted AI row is authoritative; after persist the backend emits one
canonical-final control event so the live bubble converges to it. These
tests pin the payload contract without running a turn.
"""

import asyncio
import json
import unittest

from src.open_llm_vtuber.conversations.conversation_utils import (
    send_canonical_final,
    send_conversation_start_signals,
)


def run(coro):
    return asyncio.new_event_loop().run_until_complete(coro)


class ChainStartIdentityTest(unittest.TestCase):
    def test_carries_history_and_request_id(self):
        sent = []
        run(
            send_conversation_start_signals(
                sent.append, history_uid="h1", request_id="chat-1"
            )
        )
        self.assertEqual(len(sent), 1)
        payload = json.loads(sent[0])
        self.assertEqual(payload["text"], "conversation-chain-start")
        self.assertEqual(payload["history_uid"], "h1")
        self.assertEqual(payload["request_id"], "chat-1")

    def test_omits_empty_identity(self):
        sent = []
        run(send_conversation_start_signals(sent.append))
        payload = json.loads(sent[0])
        self.assertNotIn("history_uid", payload)
        self.assertNotIn("request_id", payload)

    def test_dead_socket_is_fail_soft(self):
        async def boom(_payload):
            raise ConnectionError("gone")

        run(send_conversation_start_signals(boom, history_uid="h", request_id="r"))

    def test_legacy_positional_call_still_works(self):
        sent = []
        run(send_conversation_start_signals(sent.append, "h9"))
        payload = json.loads(sent[0])
        self.assertEqual(payload["history_uid"], "h9")
        self.assertNotIn("request_id", payload)


class CanonicalFinalTest(unittest.TestCase):
    def test_payload_contract(self):
        sent = []
        run(
            send_canonical_final(
                sent.append,
                history_uid="h1",
                request_id="chat-9",
                text="aku cuma mau kamu.",
            )
        )
        self.assertEqual(len(sent), 1)
        payload = json.loads(sent[0])
        self.assertEqual(payload["type"], "ai-final")
        self.assertEqual(payload["history_uid"], "h1")
        self.assertEqual(payload["request_id"], "chat-9")
        # Verbatim canonical text: never truncated, never rebuilt.
        self.assertEqual(payload["text"], "aku cuma mau kamu.")

    def test_guards_refuse_empty_fields(self):
        sent = []
        run(send_canonical_final(sent.append, history_uid="h", request_id="r", text=""))
        run(send_canonical_final(sent.append, history_uid="", request_id="r", text="x"))
        run(send_canonical_final(sent.append, history_uid="h", request_id="", text="x"))
        self.assertEqual(sent, [])

    def test_dead_socket_is_fail_soft(self):
        async def boom(_payload):
            raise ConnectionError("gone")

        run(
            send_canonical_final(
                boom, history_uid="h", request_id="r", text="full text here"
            )
        )


if __name__ == "__main__":
    unittest.main()
