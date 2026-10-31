"""Regression tests for WebSocket client initialization and disconnect handling.

Pins the contract that:
- Normal client disconnections during handshake (WebSocketDisconnect) are
  handled as INFO logs and do not trigger false-alarm ERROR logs.
- Disconnect during service context init or initial message sending runs
  idempotent cleanup for that client.
- Real system exceptions (e.g. RuntimeError) are logged with logger.exception
  tracebacks so real bugs are not obscured.
- Cleanup of a failed client is strictly isolated and never touches or removes
  active peer connections or their state.
"""

from __future__ import annotations

import asyncio
import json
import unittest
from io import StringIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger

from src.open_llm_vtuber.service_context import ServiceContext
from src.open_llm_vtuber.websocket_handler import WebSocketHandler


class _MockWebSocket:
    def __init__(self, send_disconnect_at: int = -1):
        self.sent: list[dict] = []
        self.send_call_count = 0
        self.send_disconnect_at = send_disconnect_at

    async def send_text(self, payload: str) -> None:
        self.send_call_count += 1
        if self.send_disconnect_at > 0 and self.send_call_count >= self.send_disconnect_at:
            raise WebSocketDisconnect(code=1000)
        self.sent.append(json.loads(payload))

    async def accept(self) -> None:
        pass


def _create_mock_default_context() -> ServiceContext:
    ctx = MagicMock(spec=ServiceContext)
    ctx.config = MagicMock()
    ctx.config.model_copy.return_value = MagicMock()
    ctx.system_config = MagicMock()
    ctx.system_config.model_copy.return_value = MagicMock()
    ctx.character_config = MagicMock()
    ctx.character_config.model_copy.return_value = MagicMock()
    ctx.character_config.conf_name = "TestChar"
    ctx.character_config.conf_uid = "id_test_01"
    ctx.live2d_model = MagicMock()
    ctx.live2d_model.model_info = {"model": "test"}
    ctx.asr_engine = MagicMock()
    ctx.tts_engine = MagicMock()
    ctx.vad_engine = MagicMock()
    ctx.translate_engine = MagicMock()
    ctx.mcp_server_registery = MagicMock()
    ctx.tool_adapter = MagicMock()
    return ctx


class TestWebSocketInitDisconnect(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.default_ctx = _create_mock_default_context()
        self.handler = WebSocketHandler(self.default_ctx)

    async def test_disconnect_during_init_service_context(self):
        """A client that disconnects while context is loading logs INFO, not ERROR."""
        ws = _MockWebSocket()
        client_uid = "client-disconnect-ctx"

        log_buffer = StringIO()
        sink_id = logger.add(log_buffer, level="DEBUG", format="{level} | {message}")

        try:
            with patch.object(
                self.handler,
                "_init_service_context",
                side_effect=WebSocketDisconnect(code=1000),
            ):
                with self.assertRaises(WebSocketDisconnect):
                    await self.handler.handle_new_connection(ws, client_uid)
        finally:
            logger.remove(sink_id)

        output = log_buffer.getvalue()
        self.assertIn("INFO | Client client-disconnect-ctx disconnected during connection initialization", output)
        self.assertNotIn("ERROR", output)
        self.assertNotIn(client_uid, self.handler.client_connections)
        self.assertNotIn(client_uid, self.handler.client_contexts)

    async def test_disconnect_during_send_initial_messages(self):
        """A client that disconnects while server sends initial frames logs INFO, not ERROR."""
        ws = _MockWebSocket(send_disconnect_at=1)
        client_uid = "client-disconnect-msg"

        mock_session_ctx = MagicMock(spec=ServiceContext)
        mock_session_ctx.character_config = SimpleNamespace(
            conf_name="Mili", conf_uid="id_mili_01"
        )
        mock_session_ctx.live2d_model = SimpleNamespace(model_info={})

        log_buffer = StringIO()
        sink_id = logger.add(log_buffer, level="DEBUG", format="{level} | {message}")

        try:
            with patch.object(
                self.handler,
                "_init_service_context",
                AsyncMock(return_value=mock_session_ctx),
            ):
                with self.assertRaises(WebSocketDisconnect):
                    await self.handler.handle_new_connection(ws, client_uid)
        finally:
            logger.remove(sink_id)

        output = log_buffer.getvalue()
        self.assertIn("INFO | Client client-disconnect-msg disconnected during connection initialization", output)
        self.assertNotIn("ERROR", output)
        self.assertNotIn(client_uid, self.handler.client_connections)
        self.assertNotIn(client_uid, self.handler.client_contexts)

    async def test_system_exception_logs_with_traceback(self):
        """An actual runtime error is logged with ERROR and full traceback."""
        ws = _MockWebSocket()
        client_uid = "client-real-error"

        log_buffer = StringIO()
        sink_id = logger.add(log_buffer, level="DEBUG", format="{level} | {message}")

        try:
            with patch.object(
                self.handler,
                "_init_service_context",
                side_effect=RuntimeError("Unexpected memory crash"),
            ):
                with self.assertRaises(RuntimeError):
                    await self.handler.handle_new_connection(ws, client_uid)
        finally:
            logger.remove(sink_id)

        output = log_buffer.getvalue()
        self.assertIn("ERROR | Failed to initialize connection for client client-real-error: Unexpected memory crash", output)
        self.assertNotIn(client_uid, self.handler.client_connections)
        self.assertNotIn(client_uid, self.handler.client_contexts)

    async def test_cleanup_is_idempotent_and_preserves_peer_state(self):
        """Cleaning up a disconnected client must never alter or wipe an active peer client's state."""
        peer_uid = "peer-healthy-client"
        peer_ws = _MockWebSocket()
        peer_ctx = MagicMock(spec=ServiceContext)

        self.handler.client_connections[peer_uid] = peer_ws
        self.handler.client_contexts[peer_uid] = peer_ctx
        self.handler.chat_group_manager.client_group_map[peer_uid] = "group-1"
        peer_task = asyncio.create_task(asyncio.sleep(10))
        self.handler.current_conversation_tasks[peer_uid] = peer_task

        dead_uid = "dead-client"
        dead_ws = _MockWebSocket()
        dead_ctx = MagicMock(spec=ServiceContext)
        self.handler.client_connections[dead_uid] = dead_ws
        self.handler.client_contexts[dead_uid] = dead_ctx
        self.handler.chat_group_manager.client_group_map[dead_uid] = "group-1"

        # First cleanup
        await self.handler._cleanup_failed_connection(dead_uid)
        self.assertNotIn(dead_uid, self.handler.client_connections)
        self.assertNotIn(dead_uid, self.handler.client_contexts)
        self.assertNotIn(dead_uid, self.handler.chat_group_manager.client_group_map)

        # Peer must be untouched
        self.assertIn(peer_uid, self.handler.client_connections)
        self.assertIn(peer_uid, self.handler.client_contexts)
        self.assertEqual(self.handler.chat_group_manager.client_group_map[peer_uid], "group-1")
        self.assertFalse(peer_task.cancelled())

        # Second cleanup (idempotency check)
        await self.handler._cleanup_failed_connection(dead_uid)
        self.assertIn(peer_uid, self.handler.client_connections)
        self.assertIn(peer_uid, self.handler.client_contexts)
        self.assertFalse(peer_task.cancelled())

        # Clean up mock peer task
        peer_task.cancel()
        try:
            await peer_task
        except asyncio.CancelledError:
            pass

    async def test_successful_connection_handshake(self):
        """A healthy connection establishes without errors and sends initial handshake frames."""
        ws = _MockWebSocket()
        client_uid = "healthy-client"

        mock_session_ctx = MagicMock(spec=ServiceContext)
        mock_session_ctx.character_config = SimpleNamespace(
            conf_name="Mili", conf_uid="id_mili_01"
        )
        mock_session_ctx.live2d_model = SimpleNamespace(model_info={"name": "mili"})

        with patch.object(
            self.handler,
            "_init_service_context",
            AsyncMock(return_value=mock_session_ctx),
        ):
            await self.handler.handle_new_connection(ws, client_uid)

        self.assertIn(client_uid, self.handler.client_connections)
        self.assertIn(client_uid, self.handler.client_contexts)
        sent_types = [m.get("type") for m in ws.sent]
        self.assertIn("full-text", sent_types)
        self.assertIn("set-model-and-conf", sent_types)
        self.assertIn("control", sent_types)


if __name__ == "__main__":
    unittest.main()
