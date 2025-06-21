import pytest
from unittest.mock import AsyncMock

from prompthelix.websocket_manager import ConnectionManager


@pytest.mark.asyncio
async def test_connect_and_disconnect():
    manager = ConnectionManager()
    ws = AsyncMock()
    await manager.connect(ws)
    assert ws in manager.active_connections
    manager.disconnect(ws)
    assert ws not in manager.active_connections


@pytest.mark.asyncio
async def test_send_and_broadcast():
    manager = ConnectionManager()
    ws1 = AsyncMock()
    ws2 = AsyncMock()
    await manager.connect(ws1)
    await manager.connect(ws2)
    await manager.send_personal_message('hi', ws1)
    await manager.broadcast('hello')

    ws1.send_text.assert_any_await('hi')
    ws1.send_text.assert_any_await('hello')
    ws2.send_text.assert_awaited_with('hello')
