"""Synchronous MCP client over Streamable HTTP.

SYNC-OVER-ASYNC BRIDGE: the official SDK is async; our agent loop is sync.
Each operation opens a session, runs, and closes (`asyncio.run`). That is a
deliberate simplification aligned with where the protocol is heading —
Streamable HTTP is designed so servers can be stateless per request — but it
does pay connection setup per call. The async-native FastAPI interface of
phase 9 could keep long-lived sessions instead; documented trade-off.

The `_connect` seam exists for tests: integration tests subclass this client
and connect to an IN-PROCESS server over memory streams — full protocol
exchange (initialize, list_tools, call_tool), zero network.
"""

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel


class RemoteToolInfo(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
    # From the server's ToolAnnotations. Missing annotations mean UNKNOWN
    # safety — we default to False (not read-only), i.e. consent required.
    # Fail closed on missing metadata; the opposite default is how "the tool
    # deleted prod and nobody was asked" incidents happen.
    read_only: bool = False


class SyncMcpClient:
    def __init__(self, url: str) -> None:
        self._url = url

    # -- protocol operations --------------------------------------------------

    def list_tools(self) -> list[RemoteToolInfo]:
        async def op(session: ClientSession) -> list[RemoteToolInfo]:
            result = await session.list_tools()
            tools = []
            for tool in result.tools:
                annotations = tool.annotations
                read_only = bool(annotations and annotations.readOnlyHint)
                tools.append(
                    RemoteToolInfo(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema,
                        read_only=read_only,
                    )
                )
            return tools

        return self._run(op)

    def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        """Returns (text, is_error) — mirroring the observation contract of
        the agent loop, so adapters translate trivially."""

        async def op(session: ClientSession) -> tuple[str, bool]:
            result = await session.call_tool(name, arguments)
            text = "\n".join(
                block.text for block in result.content if block.type == "text"
            )
            return text or "(empty result)", bool(result.isError)

        return self._run(op)

    # -- plumbing --------------------------------------------------------------

    def _run(self, op: Callable[[ClientSession], Awaitable[Any]]) -> Any:
        async def runner() -> Any:
            async with self._connect() as session:
                return await op(session)

        return asyncio.run(runner())

    @asynccontextmanager
    async def _connect(self) -> AsyncIterator[ClientSession]:
        async with streamablehttp_client(self._url) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()
                yield session
