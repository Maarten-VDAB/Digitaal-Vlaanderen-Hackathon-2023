"""Callback handlers used in the app."""
from typing import Any

from schemas import ChatResponse
from langchain.callbacks.base import AsyncCallbackHandler


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """Callback handler for streaming LLM responses."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ChatResponse(sender="bot", message=token, type="stream")
        await self.websocket.send_json(resp.dict())

    async def on_text(self, text: str, **kwargs: Any) -> None:
        start = ChatResponse(sender="bot", message="", type="start")
        await self.websocket.send_json(start.dict())