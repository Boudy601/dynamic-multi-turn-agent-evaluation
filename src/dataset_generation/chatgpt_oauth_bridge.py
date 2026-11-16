"""Local ChatGPT OAuth -> Responses API adapter for the Task Closure generator.

The login module stays outside the repository; this adapter imports it by path
and asks it for short-lived credentials. Never copy OAuth tokens into .env.
"""
from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


class ChatGPTOAuthResponsesClient:
    """An async provider compatible with generate_golden_dataset.request_with_backoff."""

    def __init__(self, auth_script: str | Path):
        path = Path(auth_script).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"OAuth script not found: {path}")
        spec = importlib.util.spec_from_file_location("task_closure_local_oauth", path)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Cannot import OAuth script: {path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        if not callable(getattr(module, "current_access_token", None)):
            raise RuntimeError("OAuth script must provide current_access_token()")
        self.auth = module

    def check_login(self) -> None:
        """Fail before processing records if authentication is missing/expired."""
        self.auth.current_access_token()  # Do not store, print, or return token.

    async def generate(self, messages: list[dict[str, str]], model: str, timeout_seconds: float) -> str:
        # Token may rotate at any time; fetch afresh for each request, including retries.
        access_token = await asyncio.to_thread(self.auth.current_access_token)
        fragments: list[str] = []
        completed = False
        async with AsyncOpenAI(api_key=access_token, base_url="https://api.openai.com/v1", max_retries=0, timeout=timeout_seconds) as api:
            stream = await api.responses.create(
                model=model,
                input=messages,
                store=False,
                stream=True,
                timeout=timeout_seconds,
            )
            try:
                async for event in stream:
                    if event.type == "response.output_text.delta":
                        fragments.append(event.delta)
                    elif event.type == "response.completed":
                        completed = True
                    elif event.type in ("response.failed", "response.incomplete", "error"):
                        raise RuntimeError(f"Responses API ended without success: {event.type}")
            finally:
                await stream.close()
        if not completed or not fragments:
            raise ValueError("Responses API returned no completed text response")
        return "".join(fragments)

    async def close(self) -> None:
        """Each request owns its short-lived HTTP client; no global connection to close."""
        return None
