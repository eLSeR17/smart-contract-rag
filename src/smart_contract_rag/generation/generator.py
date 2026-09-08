"""LLM-based answer generator calling a local model over HTTP.

The generator talks to Ollama's ``/api/chat`` endpoint. Ollama is reachable at
``http://ollama:11434`` *inside the ``docker_default`` network*; it does not
expose a port to the host WSL machine. That is why we never hardcode
``localhost`` here — the URL is always injected via the environment.
"""

from __future__ import annotations

from typing import Protocol

import httpx

from ..models import Chunk
from .prompt import build_user_prompt


class SystemPromptError(Exception):
    """Raised when a prompt template fails to validate or build."""


class Generator(Protocol):
    """Anything able to produce a textual answer from context chunks."""

    def generate(self, query: str, chunks: list[Chunk], system_prompt: str) -> str:
        """Return the model's answer text for the given context."""
        ...


class LLMGenerator:
    """Generates answers via a local Ollama chat endpoint."""

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        temperature: float = 0.2,
        max_tokens: int = 512,
        timeout: float = 120.0,
    ) -> None:
        if not base_url:
            raise SystemPromptError("OLLAMA_URL must be provided")
        if not model:
            raise SystemPromptError("OLLAMA_MODEL must be provided")
        # httpx is thread-safe and connection-pooled; no threads are spawned.
        self._client = httpx.Client(base_url=base_url, timeout=timeout)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url

    # ------------------------------------------------------------------
    def generate(
        self,
        query: str,
        chunks: list[Chunk],
        system_prompt: str,
    ) -> str:
        if not chunks:
            # With no evidence there is nothing to anchor on; the pipeline's
            # grounding layer will refuse. We still call the model so the flow
            # is uniform, but signal emptiness is being passed.
            raise SystemPromptError("No context chunks provided for generation")
        user_prompt = build_user_prompt(query, chunks)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }
        response = self._client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()
        content = (data.get("message") or {}).get("content", "")
        return content.strip()
