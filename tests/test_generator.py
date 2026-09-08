"""Tests for the LLM generator and prompt templates (no real network)."""

from __future__ import annotations

import httpx
import pytest

from smart_contract_rag.generation.generator import LLMGenerator, SystemPromptError
from smart_contract_rag.generation.prompt import build_system_prompt, build_user_prompt
from smart_contract_rag.models import Chunk, SourceRef


class TestSystemPrompt:
    def test_prompt_defines_source_numbering_and_dont_know(self) -> None:
        prompt = build_system_prompt()
        assert "[1]" in prompt or "numbered" in prompt.lower()
        assert "I DON'T KNOW" in prompt
        assert "ONLY" in prompt

    def test_prompt_rejects_embedded_instructions(self) -> None:
        prompt = build_system_prompt()
        assert "do not follow" in prompt.lower()


class TestUserPrompt:
    def test_chunks_numbered_and_quoted(self) -> None:
        chunks = [
            Chunk(text="first finding", source=SourceRef(doc_id="doc-a", page=2), index=0),
            Chunk(text="second finding", source=SourceRef(doc_id="doc-b"), index=1),
        ]
        prompt = build_user_prompt("Is there reentrancy?", chunks)
        assert "[1] doc-a (page 2): first finding" in prompt
        assert "[2] doc-b: second finding" in prompt
        assert "QUESTION: Is there reentrancy?" in prompt


class TestLLMGenerator:
    def _generator_with_mock(self, reply: dict) -> tuple[LLMGenerator, list[httpx.Request]]:
        captured: list[httpx.Request] = []

        def handler(request: httpx.Request) -> httpx.Response:
            captured.append(request)
            return httpx.Response(200, json=reply)

        transport = httpx.MockTransport(handler)
        gen = LLMGenerator(base_url="http://ollama:11434", model="qwen2.5-coder:7b")
        gen._client = httpx.Client(transport=transport, base_url="http://ollama:11434")
        return gen, captured

    def test_generate_returns_content(self) -> None:
        gen, _ = self._generator_with_mock(
            {"message": {"content": "The withdraw is vulnerable [1]."}}
        )
        chunks = [Chunk(text="withdraw vulnerable", source=SourceRef(doc_id="d"), index=0)]
        out = gen.generate("reentrancy?", chunks, build_system_prompt())
        assert out == "The withdraw is vulnerable [1]."

    def test_generate_sends_expected_payload(self) -> None:
        gen, captured = self._generator_with_mock({"message": {"content": "ok"}})
        chunks = [Chunk(text="some corpus text", source=SourceRef(doc_id="d"), index=0)]
        gen.generate("question?", chunks, build_system_prompt())
        assert len(captured) == 1
        assert captured[0].url.path == "/api/chat"
        payload = captured[0].read()
        assert b"qwen2.5-coder:7b" in payload
        assert b"some corpus text" in payload

    def test_generate_no_chunks_raises(self) -> None:
        gen, _ = self._generator_with_mock({"message": {"content": "x"}})
        with pytest.raises(SystemPromptError):
            gen.generate("q", [], build_system_prompt())

    def test_missing_base_url_raises(self) -> None:
        with pytest.raises(SystemPromptError):
            LLMGenerator(base_url="", model="m")

    def test_missing_model_raises(self) -> None:
        with pytest.raises(SystemPromptError):
            LLMGenerator(base_url="http://ollama:11434", model="")
