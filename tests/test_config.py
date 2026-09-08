"""Tests for environment-driven configuration."""

from __future__ import annotations

from smart_contract_rag.config import Settings


class TestSettings:
    def test_defaults(self) -> None:
        settings = Settings.from_env(environ={})
        assert settings.ollama_url == "http://ollama:11434"  # docker network, not localhost
        assert settings.ollama_model == "qwen2.5-coder:7b"
        assert settings.embedding_model == "all-MiniLM-L6-v2"

    def test_overrides(self) -> None:
        settings = Settings.from_env(
            environ={
                "OLLAMA_MODEL": "some-other-model",
                "TOP_K": "8",
                "CORPUS_DIR": "/custom/corpus",
            }
        )
        assert settings.ollama_model == "some-other-model"
        assert settings.top_k == 8
        assert str(settings.corpus_dir) == "/custom/corpus"
        assert str(settings.manifest_path) == "/custom/corpus/manifest.json"
