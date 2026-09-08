"""Environment-driven configuration for the RAG system.

All external endpoints and paths are read from environment variables (see
``.env.example``) so the same code runs on a developer laptop, inside the
docker network, or in CI — with no hardcoded secrets or hosts. Naming follows
the variables documented in ``.env.example``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

_DEFAULTS: dict[str, str] = {
    "OLLAMA_URL": "http://ollama:11434",  # inside docker_default; never localhost
    "OLLAMA_MODEL": "qwen2.5-coder:7b",
    "EMBEDDING_MODEL": "all-MiniLM-L6-v2",
    "CHROMA_DIR": "./data/chroma",
    "CORPUS_DIR": "./data/corpus",
    "MANIFEST_FILE": "manifest.json",
    # Anti-hallucination grounding knobs.
    "GROUNDING_MODE": "lexical",  # valid: "lexical" | "semantic"
    "GROUNDING_MODEL": "cross-encoder/nli-MiniLM-L6-v2",
    "GROUNDING_THRESHOLD": "0.5",
}


@dataclass
class Settings:
    """Typed configuration snapshot loaded from the environment."""

    ollama_url: str
    ollama_model: str
    embedding_model: str
    chroma_dir: Path
    corpus_dir: Path
    manifest_path: Path
    # Optional verbosity/behaviour knobs.
    top_k: int = field(default=5)
    # Anti-hallucination grounding knobs.
    grounding_mode: str = field(default="lexical")
    grounding_model: str = field(default="cross-encoder/nli-MiniLM-L6-v2")
    grounding_threshold: float = field(default=0.5)

    @classmethod
    def from_env(cls, environ: dict[str, str] | None = None) -> "Settings":
        env = environ if environ is not None else os.environ
        value = lambda key: env.get(key, _DEFAULTS[key])  # noqa: E731
        corpus_dir = Path(value("CORPUS_DIR"))
        return cls(
            ollama_url=value("OLLAMA_URL"),
            ollama_model=value("OLLAMA_MODEL"),
            embedding_model=value("EMBEDDING_MODEL"),
            chroma_dir=Path(value("CHROMA_DIR")),
            corpus_dir=corpus_dir,
            manifest_path=corpus_dir / value("MANIFEST_FILE"),
            top_k=int(env.get("TOP_K", "5")),
            grounding_mode=value("GROUNDING_MODE"),
            grounding_model=value("GROUNDING_MODEL"),
            grounding_threshold=float(value("GROUNDING_THRESHOLD")),
        )
