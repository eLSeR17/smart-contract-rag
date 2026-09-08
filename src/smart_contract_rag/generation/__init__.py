"""Generation layer: prompts, LLM calls, and anti-hallucination grounding."""

from __future__ import annotations

from .generator import LLMGenerator, SystemPromptError
from .grounding import GroundedTextCheck, NaiveGroundedTextCheck
from .prompt import build_system_prompt, build_user_prompt

__all__ = [
    "GroundedTextCheck",
    "LLMGenerator",
    "NaiveGroundedTextCheck",
    "SystemPromptError",
    "build_system_prompt",
    "build_user_prompt",
]
