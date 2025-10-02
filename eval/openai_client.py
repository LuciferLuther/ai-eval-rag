"""Lightweight OpenAI client abstraction with an optional mock for testing."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol


class CompletionClient(Protocol):
    def generate(self, prompt: str, **kwargs) -> str:
        ...


@dataclass
class OpenAIChatClient(CompletionClient):
    api_key: str
    model: str = "gpt-4o-mini"
    base_url: Optional[str] = None
    default_kwargs: Dict[str, object] = field(default_factory=lambda: {"temperature": 0.0, "max_output_tokens": 512})

    def __post_init__(self) -> None:
        from openai import OpenAI  # lazy import to keep startup lean

        self._client = OpenAI(api_key=self.api_key, base_url=self.base_url)

    def generate(self, prompt: str, **kwargs) -> str:
        payload = {**self.default_kwargs, **kwargs}
        model = payload.pop("model", self.model)
        response = self._client.responses.create(model=model, input=prompt, **payload)
        # The Responses API returns a helper for concatenated text.
        text = getattr(response, "output_text", None)
        if text:
            return text.strip()
        # Fallback if output_text is unavailable (older SDKs)
        if response.output and response.output[0].content:
            return "".join(block.text for block in response.output[0].content if getattr(block, "text", None)).strip()
        raise RuntimeError("OpenAI response did not include text content")


@dataclass
class MockEchoClient(CompletionClient):
    suffix: str = ""

    def generate(self, prompt: str, **kwargs) -> str:  # pragma: no cover - trivial
        return f"[mock]{self.suffix} {prompt}".strip()


def build_client(kind: str = "openai") -> CompletionClient:
    """Factory that returns either a real OpenAI client or a mock."""

    resolved_kind = kind.lower()
    if resolved_kind == "mock" or os.getenv("OPENAI_MOCK", "false").lower() == "true":
        return MockEchoClient()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set. Provide it or use --mock/OPENAI_MOCK=true.")

    base_url = os.getenv("OPENAI_BASE_URL") or None
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    return OpenAIChatClient(api_key=api_key, base_url=base_url, model=model)