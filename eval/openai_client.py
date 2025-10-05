"""Lightweight OpenAI client abstraction with an optional mock for testing."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Optional, Protocol
from dotenv import load_dotenv

# Load environment variables from .env files
load_dotenv()  # Current directory
load_dotenv("../.env")  # Parent directory
load_dotenv("../../.env")  # Grandparent directory


class CompletionClient(Protocol):
    def generate(self, prompt: str, **kwargs) -> str:
        ...


@dataclass
class OpenAIChatClient(CompletionClient):
    api_key: str
    model: str = "gpt-4.1-mini"
    base_url: Optional[str] = None
    default_kwargs: Dict[str, object] = field(default_factory=lambda: {"temperature": 0.0, "max_output_tokens": 1024})

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

    def generate(self, prompt: str, **kwargs) -> str:  # pragma: no cover - lightweight shim
        """Return canned answers so the mock behaves like a tiny FAQ."""

        def reply(text: str) -> str:
            suffix = f" {self.suffix}" if self.suffix else ""
            return f"{text}{suffix}".strip()

        prompt_lc = prompt.lower()

        if "library card" in prompt_lc or ("card" in prompt_lc and "libr" in prompt_lc):
            return reply("Bring a photo ID and proof of address to the front desk to get your library card.")
        if "renew" in prompt_lc and "book" in prompt_lc:
            return reply("Renew twice from the My Account page, but holds stop additional renewals.")
        if "late fee" in prompt_lc or "forget to return" in prompt_lc:
            return reply("Overdue books cost 25 cents per day with a five dollar maximum.")
        if "print" in prompt_lc or "printing" in prompt_lc:
            return reply("Ten cents per page for black-and-white prints and fifty cents per page for color.")
        if "storytime" in prompt_lc:
            return reply("Children's storytime runs Tuesdays and Thursdays at 10 a.m. in the community room.")
        if ("meeting room" in prompt_lc) or ("reserve" in prompt_lc and "room" in prompt_lc):
            return reply("Reserve the meeting room up to four hours per week and book online up to two weeks ahead.")
        if "wi-fi" in prompt_lc or "wifi" in prompt_lc:
            return reply("The Wi-Fi password is posted on signs at every table.")
        if "volunteer" in prompt_lc:
            return reply("Fill out the volunteer interest form and attend the monthly orientation to get started.")
        if "donate" in prompt_lc:
            return reply("We accept gently used books from the last five years; drop them off Saturdays 9 a.m. to noon.")
        if "cross-validation" in prompt_lc or "m&a" in prompt_lc:
            return reply("Confirm the five critical financial metrics, especially revenue, before raising an alert.")
        if "fire safety" in prompt_lc and ("workflow" in prompt_lc or "comms" in prompt_lc):
            return reply("Send the primaries on Monday and Wednesday, then follow up Tuesday through Friday as scheduled.")
        if "inspection" in prompt_lc and ("automation" in prompt_lc or "outreach" in prompt_lc):
            return reply("It delivered a 70% drop in admin cost across SMS and WhatsApp outreach.")
        if "feature flag" in prompt_lc or "non-technical" in prompt_lc:
            return reply("Feature flags let non-technical teams toggle access without engineering support.")
        if "medical" in prompt_lc and "booking" in prompt_lc:
            return reply("We cited a 30% accuracy lift driven by the hybrid vector retrieval stack.")

        return reply(f"[mock] {prompt}")


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
