"""Guardrail utilities for the RAG API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass
class GuardrailResult:
    triggered: bool
    reason: str | None = None


class QueryDenylistGuardrail:
    """Rejects queries that contain high-risk substrings (PII/credentials)."""

    def __init__(self, denylist: Iterable[str] | None = None) -> None:
        self.denylist: List[str] = [
            "social security",
            "credit card",
            "ssn",
            "password dump",
        ]
        if denylist:
            for term in denylist:
                needle = term.strip().lower()
                if needle and needle not in self.denylist:
                    self.denylist.append(needle)

    def check(self, query: str) -> GuardrailResult:
        lowered = query.lower()
        for term in self.denylist:
            if term in lowered:
                return GuardrailResult(True, f"Query blocked by denylist term '{term}'")
        return GuardrailResult(False)


_DEFAULT_GUARDRAIL = QueryDenylistGuardrail()


def check_guardrails(query: str) -> GuardrailResult:
    """Runs the configured guardrails for a query."""

    return _DEFAULT_GUARDRAIL.check(query)