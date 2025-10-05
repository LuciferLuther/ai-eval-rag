"""Guardrail utilities for the RAG API."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

@dataclass
class GuardrailResult:
    triggered: bool
    reason: str | None = None

class QueryDenylistGuardrail:
    """Rejects queries that contain high-risk substrings (PII/credentials/prompt-injection)."""

    def __init__(self, denylist: Iterable[str] | None = None) -> None:
        # Base list (can be tuned per deployment)
        self.denylist: List[str] = [
            "social security",
            "ssn",
            "credit card",
            "password",
            "password dump",
            "api key",
            "apikey",
            "ssh key",
            "system prompt",
            "ignore previous",
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

class QueryBudgetGuardrail:
    """Rejects overly long queries to control latency/cost. Simple word-count proxy for tokens."""
    def __init__(self, max_words: int = 128) -> None:
        self.max_words = max_words

    def check(self, query: str) -> GuardrailResult:
        words = len(query.split())
        if words > self.max_words:
            return GuardrailResult(True, f"Query exceeds budget ({words} > {self.max_words} words)")
        return GuardrailResult(False)

# Compose guardrails here (order matters: cheap checks first)
_DENY = QueryDenylistGuardrail()
_BUDGET = QueryBudgetGuardrail(max_words=128)

def check_guardrails(query: str) -> GuardrailResult:
    """Runs the configured guardrails for a query."""
    for guard in (_BUDGET, _DENY):
        res = guard.check(query)
        if res.triggered:
            return res
    return GuardrailResult(False)
