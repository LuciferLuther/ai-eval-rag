"""Evaluation harness for prompt reliability experiments.

Usage:
    python eval/eval_harness.py --tests eval/tests.json --model gpt-4o-mini --mock

The harness will call into the OpenAI client wrapper and evaluate responses
against simple heuristics (regex/contains/exact) while also computing invariance
consistency metrics for variants and perturbations.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from openai_client import build_client, CompletionClient


SUPPORTED_METHODS = {"regex", "exact", "contains"}


@dataclass
class TestCase:
    """Represents a single prompt-and-expected pair from tests.json."""

    id: str
    prompt: str
    expected_meta: Dict[str, str]
    evaluation_method: str
    variant_of: Optional[str]
    perturbs: Sequence[str]

    @classmethod
    def from_dict(cls, raw: Dict[str, object]) -> "TestCase":
        method = str(raw.get("expected_evaluation", {}).get("method", "")).lower()
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported evaluation method '{method}' in test {raw.get('id')}.")

        return cls(
            id=str(raw["id"]),
            prompt=str(raw["prompt"]),
            expected_meta={k: str(v) for k, v in raw.get("expected_evaluation", {}).items()},
            evaluation_method=method,
            variant_of=str(raw.get("variant_of")) if raw.get("variant_of") else None,
            perturbs=tuple(str(pid) for pid in raw.get("perturbs", []) if pid),
        )


@dataclass
class TestResult:
    case: TestCase
    response: str
    passed: bool
    score: float
    failure_reason: Optional[str] = None


@dataclass
class ConsistencyMetrics:
    invariance_pairs: int
    invariance_consistent: int
    perturbation_pairs: int
    perturbation_diverged: int

    @property
    def invariance_rate(self) -> float:
        return self._safe_div(self.invariance_consistent, self.invariance_pairs)

    @property
    def perturbation_rate(self) -> float:
        return self._safe_div(self.perturbation_diverged, self.perturbation_pairs)

    @staticmethod
    def _safe_div(numerator: int, denominator: int) -> float:
        return (numerator / denominator) if denominator else 0.0


def load_test_cases(path: Path) -> List[TestCase]:
    with path.open("r", encoding="utf-8") as fp:
        raw_cases = json.load(fp)
    return [TestCase.from_dict(item) for item in raw_cases]


def evaluate(case: TestCase, response: str) -> TestResult:
    method = case.evaluation_method
    expected = case.expected_meta
    text = response.strip()

    if method == "regex":
        pattern = expected.get("pattern", "")
        try:
            passed = bool(re.search(pattern, text))
        except re.error as exc:
            raise ValueError(f"Invalid regex in test {case.id}: {pattern}") from exc
        score = 1.0 if passed else 0.0
        reason = None if passed else f"Regex '{pattern}' not found"
    elif method == "exact":
        target = expected.get("value", "")
        passed = text == target
        score = 1.0 if passed else 0.0
        reason = None if passed else f"Exact match failed. Expected '{target}', got '{text}'"
    elif method == "contains":
        target = expected.get("value", "")
        passed = target.lower() in text.lower()
        score = 1.0 if passed else 0.0
        reason = None if passed else f"Missing expected substring '{target}'"
    else:
        raise ValueError(f"Unknown evaluation method: {method}")

    return TestResult(case=case, response=text, passed=passed, score=score, failure_reason=reason)


def aggregate_consistency(results: List[TestResult]) -> ConsistencyMetrics:
    by_id = {result.case.id: result for result in results}
    invariance_pairs = invariance_consistent = 0
    perturb_pairs = perturb_diverged = 0

    # For invariance, compare variant response to canonical parent response.
    for result in results:
        parent_id = result.case.variant_of
        if not parent_id:
            continue
        parent = by_id.get(parent_id)
        if not parent:
            continue
        invariance_pairs += 1
        if normalize_response(result.response) == normalize_response(parent.response):
            invariance_consistent += 1

    # For perturbations we count when the perturbed prompt leads to a different answer.
    for result in results:
        for parent_id in result.case.perturbs:
            parent = by_id.get(parent_id)
            if not parent:
                continue
            perturb_pairs += 1
            if normalize_response(result.response) != normalize_response(parent.response):
                perturb_diverged += 1

    return ConsistencyMetrics(
        invariance_pairs=invariance_pairs,
        invariance_consistent=invariance_consistent,
        perturbation_pairs=perturb_pairs,
        perturbation_diverged=perturb_diverged,
    )


def normalize_response(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def run_eval(tests: Sequence[TestCase], client: CompletionClient, args: argparse.Namespace) -> Dict[str, object]:
    results: List[TestResult] = []

    for case in tests:
        if args.dry_run:
            raw_response = f"[dry-run] {case.id}"
        else:
            raw_response = client.generate(case.prompt, model=args.model, temperature=args.temperature)
        test_result = evaluate(case, raw_response)
        results.append(test_result)
        status = "PASS" if test_result.passed else "FAIL"
        print(f"[{status}] {case.id}: score={test_result.score:.1f}")
        if test_result.failure_reason:
            print(f"    reason: {test_result.failure_reason}")

    accuracy = sum(r.passed for r in results) / len(results) if results else 0.0
    consistency = aggregate_consistency(results)

    summary = {
        "total_cases": len(results),
        "accuracy": accuracy,
        "invariance_pairs": consistency.invariance_pairs,
        "invariance_rate": consistency.invariance_rate,
        "perturbation_pairs": consistency.perturbation_pairs,
        "perturbation_divergence_rate": consistency.perturbation_rate,
    }

    if args.output_dir:
        path = Path(args.output_dir)
        path.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        payload = {
            "summary": summary,
            "results": [
                {
                    "id": r.case.id,
                    "prompt": r.case.prompt,
                    "response": r.response,
                    "passed": r.passed,
                    "score": r.score,
                    "failure_reason": r.failure_reason,
                    "variant_of": r.case.variant_of,
                    "perturbs": list(r.case.perturbs),
                }
                for r in results
            ],
        }
        outfile = path / f"eval_{timestamp}.json"
        outfile.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"Saved detailed report to {outfile}")

    return summary


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prompt reliability evaluation harness")
    parser.add_argument("--tests", type=Path, default=Path("eval/tests.json"))
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--mock", action="store_true", help="Use a deterministic echo client")
    parser.add_argument("--dry-run", action="store_true", help="Skip LLM call and echo test ids")
    parser.add_argument("--output-dir", type=Path, help="Optional folder for JSON reports")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    tests = load_test_cases(args.tests)

    if args.mock:
        client = build_client(kind="mock")
    else:
        client = build_client()

    summary = run_eval(tests, client, args)
    print("=== SUMMARY ===")
    for key, value in summary.items():
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())