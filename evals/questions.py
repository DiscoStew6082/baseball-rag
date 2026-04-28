"""CI-safe golden eval runner for ``evals/questions.yaml``."""

from __future__ import annotations

import argparse
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

from baseball_rag.provenance import StructuredAnswer

AnswerFn = Callable[[str], StructuredAnswer]


DEFAULT_QUESTIONS_PATH = Path(__file__).with_name("questions.yaml")
LIVE_SOURCE_TYPES = {"chroma"}
LIVE_INTENTS = {"freeform_query", "player_biography", "general_explanation"}


@dataclass(frozen=True)
class EvalCase:
    """A single golden question and its expected response properties."""

    id: str
    question: str
    spec: dict[str, Any]

    @property
    def required_sources(self) -> set[str]:
        return {str(source) for source in self.spec.get("required_sources", [])}

    @property
    def intent(self) -> str | None:
        intent = self.spec.get("intent")
        return str(intent) if intent is not None else None

    @property
    def ci_safe(self) -> bool:
        return bool(self.spec.get("ci_safe", False))

    def requires_live_services(self) -> bool:
        """Return True when the case is expected to need LLM or live Chroma."""
        if self.required_sources & LIVE_SOURCE_TYPES:
            return True
        return self.intent in LIVE_INTENTS

    def should_run(self, *, include_live: bool = False) -> bool:
        """Select deterministic cases by default, plus explicit opt-ins."""
        if include_live or self.ci_safe:
            return True
        return (
            self.intent == "stat_query"
            and self.required_sources == {"duckdb"}
            and not self.spec.get("expected_unsupported", False)
            and not self.requires_live_services()
        )


@dataclass
class EvalCaseResult:
    """Outcome for one eval case."""

    case_id: str
    status: str
    failures: list[str] = field(default_factory=list)
    reason: str | None = None


@dataclass
class EvalRunResult:
    """Aggregate outcome for an eval run."""

    passed: list[EvalCaseResult] = field(default_factory=list)
    failed: list[EvalCaseResult] = field(default_factory=list)
    skipped: list[EvalCaseResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failed

    @property
    def attempted(self) -> int:
        return len(self.passed) + len(self.failed)


def load_cases(path: Path = DEFAULT_QUESTIONS_PATH) -> list[EvalCase]:
    """Load eval cases from the YAML manifest."""
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    questions = raw.get("questions")
    if not isinstance(questions, list):
        raise ValueError(f"{path} must contain a top-level questions list")

    cases: list[EvalCase] = []
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Question #{index} must be a mapping")
        case_id = item.get("id")
        question = item.get("question")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"Question #{index} must have a non-empty string id")
        if not isinstance(question, str) or not question:
            raise ValueError(f"Question {case_id!r} must have a non-empty question")
        cases.append(EvalCase(id=case_id, question=question, spec=item))
    return cases


def selected_cases(cases: list[EvalCase], *, include_live: bool = False) -> list[EvalCase]:
    """Return cases runnable under the selected service constraints."""
    return [case for case in cases if case.should_run(include_live=include_live)]


def run_cases(
    cases: list[EvalCase],
    *,
    answer_fn: AnswerFn | None = None,
    include_live: bool = False,
) -> EvalRunResult:
    """Run selected cases through ``baseball_rag.service.answer`` and validate them."""
    runner: AnswerFn
    if answer_fn is None:
        from baseball_rag.service import answer as service_answer

        def service_runner(value: str) -> StructuredAnswer:
            return service_answer(value)

        runner = service_runner

    else:
        runner = answer_fn

    result = EvalRunResult()
    for case in cases:
        if not case.should_run(include_live=include_live):
            result.skipped.append(
                EvalCaseResult(case_id=case.id, status="skipped", reason="not CI-safe")
            )
            continue

        try:
            answer = runner(case.question)
            failures = validate_case(case, answer)
        except Exception as exc:  # noqa: BLE001 - evals should report all case failures
            failures = [f"{type(exc).__name__}: {exc}"]
        case_result = EvalCaseResult(
            case_id=case.id,
            status="failed" if failures else "passed",
            failures=failures,
        )
        if failures:
            result.failed.append(case_result)
        else:
            result.passed.append(case_result)
    return result


def validate_case(case: EvalCase, answer: StructuredAnswer) -> list[str]:
    """Validate supported golden expectations against a structured answer."""
    failures: list[str] = []
    spec = case.spec

    expected_intent = spec.get("intent")
    if expected_intent is not None and answer.intent != expected_intent:
        failures.append(f"intent: expected {expected_intent!r}, got {answer.intent!r}")

    if "expected_unsupported" in spec and answer.unsupported is not bool(
        spec["expected_unsupported"]
    ):
        failures.append(
            f"unsupported: expected {bool(spec['expected_unsupported'])!r}, "
            f"got {answer.unsupported!r}"
        )

    answer_text = _normalized_text(answer.answer)
    for needle in spec.get("expected_answer_contains", []) or []:
        if _normalized_text(str(needle)) not in answer_text:
            failures.append(f"answer missing substring {needle!r}")

    source_types = [source.type for source in answer.sources]
    for source_type in spec.get("required_sources", []) or []:
        if source_type not in source_types:
            failures.append(f"sources missing required type {source_type!r}")

    row_count = _row_count(answer)
    min_rows = spec.get("expected_min_rows")
    if min_rows is not None and row_count < int(min_rows):
        failures.append(f"row count: expected >= {min_rows}, got {row_count}")

    max_rows = spec.get("expected_max_rows")
    if max_rows is not None and row_count > int(max_rows):
        failures.append(f"row count: expected <= {max_rows}, got {row_count}")

    for field_name in spec.get("required_source_manifest_fields", []) or []:
        if not any(
            source.data_manifest and field_name in source.data_manifest for source in answer.sources
        ):
            failures.append(f"source manifest missing field {field_name!r}")

    if spec.get("expected_sql_visible") and not any(source.sql for source in answer.sources):
        failures.append("expected visible SQL on at least one source")

    for needle in spec.get("expected_sql_contains", []) or []:
        if not any(source.sql and str(needle) in source.sql for source in answer.sources):
            failures.append(f"SQL missing substring {needle!r}")

    return failures


def _row_count(answer: StructuredAnswer) -> int:
    return sum(len(source.rows) for source in answer.sources)


def _normalized_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    without_accents = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_accents.casefold()


def main(argv: list[str] | None = None) -> int:
    """Run evals from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS_PATH)
    parser.add_argument(
        "--include-live",
        action="store_true",
        help="also run cases that may require LLM or Chroma services",
    )
    args = parser.parse_args(argv)

    result = run_cases(load_cases(args.questions), include_live=args.include_live)
    print(
        f"evals: {len(result.passed)} passed, {len(result.failed)} failed, "
        f"{len(result.skipped)} skipped"
    )
    for failed in result.failed:
        print(f"- {failed.case_id}: " + "; ".join(failed.failures))
    return 0 if result.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
