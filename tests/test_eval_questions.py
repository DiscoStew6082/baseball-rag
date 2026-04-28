"""Tests for the golden eval question runner."""

from pathlib import Path

import yaml

from baseball_rag.provenance import SourceRecord, StructuredAnswer
from evals.questions import (
    StrategyRunResult,
    format_strategy_summary,
    load_cases,
    run_cases,
    run_strategy_cases,
    selected_cases,
    selected_strategy_cases,
    validate_case,
)


def _answer(
    *,
    answer: str = "Davis had 153 RBI",
    intent: str = "stat_query",
    unsupported: bool = False,
    source_type: str = "duckdb",
    rows: list[dict] | None = None,
) -> StructuredAnswer:
    return StructuredAnswer(
        answer=answer,
        intent=intent,
        unsupported=unsupported,
        sources=[
            SourceRecord(
                type=source_type,  # type: ignore[arg-type]
                label="test source",
                rows=rows if rows is not None else [{"name": "Tommy Davis"}],
                sql="SELECT * FROM batting WHERE yearID = ?",
                data_manifest={"dataset": {}, "files": [], "coverage": {}, "download": {}},
            )
        ],
    )


def test_load_cases_reads_yaml_manifest(tmp_path: Path):
    path = tmp_path / "questions.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "questions": [
                    {
                        "id": "stat_rbi_1962",
                        "question": "who had the most RBIs in 1962",
                        "intent": "stat_query",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    cases = load_cases(path)

    assert [case.id for case in cases] == ["stat_rbi_1962"]
    assert cases[0].question == "who had the most RBIs in 1962"


def test_selected_cases_defaults_to_ci_safe_stat_queries():
    cases = load_cases()

    selected_ids = {case.id for case in selected_cases(cases)}

    assert "stat_rbi_1962" in selected_ids
    assert "player_bio_babe_ruth" not in selected_ids
    assert "freeform_braves_1936" not in selected_ids


def test_ci_safe_flag_allows_explicit_live_opt_in():
    cases = [
        load_cases()[0],
        load_cases()[0].__class__(
            id="bio",
            question="who was Babe Ruth",
            spec={
                "id": "bio",
                "question": "who was Babe Ruth",
                "intent": "player_biography",
                "required_sources": ["chroma"],
                "ci_safe": True,
            },
        ),
    ]

    assert [case.id for case in selected_cases(cases)] == ["stat_rbi_1962", "bio"]


def test_selected_strategy_cases_filters_to_retrieval_relevant_cases():
    stat_case = load_cases()[0]
    bio_case = stat_case.__class__(
        id="bio",
        question="who was Babe Ruth",
        spec={
            "id": "bio",
            "question": "who was Babe Ruth",
            "intent": "player_biography",
            "required_sources": ["chroma"],
        },
    )
    unsupported_case = stat_case.__class__(
        id="ambiguous",
        question="who was Smith",
        spec={
            "id": "ambiguous",
            "question": "who was Smith",
            "expected_unsupported": True,
        },
    )

    assert selected_strategy_cases([stat_case, bio_case, unsupported_case]) == [bio_case]


def test_validate_case_checks_core_expectations():
    case = load_cases()[0]

    failures = validate_case(case, _answer(answer="Tommy Davis finished with 153 RBI"))

    assert failures == []


def test_validate_case_reports_mismatches():
    case = load_cases()[0]

    failures = validate_case(
        case,
        _answer(answer="not enough", intent="general_explanation", source_type="chroma", rows=[]),
    )

    assert "intent: expected 'stat_query', got 'general_explanation'" in failures
    assert "answer missing substring 'Davis'" in failures
    assert "sources missing required type 'duckdb'" in failures
    assert "row count: expected >= 1, got 0" in failures


def test_run_cases_uses_mocked_answer_for_selected_cases_only():
    cases = [
        load_cases()[0],
        load_cases()[0].__class__(
            id="bio",
            question="who was Babe Ruth",
            spec={
                "id": "bio",
                "question": "who was Babe Ruth",
                "intent": "player_biography",
                "required_sources": ["chroma"],
            },
        ),
    ]
    asked: list[str] = []

    def answer_fn(question: str) -> StructuredAnswer:
        asked.append(question)
        return _answer(answer="Tommy Davis finished with 153 RBI")

    result = run_cases(cases, answer_fn=answer_fn)

    assert result.ok
    assert result.attempted == 1
    assert len(result.skipped) == 1
    assert asked == ["who had the most RBIs in 1962"]


def test_run_strategy_cases_runs_each_strategy_with_answer_factory():
    base_case = load_cases()[0]
    cases = [
        base_case,
        base_case.__class__(
            id="bio",
            question="who was Babe Ruth",
            spec={
                "id": "bio",
                "question": "who was Babe Ruth",
                "intent": "player_biography",
                "required_sources": ["chroma"],
                "expected_answer_contains": ["Babe Ruth"],
            },
        ),
    ]
    calls: list[tuple[str, str]] = []

    def answer_factory(strategy: str):
        def answer_fn(question: str) -> StructuredAnswer:
            calls.append((strategy, question))
            return _answer(
                answer="Babe Ruth biography",
                intent="player_biography",
                source_type="chroma",
            )

        return answer_fn

    results = run_strategy_cases(
        cases,
        strategies=["semantic_chroma", "hybrid_player_bio"],
        answer_factory=answer_factory,
        include_live=True,
    )

    assert list(results) == ["semantic_chroma", "hybrid_player_bio"]
    assert all(result.ok for result in results.values())
    assert calls == [
        ("semantic_chroma", "who was Babe Ruth"),
        ("hybrid_player_bio", "who was Babe Ruth"),
    ]


def test_format_strategy_summary_renders_table():
    result = StrategyRunResult()
    result.by_strategy["exact_player_id"] = run_cases(
        [load_cases()[0]],
        answer_fn=lambda _question: _answer(answer="Tommy Davis finished with 153 RBI"),
    )
    result.by_strategy["semantic_chroma"] = run_cases(
        [load_cases()[0]],
        answer_fn=lambda _question: _answer(answer="wrong", rows=[]),
    )

    summary = format_strategy_summary(result)

    assert "strategy" in summary
    assert "exact_player_id" in summary
    assert "semantic_chroma" in summary
    assert "passed" in summary
    assert "failed" in summary
    assert "skipped" in summary
