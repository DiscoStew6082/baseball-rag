"""Structured answer and provenance models for grounded responses."""

from dataclasses import dataclass, field
from typing import Any, Literal

SourceType = Literal["duckdb", "chroma", "system"]


@dataclass
class SourceRecord:
    """A single source used to ground an answer."""

    type: SourceType
    label: str
    detail: str | None = None
    sql: str | None = None
    rows: list[dict[str, Any]] = field(default_factory=list)
    columns: list[str] = field(default_factory=list)
    score: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable source record."""
        return {
            "type": self.type,
            "label": self.label,
            "detail": self.detail,
            "sql": self.sql,
            "rows": self.rows,
            "columns": self.columns,
            "score": self.score,
        }


@dataclass
class StructuredAnswer:
    """Grounded answer returned by the shared answer service."""

    answer: str
    intent: str
    sources: list[SourceRecord] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    unsupported: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable answer payload."""
        return {
            "answer": self.answer,
            "intent": self.intent,
            "sources": [source.to_dict() for source in self.sources],
            "warnings": self.warnings,
            "unsupported": self.unsupported,
        }
