"""Central registry of SQL-addressable baseball statistics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

StatTable = Literal["batting", "pitching", "fielding"]


@dataclass(frozen=True)
class StatDefinition:
    """A whitelisted statistic that can be referenced in generated SQL."""

    canonical: str
    table: StatTable
    column: str | None
    sql_expr: str | None = None
    min_sample_clause: str | None = None
    higher_is_better: bool = True

    def expression(self, alias: str) -> str:
        """Return a SQL expression for this stat using a trusted table alias."""
        if self.sql_expr is not None:
            return self.sql_expr.format(alias=alias)
        if self.column is None:
            raise ValueError(f"Stat {self.canonical} has no column or expression")
        return f"{alias}.{quote_identifier(self.column)}"


def quote_identifier(identifier: str) -> str:
    """Quote a trusted SQL identifier for DuckDB."""
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


_REGISTRY: dict[str, StatDefinition] = {
    "HR": StatDefinition("HR", "batting", "HR"),
    "RBI": StatDefinition("RBI", "batting", "RBI"),
    "H": StatDefinition("H", "batting", "H"),
    "AB": StatDefinition("AB", "batting", "AB"),
    "R": StatDefinition("R", "batting", "R"),
    "2B": StatDefinition("2B", "batting", "2B"),
    "3B": StatDefinition("3B", "batting", "3B"),
    "SB": StatDefinition("SB", "batting", "SB"),
    "BB": StatDefinition("BB", "batting", "BB"),
    "SO": StatDefinition("SO", "batting", "SO"),
    "AVG": StatDefinition(
        "AVG",
        "batting",
        None,
        sql_expr="CAST({alias}.H AS DOUBLE) / NULLIF({alias}.AB, 0)",
        min_sample_clause="{alias}.AB >= 100",
    ),
    "W": StatDefinition("W", "pitching", "W"),
    "L": StatDefinition("L", "pitching", "L"),
    "G": StatDefinition("G", "pitching", "G"),
    "GS": StatDefinition("GS", "pitching", "GS"),
    "SV": StatDefinition("SV", "pitching", "SV"),
    "ERA": StatDefinition(
        "ERA",
        "pitching",
        "ERA",
        min_sample_clause="{alias}.IPouts >= 300",
        higher_is_better=False,
    ),
    "PO": StatDefinition("PO", "fielding", "PO"),
}

_ALIASES = {
    "K": "SO",
    "STRIKEOUTS": "SO",
    "HITS": "H",
    "HOMER": "HR",
    "HOMERS": "HR",
    "HOME_RUNS": "HR",
    "RUNS_BATTED_IN": "RBI",
    "BAT_AVG": "AVG",
    "BATTING_AVERAGE": "AVG",
}


def get_stat(stat: str, *, table: StatTable | None = None) -> StatDefinition:
    """Return a whitelisted stat definition or raise ValueError."""
    canonical = normalize_stat(stat)
    definition = _REGISTRY.get(canonical)
    if definition is None:
        supported = ", ".join(supported_stats())
        raise ValueError(f"Unsupported stat '{stat}'. Supported stats: {supported}")
    if table is not None and definition.table != table:
        raise ValueError(f"Stat '{canonical}' belongs to {definition.table}, not {table}")
    return definition


def normalize_stat(stat: str) -> str:
    """Normalize common user/model stat spellings to registry keys."""
    key = stat.strip().upper().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(key, key)


def supported_stats(table: StatTable | None = None) -> list[str]:
    """Return supported canonical stat names."""
    return sorted(
        name for name, definition in _REGISTRY.items() if table is None or definition.table == table
    )


def supported_tables() -> set[StatTable]:
    """Return tables that may be used by generated query specs."""
    return {"batting", "pitching", "fielding"}
