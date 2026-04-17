"""Query routing - classify user intent and extract structured arguments."""
import datetime
import re
from dataclasses import dataclass


@dataclass
class RouteResult:
    """Parsed result from classifying a user query."""
    intent: str           # "stat_query" | "general_explanation"
    stat: str | None      # e.g., "RBI", "HR"
    year: int | None      # e.g., 1962
    position: str | None  # e.g., "OF", "CF"
    player_name: str | None = None  # e.g., "Mike Trout"
    raw_question: str = ""          # original question text


# Stat keywords that imply a stat query
STAT_KEYWORDS = {
    "rbi", "rbis",
    "home run", "hr", "hrs",
    "batting average", "avg", ".400", ".406",
    " ERA ", "whip", "strikeout", "so", "k",
    "win", "loss", "w-l", "wl",
    "stolen base", "sb", "steal",
    "double", "2b", "triple", "3b",
    "putout", "po", "assist", "a",
    "on-base", "ops", "slugging",
}

# Compiled pattern for stat-query detection
_LEADER_RE = re.compile(r"\b(most|least|highest|lowest|lead|leader|top|bottom)\b", re.I)


def _extract_year(text: str) -> int | None:
    """Pull a year out of the query text."""
    t = text.lower()
    # Handle "this year", "current year", "this season", "current season"
    if any(phrase in t for phrase in ("this year", "current year", "this season", "current season")):
        return datetime.datetime.now().year
    # Try 4-digit first
    m = re.search(r"\b(20\d{2}|19\d{2}|18\d{2})\b", text)
    if m:
        return int(m.group(1))
    # Fall back to '52, '62 style -- assumes 1900s or 2000s
    m = re.search(r"['\"](\d{2})\b", text)
    if m:
        y = int(m.group(1))
        return 1900 + y if y > 30 else 2000 + y
    return None


def _extract_stat(text: str) -> str | None:
    """Map query keywords to canonical stat names."""
    t = text.lower()
    for kw, stat in [
        ("rbis", "RBI"), ("rbi", "RBI"),
        ("home run", "HR"), ("hrs", "HR"), ("hr", "HR"),
        (".400", "AVG"), (".406", "AVG"), ("batting average", "AVG"), ("avg", "AVG"),
        (" ERA ", "ERA"), ("era", "ERA"),
        ("whip", "WHIP"),
        ("strikeout", "SO"), ("strike outs", "SO"), ("so ", "SO"), ("k", "SO"),
        ("win", "W"), ("loss", "L"), ("w-l", "WL"),
        ("stolen base", "SB"), ("sb", "SB"), ("steal", "SB"),
        ("double", "2B"), ("2b", "2B"),
        ("triple", "3B"), ("3b", "3B"),
        ("putout", "PO"), ("po ", "PO"), ("catch", "PO"),
    ]:
        if kw in t:
            return stat
    return None


def _extract_player(text: str) -> str | None:
    """Pull a player name out of the query text.

    Looks for patterns like 'does Mike Trout have' or 'did Aaron Judge have'
    where NAME is Title Case word(s).
    """
    m = re.search(r"(?:does|did|has|had|are)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)", text)
    if m:
        return m.group(1)
    return None


def route(question: str) -> RouteResult:
    """Classify a natural language question into intent + extracted arguments.

    Simple keyword-based classifier -- good enough for portfolio demo.
    For production, replace with an LLM call that returns JSON.

    Returns:
        RouteResult with intent and any extracted args.
    """
    stat = _extract_stat(question)
    year = _extract_year(question)
    player_name = _extract_player(question)

    # Outfield / position detection
    pos: str | None = None
    if "outfield" in question.lower() or re.search(r"\bOF\b", question):
        pos = "OF"

    is_stat_query = stat is not None or bool(_LEADER_RE.search(question))

    return RouteResult(
        intent="stat_query" if is_stat_query else "general_explanation",
        stat=stat,
        year=year,
        position=pos,
        player_name=player_name,
        raw_question=question,
    )
