"""Query routing - classify user intent and extract structured arguments via LLM."""

import json
from dataclasses import dataclass

from baseball_rag.arch.tracing import traced


@dataclass
class RouteResult:
    """Parsed result from classifying a user query."""

    intent: str  # "stat_query" | "player_biography" | "freeform_query" | "general_explanation"
    stat: str | None  # e.g., "RBI", "HR"
    year: int | None  # e.g., 1962
    position: str | None  # e.g., "OF", "CF"
    player_name: str | None = None  # e.g., "Mike Trout"
    raw_question: str = ""  # original question text


_ROUTING_PROMPT = (
    "You are a baseball query classifier. Given a user question, "
    "respond with ONLY valid JSON (no markdown, no explanation).\n\n"
    "Identify:\n"
    "- intent: 'stat_query' if asking about a specific stat for a player or "
    "league-wide leaders; 'player_biography' if asking about a player's "
    "career history, teams, biographical info (e.g., 'who was Wally Pipp', "
    "'what teams did he play for', 'tell me about this player'); "
    "'freeform_query' if the question requires querying the database with "
    "arbitrary filters or conditions not covered by stat_query; "
    "'general_explanation' only when none of the above apply\n"
    "- stat: the canonical stat name if detectable (RBI, HR, AVG, ERA, WHIP, SO, SB, 2B,"
    " 3B, W, L, PO, etc.); null otherwise\n"
    "- year: the season/year mentioned (integer); null if none\n"
    "- position: 'OF' or 'CF' etc. if a defensive position is specified; null otherwise\n"
    "- player_name: the full player name if one is mentioned "
    '(e.g., "Ronald Acuna Jr."); null otherwise\n\n'
    "Stat name mapping: RBI, HR, AVG, ERA, WHIP, SO, SB, 2B, 3B, W, L, PO, BB, H, IP, K\n"
    "Never guess — return null for any field not explicitly in the question.\n\n"
    "Examples:\n"
    '- "who led MLB in RBIs in 2022" → '
    '{{"intent":"stat_query","stat":"RBI","year":2022,"position":null,'
    '"player_name":null}}\n'
    '- "how many HRs did Aaron Judge have last year" → '
    '{{"intent":"stat_query","stat":"HR","year":null,"position":null,'
    '"player_name":"Aaron Judge"}}\n'
    '- "who was Wally Pipp" → '
    '{{"intent":"player_biography","stat":null,"year":null,"position":null,'
    '"player_name":"Wally Pipp"}}\n'
    '- "what teams did he play for" → '
    '{{"intent":"player_biography","stat":null,"year":null,"position":null,'
    '"player_name":null}}\n'
    '- "tell me about this player" → '
    '{{"intent":"player_biography","stat":null,"year":null,"position":null,'
    '"player_name":null}}\n'
    "- 'what is a forced play in baseball' → "
    '{{"intent":"general_explanation","stat":null,"year":null,'
    '"position":null,"player_name":null}}\n'
    '- "who played for the Braves in 1936" → '
    '{{"intent":"freeform_query","stat":null,"year":1936,"position":null,'
    '"player_name":null}}\n'
    '- "list all pitchers with over 300 wins" → '
    '{{"intent":"freeform_query","stat":null,"year":null,"position":null,'
    '"player_name":null}}\n'
    "\nQuestion: {question}"
)


def _extract_json_blocks(text: str) -> list[tuple[int, int]]:
    """Find all candidate JSON objects in text (start brace → balanced end brace).

    Returns list of (start, end+1) byte positions for each {...} block found.
    """
    blocks = []
    depth = 0
    start = -1
    for i, c in enumerate(text):
        if c == "{":
            if depth == 0:
                start = i
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0 and start >= 0:
                blocks.append((start, i + 1))
    return blocks


def _parse_llm_json(raw: str) -> dict | None:
    """Parse LLM JSON response.

    Gemma 4 often wraps its output in a reasoning/thinking block even when
    instructed to return only JSON. We find the {...} block that actually
    parses as valid RouteResult-shaped JSON.
    """
    text = raw.strip()

    # Try stripping markdown fences first
    for fence in ("```json", "```"):
        if text.startswith(fence):
            text = text[len(fence) :].strip().rstrip("`").strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find all {...} blocks and try each one
    for start, end in _extract_json_blocks(text):
        candidate = text[start:end]
        try:
            data = json.loads(candidate)
            # Sanity-check: must have 'intent' field
            if isinstance(data, dict) and "intent" in data:
                return data
        except json.JSONDecodeError:
            continue

    return None


@traced(component_id="query-router", label="Route Query")
def route(question: str) -> RouteResult:
    """Classify a natural language question using the LLM.

    Falls back to a simple heuristic if LM Studio is unavailable.
    """
    try:
        from baseball_rag.generation.llm import make_request

        prompt = _ROUTING_PROMPT.format(question=question)
        response = make_request(prompt, max_tokens=500, temperature=0.1)
        data = _parse_llm_json(response.content)

        if data and data.get("intent") in (
            "stat_query",
            "player_biography",
            "freeform_query",
            "general_explanation",
        ):
            return RouteResult(
                intent=data["intent"],
                stat=data.get("stat"),
                year=data.get("year"),
                position=data.get("position"),
                player_name=data.get("player_name"),
                raw_question=question,
            )
    except ConnectionError:
        pass  # Fall through to heuristic

    # LM Studio unavailable or LLM returned garbled — use safe fallback
    return _heuristic_route(question)


def _heuristic_route(question: str) -> RouteResult:
    """Fallback routing when the LLM is unavailable.

    Only handles explicit leaderboard queries (who had most/least/top N).
    Player-specific stat lookups always go through the LLM path.
    """
    import re

    # Only classify as stat_query if it's clearly a league-wide leader request
    leader_re = re.compile(r"\b(most|least|highest|lowest|lead|leader|leaders|top|bottom)\b")
    is_leaderboard = bool(leader_re.search(question))

    # Extract year (4-digit only for fallback)
    year: int | None = None
    m = re.search(r"\b(20\d{2}|19\d{2})\b", question)
    if m:
        year = int(m.group(1))

    return RouteResult(
        intent="stat_query" if is_leaderboard else "general_explanation",
        stat=None,
        year=year,
        position=None,
        player_name=None,  # Never extract players via regex — too error-prone
        raw_question=question,
    )
