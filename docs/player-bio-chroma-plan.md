# Plan: Populate ChromaDB with Player Bios for Full-Coverage RAG

## Goal

Replace the current useless 5-HOF-bio ChromaDB layer with a **one-doc-per-player corpus** generated from DuckDB at ingest time. This enables natural language queries about any of the ~24,000 MLB players in the dataset — including random guys from the 1950s.

Example queries this fixes:
- "what teams did Wally Pipp play for?"
- "who was Tiant and what years did he play?"
- "tell me about Gil English"
- "how many RBIs did Ruben Jones have?"

## Architecture After

```
User Question
     │
     ▼
┌─────────────┐
│   Router    │ → intent: stat_query | general_explanation | player_biography
└─────────────┘
     │
     ├─── stat_query ──────────────────► DuckDB (existing)
     │                                     get_stat_leaders / get_player_stat
     │
     ├─── general_explanation ──────────► ChromaDB (stat defs + HOF bios, existing)
     │                                     retrieve() → LLM generation
     │
     └─── player_biography ─────────────► ChromaDB (player bio docs, NEW)
                                         retrieve() → "Tell me about {name}" prompt
```

## Data Flow

At **ingest time**:
1. Query DuckDB for all distinct players with batting records
2. For each player, generate a markdown bio doc from their data
3. Index all ~24k bios into ChromaDB alongside existing stat definitions and HOF bios

At **query time**:
1. Router detects `player_biography` intent (has `player_name`, not a stat query)
2. Retrieve that player's bio from ChromaDB by name match
3. LLM synthesizes answer using the retrieved bio as context

---

## Step 1: Build Player Bio Generator

**File:** `src/baseball_rag/corpus/player_bios.py` (new)

### Test first (`tests/test_player_bio.py`)
```python
def test_build_player_bio():
    """Matt Olson should have ATL and OAK teams, correct year range."""
    bio = build_player_bio("olso001mat-01")  # retro ID or similar
    assert "Atlanta" in bio or "ATL" in bio
    assert "Oakland" in bio or "OAK" in bio
```

### Implementation: `build_player_bio(player_id) -> str`

Query DuckDB for a single player and return a markdown document:

```
---
title: Matt Olson
player_id: olso001mat-01
category: player_biography
---

# Matt Olson

**Full name:** Matthew James Olson
**Birth:** September 20, 1988 (Tampa, FL) — if available in People table
**Bats/Throws:** R/R
**Primary position:** First Base

## Career Summary

Matt Olson played for the **Oakland Athletics** (2016–2021, 7 seasons: 2016-2021) and
the **Atlanta Braves** (2022–present).

Career span: 2016–2025 (10+ seasons through 2024)

Notable: MLB home run leader (2023), Silver Slugger Award winner.
```

Fields to pull from DuckDB:
- `people`: nameFirst, nameLast, birthCity, birthState, bats, throws
- `batting` per player: DISTINCT teamID + MIN/MAX yearID per team

### Key design decision
The bio is a **static snapshot** generated at ingest time. It won't reflect mid-season updates but that's fine — re-ingest periodically or on demand.

---

## Step 2: Add Player Biography Routing Intent

**File:** `src/baseball_rag/routing/query_router.py`

Update `_ROUTING_PROMPT` to detect player biography queries:

```
- intent: 'player_biography' if asking about a specific player's career history,
  teams, biographical info (e.g. "who was Wally Pipp", "what teams did he play for",
  "tell me about this player"); null otherwise
```

Update `RouteResult` dataclass:
```python
@dataclass
class RouteResult:
    intent: str       # now includes "player_biography"
    ...
```

Existing `"general_explanation"` stays for pure baseball knowledge questions ("what is the infield fly rule").

---

## Step 3: Ingest All Player Bios into ChromaDB

**File:** `src/baseball_rag/corpus/ingest.py`

Modify `build_index()`:
1. Keep existing stat_defs + hof_bios path (lines unchanged)
2. Add: query DuckDB for all distinct playerIDs in batting table
3. For each playerID, call `build_player_bio(player_id)` to get doc text
4. Add to ChromaDB with id=`player:{playerID}`, category=`player_biography`

**Estimated time:** ~24k players × simple DuckDB query + string concat = fast (<1 min)

**CLI flag:** `--include-player-bios` (default: True) so CI tests can skip if needed.

---

## Step 4: Add Biographical Retrieval + Generation Path

**File:** `src/baseball_rag/cli.py`

In the `answer()` function, add a third branch after `stat_query`:

```python
elif decision.intent == "player_biography":
    from baseball_rag.db.duckdb_schema import get_duckdb
    conn = get_duckdb()

    # Try to find player by name in DuckDB first
    # Then retrieve their bio doc from ChromaDB
    ...
```

**Prompt template (new):** `PLAYER_BIO_TEMPLATE` in `generation/prompt.py`

---

## Step 5: Update Documentation

- `docs/architecture.md`: Update data layer diagram, routing section
- `docs/corpus.md`: Remove reference to "15 documents" — now ~24k+ player bios + stat defs + HOF bios
- `README.md`: Update example queries to include player biography examples

---

## File Changes Summary

| Step | File | Change |
|------|------|--------|
| 1 (new) | `src/baseball_rag/corpus/player_bios.py` | Bio generation from DuckDB |
| 1 (test) | `tests/test_player_bio.py` | Tests for bio generator |
| 2 | `src/baseball_rag/routing/query_router.py` | Add player_biography intent |
| 3 | `src/baseball_rag/corpus/ingest.py` | Index all player bios |
| 4 | `src/baseball_rag/cli.py` | Handle player_biography route |
| 4 (new) | `src/baseball_rag/generation/prompt.py` | Add bio prompt template |

## Verification

```bash
# Re-ingest corpus with player bios
uv run python -m baseball_rag.corpus.ingest

# Test queries
uv run baseball-rag "what teams did Wally Pipp play for?"
uv run baseball-rag "who was Gil English and what years did he play?"
uv run baseball-rag "tell me about Tiant"
```