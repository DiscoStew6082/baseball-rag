# Corpus

The corpus is the knowledge base that grounds LLM responses in factual baseball content. It lives in `src/baseball_rag/corpus/`.

## Structure

```
corpus/
├── __init__.py              # Path constants + helpers: get_stat_defs(), get_hof_bios()
├── diagnostics.py           # Index/manifest/environment diagnostics
├── frontmatter.py           # YAML frontmatter parser
├── ingest.py                # ChromaDB index builder (build_index)
├── stat_definitions/        # 10 markdown files — one per stat
│   ├── AVG.md
│   ├── BB.md
│   ├── 2B.md
│   ├── ERA.md
│   ├── HR.md
│   ├── OPS.md
│   ├── PO.md
│   ├── RBI.md
│   ├── SB.md
│   └── WHIP.md
└── hof/                     # 5 Hall of Fame player biographies
    ├── Babe_Ruth.md
    ├── Hank_Aaron.md
    ├── Mickey_Mantle.md
    ├── Ted_Williams.md
    └── Willie_Mays.md
```

## Document Format

Every corpus document uses YAML frontmatter:

```markdown
---
title: Home Runs (HR)
category: stat_definition
tags:
  - hitting
  - power
---

A home run occurs when a batter hits the ball over the outfield fence...
The dead-ball era (roughly 1900-1919) saw dramatically lower home run totals...
```

### Frontmatter Fields

| Field | Required | Purpose |
|-------|----------|---------|
| `title` | Yes | Display name used in LLM citations `[Source: HR.md]` — should match the filename stem |
| `category` | Yes | Controls routing: `stat_definition` or `hof_bio` |
| `tags` | No | Freeform labels for future filtering / expansion |

### Body

Prose content explaining the concept. Supports multiple paragraphs, historical context, notable records, and player references.

## How Documents Feed Into RAG

### 1. Parsing (`frontmatter.py`)

```python
parse_frontmatter(content)
  → {"metadata": {"title": ..., "category": ..., "tags": [...]}, "body": "..."}
```

Splits on the first `---...---` fence, parses YAML with `yaml.safe_load`, returns metadata dict + body string.

### 2. Ingestion (`ingest.py` → `build_index()`)

```python
for path in [*get_stat_defs(), *get_hof_bios()]:
    result = parse_frontmatter(path.read_text())
    text = f"{result['metadata']['title']}\n\n{result['body'].strip()}"
```

Each document is formatted as:

```
{title}

{body}
```

Then added to ChromaDB with:

| ChromaDB Field | Value |
|----------------|-------|
| `id` | Filename stem (e.g., `Babe_Ruth`) |
| `text` | Title + double-newline + body |
| `metadata.source` | Full filename (`Babe_Ruth.md`) |
| `metadata.category` | From frontmatter |
| `metadata.title` | From frontmatter |

### 3. Retrieval

At query time, ChromaDB performs cosine-similarity search over indexed corpus
documents. Top-k results are passed to the prompt layer. Chroma's files under
`data/` are generated local state; the durable source is this Markdown corpus.

For generated player profiles, player-name questions first resolve the local
`playerID` from DuckDB. Retrieval then applies a Chroma metadata filter such as
`where={"player_id": "ruthba01"}` before falling back to semantic search. This
keeps ambiguous names from silently retrieving the wrong player.

### 4. Prompt Grounding (`generation/prompt.py`)

Retrieved chunks appear in the prompt as:

```
[Source: HR.md]
Home Runs (HR)
A home run occurs when a batter hits the ball over the outfield fence...

---
Question: who had the most home runs in 1920
```

The LLM is instructed to cite sources explicitly: `[Source: HR.md]`.

## Adding New Documents

### A new stat definition

1. Create `src/baseball_rag/corpus/stat_definitions/{STAT_NAME}.md`
2. Include frontmatter with `title`, `category: stat_definition`, and optionally `tags`
3. Rebuild the index:

```bash
uv run python -m baseball_rag.corpus --static-only
```

### A new Hall of Fame bio

1. Create `src/baseball_rag/corpus/hof/{Player_Name}.md`
2. Include frontmatter with `title` and `category: hof_bio`
3. Rebuild the index as above

## Current Corpus Size

15 checked-in Markdown documents total (10 stat definitions + 5 HOF bios). The
index is wiped and rebuilt on every run for reproducibility. A larger
experimental mode can also index generated player bios from DuckDB:

```bash
uv run python -m baseball_rag.corpus
```

Full player-profile builds also write `data/corpus_manifest.json`. That file is
generated and ignored by git. It records the Chroma collection name, static docs,
generated player profile count, document IDs, `player_id`, title, doc kind, and
source tables used to produce each profile.

Generated player profiles use frontmatter like:

```yaml
---
title: Babe Ruth
player_id: ruthba01
category: player_biography
doc_kind: generated_player_profile
source_tables:
  - people
  - batting
  - pitching
  - fielding
---
```

## Diagnostics

Print corpus diagnostics as JSON:

```bash
uv run python -m baseball_rag.corpus diagnostics --persist-dir data
```

The report includes:

- resolved `persist_dir`
- checked-in static corpus counts and file stems
- Chroma collection existence and indexed count
- Chroma metadata counts by `category` and `doc_kind`
- `corpus_manifest.json` presence, generated timestamp, and document counts
- `CHROMA_PERSIST_DIR`, `LMSTUDIO_BASE_URL`, and `LMSTUDIO_EMBEDDING_MODEL` hints

Diagnostics do not require a healthy index. Missing directories, missing
collections, missing manifests, and corrupt manifests are reported in the JSON
instead of raising.

During a full ingest, keep a second terminal open and run the same diagnostics
command against the target persist directory. Early or interrupted builds may
show a collection without a final manifest, or a manifest whose generated-player
count is still zero; that is expected until ingest finishes.

## Retrieval Strategy Benchmark

After building the full player-profile index, compare Chroma-backed retrieval
strategies without changing the corpus:

```bash
uv run python -m evals.questions --all-strategies --retrieval-only
```

This runs only eval cases where retrieval strategy can affect the answer and
prints a per-strategy pass/fail/skip summary.
