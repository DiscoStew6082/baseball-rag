# Corpus

The corpus is the knowledge base that grounds LLM responses in factual baseball content. It lives in `src/baseball_rag/corpus/`.

## Structure

```
corpus/
├── __init__.py              # Path constants + helpers: get_stat_defs(), get_hof_bios()
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

At query time, ChromaDB performs cosine-similarity search over all corpus documents. Top-k results are passed to the prompt layer.

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
uv run python -m baseball_rag.corpus.ingest
```

### A new Hall of Fame bio

1. Create `src/baseball_rag/corpus/hof/{Player_Name}.md`
2. Include frontmatter with `title` and `category: hof_bio`
3. Rebuild the index as above

## Current Corpus Size

15 documents total (10 stat definitions + 5 HOF bios). The index is wiped and rebuilt on every run for reproducibility.