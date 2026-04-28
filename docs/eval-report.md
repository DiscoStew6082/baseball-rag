# Baseball RAG Eval Report

- Command: `python -m evals.questions --report docs/eval-report.md`
- Mode: answer
- Cases loaded: 61
- Attempted: 10
- Passed: 10
- Failed: 0
- Skipped: 51

## Service Requirements

- Deterministic/CI-safe mode was used; non-default cases were skipped. 51 case(s) are available behind `--include-live`; 28 skipped case(s) may require Chroma, corpus, and LLM services.

## Suite Coverage

- stat query: `stat_rbi_1962` - who had the most RBIs in 1962
- unsupported/guardrail: `stat_unsupported_column` - who led the league in vibes in 1999
- freeform SQL query: `stat_sql_injection_team` - who played for the Braves%' OR 1=1 -- in 1936
- player biography retrieval: `player_bio_babe_ruth` - who was Babe Ruth
- baseball explanation retrieval: `strategy_broad_bio_query_yankees_slugger` - which indexed player biography talks about a Yankees switch-hitting slugger

## Failed Cases

- None
