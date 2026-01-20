# Planner Module

This document explains how to run and test the planner module on its own.

## What the Planner Does

Given a user question, the planner:
- Loads dataset schemas from `src/portfolio_agent/config/file_metadata.yaml`.
- Renders `src/portfolio_agent/prompts/planner_prompt.j2`.
- Calls the configured LLM (Gemini or OpenAI).
- Returns a JSON plan describing the technical question, required files, tool actions, and API calls.

## Output Schema (Summary)

The planner returns JSON with these keys:
- `technical_question`
- `tickers`
- `date_range` (start/end or null)
- `required_files`
- `api_calls` (e.g., Yahoo price history)
- `tool_actions` (e.g., load_trades/load_dividends/load_positions)

## Run Planner Only

Run the planner directly from the repo root:

```
.venv-1/bin/python - <<'PY'
from portfolio_agent.planner.planner import run_planner
result = run_planner("How is my portfolio doing compared to Amazon since 2024")
print(result.model_dump())
PY
```

.venv-1/bin/python - <<'PY'
from portfolio_agent.planner.planner import run_planner
print(run_planner("Show me Tesla dividend income last year").model_dump())
PY




## Notes

- The planner requires network access for the LLM call.
- If network access is restricted, the planner will return an empty plan.
- Logs are written to `logs/portfolio_agent.log`.
