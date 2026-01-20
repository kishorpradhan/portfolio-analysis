# Portfolio Agent

An agentic portfolio analytics tool that evaluates trades, positions, and portfolio performance.

Agentic portfolio analysis with planning, local data preparation, LLM-based code generation, and sandbox execution.

## Architecture

Data preparation: `src/portfolio_agent/clients/robinhood_client/get_trades_robinhood.py`
This extracts three files: trades (transactions), positions, and dividends.

Portfolio Q&A:

Pipeline stages:
1) Planner: converts the user question into required datasets and the ticker prices needed (using the Yahoo Finance API).
2) Data prep: selected files are copied to `data/processed/<run_id>/`. Market data (end-of-day prices for tickers) are also copied into this directory.
3) Code generation: generates analysis code using schemas and sample rows.
4) Sandbox execution: runs generated code against the processed artifacts.
5) Final answer: summarizes JSON results with an LLM.

Key modules:
- Planner: `src/portfolio_agent/planner/planner.py`
- Data prep: `src/portfolio_agent/data_access/prep.py`
- Codegen: `src/portfolio_agent/agents/codegen.py`
- Sandbox: `src/portfolio_agent/agents/sandbox.py`
- Orchestrator: `src/portfolio_agent/workflows/portfolio_analysis.py`

## Quick Start

1) Install dependencies:

```
.venv-1/bin/pip install -e .
```

2) Run a question (full pipeline):

```
.venv-1/bin/portfolio-agent --question "Show me the daily share price of Tesla"
```

3) Run code generation + execution only:

```
.venv-1/bin/portfolio-agent --code-only --question "Compare my portfolio vs AMZN in 2024"
```

Logs are written to `logs/portfolio_agent.log`.

## Planner Module

The planner converts a user question into a technical plan describing:
- Required datasets (trades, positions, dividends)
- Required API calls (Yahoo historical prices)
- Tickers and date ranges

Planner prompt:
- `src/portfolio_agent/prompts/planner_prompt.j2`

Planner usage docs:
- `docs/planner.md`

## Data Files

Expected CSV files in `data/`:
- `trades.csv`
- `positions.csv`
- `dividends.csv`

Processed session artifacts are written to:
- `data/processed/<run_id>/`

Yahoo history is consolidated into:
- `ticker_eod_price_history.csv`

## Schemas

Dataset schemas are defined in YAML and used for code generation:
- `data/schemas/*.yml`

When schema metadata is missing, the code generation prompt is augmented with inferred column types and sample rows from the CSVs.
