from __future__ import annotations

from typing import Optional

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from portfolio_agent.clients.yahoo import get_history
from portfolio_agent.data_access.dividends import load_dividends
from portfolio_agent.data_access.positions import load_positions
from portfolio_agent.data_access.trades import load_trades
from portfolio_agent.tools.summarizer import summarize_json


class HistoryInput(BaseModel):
    symbol: str = Field(..., description="Stock ticker symbol, e.g., TSLA")
    start: str = Field(..., description="Start date (YYYY-MM-DD)")
    end: str = Field(..., description="End date (YYYY-MM-DD)")
    interval: str = Field(
        default="1d",
        description="Price interval: 1d, 1wk, or 1mo",
    )


class LoadTradesInput(BaseModel):
    symbol: Optional[str] = Field(None, description="Stock ticker to filter, e.g., TSLA")
    start: Optional[str] = Field(None, description="Start datetime (ISO 8601)")
    end: Optional[str] = Field(None, description="End datetime (ISO 8601)")


class LoadDividendsInput(BaseModel):
    symbol: Optional[str] = Field(None, description="Ticker to filter, e.g., TSLA")
    start: Optional[str] = Field(None, description="Start datetime (ISO 8601)")
    end: Optional[str] = Field(None, description="End datetime (ISO 8601)")


class LoadPositionsInput(BaseModel):
    pass


class SummarizeInput(BaseModel):
    payload: object = Field(..., description="JSON-like data to summarize (dict/list/JSON string).")
    max_bullets: int = Field(default=5, description="Maximum bullet points in the summary.")


def get_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="get_history",
            description="Fetch historical OHLCV prices for a symbol between start/end dates.",
            func=lambda symbol, start, end, interval="1d": get_history(symbol, start, end, interval).to_dict(orient="records"),
            args_schema=HistoryInput,
        ),
        StructuredTool.from_function(
            name="load_trades",
            description="Load trades/orders from CSV, optionally filtered by symbol and date range.",
            func=lambda symbol=None, start=None, end=None: load_trades(symbol=symbol, start=start, end=end).to_dict(orient="records"),
            args_schema=LoadTradesInput,
        ),
        StructuredTool.from_function(
            name="load_dividends",
            description="Load dividends from CSV, optionally filtered by ticker and date range.",
            func=lambda symbol=None, start=None, end=None: load_dividends(symbol=symbol, start=start, end=end).to_dict(orient="records"),
            args_schema=LoadDividendsInput,
        ),
        StructuredTool.from_function(
            name="load_positions",
            description="Load positions from CSV.",
            func=lambda: load_positions().to_dict(orient="records"),
            args_schema=LoadPositionsInput,
        ),
        StructuredTool.from_function(
            name="summarize_json",
            description="Summarize JSON-like metrics or records into concise bullets using the configured LLM.",
            func=lambda payload, max_bullets=5: summarize_json(payload, max_bullets=max_bullets),
            args_schema=SummarizeInput,
        ),
    ]
