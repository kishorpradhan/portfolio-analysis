from __future__ import annotations

import datetime as dt
from typing import Iterable, Optional

import pandas as pd

from portfolio_agent.config.settings import get_settings
from portfolio_agent.utils.logger import get_logger

logger = get_logger(__name__)


class YahooFinanceUnavailable(Exception):
    """Raised when yfinance is not installed or cannot fetch data."""


def _coerce_date(value: str | dt.date | dt.datetime) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def get_history(
    symbol: str,
    start: str | dt.date | dt.datetime,
    end: str | dt.date | dt.datetime,
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a symbol between start and end dates (inclusive).
    Returns a DataFrame with columns: date, open, high, low, close, volume.
    """
    try:
        import yfinance as yf  # type: ignore
    except Exception as exc:  # pragma: no cover - import guard
        raise YahooFinanceUnavailable(
            "yfinance is required for historical price fetches. Install yfinance."
        ) from exc

    symbol = (symbol or "").strip()
    if not symbol:
        raise ValueError("symbol is required")

    start_date = _coerce_date(start)
    end_date = _coerce_date(end)
    if start_date > end_date:
        raise ValueError("start must be on or before end")

    if interval not in {"1d", "1wk", "1mo"}:
        raise ValueError("interval must be one of {'1d','1wk','1mo'}")

    logger.info("Fetching history for %s from %s to %s (interval=%s)", symbol, start_date, end_date, interval)
    ticker = yf.Ticker(symbol)
    df = ticker.history(start=start_date, end=end_date + dt.timedelta(days=1), interval=interval)

    if df.empty:
        raise ValueError(f"No historical data returned for {symbol} in range {start_date} to {end_date}")

    df = df.reset_index()
    # Normalize column casing and names
    df = df.rename(
        columns={
            "Date": "date",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    # Keep expected columns only
    expected = ["date", "open", "high", "low", "close", "volume"]
    df = df[expected]

    return df
