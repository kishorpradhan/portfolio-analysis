from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from portfolio_agent.config.settings import get_settings
from portfolio_agent.utils.logger import get_logger

logger = get_logger(__name__)


class DividendsNotFound(Exception):
    """Raised when expected dividends CSV is missing."""


def load_dividends(
    symbol: Optional[str] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load dividends CSV and optionally filter by ticker and date range.
    Expected columns: date, ticker, amount, state.
    """
    settings = get_settings()
    csv_path = path or (settings.data_dir / "dividends.csv")
    if not csv_path.exists():
        raise DividendsNotFound(f"Dividends file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if symbol:
        if "ticker" in df.columns:
            df = df[df["ticker"].str.upper() == symbol.upper()]

    if start is not None:
        df = df[df["date"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["date"] <= pd.to_datetime(end)]

    return df.reset_index(drop=True)
