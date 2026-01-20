from __future__ import annotations

import pandas as pd
from pathlib import Path
from typing import Optional

from portfolio_agent.config.settings import get_settings
from portfolio_agent.utils.logger import get_logger

logger = get_logger(__name__)


class TradesNotFound(Exception):
    """Raised when expected trades/orders CSV is missing."""


def load_trades(
    symbol: Optional[str] = None,
    start: Optional[pd.Timestamp] = None,
    end: Optional[pd.Timestamp] = None,
    path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    Load trades/orders CSV and optionally filter by symbol and date range.
    Expected columns: id, symbol, side, quantity, price, created_at, state.
    """
    settings = get_settings()
    csv_path = path or (settings.data_dir / "orders.csv")
    if not csv_path.exists():
        raise TradesNotFound(f"Trades/orders file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    if "created_at" in df.columns:
        df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")

    if symbol:
        df = df[df["symbol"].str.upper() == symbol.upper()]

    if start is not None:
        df = df[df["created_at"] >= pd.to_datetime(start)]
    if end is not None:
        df = df[df["created_at"] <= pd.to_datetime(end)]

    return df.reset_index(drop=True)
