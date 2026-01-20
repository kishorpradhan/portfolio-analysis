from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from portfolio_agent.config.settings import get_settings
from portfolio_agent.utils.logger import get_logger

logger = get_logger(__name__)


class PositionsNotFound(Exception):
    """Raised when expected positions CSV is missing."""


def load_positions(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load positions CSV.
    Expected columns: symbol, quantity, average_price, price, pe_ratio, list_date, sector.
    """
    settings = get_settings()
    csv_path = path or (settings.data_dir / "portfolio.csv")
    if not csv_path.exists():
        raise PositionsNotFound(f"Positions file not found at {csv_path}")

    df = pd.read_csv(csv_path)
    return df.reset_index(drop=True)
