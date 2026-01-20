from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from portfolio_agent.config.settings import get_settings
from portfolio_agent.utils.logger import get_logger

logger = get_logger(__name__)


def ensure_processed_dir(symbol: str | None = None) -> Path:
    settings = get_settings()
    base = settings.data_dir / "processed"
    if symbol:
        base = base / symbol.upper()
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_records_to_csv(records: List[dict], dest: Path, columns: List[str] | None = None) -> Path:
    df = pd.DataFrame.from_records(records)
    if columns:
        df = df[[col for col in columns if col in df.columns]]
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(dest, index=False)
    logger.info("Wrote %d rows to %s", len(df), dest)
    return dest


def infer_schema(path: Path) -> Dict[str, str]:
    df = pd.read_csv(path, nrows=1)
    return {col: str(dtype) for col, dtype in df.dtypes.items()}


def persist_artifacts(
    artifacts: Dict[str, list],
    symbol: str | None = None,
) -> Dict[str, str]:
    """
    Write artifacts (e.g., trades/history records) to CSVs in processed dir.
    Returns mapping of artifact name to file path.
    """
    out: Dict[str, str] = {}
    base = ensure_processed_dir(symbol)

    for name, records in (artifacts or {}).items():
        if not records:
            continue
        dest = base / f"{symbol or name}_{name}.csv"
        write_records_to_csv(records, dest)
        out[f"tool_{name}"] = str(dest)

    return out
