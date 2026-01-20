from __future__ import annotations

import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Mapping

import pandas as pd

from portfolio_agent.data_access.persist import ensure_processed_dir
from portfolio_agent.utils.logger import get_logger

logger = get_logger(__name__)


def _write_output(records: Any, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.suffix.lower() == ".json":
        with dest.open("w", encoding="utf-8") as handle:
            json.dump(records, handle, default=str, indent=2)
        return
    df = pd.DataFrame.from_records(records)
    df.to_csv(dest, index=False)


def _execute_tool(tool_name: str, tool_map: Mapping[str, Any], args: dict) -> Any:
    if tool_name not in tool_map:
        raise KeyError(f"Tool '{tool_name}' is not registered.")
    func = tool_map[tool_name]
    return func(**args)


def prepare_data(plan: Dict[str, Any], tool_map: Mapping[str, Any], run_id: str) -> Dict[str, Dict[str, Any]]:
    """
    Execute planner tool_actions and api_calls locally and persist outputs.
    Returns artifact records and file path mapping for codegen.
    """
    artifacts: Dict[str, list] = {}
    file_paths: Dict[str, str] = {}
    base_dir = ensure_processed_dir(run_id)

    required_files = plan.get("required_files") or []
    api_calls = plan.get("api_calls") or []

    logger.info(
        "Preparing data with required_files=%s api_calls=%s run_id=%s",
        required_files,
        [call.get("tool") for call in api_calls],
        run_id,
    )

    if "trades.csv" in required_files:
        dest = base_dir / "trades.csv"
        result = _execute_tool("load_trades", tool_map, {})
        records = result.to_dict(orient="records") if isinstance(result, pd.DataFrame) else list(result)
        artifacts["load_trades"] = records
        _write_output(records, dest)
        file_paths["trades.csv"] = str(dest)
        logger.info("Prepared trades: %d rows -> %s", len(records), dest)

    if "dividends.csv" in required_files:
        dest = base_dir / "dividends.csv"
        result = _execute_tool("load_dividends", tool_map, {})
        records = result.to_dict(orient="records") if isinstance(result, pd.DataFrame) else list(result)
        artifacts["load_dividends"] = records
        _write_output(records, dest)
        file_paths["dividends.csv"] = str(dest)
        logger.info("Prepared dividends: %d rows -> %s", len(records), dest)

    if "positions.csv" in required_files:
        dest = base_dir / "positions.csv"
        result = _execute_tool("load_positions", tool_map, {})
        records = result.to_dict(orient="records") if isinstance(result, pd.DataFrame) else list(result)
        artifacts["load_positions"] = records
        _write_output(records, dest)
        file_paths["positions.csv"] = str(dest)
        logger.info("Prepared positions: %d rows -> %s", len(records), dest)

    history_records: list[dict] = []
    history_tickers: list[str] = []
    history_start: Optional[str] = None
    history_end: Optional[str] = None
    for call in api_calls:
        tool = call.get("tool")
        tickers = call.get("tickers") or []
        date_range = call.get("date_range") or {}
        if tool != "get_history":
            continue
        if not tickers:
            logger.warning("Skipping get_history call with no tickers.")
            continue

        start = date_range.get("start")
        end = date_range.get("end")
        if not start or not end:
            today = dt.date.today()
            start = (today - dt.timedelta(days=730)).isoformat()
            end = today.isoformat()
            logger.info("No date_range provided; defaulting to %s -> %s", start, end)

        history_start = history_start or start
        history_end = history_end or end

        for ticker in tickers:
            logger.info("Fetching Yahoo history for %s (%s to %s)", ticker, start, end)
            result = _execute_tool(
                "get_history",
                tool_map,
                {"symbol": ticker, "start": start, "end": end, "interval": "1d"},
            )
            records = result.to_dict(orient="records") if isinstance(result, pd.DataFrame) else list(result)
            for row in records:
                row["ticker"] = ticker
            history_records.extend(records)
            history_tickers.append(ticker)

    if history_records:
        dest = base_dir / "ticker_eod_price_history.csv"
        _write_output(history_records, dest)
        artifacts["get_history"] = history_records
        file_paths["ticker_eod_price_history.csv"] = str(dest)
        logger.info(
            "Prepared history for %s tickers (%s to %s): %d rows -> %s",
            sorted(set(history_tickers)),
            history_start,
            history_end,
            len(history_records),
            dest,
        )

    return {"artifacts": artifacts, "artifact_paths": file_paths}
