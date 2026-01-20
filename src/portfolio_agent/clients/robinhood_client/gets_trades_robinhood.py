from __future__ import annotations
import argparse
import csv
from pathlib import Path
from typing import Any, Dict, List, Optional

from portfolio_agent.clients.robinhood_client.config import load_config
from portfolio_agent.clients.robinhood_client.txn_loaders_from_csv import (
    load_orders_from_csv,
    load_positions_from_csv,
)
from portfolio_agent.config.settings import get_settings
from portfolio_agent.utils.logger import get_logger
import pandas as pd
from datetime import datetime


logger = get_logger(__name__)


try:
    import robin_stocks.robinhood as r
    HAS_RH_LIB = True
except ImportError:
    HAS_RH_LIB = False

DEFAULT_DATA_DIR = get_settings().data_dir


class RobinhoodClient:
    """
    Abstraction over live Robinhood and CSV.
    """

    def __init__(self, config=None):
        self.config = config or load_config()
        self._logged_in = False
        self._instrument_symbol_cache: Dict[str, str] = {}
        self._fundamentals_cache: Dict[str, Dict[str, Any]] = {}

    # ---------- private ----------
    def _login(self):
        if self._logged_in:
            return
        if not self.config.use_live:
            # CSV mode, no login
            return
        if not HAS_RH_LIB:
            raise RuntimeError("robin_stocks is not installed but ROBINHOOD_USE_LIVE=true")
        r.login(
            username=self.config.username,
            password=self.config.password,
            mfa_code=self.config.totp,
        )
        self._logged_in = True

    def _get_symbol_from_instrument(self, instrument_url: Optional[str]) -> str:
        """
        Resolve a symbol from a Robinhood instrument URL with simple caching to
        avoid repeated API calls when multiple orders reference the same
        instrument.
        """
        if not instrument_url:
            return "UNKNOWN"

        cached_symbol = self._instrument_symbol_cache.get(instrument_url)
        if cached_symbol:
            return cached_symbol

        if not HAS_RH_LIB:
            return "UNKNOWN"

        try:
            instrument = r.get_instrument_by_url(instrument_url)
        except Exception as exc:  # pragma: no cover - defensive, network dependent
            logger.warning(f"Failed to fetch instrument from {instrument_url}: {exc}")
            symbol = "UNKNOWN"
        else:
            symbol = instrument.get("symbol", "UNKNOWN") if instrument else "UNKNOWN"

        self._instrument_symbol_cache[instrument_url] = symbol
        return symbol

    def _get_fundamentals(self, symbol: Optional[str]) -> Dict[str, Any]:
        """
        Fetch a subset of Robinhood fundamentals for a symbol, caching results
        to avoid redundant network calls when multiple positions reference the
        same equity.
        """
        if not symbol:
            return {}

        cached_fundamentals = self._fundamentals_cache.get(symbol)
        if cached_fundamentals is not None:
            return cached_fundamentals

        if not HAS_RH_LIB:
            self._fundamentals_cache[symbol] = {}
            return {}

        try:
            fundamentals_list = r.stocks.get_fundamentals(symbol)
        except Exception as exc:  # pragma: no cover - defensive, network dependent
            logger.warning(f"Failed to fetch fundamentals for {symbol}: {exc}")
            fundamentals_data: Dict[str, Any] = {}
        else:
            fundamentals = fundamentals_list[0] if fundamentals_list and isinstance(fundamentals_list, list) else {}
            fundamentals_data = {
                "pe_ratio": self._to_float(fundamentals.get("pe_ratio")),
                "list_date": fundamentals.get("list_date"),
                "sector": fundamentals.get("sector"),
            }

        self._fundamentals_cache[symbol] = fundamentals_data
        return fundamentals_data

    @staticmethod
    def _to_float(value: Any) -> Optional[float]:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    # ---------- public ----------
    def get_positions(self) -> List[Dict[str, Any]]:
        """
        Return list of {symbol, quantity, average_price, price}
        """
        if self.config.use_live:
            self._login()
            holdings = r.account.build_holdings()
            out = []
            for symbol, data in holdings.items():
                fundamentals = self._get_fundamentals(symbol)
                out.append({
                    "symbol": symbol,
                    "quantity": float(data["quantity"]),
                    "average_price": float(data["average_buy_price"]),
                    "price": float(data["price"]),
                    "fundamentals": fundamentals,
                })
            return out
        else:
            if not self.config.csv_path:
                raise RuntimeError("CSV mode enabled but ROBINHOOD_CSV_PATH not set")
            return load_positions_from_csv(self.config.csv_path)

    def get_orders(self) -> List[Dict[str, Any]]:
        """
        Return list of orders/trades.
        """
        if self.config.use_live:
            self._login()
            orders = r.orders.get_all_stock_orders()
            logger.info(f"Fetched {len(orders)} raw orders")
            logger.debug(f"First order sample: {orders[0] if orders else 'none'}")
            logger.info("Starting monthly investment computation...")
            
            cleaned = []
            for o in orders:
                # instrument is a URL; we can skip it or fetch symbol if needed
                instrument_url = o.get("instrument")
                symbol = o.get("symbol") or self._get_symbol_from_instrument(instrument_url)
                cleaned.append({
                    "id": o["id"],
                    "symbol": symbol,
                    "side": o["side"],
                    "quantity": float(o["quantity"]),
                    "price": float(o["price"]) if o["price"] else 0.0,
                    "created_at": o["created_at"],
                    "state": o["state"],
                })
            return cleaned
        else:
            if not self.config.csv_path:
                raise RuntimeError("CSV mode enabled but ROBINHOOD_CSV_PATH not set")
            return load_orders_from_csv(self.config.csv_path)

    def get_dividends(self) -> List[Dict[str, Any]]:
        """
        Return list of dividend payments as {date, ticker, amount, state}.
        """
        if self.config.use_live:
            self._login()
            dividends = r.account.get_dividends()
            cleaned: List[Dict[str, Any]] = []
            for d in dividends:
                ticker = self._get_symbol_from_instrument(d.get("instrument"))
                raw_date = d.get("paid_at") or d.get("payable_date") or d.get("record_date")
                # Parse the string and format it as YYYY-MM-DD
                clean_date = pd.to_datetime(raw_date).strftime('%Y-%m-%d') if raw_date else None
                cleaned.append(
                    {
                        "date": clean_date,
                        "ticker": ticker,
                        "amount": float(d.get("amount") or 0),
                        "state": d.get("state"),
                    }
                )
            return cleaned
        else:
            raise RuntimeError("CSV mode enabled but dividends CSV not configured")

    def info(self) -> dict:
        return {
            "mode": "live" if self.config.use_live else "csv",
            "csv_path": self.config.csv_path,
            "has_rh_lib": HAS_RH_LIB,
            "logged_in": self._logged_in,
        }

    # ---------- writers ----------
    def _resolve_output_path(self, output_dir: Optional[str | Path], filename: str) -> Path:
        base_dir = Path(output_dir) if output_dir else DEFAULT_DATA_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir / filename

    def write_positions_to_csv(
        self,
        output_dir: Optional[str | Path] = None,
        filename: str = "portfolio.csv",
    ) -> Path:
        """
        Persist portfolio/positions data to CSV for later reuse.
        """
        positions = self.get_positions()
        output_path = self._resolve_output_path(output_dir, filename)
        fieldnames = ["symbol", "quantity", "average_price", "price", "pe_ratio", "list_date", "sector"]

        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for position in positions:
                fundamentals = position.get("fundamentals") or {}
                writer.writerow({
                    "symbol": position.get("symbol"),
                    "quantity": position.get("quantity"),
                    "average_price": position.get("average_price"),
                    "price": position.get("price"),
                    "pe_ratio": fundamentals.get("pe_ratio"),
                    "list_date": fundamentals.get("list_date"),
                    "sector": fundamentals.get("sector"),
                })

        logger.info(f"Wrote {len(positions)} positions to {output_path}")
        return output_path

    def write_orders_to_csv(
        self,
        output_dir: Optional[str | Path] = None,
        filename: str = "orders.csv",
    ) -> Path:
        """
        Persist orders/trades data to CSV for later reuse.
        """
        orders = self.get_orders()
        output_path = self._resolve_output_path(output_dir, filename)
        fieldnames = ["id", "symbol", "side", "quantity", "price", "created_at", "state"]

        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for order in orders:
                writer.writerow({
                    "id": order.get("id"),
                    "symbol": order.get("symbol"),
                    "side": order.get("side"),
                    "quantity": order.get("quantity"),
                    "price": order.get("price"),
                    "created_at": order.get("created_at"),
                    "state": order.get("state"),
                })

        logger.info(f"Wrote {len(orders)} orders to {output_path}")
        return output_path

    def write_dividends_to_csv(
        self,
        output_dir: Optional[str | Path] = None,
        filename: str = "dividends.csv",
    ) -> Path:
        """
        Persist dividend payments to CSV for later reuse.
        """
        dividends = self.get_dividends()
        output_path = self._resolve_output_path(output_dir, filename)
        fieldnames = ["date", "ticker", "amount", "state"]

        with output_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for dividend in dividends:
                writer.writerow(
                    {
                        "date": dividend.get("date"),
                        "ticker": dividend.get("ticker"),
                        "amount": dividend.get("amount"),
                        "state": dividend.get("state"),
                    }
                )

        logger.info(f"Wrote {len(dividends)} dividends to {output_path}")
        return output_path


def main():
    parser = argparse.ArgumentParser(description="Fetch Robinhood portfolio and orders, then write them to CSV.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory to write CSV files to (default: repo data directory).",
    )
    parser.add_argument(
        "--portfolio-filename",
        default="portfolio.csv",
        help="Filename for portfolio CSV (default: portfolio.csv).",
    )
    parser.add_argument(
        "--orders-filename",
        default="orders.csv",
        help="Filename for orders CSV (default: orders.csv).",
    )
    parser.add_argument(
        "--dividends-filename",
        default="dividends.csv",
        help="Filename for dividends CSV (default: dividends.csv).",
    )

    args = parser.parse_args()
    client = RobinhoodClient()

    portfolio_path = client.write_positions_to_csv(
        output_dir=args.output_dir,
        filename=args.portfolio_filename,
    )
    orders_path = client.write_orders_to_csv(
        output_dir=args.output_dir,
        filename=args.orders_filename,
    )
    dividends_path = client.write_dividends_to_csv(
        output_dir=args.output_dir,
        filename=args.dividends_filename,
    )

    print(f"Portfolio written to {portfolio_path}")
    print(f"Orders written to {orders_path}")
    print(f"Dividends written to {dividends_path}")


if __name__ == "__main__":  # pragma: no cover
    main()
