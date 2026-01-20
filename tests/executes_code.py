import pandas as pd
import numpy as np
from portfolio_agent.config.settings import get_settings

data_dir = get_settings().data_dir
positions = pd.read_csv(data_dir / "portfolio.csv")
trades = pd.read_csv(data_dir / "trades.csv")

positions['unrealized_pnl'] = (positions['price'] - positions['average_price']) * positions['quantity']

total_profit = positions['unrealized_pnl'].sum()

google_profit = positions[positions['symbol'].isin(['GOOG', 'GOOGL'])]['unrealized_pnl'].sum()

if total_profit != 0:
    google_share_of_profit = google_profit / total_profit
else:
    google_share_of_profit = 0.0

result = {
    "google_profit": google_profit,
    "total_profit": total_profit,
    "google_share_of_profit": google_share_of_profit
}

print(result)
