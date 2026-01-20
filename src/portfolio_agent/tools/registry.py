from portfolio_agent.clients.yahoo import get_history
from portfolio_agent.data_access.trades import load_trades
from portfolio_agent.data_access.dividends import load_dividends
from portfolio_agent.data_access.positions import load_positions

TOOL_MAP = {
    "get_history": get_history,
    "load_trades": load_trades,
    "load_dividends": load_dividends,
    "load_positions": load_positions,
}
