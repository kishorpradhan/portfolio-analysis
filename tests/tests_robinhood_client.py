from portfolio_agent.clients.robinhood_client.gets_trades_robinhood import RobinhoodClient
from portfolio_agent.utils.logger import get_logger

logger = get_logger(__name__)




def main():
    logger.info("=== Running Robinhood Client test ===")
    rh = RobinhoodClient()
    print("INFO:", rh.info())

    print("\n=== Positions ===")
    positions = rh.get_positions()
    for p in positions:
        print(p)

    print("\n=== Orders (first 5) ===")
    orders = rh.get_orders()
    for o in orders[:5]:
        print(o)
    logger.info(f"Fetched {len(orders)} orders")
    logger.debug(f"Sample: {orders[:2]}")

if __name__ == "__main__":
    main()
