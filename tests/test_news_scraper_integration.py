import os
import unittest

from agent_layer.NewsScraperAgent import NewsScraperAgent
from portfolio_agent.utils.logger import get_logger


RUN_INTEGRATION = os.getenv("RUN_NEWS_SCRAPER_INTEGRATION") == "1"
logger = get_logger(__name__)


@unittest.skipUnless(
    RUN_INTEGRATION,
    "Set RUN_NEWS_SCRAPER_INTEGRATION=1 to enable this live Yahoo Finance scrape.",
)
class TestNewsScraperIntegration(unittest.TestCase):
    def test_fetch_live_news(self) -> None:
        ticker = os.getenv("NEWS_SCRAPER_TICKER", "AAPL")
        logger.debug("Starting integration test for ticker %s", ticker)

        agent = NewsScraperAgent()
        articles = agent.get_top_news(ticker)
        logger.debug("Fetched %d Yahoo Finance articles for %s", len(articles), ticker)
        self.assertTrue(
            articles,
            "Expected at least one article from Yahoo Finance. Verify yfinance install or HTTP fallback.",
        )


if __name__ == "__main__":
    unittest.main()
