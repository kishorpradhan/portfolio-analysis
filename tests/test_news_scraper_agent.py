import unittest
from unittest.mock import MagicMock, patch

from agent_layer.NewsScraperAgent import NewsScraperAgent, NewsArticle


SAMPLE_HTML = """
<html>
  <body>
    <ul>
      <li data-test-id="news-stream-list-item">
        <h3><a href="/news/sample-story-1">Headline One</a></h3>
      </li>
      <li data-test-id="news-stream-list-item">
        <h3><a href="/news/sample-story-2">Headline Two</a></h3>
      </li>
      <li data-test-id="news-stream-list-item">
        <h3><a href="/news/sample-story-3">Headline Three</a></h3>
      </li>
      <li data-test-id="news-stream-list-item">
        <h3><a href="/news/sample-story-4">Headline Four</a></h3>
      </li>
    </ul>
  </body>
</html>
"""


class TestNewsScraperAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.scraper = NewsScraperAgent(max_articles=3)

    def test_parse_news_returns_unique_articles(self) -> None:
        articles = self.scraper._parse_news(SAMPLE_HTML)
        self.assertEqual(len(articles), 3)
        self.assertTrue(
            all(article.url.startswith("https://finance.yahoo.com") for article in articles)
        )
        self.assertEqual(articles[0].title, "Headline One")

    def test_get_top_news_prefers_yfinance(self) -> None:
        yfinance_articles = [
            NewsArticle(title="A", url="https://example.com/a"),
            NewsArticle(title="B", url="https://example.com/b"),
        ]
        with patch.object(
            self.scraper,
            "_fetch_via_yfinance",
            MagicMock(return_value=yfinance_articles),
        ) as mock_yf, patch.object(
            self.scraper,
            "_fetch_via_requests",
            MagicMock(return_value=""),
        ) as mock_http:
            articles = self.scraper.get_top_news("aapl")

        mock_yf.assert_called_once()
        mock_http.assert_not_called()
        self.assertEqual(articles, yfinance_articles)

    def test_get_top_news_falls_back_to_requests(self) -> None:
        with patch.object(
            self.scraper, "_fetch_via_yfinance", MagicMock(return_value=[])
        ) as mock_yf, patch.object(
            self.scraper, "_fetch_via_requests", MagicMock(return_value=SAMPLE_HTML)
        ) as mock_http:
            articles = self.scraper.get_top_news("aapl")

        mock_yf.assert_called_once()
        mock_http.assert_called_once()
        self.assertEqual(len(articles), 3)
        self.assertEqual(articles[0].title, "Headline One")

    def test_get_top_news_requires_ticker(self) -> None:
        with self.assertRaises(ValueError):
            self.scraper.get_top_news("   ")


if __name__ == "__main__":
    unittest.main()
