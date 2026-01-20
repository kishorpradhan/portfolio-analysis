from langchain_community.tools.tavily_search import TavilySearchResults

from portfolio_agent.config.settings import get_settings


def build_search_tool() -> TavilySearchResults:
    settings = get_settings()
    if not settings.tavily_api_key:
        raise EnvironmentError("TAVILY_API_KEY is not configured for search.")
    return TavilySearchResults(
        max_results=3, include_answer=True, api_key=settings.tavily_api_key
    )
