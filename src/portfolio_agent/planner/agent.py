from __future__ import annotations

import json
from typing import Any, Dict

from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from portfolio_agent.config.settings import get_settings
from portfolio_agent.tools.catalog import get_tools
from portfolio_agent.utils.logger import get_logger

logger = get_logger(__name__)


PLANNER_SYSTEM_PROMPT = """
You are a planning agent for portfolio analysis. You can call tools to load trades from local file 
and get historical prices for tickers.
Goal: Decide what data to fetch for the user's question, call tools to fetch it, then return a single JSON plan.

Tools available:
- load_trades(symbol=None, start=None, end=None) -> list of trades {id, symbol, side, quantity, price, created_at, state}
- get_history(symbol, start, end, interval="1d") -> list of {date, open, high, low, close, volume}
- summarize_json(payload, max_bullets=5) -> concise bullet summary (rarely needed at planning time)

Output format (JSON only):
{
  "symbols": ["TSLA"],
  "date_range": {"start": "YYYY-MM-DD" | null, "end": "YYYY-MM-DD" | null},
  "artifacts": {
    "trades": [...],    # list of trade records if fetched
    "history": [...]    # list of price records if fetched
  }
}

Rules:
- Call tools as needed to populate artifacts; prefer trades for mentioned symbols.
- For market-performance questions about a single ticker (e.g., "how is NVDA doing", "price between dates"), call get_history.
- If dates are not specified, leave them null.
- Final answer must be ONLY the JSON object above. No markdown, no extra text.
"""


def _build_planner_model():
    settings = get_settings()
    provider = settings.llm_provider.lower()
    if provider in {"gemini", "google"}:
        return ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0.0)
    if provider in {"openai", "gpt", "chatgpt"}:
        return ChatOpenAI(model=settings.openai_model, temperature=0.0)
    raise ValueError(f"Unsupported LLM provider '{settings.llm_provider}' for planner.")


def run_planner(question: str) -> Dict[str, Any]:
    """
    Run a multi-turn planner agent that can call tools, then returns a JSON plan
    with symbols, date_range, and fetched artifacts.
    """
    captured: Dict[str, list] = {}

    def make_capturing_tool(t):
        original_func = t.func

        def wrapped(*args, **kwargs):
            result = original_func(*args, **kwargs)
            try:
                captured[t.name] = result if isinstance(result, list) else [result]
            except Exception:
                logger.debug("Failed to capture tool result for %s", t.name)
            return result

        # Rebuild StructuredTool with wrapped func to preserve schema/name
        from langchain_core.tools import StructuredTool

        return StructuredTool.from_function(
            name=t.name,
            description=t.description,
            func=wrapped,
            args_schema=t.args_schema,
        )

    tools = [make_capturing_tool(t) for t in get_tools()]

    agent = create_agent(
        model=_build_planner_model(),
        system_prompt=PLANNER_SYSTEM_PROMPT,
        tools=tools,
    )
    try:
        response = agent.invoke({"messages": [{"role": "user", "content": question}]})
        content = response.get("messages", [])[-1].content if isinstance(response, dict) else None
        if isinstance(content, list):
            content = "".join(part.get("text", "") for part in content if isinstance(part, dict))
        if not isinstance(content, str):
            content = str(content)
        plan = json.loads(content)
        if not isinstance(plan, dict):
            raise ValueError("Planner did not return a JSON object")
        return {"plan": plan, "artifacts": captured}
    except Exception as exc:
        logger.warning("Planner failed; returning empty plan. Error: %s", exc)
        return {"plan": {"symbols": [], "date_range": {"start": None, "end": None}}, "artifacts": {}}
