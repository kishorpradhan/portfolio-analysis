from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from portfolio_agent.config.settings import get_settings
from portfolio_agent.utils.logger import get_logger

logger = get_logger(__name__)


def _build_llm(temperature: float = 0.3):
    settings = get_settings()
    provider = settings.llm_provider.lower()
    if provider in {"gemini", "google"}:
        return ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=temperature)
    if provider in {"openai", "gpt", "chatgpt"}:
        return ChatOpenAI(model=settings.openai_model, temperature=temperature)
    raise ValueError(f"Unsupported LLM provider for summarizer: {settings.llm_provider}")


def summarize_json(payload: Any, max_bullets: int = 5) -> str:
    """
    Summarize a JSON-like payload (dict/list/str) into concise bullets.
    """
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except Exception:
            # keep as raw string
            pass

    try:
        payload_text = json.dumps(payload, default=str)
    except Exception:
        payload_text = str(payload)

    system = SystemMessage(content="You are a concise financial summarizer. Highlight profit/loss, counts, and key drivers.")
    human = HumanMessage(
        content=(
            f"Summarize the following JSON in up to {max_bullets} bullets. "
            "Be precise, avoid speculation, and keep it short.\n"
            f"DATA:\n{payload_text}"
        )
    )

    llm = _build_llm()
    response = llm.invoke([system, human])
    content = getattr(response, "content", "") or ""
    logger.debug("Summarizer LLM response length: %d", len(content))
    return content.strip()
