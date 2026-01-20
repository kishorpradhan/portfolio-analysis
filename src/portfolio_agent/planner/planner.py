from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from jinja2 import Template
from pydantic import BaseModel, Field, ValidationError

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from portfolio_agent.config.settings import get_settings
from portfolio_agent.utils.logger import get_logger

logger = get_logger(__name__)


class DateRange(BaseModel):
    start: Optional[str] = Field(default=None, description="YYYY-MM-DD or null")
    end: Optional[str] = Field(default=None, description="YYYY-MM-DD or null (optional)")


class ApiCall(BaseModel):
    tool: str
    tickers: List[str] = Field(default_factory=list)
    date_range: DateRange


class PlannerOutput(BaseModel):
    technical_question: str
    required_files: List[str] = Field(default_factory=list)
    api_calls: List[ApiCall] = Field(default_factory=list)
    tickers: List[str] = Field(default_factory=list)


def _load_dataset_metadata(path: Path) -> str:
    logger.info("Planner loading dataset metadata from %s", path)
    raw = yaml.safe_load(path.read_text())
    datasets = raw.get("datasets", {})
    logger.info("Planner datasets loaded: %s", list(datasets.keys()))
    lines = []
    for name, metadata in datasets.items():
        lines.append(f"DATASET: {name}")
        lines.append(f"DESCRIPTION: {metadata.get('description', '')}")
        lines.append("SCHEMA:")
        for col in metadata.get("columns", []):
            col_str = f"  - {col['name']} (Type: {col['type']})"
            constraints = col.get("constraints", {})
            if constraints:
                constraint_str = ", ".join([f"{k}: {v}" for k, v in constraints.items()])
                col_str += f" | Constraints: [{constraint_str}]"
            lines.append(col_str)
        examples = metadata.get("examples", [])
        if examples:
            lines.append("SAMPLE ROW:")
            lines.append(f"  {json.dumps(examples[0])}")
        lines.append("-" * 20)
    
    return "\n".join(lines)
    




def _render_planner_prompt(question: str) -> str:
    logger.info("Planner rendering prompt for question: %s", question)
    base_dir = Path(__file__).resolve().parents[1]
    template_path = base_dir / "prompts" / "planner_prompt.j2"
    metadata_path = base_dir / "config" / "file_metadata.yaml"
    logger.info("Planner template path: %s", template_path)
    logger.info("Planner metadata path: %s", metadata_path)
    template = Template(template_path.read_text())
    dataset_metadata = _load_dataset_metadata(metadata_path)
    prompt = template.render(
        user_question=question,
        dataset_metadata=dataset_metadata,
    )
    logger.info("Planner prompt:\n%s", prompt)
    return prompt



def _build_llm(temperature: float = 0.3):
    settings = get_settings()
    provider = settings.llm_provider.lower()
    logger.info("Planner LLM provider: %s", settings.llm_provider)
    if provider in {"gemini", "google"}:
        logger.info("Planner using Gemini model: %s", settings.gemini_model)
        return ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=temperature)
    if provider in {"openai", "gpt", "chatgpt"}:
        logger.info("Planner using OpenAI model: %s", settings.openai_model)
        return ChatOpenAI(model=settings.openai_model, temperature=temperature)
    raise ValueError(f"Unsupported LLM provider '{settings.llm_provider}' for planner.")


def run_planner(question: str) -> PlannerOutput:
    logger.info("Planner start.")
    prompt = _render_planner_prompt(question)
    logger.info("Planner prompt (run_planner):\n%s", prompt)
    llm = _build_llm()
    logger.info("Planner prompt length: %d", len(prompt))
    logger.info("Planner LLM call starting.")
    response = llm.invoke(prompt)
    logger.info("Planner LLM call complete.")
    content = getattr(response, "content", "") or ""
    if not isinstance(content, str):
        content = str(content)
    try:
        logger.info("Planner parsing LLM response.")
        payload = json.loads(content)
        logger.info("Planner JSON parsed successfully.")
        return PlannerOutput.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        logger.warning("Planner returned invalid JSON; falling back to empty plan: %s", exc)
        return PlannerOutput(
            technical_question=question,
            tickers=[],
            required_files=[],
        )
