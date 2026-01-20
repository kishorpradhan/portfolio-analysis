import json
import yaml
from pathlib import Path
from typing import Dict, Mapping
import pandas as pd
import importlib.resources as pkg_resources

from deepagents.backends.filesystem import FilesystemBackend
from deepagents.middleware.filesystem import FilesystemMiddleware
from jinja2 import Template
from langchain.agents import create_agent
from langchain_core.rate_limiters import InMemoryRateLimiter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

from portfolio_agent.config.settings import get_settings
from portfolio_agent.tools.catalog import get_tools
from portfolio_agent.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()

DATA_DIR = settings.data_dir
if not DATA_DIR.exists():
    raise FileNotFoundError(f"Expected data directory at {DATA_DIR}")

filesystem_middleware = FilesystemMiddleware(
    backend=FilesystemBackend(root_dir=DATA_DIR, virtual_mode=True)
)

rate_limiter = InMemoryRateLimiter(
    requests_per_second=1,
    check_every_n_seconds=1,
    max_bucket_size=1,
)

logger.info("Using data directory at: %s", DATA_DIR)
logger.info("Using LLM Provider: %s", settings.llm_provider)


def _load_file_text(package: str, filename: str) -> str:
    resource = pkg_resources.files(package).joinpath(filename)
    return resource.read_text()


def load_dataset_metadata() -> str:
    raw = yaml.safe_load(_load_file_text("portfolio_agent.config", "file_metadata.yaml"))
    datasets = raw.get("datasets", {})
    lines = []

    for name, metadata in datasets.items():
        lines.append(f"DATASET: {name}")
        lines.append(f"DESCRIPTION: {metadata.get('description', '')}")

        lines.append("SCHEMA:")
        for col in metadata.get("columns", []):
            col_str = f"  - {col['name']} (Type: {col['type']})"
            constraints = col.get("constraints", {})
            if constraints:
                constraint_str = ", ".join(
                    [f"{k}: {v}" for k, v in constraints.items()]
                )
                col_str += f" | Constraints: [{constraint_str}]"
            lines.append(col_str)

        examples = metadata.get("examples", [])
        if examples:
            lines.append("SAMPLE ROW:")
            lines.append(f"  {json.dumps(examples[0])}")

        lines.append("-" * 20)

    return "\n".join(lines)


def _load_yaml_schema_map() -> dict:
    raw = yaml.safe_load(_load_file_text("portfolio_agent.config", "file_metadata.yaml"))
    datasets = raw.get("datasets", {})
    schema_map = {}
    for name, metadata in datasets.items():
        cols = metadata.get("columns", [])
        schema_map[name] = {
            "description": metadata.get("description", ""),
            "columns": [
                {
                    "name": col.get("name"),
                    "type": col.get("type", "unknown"),
                    "description": col.get("description", ""),
                    "constraints": col.get("constraints", {}),
                }
                for col in cols
            ],
            "examples": metadata.get("examples", [])[:3],
        }
    return schema_map


def _load_sample_rows(path: str, limit: int = 3) -> list[dict]:
    try:
        if path.endswith(".json"):
            data = json.loads(Path(path).read_text())
            if isinstance(data, list):
                return data[:limit]
            if isinstance(data, dict):
                return [data]
            return []
        df = pd.read_csv(path, nrows=limit)
        return df.to_dict(orient="records")
    except Exception:
        return []


def _infer_schema(path: str) -> dict:
    if path.endswith(".json"):
        try:
            data = json.loads(Path(path).read_text())
            if isinstance(data, list) and data:
                return {k: type(v).__name__ for k, v in data[0].items()}
            if isinstance(data, dict):
                return {k: type(v).__name__ for k, v in data.items()}
        except Exception:
            return {}
        return {}
    try:
        df = pd.read_csv(path, nrows=1)
        return {col: str(dtype) for col, dtype in df.dtypes.items()}
    except Exception:
        return {}


def load_and_render_prompt(
    user_query: str,
    file_paths: Mapping[str, str],
    dataset_metadata: str,
    file_schemas: Mapping[str, dict] | None = None,
) -> str:
    template_text = _load_file_text("portfolio_agent.prompts", "codegen_prompt.j2")
    template = Template(template_text)
    file_paths_text = yaml.dump(dict(file_paths), default_flow_style=False)
    file_schemas_text = yaml.dump(dict(file_schemas or {}), default_flow_style=False)

    return template.render(
        user_query=user_query,
        file_paths=file_paths_text,
        dataset_metadata=dataset_metadata,
        file_schemas=file_schemas_text,
    )


def build_chat_model(*, temperature: float):
    provider = settings.llm_provider.lower()
    if provider in {"gemini", "google"}:
        return ChatGoogleGenerativeAI(
            model=settings.gemini_model,
            temperature=temperature,
            rate_limiter=rate_limiter,
        )
    if provider in {"openai", "gpt", "chatgpt"}:
        return ChatOpenAI(
            model=settings.openai_model,
            temperature=temperature,
        )
    raise ValueError(
        f"Unsupported LLM_PROVIDER '{settings.llm_provider}'. Expected one of gemini/google or openai/gpt/chatgpt."
    )


def create_codegen_agent(system_prompt: str | None = None):
    tools = get_tools()
    return create_agent(
        model=build_chat_model(temperature=0.35),
        system_prompt=system_prompt,
        tools=tools,
    )


def generate_python_code(
    user_query: str,
    file_paths: Mapping[str, str],
    required_files: list[str] | None = None,
    schema_paths: Mapping[str, str] | None = None,
) -> str:
    datasets_metadata = load_dataset_metadata()
    yaml_schema_map = _load_yaml_schema_map()

    file_schemas: dict = {}
    for name, path in file_paths.items():
        schema_path = (schema_paths or {}).get(name, path)
        yaml_schema = yaml_schema_map.get(name)
        if required_files and name in required_files and yaml_schema:
            file_schemas[name] = yaml_schema
            if not yaml_schema.get("examples"):
                file_schemas[name]["sample_rows"] = _load_sample_rows(schema_path)
            else:
                file_schemas[name]["sample_rows"] = yaml_schema.get("examples", [])[:3]
        else:
            file_schemas[name] = {
                "columns": _infer_schema(schema_path),
                "sample_rows": _load_sample_rows(schema_path),
            }

    prompt = load_and_render_prompt(
        user_query=user_query,
        file_paths=file_paths,
        dataset_metadata=datasets_metadata,
        file_schemas=file_schemas,
    )

    agent = create_codegen_agent(prompt)
    logger.info("Codegen LLM input: %s", prompt)

    response = agent.invoke({"messages": [{"role": "user", "content": user_query}]})

    content = response.get("messages", [])[-1].content
    if isinstance(content, list):
        content = "".join(part.get("text", "") for part in content)
    elif not isinstance(content, str):
        content = str(content)
    code = content.strip()
    if not code:
        raise ValueError("LLM returned empty content")
    return code
