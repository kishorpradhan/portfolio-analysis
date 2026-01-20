import json
import select
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence
from uuid import uuid4

from jinja2 import Template
from portfolio_agent.agents.codegen import generate_python_code
from portfolio_agent.agents.sandbox import CodeInterpreterFunctionTool
from portfolio_agent.clients.llm import get_embedding_client
from portfolio_agent.clients.vectorstore import get_vector_store
from portfolio_agent.config.settings import get_settings
from portfolio_agent.data_access.prep import prepare_data
from portfolio_agent.planner.planner import run_planner
from portfolio_agent.tools.registry import TOOL_MAP
from portfolio_agent.utils.logger import get_logger
from portfolio_agent.workflows.schema import FinalAnswerSchema

SANDBOX_DATA_DIR = "/home/user/data"


def _confirm_cache_insert(timeout_seconds: int = 60) -> bool:
    prompt = (
        f"Cache generated code for future runs? [Y/n] "
        f"(auto-saves in {timeout_seconds}s): "
    )
    print(prompt, end="", flush=True)
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
        if not ready:
            print("\nNo response detected. Defaulting to cache insertion.")
            return True
        answer = sys.stdin.readline().strip().lower()
    except Exception:
        print("\nInput unavailable. Defaulting to cache insertion.")
        return True

    return answer not in {"n", "no"}


def _extract_json_payload(raw: str) -> Optional[Sequence[dict]]:
    if not raw:
        return None

    raw = raw.strip()
    for candidate in (raw, raw[raw.find("{") :] if "{" in raw else ""):
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, (list, dict)):
                return parsed if isinstance(parsed, list) else [parsed]
        except Exception:
            continue
    return None


def _cache_lookup(question: str, datasets: List[str], skip_cache: bool) -> dict:
    settings = get_settings()
    logger = get_logger(__name__)
    if not settings.pinecone_api_key or skip_cache:
        return {"cached_code": None, "embedding": None, "vector_store": None}

    try:
        embedding_client = get_embedding_client()
        embedding = embedding_client.embed_query(question)
        vector_store = get_vector_store(len(embedding))
        top_entries = vector_store.top_records()
        logger.info("Top %d cached entries preview: %s", len(top_entries), top_entries)
        hit = vector_store.search_by_query(
            query=question,
            embedding=embedding,
            datasets=datasets,
        )
        logger.info("embedding dimensions: %d", len(embedding))
        if hit and hit.get("python_code"):
            logger.info("Vector store cache HIT for prompt; skipping LLM call.")
            return {"cached_code": hit["python_code"], "embedding": embedding, "vector_store": vector_store}
        return {"cached_code": None, "embedding": embedding, "vector_store": vector_store}
    except Exception as exc:
        logger.warning("Vector store lookup failed, falling back to LLM: %s", exc)
        return {"cached_code": None, "embedding": None, "vector_store": None}


def _generate_code(
    question: str,
    file_paths_prompt: Dict[str, str],
    required_files: List[str],
    cache: dict,
    schema_paths: Dict[str, str],
) -> dict:
    logger = get_logger(__name__)
    cached_code = cache.get("cached_code")
    if cached_code:
        logger.info("Using cached python code.")
        return {"code": cached_code, "code_source": "cache"}

    code = generate_python_code(
        user_query=question,
        file_paths=file_paths_prompt,
        required_files=required_files,
        schema_paths=schema_paths,
    )
    logger.info("Prepared %d characters of Python from LLM.", len(code))
    logger.debug("Generated code:\n%s", code)
    return {"code": code, "code_source": "llm"}


def _execute_code(code: str, run_dir: Path, run_id: str, code_source: str) -> dict:
    logger = get_logger(__name__)
    logger.info("Execution %s starting. Code source=%s", run_id, code_source)
    interpreter = CodeInterpreterFunctionTool()
    try:
        uploaded = interpreter.upload_directory(str(run_dir), sandbox_dir=SANDBOX_DATA_DIR)
        logger.info(
            "Execution %s uploaded %d files from %s to %s",
            run_id,
            len(uploaded),
            run_dir,
            SANDBOX_DATA_DIR,
        )

        sandbox_ready_code = code.replace(str(run_dir), SANDBOX_DATA_DIR)
        if sandbox_ready_code == code:
            logger.warning(
                "Execution %s: generated code did not reference %s; executing original code.",
                run_id,
                run_dir,
            )
        logger.debug("Execution %s code:\n%s", run_id, sandbox_ready_code)
        execution = interpreter.langchain_call(sandbox_ready_code)
        logger.info(
            "Execution %s complete. error=%s stderr_len=%d stdout_len=%d",
            run_id,
            bool(execution.get("error")),
            len(execution.get("stderr") or ""),
            len(execution.get("stdout") or ""),
        )
        if execution.get("stderr"):
            logger.debug("Execution %s stderr:\n%s", run_id, execution.get("stderr"))
        if execution.get("stdout"):
            logger.debug("Execution %s stdout:\n%s", run_id, execution.get("stdout"))
    finally:
        interpreter.close()

    return {"execution": execution, "run_id": run_id, "sandbox_code": sandbox_ready_code}


def _cache_write(question: str, datasets: List[str], code: str, cache: dict, execution: dict, skip_cache: bool) -> None:
    logger = get_logger(__name__)
    if skip_cache:
        return
    if cache.get("cached_code"):
        return
    vector_store = cache.get("vector_store")
    embedding = cache.get("embedding")
    if not vector_store or embedding is None:
        return
    if execution.get("error") or execution.get("stderr"):
        logger.info("Skipping cache write due to execution errors.")
        return
    if not _confirm_cache_insert():
        logger.info("User declined caching the generated code.")
        return
    try:
        vector_store.upsert(
            embedding=embedding,
            code=code,
            query=question,
            datasets=datasets,
        )
        logger.info("Stored generated code in vector store cache.")
    except Exception as exc:
        logger.warning("Failed to upsert code into vector store: %s", exc)


def _final_answer(payload: dict) -> FinalAnswerSchema:
    settings = get_settings()
    logger = get_logger(__name__)
    provider = settings.llm_provider.lower()
    if provider in {"gemini", "google"}:
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = ChatGoogleGenerativeAI(model=settings.gemini_model, temperature=0.0)
    else:
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(model=settings.openai_model, temperature=0.0)

    base_dir = Path(__file__).resolve().parents[1]
    template_path = base_dir / "prompts" / "final_answer_prompt.j2"
    payload_text = json.dumps(payload, default=str)
    prompt = Template(template_path.read_text()).render(payload=payload_text)
    result = model.with_structured_output(FinalAnswerSchema, method="function_calling").invoke(prompt)
    logger.info("Final answer generated.")
    return result


def run_question(question: str, *, skip_cache: bool = False) -> dict:
    logger = get_logger(__name__)
    settings = get_settings()
    run_id = uuid4().hex
    run_dir = settings.data_dir / "processed" / run_id

    logger.info("Planner starting for question: %s", question)
    plan = run_planner(question).model_dump()
    logger.info("Planner output required_files=%s api_calls=%s", plan.get("required_files"), plan.get("api_calls"))

    prep = prepare_data(plan, TOOL_MAP, run_id=run_id)
    artifact_paths = prep.get("artifact_paths") or {}

    if not artifact_paths:
        logger.warning("No artifact paths produced by data prep.")

    file_paths_prompt: Dict[str, str] = {}
    for name, local_path in artifact_paths.items():
        try:
            rel = Path(local_path).resolve().relative_to(run_dir.resolve())
            file_paths_prompt[name] = str(Path(SANDBOX_DATA_DIR) / rel.as_posix())
        except Exception:
            file_paths_prompt[name] = str(Path(SANDBOX_DATA_DIR) / Path(local_path).name)

    datasets = list(file_paths_prompt.keys())

    cache = _cache_lookup(question, datasets, skip_cache)
    code_info = _generate_code(
        question,
        file_paths_prompt,
        plan.get("required_files") or [],
        cache,
        schema_paths=artifact_paths,
    )
    exec_info = _execute_code(code_info["code"], run_dir, run_id, code_info["code_source"])
    _cache_write(question, datasets, code_info["code"], cache, exec_info["execution"], skip_cache)

    stdout = exec_info["execution"].get("stdout") or ""
    results = exec_info["execution"].get("results") or ""
    if isinstance(stdout, list):
        stdout = "\n".join(map(str, stdout))
    if isinstance(results, list):
        results = "\n".join(map(str, results))

    parsed_json = _extract_json_payload(str(stdout) + str(results))

    payload = {
        "question": question,
        "code": exec_info.get("sandbox_code"),
        "execution": {
            "error": (exec_info.get("execution") or {}).get("error"),
            "stderr": (exec_info.get("execution") or {}).get("stderr"),
            "stdout": (exec_info.get("execution") or {}).get("stdout"),
        },
        "result": parsed_json,
    }

    return {
        "plan": plan,
        "artifact_paths": artifact_paths,
        "code": code_info["code"],
        "execution": exec_info["execution"],
        "parsed_json": parsed_json,
        "payload": payload,
    }


def run_codegen_only(question: str, *, skip_cache: bool = False) -> dict:
    result = run_question(question, skip_cache=skip_cache)
    return {"code": result["code"], "execution": result["execution"]}


def answer_question(question: str, *, skip_cache: bool = False) -> str:
    result = run_question(question, skip_cache=skip_cache)
    final = _final_answer(result["payload"])
    if hasattr(final, "answer"):
        return final.answer
    return str(final)
