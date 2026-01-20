import json
import os
import shlex
from pathlib import Path
from typing import Any, List, Optional

from e2b_code_interpreter import Sandbox
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import Tool
from pydantic import BaseModel, Field

from portfolio_agent.config.settings import get_settings
from portfolio_agent.utils.logger import get_logger

settings = get_settings()
E2B_API_KEY = settings.e2b_api_key
logger = get_logger(__name__)

class LangchainCodeInterpreterToolInput(BaseModel):
    code: str = Field(description="Python code to execute.")


class CodeInterpreterFunctionTool:
    """
    This class calls arbitrary code against a Python Jupyter notebook.
    It requires an E2B_API_KEY to create a sandbox.
    """

    tool_name: str = "code_interpreter"

    def __init__(self):
        # Instantiate the E2B sandbox - this is a long lived object
        # that's pinging E2B cloud to keep the sandbox alive.
        self.code_interpreter = Sandbox.create()

    def close(self):
        self.code_interpreter.kill()

    def _require_upload_confirmation(self, transfers: List[str]) -> None:
        """
        Simple human-in-the-loop gate before copying local data into the sandbox.
        """
        if not transfers:
            raise ValueError("No files queued for upload.")

        print("About to upload the following files to the sandbox:")
        for entry in transfers:
            print(f"  - {entry}")

        try:
            answer = input("Do you want me to upload files from local to sandbox? [y/N]: ")
        except EOFError:
            answer = ""

        normalized = answer.strip().lower()
        if normalized not in {"y", "yes"}:
            raise SystemExit("Upload cancelled by user.")

    def upload_file(self, local_path: str, sandbox_path: Optional[str] = None) -> str:
        """
        Upload a single file from the local filesystem into the sandbox.
        """

        path = Path(local_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Local file not found: {path}")

        destination = (sandbox_path or f"/home/user/{path.name}").strip()
        if not destination:
            raise ValueError("sandbox_path cannot be empty.")

        transfer_desc = f"{path}  ->  {destination}"
        self._require_upload_confirmation([transfer_desc])

        return self._upload_single_file(path, destination)

    def upload_directory(
        self,
        local_dir: str,
        sandbox_dir: Optional[str] = None,
    ) -> List[str]:
        """
        Recursively upload every file from a local directory to the sandbox.
        """
        base_dir = Path(local_dir).expanduser().resolve()
        if not base_dir.is_dir():
            raise NotADirectoryError(f"Local directory not found: {base_dir}")

        target_root = sandbox_dir or f"/home/user/{base_dir.name}"
        uploaded: List[str] = []

        transfers: List[tuple[Path, str]] = []
        for file_path in base_dir.rglob("*"):
            if not file_path.is_file():
                continue
            relative = file_path.relative_to(base_dir)
            sandbox_path = str(Path(target_root) / relative.as_posix())
            transfers.append((file_path, sandbox_path))

        if not transfers:
            logger.warning("No files found in directory %s to upload.", base_dir)
            return []

        descriptions = [f"{src}  ->  {dest}" for src, dest in transfers]
        self._require_upload_confirmation(descriptions)

        for src, dest in transfers:
            uploaded.append(self._upload_single_file(src, dest))

        return uploaded

    def _upload_single_file(self, src: Path, destination: str) -> str:
        directory = os.path.dirname(destination)
        if directory and directory not in ("/", "."):
            self.code_interpreter.commands.run(f"mkdir -p {shlex.quote(directory)}")

        with src.open("rb") as file_handle:
            self.code_interpreter.files.write(destination, file_handle.read())

        logger.info("Uploaded %s to sandbox path %s", src, destination)
        return destination

    def call(self, parameters: dict, **kwargs: Any):
        code = parameters.get("code", "")
        print(f"***Code Interpreting...\n{code}\n====")
        execution = self.code_interpreter.run_code(code)
        return {
            "results": execution.results,
            "stdout": execution.logs.stdout,
            "stderr": execution.logs.stderr,
            "error": execution.error,
        }

    # langchain does not return a dict as a parameter, only a code string
    def langchain_call(self, code: str):
        return self.call({"code": code})

    def to_langchain_tool(self) -> Tool:
        tool = Tool(
            name=self.tool_name,
            description="Execute python code in a Jupyter notebook cell and returns any rich data (eg charts), stdout, stderr, and error.",
            func=self.langchain_call,
        )
        tool.args_schema = LangchainCodeInterpreterToolInput
        return tool

    @staticmethod
    def format_to_tool_message(
        message_log: List[BaseMessage],
        observation: dict,
        tool_call_id: Optional[str] = None,
    ) -> List[BaseMessage]:
        """
        Format the output of the CodeInterpreter tool to be returned as a ToolMessage.
        """
        new_messages = list(message_log)

        # TODO: Add info about the results for the LLM
        content = json.dumps(
            {k: v for k, v in observation.items() if k not in ("results")}, indent=2
        )
        new_messages.append(
            ToolMessage(
                content=content,
                tool_call_id=tool_call_id or CodeInterpreterFunctionTool.tool_name,
            )
        )

        return new_messages
    
