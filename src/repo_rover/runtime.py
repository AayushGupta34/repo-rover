"""Shared runtime pieces for the RepoRover crews.

- GroqChatLLM: lets CrewAI agents run on a `langchain_groq.ChatGroq` instance.
  CrewAI 1.x no longer executes LangChain chat models directly (it would read
  only the model name off the object and route it to OpenAI), so this adapter
  implements CrewAI's BaseLLM interface and delegates every call to ChatGroq.
- RepoDirectoryReadTool / BoundedScrapeWebsiteTool: the stock crewai_tools
  tools with their output bounded so a single observation cannot blow through
  Groq's per-minute token limit.
"""

from dotenv import load_dotenv

load_dotenv()

import os
from typing import Any

from crewai.llms.base_llm import BaseLLM
from crewai_tools import DirectoryReadTool, ScrapeWebsiteTool
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from pydantic import PrivateAttr

from repo_rover import resilience

GROQ_MODEL = "qwen/qwen3.8-27b"
GROQ_CONTEXT_WINDOW = 131_072

IGNORED_DIRECTORIES = frozenset(
    {
        ".git", ".hg", ".svn", ".venv", "venv", "env", "node_modules",
        "__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox",
        ".idea", ".vscode", "dist", "build", "target", ".next", ".nuxt",
        ".turbo", ".cache", "coverage", ".gradle", "Pods", "vendor",
    }
)
IGNORED_SUFFIXES = (
    ".pyc", ".pyo", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".webp", ".svg",
    ".woff", ".woff2", ".ttf", ".eot", ".mp3", ".mp4", ".mov", ".zip", ".gz",
    ".tar", ".7z", ".pdf", ".exe", ".dll", ".so", ".dylib", ".class", ".jar",
    ".lock", ".map", ".min.js", ".min.css",
)
MAX_LISTED_FILES = 400
MAX_SCRAPE_CHARS = 12_000


def require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"{name} is not set. Add it to the .env file.")
    return value


def build_chat(temperature: float) -> ChatOpenAI:
    """Chat model for any OpenAI-compatible endpoint, configured from .env.

    LLM_BASE_URL, LLM_MODEL and LLM_API_KEY select the provider (Cerebras,
    Gemini's OpenAI-compatible endpoint, OpenRouter, Groq, ...).
    """
    return _chat_for_model(require_env("LLM_MODEL"), temperature)


# Smaller models to fall back to when LLM_FALLBACK_MODELS is not set.
DEFAULT_FALLBACK_MODELS = {
    "llama-3.3-70b-versatile": ["llama-3.1-8b-instant"],
}
LLM_MAX_ATTEMPTS = 4
LLM_BASE_DELAY_SECONDS = 1.0
LLM_MAX_DELAY_SECONDS = 30.0
LLM_REQUEST_TIMEOUT_SECONDS = 120.0


def _chat_for_model(model: str, temperature: float) -> ChatOpenAI:
    # max_retries=0: retries, backoff and model fallback are handled by
    # GroqChatLLM so they are bounded, jittered and visible in one place.
    return ChatOpenAI(
        model=model,
        base_url=require_env("LLM_BASE_URL"),
        api_key=require_env("LLM_API_KEY"),
        temperature=temperature,
        max_retries=0,
        timeout=LLM_REQUEST_TIMEOUT_SECONDS,
    )


def fallback_model_names() -> list[str]:
    """Models to try, in order, after LLM_MODEL fails.

    LLM_FALLBACK_MODELS (comma-separated) overrides the built-in defaults.
    """
    primary = os.getenv("LLM_MODEL", "")
    configured = os.getenv("LLM_FALLBACK_MODELS")
    if configured is not None:
        names = [name.strip() for name in configured.split(",") if name.strip()]
    else:
        names = list(DEFAULT_FALLBACK_MODELS.get(primary, []))
    return [name for name in names if name != primary]


def build_fallback_chats(temperature: float) -> list[ChatOpenAI]:
    return [_chat_for_model(name, temperature) for name in fallback_model_names()]


class LLMUnavailableError(RuntimeError):
    """Every model in the fallback chain failed."""


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict):
                parts.append(str(part.get("text", "")))
            else:
                parts.append(str(part))
        return "\n".join(p for p in parts if p)
    return str(content)


def _to_langchain_messages(messages: str | list[dict[str, Any]]) -> list[BaseMessage]:
    if isinstance(messages, str):
        return [HumanMessage(content=messages)]
    converted: list[BaseMessage] = []
    for message in messages:
        role = message.get("role", "user")
        text = _content_to_text(message.get("content"))
        if role == "system":
            converted.append(SystemMessage(content=text))
        elif role == "assistant":
            converted.append(AIMessage(content=text))
        else:
            converted.append(HumanMessage(content=text))
    return converted


class GroqChatLLM(BaseLLM):
    """CrewAI LLM that delegates completions to a chain of chat models.

    Each call tries the primary model with exponential backoff and full
    jitter on transient failures (429, 5xx, timeouts, empty answers). If the
    primary keeps failing, or fails with a non-transient error, the next model
    in the fallback chain is tried. LLMUnavailableError is raised only when
    every model has failed.
    """

    llm_type: str = "groq_chat"
    _chat: Any = PrivateAttr()
    _chain: list[Any] = PrivateAttr(default_factory=list)
    _events: list[dict[str, Any]] = PrivateAttr(default_factory=list)

    def __init__(self, chat: Any, fallbacks: list[Any] | None = None, **kwargs: Any) -> None:
        super().__init__(
            model=chat.model_name,
            temperature=chat.temperature,
            provider="groq",
            context_window_size=GROQ_CONTEXT_WINDOW,
            **kwargs,
        )
        self._chat = chat
        self._chain = [chat, *(fallbacks or [])]
        self._events = []

    @property
    def resilience_events(self) -> list[dict[str, Any]]:
        """Retries and fallbacks that happened, for logs and chaos reports."""
        return self._events

    def _invoke_once(self, chat: Any, messages: list[BaseMessage], stop: list[str] | None) -> str:
        response = chat.invoke(messages, stop=stop)
        usage = getattr(response, "usage_metadata", None)
        if usage:
            self._track_token_usage_internal(
                {
                    "prompt_tokens": usage.get("input_tokens", 0),
                    "completion_tokens": usage.get("output_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            )
        text = _content_to_text(response.content)
        if not text.strip():
            raise resilience.EmptyCompletionError(
                f"model {getattr(chat, 'model_name', '?')} returned an empty response"
            )
        return text

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
    ) -> str:
        stop = self.stop_sequences or None
        converted = _to_langchain_messages(messages)
        last_error: BaseException | None = None

        for index, chat in enumerate(self._chain):
            model_name = getattr(chat, "model_name", f"model-{index}")

            def record_retry(attempt: int, error: BaseException, delay: float, model=model_name) -> None:
                self._events.append(
                    {"event": "retry", "model": model, "attempt": attempt,
                     "error": type(error).__name__, "delay": round(delay, 3)}
                )

            try:
                return resilience.retry_with_backoff(
                    lambda chat=chat: self._invoke_once(chat, converted, stop),
                    is_retryable=resilience.is_transient_llm_error,
                    max_attempts=LLM_MAX_ATTEMPTS,
                    base_delay=LLM_BASE_DELAY_SECONDS,
                    max_delay=LLM_MAX_DELAY_SECONDS,
                    on_retry=record_retry,
                )
            except Exception as error:
                last_error = error
                if index + 1 < len(self._chain):
                    self._events.append(
                        {"event": "fallback", "from": model_name,
                         "to": getattr(self._chain[index + 1], "model_name", "?"),
                         "error": type(error).__name__}
                    )

        raise LLMUnavailableError(
            f"All {len(self._chain)} model(s) failed; last error: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def supports_function_calling(self) -> bool:
        # Agents use CrewAI's text (ReAct) tool protocol, which only needs
        # plain completions and stop sequences from the model.
        return False

    def supports_stop_words(self) -> bool:
        return True


class RepoDirectoryReadTool(DirectoryReadTool):
    """DirectoryReadTool that skips dependency/VCS/build noise and caps the listing."""

    def _run(self, **kwargs: Any) -> Any:
        directory = os.path.realpath(kwargs.get("directory") or self.directory or "")
        if not os.path.isdir(directory):
            return f"Error: '{directory}' is not a directory."

        listed: list[str] = []
        truncated = False
        for root, dirs, files in os.walk(directory):
            dirs[:] = sorted(d for d in dirs if d not in IGNORED_DIRECTORIES)
            for filename in sorted(files):
                if filename.lower().endswith(IGNORED_SUFFIXES):
                    continue
                if len(listed) >= MAX_LISTED_FILES:
                    truncated = True
                    break
                relative = os.path.relpath(os.path.join(root, filename), directory)
                listed.append(relative.replace(os.sep, "/"))
            if truncated:
                break

        if not listed:
            return f"No readable source files found under {directory}."
        note = (
            f"\n(Listing truncated at {MAX_LISTED_FILES} files.)" if truncated else ""
        )
        return (
            f"Files under {directory} (paths are relative to it; pass them "
            "unchanged as file_path to the file reading tool):\n- "
            + "\n- ".join(listed)
            + note
        )


class BoundedScrapeWebsiteTool(ScrapeWebsiteTool):
    """ScrapeWebsiteTool whose returned page text is capped in length."""

    def _run(self, **kwargs: Any) -> Any:
        try:
            text = super()._run(**kwargs)
        except Exception as exc:
            return f"Error: could not scrape the page: {exc}"
        text = str(text)
        if len(text) > MAX_SCRAPE_CHARS:
            return text[:MAX_SCRAPE_CHARS] + "\n[...page truncated...]"
        return text
