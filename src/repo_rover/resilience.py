"""Retry with exponential backoff and full jitter, and error classification.

Used by RepoRover's own LLM and database calls. `SLEEP` and `RNG` are module
attributes looked up at call time, so tests and chaos experiments can replace
them and run instantly and reproducibly.
"""

from __future__ import annotations

import random
import time
from typing import Any, Callable, TypeVar

T = TypeVar("T")

SLEEP: Callable[[float], None] = time.sleep
RNG: random.Random = random.Random()

RETRYABLE_HTTP_STATUS = frozenset({408, 409, 425, 429, 500, 502, 503, 504, 529})
RETRYABLE_LLM_ERROR_NAMES = frozenset(
    {"RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError",
     "ServiceUnavailableError", "EmptyCompletionError"}
)


class SimulatedHTTPError(Exception):
    """An HTTP failure with a status code, as raised by fault injectors."""

    def __init__(self, status_code: int, message: str = "", retry_after: float | None = None):
        super().__init__(message or f"simulated HTTP {status_code}")
        self.status_code = status_code
        self.retry_after = retry_after


class EmptyCompletionError(Exception):
    """The model returned no text."""


class RetryExhausted(Exception):
    """Every attempt failed with a retryable error."""

    def __init__(self, attempts: int, last_error: BaseException):
        super().__init__(f"gave up after {attempts} attempt(s): {type(last_error).__name__}: {last_error}")
        self.attempts = attempts
        self.last_error = last_error


def full_jitter_delay(attempt: int, base_delay: float, max_delay: float, rng: random.Random | None = None) -> float:
    """Full jitter: a uniform random delay in [0, min(max_delay, base * 2^attempt)].

    `attempt` is 0 for the wait after the first failure.
    """
    ceiling = min(max_delay, base_delay * (2 ** attempt))
    return (rng or RNG).uniform(0.0, ceiling)


def status_code_of(error: BaseException) -> int | None:
    for name in ("status_code", "status", "http_status"):
        value = getattr(error, name, None)
        if isinstance(value, int):
            return value
    response = getattr(error, "response", None)
    value = getattr(response, "status_code", None)
    return value if isinstance(value, int) else None


def is_transient_llm_error(error: BaseException) -> bool:
    """Rate limits, overload, timeouts, dropped connections and empty answers."""
    status = status_code_of(error)
    if status is not None:
        return status in RETRYABLE_HTTP_STATUS
    if isinstance(error, (TimeoutError, ConnectionError, EmptyCompletionError)):
        return True
    return type(error).__name__ in RETRYABLE_LLM_ERROR_NAMES


def is_transient_db_error(error: BaseException) -> bool:
    """Connection failures, statement timeouts, deadlocks and serialization
    failures. Constraint violations and SQL errors are not retried."""
    try:
        import psycopg2
        from psycopg2 import errors
    except ImportError:  # pragma: no cover
        return isinstance(error, (TimeoutError, ConnectionError))
    if isinstance(error, (psycopg2.OperationalError, psycopg2.InterfaceError)):
        return True  # includes QueryCanceled (statement timeout) and lost connections
    if isinstance(error, (errors.DeadlockDetected, errors.SerializationFailure)):
        return True
    return isinstance(error, (TimeoutError, ConnectionError))


def retry_with_backoff(
    operation: Callable[[], T],
    *,
    is_retryable: Callable[[BaseException], bool],
    max_attempts: int = 4,
    base_delay: float = 0.5,
    max_delay: float = 20.0,
    on_retry: Callable[[int, BaseException, float], Any] | None = None,
) -> T:
    """Run `operation`, retrying retryable failures with full-jitter backoff.

    A non-retryable error is raised immediately. After `max_attempts`
    retryable failures, RetryExhausted is raised from the last error. A
    `retry_after` attribute on the error (seconds) raises the wait to at least
    that long, capped at `max_delay`.
    """
    attempt = 0
    while True:
        try:
            return operation()
        except Exception as error:
            if not is_retryable(error):
                raise
            attempt += 1
            if attempt >= max_attempts:
                raise RetryExhausted(attempt, error) from error
            delay = full_jitter_delay(attempt - 1, base_delay, max_delay)
            retry_after = getattr(error, "retry_after", None)
            if isinstance(retry_after, (int, float)) and retry_after > 0:
                delay = max(delay, min(float(retry_after), max_delay))
            if on_retry is not None:
                on_retry(attempt, error, delay)
            SLEEP(delay)
