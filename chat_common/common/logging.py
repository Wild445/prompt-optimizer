"""Request-scoped context and structured JSON logging.

No module configures its own logger. Importing ``logger`` from here is what
guarantees a single transaction id appears on every line the service emits,
including lines written from inside a domain graph node.

The contextvars live here too, because the logger is their primary consumer.
Graph nodes do NOT read them â€” they read ``runtime.context`` instead, which is
what keeps nodes callable in isolation. The API layer is the bridge: it reads
these once and freezes them into a ``GraphContext``.
"""

import contextvars
import logging
from datetime import datetime
from typing import Any

from pythonjsonlogger.json import JsonFormatter

# ---------------------------------------------------------------------------
# Request-scoped context
# ---------------------------------------------------------------------------
transaction_id_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("transaction_id", default=None)
access_token_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("access_token", default=None)
request_headers_var: contextvars.ContextVar[Any] = contextvars.ContextVar("request_headers", default=None)
security_wallet_var: contextvars.ContextVar[Any] = contextvars.ContextVar("security_wallet", default=None)
user_email_var: contextvars.ContextVar[str | None] = contextvars.ContextVar("user_email", default=None)


def current_transaction_id() -> str:
    """Return the active transaction id, or ``"N/A"`` outside a request."""
    return transaction_id_var.get() or "N/A"


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOGGER_NAME = "chat_service"

RESERVED_LOG_FIELDS = frozenset(
    {
        "args", "asctime", "created", "exc_info", "exc_text", "filename", "funcName", "levelname", "levelno",
        "lineno", "message", "module", "msecs", "msg", "name", "pathname", "process", "processName",
        "relativeCreated", "stack_info", "taskName", "thread", "threadName",
    }
)  # fmt: skip
"""Names ``logging`` refuses to accept via ``extra``.

Passing one â€” ``extra={"message": ...}`` is the easy mistake â€” raises
``KeyError: Attempt to overwrite 'message' in LogRecord`` from inside
``makeRecord``. On an error-handling path that turns a clean 4xx into an
unhandled 500, so the platform logger sanitises rather than trusting callers.
"""


class CustomJsonFormatter(JsonFormatter):
    """JSON formatter that stamps severity, timestamp, and transaction id."""

    def add_fields(self, log_record, record, message_dict):
        """Add platform fields â€” severity, timestamp, transaction id â€” to every log record."""
        super().add_fields(log_record, record, message_dict)
        log_record["severity"] = "ERROR" if record.levelno >= logging.ERROR else record.levelname
        log_record["dateTime"] = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "+0000"
        if record.exc_info:
            log_record["exc_info"] = self.formatException(record.exc_info)
        try:
            log_record["transactionId"] = transaction_id_var.get() or "N/A"
        except LookupError:
            log_record["transactionId"] = "N/A"
        if "extra" in message_dict:
            for key, value in message_dict["extra"].items():
                if key not in log_record:
                    log_record[key] = value


class SafeExtraLogger(logging.LoggerAdapter):
    """Logger that renames ``extra`` keys colliding with LogRecord internals.

    Every module logs through this adapter, so a domain graph author can pass any
    ``extra`` dict without knowing the reserved list.
    """

    def process(self, msg, kwargs):
        """Suffix reserved keys in ``extra`` so the record can be built."""
        extra = kwargs.get("extra")
        if extra:
            kwargs["extra"] = {(f"{key}_" if key in RESERVED_LOG_FIELDS else key): value for key, value in extra.items()}
        return msg, kwargs


def build_logger(level: int = logging.INFO) -> SafeExtraLogger:
    """Create the platform logger with a single JSON handler."""
    instance = logging.getLogger(LOGGER_NAME)
    if not instance.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(CustomJsonFormatter("%(process)d %(asctime)s %(levelname)s %(message)s %(transactionId)s"))
        instance.addHandler(handler)
    instance.setLevel(level)
    instance.propagate = False
    return SafeExtraLogger(instance, {})


logger = build_logger()