"""
Structured logging helpers — CustomLogger with optional structlog JSON output.

All modules obtain their logger via the module-level ``get_logger(__name__)``
function.  Logging is configured once per process (idempotent); subsequent
calls to ``CustomLogger.__init__`` are no-ops.

File logging writes a timestamped rotating log to ``logs/`` by default.
When ``structlog`` is installed, all log output is emitted as JSON;
otherwise standard ``logging`` is used transparently.

Public API (backwards-compatible with existing codebase):
  setup_logging()         — configure root logger (called once in api.py)
  get_logger(name)        — return a logger for the given module name
"""

import logging
import logging.handlers
import os
from datetime import datetime

try:
    import structlog as _structlog
except ImportError:           # structlog is optional; fall back to stdlib
    _structlog = None

_LOG_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"


class CustomLogger:
    """Configures the Python logging subsystem exactly once per process.

    Class-level ``configured`` flag acts as a lightweight singleton guard so
    that importing any module multiple times never re-creates handlers.
    """

    configured: bool = False
    log_file_path: str | None = None

    def __init__(self, log_dir: str = "logs", enable_file_logging: bool = True):
        self.enable_file_logging = enable_file_logging

        if not CustomLogger.configured:
            if enable_file_logging:
                logs_dir = os.path.join(os.getcwd(), log_dir)
                os.makedirs(logs_dir, exist_ok=True)
                log_file = f"{datetime.now().strftime('%m_%d_%Y_%H_%M_%S')}.log"
                CustomLogger.log_file_path = os.path.join(logs_dir, log_file)

            self._configure_logging()
            CustomLogger.configured = True

        self.log_file_path = CustomLogger.log_file_path

    # ------------------------------------------------------------------

    def _configure_logging(self) -> None:
        handlers: list[logging.Handler] = []

        console = logging.StreamHandler()
        console.setLevel(logging.INFO)
        console.setFormatter(logging.Formatter(_LOG_FMT))
        handlers.append(console)

        if self.enable_file_logging and CustomLogger.log_file_path:
            fh = logging.handlers.RotatingFileHandler(
                CustomLogger.log_file_path,
                maxBytes=10 * 1024 * 1024,   # 10 MB
                backupCount=5,
                encoding="utf-8",
                mode="a",
            )
            fh.setLevel(logging.INFO)
            fh.setFormatter(logging.Formatter(_LOG_FMT))
            handlers.append(fh)

        root = logging.getLogger()
        root.handlers.clear()
        logging.basicConfig(
            level=logging.INFO,
            format=_LOG_FMT,
            handlers=handlers,
            force=True,
        )

        # Quieten noisy third-party loggers
        for noisy in ("httpx", "pinecone", "openai", "groq", "urllib3"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        # Configure structlog for JSON output when available
        if _structlog is not None:
            _structlog.configure(
                processors=[
                    _structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp"),
                    _structlog.processors.add_log_level,
                    _structlog.processors.EventRenamer(to="event"),
                    _structlog.processors.JSONRenderer(),
                ],
                logger_factory=_structlog.stdlib.LoggerFactory(),
                cache_logger_on_first_use=True,
            )

    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """Return a (struct)log logger for *name*."""
        logger_name = os.path.basename(name)
        if _structlog is None:
            return logging.getLogger(logger_name)
        return _structlog.get_logger(logger_name)


# ── Module-level init — runs once on first import ─────────────────────────────
_default = CustomLogger()


# ── Public API (backwards-compatible) ────────────────────────────────────────

def setup_logging(log_dir: str = "logs", enable_file_logging: bool = True) -> None:
    """Configure the root logger. Idempotent — safe to call multiple times."""
    CustomLogger(log_dir=log_dir, enable_file_logging=enable_file_logging)


def setup_logger(name: str, log_dir: str = "logs", enable_file_logging: bool = True):
    """Configure logging and return a logger for *name*."""
    CustomLogger(log_dir=log_dir, enable_file_logging=enable_file_logging)
    return CustomLogger.get_logger(name)


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger for *name*. Configures root if needed."""
    return CustomLogger.get_logger(name)
