"""Logging setup: rotating UTF-8 file logs (5 MB x 3) plus console output.

The file handler always uses UTF-8 so Arabic messages are stored correctly.
The console handler degrades gracefully on terminals without UTF-8 support
by replacing unencodable characters instead of raising.
"""

from __future__ import annotations

import logging
import logging.handlers
import sys
from pathlib import Path

LOG_FILE_NAME = "changemaster.log"
MAX_BYTES = 5 * 1024 * 1024  # 5 MB per file
BACKUP_COUNT = 3
_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_ROOT_LOGGER_NAME = "changemaster"


class _SafeStreamHandler(logging.StreamHandler):
    """Console handler that never crashes on non-encodable characters."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
            stream = self.stream
            encoding = getattr(stream, "encoding", None) or "utf-8"
            msg.encode(encoding)
            stream.write(msg + self.terminator)
            self.flush()
        except UnicodeEncodeError:
            encoding = getattr(self.stream, "encoding", None) or "ascii"
            safe = self.format(record).encode(encoding, errors="replace").decode(encoding)
            self.stream.write(safe + self.terminator)
            self.flush()
        except RecursionError:
            raise
        except Exception:  # noqa: BLE001 - mirror logging.Handler behaviour
            self.handleError(record)


def default_log_dir() -> Path:
    """Return the default log directory (next to the config directory)."""
    from changemaster.core.config import default_config_dir

    return default_config_dir() / "logs"


def setup_logging(
    log_dir: Path | None = None,
    level: int | str = logging.INFO,
    console: bool = True,
) -> logging.Logger:
    """Configure and return the application root logger.

    Parameters
    ----------
    log_dir:
        Directory for log files (created if needed). Defaults to the
        platform configuration directory.
    level:
        Logging level (name or numeric value).
    console:
        Whether to also log to stderr.

    Returns
    -------
    logging.Logger
        The configured ``"changemaster"`` logger. Calling this function again
        replaces previous handlers (idempotent re-configuration).
    """
    if isinstance(level, str):
        level = logging.getLevelName(level.upper())
        if not isinstance(level, int):
            level = logging.INFO

    logger = logging.getLogger(_ROOT_LOGGER_NAME)
    logger.setLevel(level)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()

    formatter = logging.Formatter(_FORMAT)

    directory = log_dir if log_dir is not None else default_log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        directory / LOG_FILE_NAME,
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    if console:
        console_handler = _SafeStreamHandler(stream=sys.stderr)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger of the application root logger.

    Parameters
    ----------
    name:
        Module-style suffix, e.g. ``"io_engine.raster"``.
    """
    return logging.getLogger(f"{_ROOT_LOGGER_NAME}.{name}")
