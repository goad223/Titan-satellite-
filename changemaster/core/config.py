"""Persistent application configuration stored as JSON.

On Windows the configuration lives under ``%APPDATA%/ChangeMaster``; on other
platforms it falls back to ``~/.config/ChangeMaster`` (used for automated
testing on Linux CI). All paths are handled with :mod:`pathlib`.
"""

from __future__ import annotations

import json
import os
import threading
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from changemaster.core.exceptions import ConfigError

_CONFIG_FILENAME = "config.json"
_APP_DIR_NAME = "ChangeMaster"


def default_config_dir() -> Path:
    """Return the platform-appropriate configuration directory.

    Windows: ``%APPDATA%/ChangeMaster``.
    Other platforms: ``$XDG_CONFIG_HOME/ChangeMaster`` or
    ``~/.config/ChangeMaster``.
    """
    appdata = os.environ.get("APPDATA")
    if os.name == "nt" and appdata:
        return Path(appdata) / _APP_DIR_NAME
    xdg = os.environ.get("XDG_CONFIG_HOME")
    base = Path(xdg) if xdg else Path.home() / ".config"
    return base / _APP_DIR_NAME


@dataclass
class AppConfig:
    """User-tunable application settings with sensible defaults.

    Attributes
    ----------
    language:
        UI language code: ``"en"`` or ``"ar"``.
    theme:
        UI theme name (``"dark"`` / ``"light"``).
    max_workers:
        Worker process count; ``0`` means auto (derived from hardware).
    tile_size:
        Tile edge length in pixels; ``0`` means auto.
    max_in_memory_mb:
        Largest image (MB) loaded fully into RAM; ``0`` means auto.
    cache_dir:
        Directory for temporary tiles/caches; empty string means default.
    recent_files:
        Most recently opened files (newest first, capped at 20).
    log_level:
        Logging level name, e.g. ``"INFO"``.
    """

    language: str = "en"
    theme: str = "dark"
    max_workers: int = 0
    tile_size: int = 0
    max_in_memory_mb: int = 0
    cache_dir: str = ""
    recent_files: list[str] = field(default_factory=list)
    log_level: str = "INFO"

    def validate(self) -> None:
        """Validate field values, raising :class:`ConfigError` if invalid."""
        if self.language not in ("en", "ar"):
            raise ConfigError(
                f"Invalid language '{self.language}'; expected 'en' or 'ar'.",
                f"لغة غير صالحة '{self.language}'؛ المتوقع 'en' أو 'ar'.",
            )
        if self.theme not in ("dark", "light"):
            raise ConfigError(
                f"Invalid theme '{self.theme}'; expected 'dark' or 'light'.",
                f"سمة غير صالحة '{self.theme}'؛ المتوقع 'dark' أو 'light'.",
            )
        for name in ("max_workers", "tile_size", "max_in_memory_mb"):
            value = getattr(self, name)
            if not isinstance(value, int) or value < 0:
                raise ConfigError(
                    f"'{name}' must be a non-negative integer, got {value!r}.",
                    f"يجب أن يكون '{name}' عدداً صحيحاً غير سالب، وجد {value!r}.",
                )
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ConfigError(
                f"Invalid log level '{self.log_level}'.",
                f"مستوى سجل غير صالح '{self.log_level}'.",
            )


class ConfigManager:
    """Thread-safe load/save manager for :class:`AppConfig`.

    Parameters
    ----------
    config_dir:
        Override of the configuration directory (mainly for tests).
    """

    def __init__(self, config_dir: Path | None = None) -> None:
        self._dir: Path = config_dir if config_dir is not None else default_config_dir()
        self._path: Path = self._dir / _CONFIG_FILENAME
        self._lock = threading.RLock()
        self._config: AppConfig = AppConfig()

    @property
    def path(self) -> Path:
        """Full path of the JSON configuration file."""
        return self._path

    @property
    def config(self) -> AppConfig:
        """The current in-memory configuration object."""
        return self._config

    def load(self) -> AppConfig:
        """Load configuration from disk, creating defaults when absent.

        Unknown keys in the file are ignored; missing keys keep defaults.
        Corrupt JSON raises :class:`ConfigError`.
        """
        with self._lock:
            if not self._path.exists():
                self._config = AppConfig()
                return self._config
            try:
                raw: Any = json.loads(self._path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise ConfigError(
                    f"Failed to read configuration file {self._path}: {exc}",
                    f"فشل في قراءة ملف الإعدادات {self._path}: {exc}",
                ) from exc
            if not isinstance(raw, dict):
                raise ConfigError(
                    f"Configuration file {self._path} must contain a JSON object.",
                    f"يجب أن يحتوي ملف الإعدادات {self._path} على كائن JSON.",
                )
            known = {f.name for f in fields(AppConfig)}
            kwargs = {k: v for k, v in raw.items() if k in known}
            config = AppConfig(**kwargs)
            config.validate()
            self._config = config
            return self._config

    def save(self) -> None:
        """Persist the current configuration atomically (write + replace)."""
        with self._lock:
            self._config.validate()
            try:
                self._dir.mkdir(parents=True, exist_ok=True)
                tmp = self._path.with_suffix(".json.tmp")
                tmp.write_text(
                    json.dumps(asdict(self._config), indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                tmp.replace(self._path)
            except OSError as exc:
                raise ConfigError(
                    f"Failed to write configuration file {self._path}: {exc}",
                    f"فشل في كتابة ملف الإعدادات {self._path}: {exc}",
                ) from exc

    def update(self, **kwargs: Any) -> AppConfig:
        """Update fields by keyword, validate, persist, and return the config.

        Raises :class:`ConfigError` for unknown field names.
        """
        with self._lock:
            known = {f.name for f in fields(AppConfig)}
            for key, value in kwargs.items():
                if key not in known:
                    raise ConfigError(
                        f"Unknown configuration key '{key}'.",
                        f"مفتاح إعدادات غير معروف '{key}'.",
                    )
                setattr(self._config, key, value)
            self.save()
            return self._config

    def add_recent_file(self, path: Path | str, limit: int = 20) -> None:
        """Insert ``path`` at the top of the recent-files list (deduplicated)."""
        with self._lock:
            entry = str(path)
            files = [f for f in self._config.recent_files if f != entry]
            files.insert(0, entry)
            self._config.recent_files = files[:limit]
            self.save()
