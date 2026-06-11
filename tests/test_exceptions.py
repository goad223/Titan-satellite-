"""Tests for changemaster.core.exceptions and logging_setup."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from changemaster.core.exceptions import (
    ChangeMasterError,
    ConfigError,
    DependencyMissingError,
    FormatNotSupportedError,
    HardwareDetectionError,
    ImageReadError,
    ImageWriteError,
    MetadataError,
    SensorProfileError,
)
from changemaster.core.logging_setup import LOG_FILE_NAME, get_logger, setup_logging


class TestExceptionHierarchy:
    @pytest.mark.parametrize(
        "exc_cls",
        [
            ConfigError,
            HardwareDetectionError,
            ImageReadError,
            ImageWriteError,
            MetadataError,
            SensorProfileError,
        ],
    )
    def test_subclasses_base(self, exc_cls: type[ChangeMasterError]) -> None:
        exc = exc_cls("english", "عربي")
        assert isinstance(exc, ChangeMasterError)
        assert exc.message_en == "english"
        assert exc.message_ar == "عربي"
        assert "english" in str(exc) and "عربي" in str(exc)

    def test_arabic_defaults_to_english(self) -> None:
        exc = ChangeMasterError("only english")
        assert exc.message_ar == "only english"
        assert exc.bilingual() == "only english"

    def test_dependency_missing_mentions_package(self) -> None:
        exc = DependencyMissingError("rasterio", "GeoTIFF reading", "قراءة GeoTIFF")
        assert exc.package == "rasterio"
        assert "pip install rasterio" in exc.message_en
        assert "rasterio" in exc.message_ar

    def test_format_not_supported_keeps_path(self) -> None:
        exc = FormatNotSupportedError("/data/file.xyz")
        assert exc.path == "/data/file.xyz"
        assert "/data/file.xyz" in exc.message_ar

    def test_catchable_as_base(self) -> None:
        with pytest.raises(ChangeMasterError):
            raise ImageReadError("boom")


class TestLoggingSetup:
    def test_creates_rotating_utf8_log(self, tmp_path: Path) -> None:
        logger = setup_logging(log_dir=tmp_path, level="DEBUG", console=False)
        logger.info("رسالة عربية مع unicode")
        log_file = tmp_path / LOG_FILE_NAME
        assert log_file.exists()
        assert "رسالة عربية" in log_file.read_text(encoding="utf-8")
        handler = logger.handlers[0]
        assert isinstance(handler, logging.handlers.RotatingFileHandler)
        assert handler.maxBytes == 5 * 1024 * 1024
        assert handler.backupCount == 3

    def test_reconfiguration_replaces_handlers(self, tmp_path: Path) -> None:
        setup_logging(log_dir=tmp_path, console=True)
        logger = setup_logging(log_dir=tmp_path, console=False)
        assert len(logger.handlers) == 1

    def test_invalid_level_falls_back_to_info(self, tmp_path: Path) -> None:
        logger = setup_logging(log_dir=tmp_path, level="NOTALEVEL", console=False)
        assert logger.level == logging.INFO

    def test_console_handler_added(self, tmp_path: Path) -> None:
        logger = setup_logging(log_dir=tmp_path, console=True)
        assert len(logger.handlers) == 2

    def test_get_logger_is_child(self, tmp_path: Path) -> None:
        setup_logging(log_dir=tmp_path, console=False)
        child = get_logger("io_engine.test")
        assert child.name == "changemaster.io_engine.test"
        child.info("child message")
        assert "child message" in (tmp_path / LOG_FILE_NAME).read_text(encoding="utf-8")
