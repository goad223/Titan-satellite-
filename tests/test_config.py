"""Tests for changemaster.core.config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from changemaster.core.config import AppConfig, ConfigManager, default_config_dir
from changemaster.core.exceptions import ConfigError


class TestDefaultConfigDir:
    def test_returns_changemaster_dir(self) -> None:
        assert default_config_dir().name == "ChangeMaster"

    def test_respects_xdg(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        monkeypatch.setattr("os.name", "posix")
        monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
        assert default_config_dir() == tmp_path / "ChangeMaster"


class TestAppConfigValidation:
    def test_defaults_valid(self) -> None:
        AppConfig().validate()

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"language": "fr"},
            {"theme": "rainbow"},
            {"max_workers": -1},
            {"tile_size": -5},
            {"log_level": "LOUD"},
        ],
    )
    def test_invalid_values_raise(self, kwargs: dict) -> None:
        with pytest.raises(ConfigError):
            AppConfig(**kwargs).validate()

    def test_arabic_language_valid(self) -> None:
        AppConfig(language="ar").validate()


class TestConfigManager:
    def test_load_missing_returns_defaults(self, tmp_path: Path) -> None:
        manager = ConfigManager(config_dir=tmp_path)
        config = manager.load()
        assert config == AppConfig()

    def test_save_and_reload_roundtrip(self, tmp_path: Path) -> None:
        manager = ConfigManager(config_dir=tmp_path)
        manager.update(language="ar", tile_size=2048)
        reloaded = ConfigManager(config_dir=tmp_path).load()
        assert reloaded.language == "ar"
        assert reloaded.tile_size == 2048

    def test_save_writes_utf8_json(self, tmp_path: Path) -> None:
        manager = ConfigManager(config_dir=tmp_path)
        manager.update(cache_dir="مجلد_مؤقت")
        raw = manager.path.read_text(encoding="utf-8")
        assert "مجلد_مؤقت" in raw
        assert json.loads(raw)["cache_dir"] == "مجلد_مؤقت"

    def test_update_unknown_key_raises(self, tmp_path: Path) -> None:
        manager = ConfigManager(config_dir=tmp_path)
        with pytest.raises(ConfigError):
            manager.update(nonexistent=True)

    def test_load_corrupt_json_raises(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("{ not json", encoding="utf-8")
        with pytest.raises(ConfigError):
            ConfigManager(config_dir=tmp_path).load()

    def test_load_non_object_raises(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text("[1, 2]", encoding="utf-8")
        with pytest.raises(ConfigError):
            ConfigManager(config_dir=tmp_path).load()

    def test_load_ignores_unknown_keys(self, tmp_path: Path) -> None:
        (tmp_path / "config.json").write_text(
            json.dumps({"language": "ar", "future_key": 1}), encoding="utf-8"
        )
        config = ConfigManager(config_dir=tmp_path).load()
        assert config.language == "ar"

    def test_recent_files_dedup_and_limit(self, tmp_path: Path) -> None:
        manager = ConfigManager(config_dir=tmp_path)
        for i in range(25):
            manager.add_recent_file(tmp_path / f"f{i}.tif")
        manager.add_recent_file(tmp_path / "f24.tif")
        files = manager.config.recent_files
        assert len(files) == 20
        assert files[0].endswith("f24.tif")
        assert len(set(files)) == len(files)
