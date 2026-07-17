"""Tests for typed configuration loading, overrides and validation."""

from __future__ import annotations

import logging

import pytest

from gesturesense.config.settings import AppConfig, load_config


def test_defaults_load_without_file():
    config = load_config(None)
    assert isinstance(config, AppConfig)
    assert config.camera.width == 1280
    assert config.tracking.max_hands == 2
    assert config.ui.theme == "dark"


def test_yaml_overrides_subset(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text(
        "camera:\n  width: 640\n  height: 480\nui:\n  theme: light\n",
        encoding="utf-8",
    )
    config = load_config(path)
    assert config.camera.width == 640
    assert config.camera.height == 480
    assert config.ui.theme == "light"
    # Untouched values keep their defaults.
    assert config.camera.fps == 30
    assert config.tracking.max_hands == 2


def test_unknown_key_warns_but_does_not_crash(tmp_path, caplog):
    path = tmp_path / "config.yaml"
    path.write_text("camera:\n  wdith: 640\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        config = load_config(path)
    assert config.camera.width == 1280  # typo ignored, default kept
    assert any("wdith" in record.message for record in caplog.records)


def test_missing_file_falls_back_to_defaults(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        config = load_config(tmp_path / "nope.yaml")
    assert config == AppConfig()


def test_malformed_yaml_falls_back_to_defaults(tmp_path, caplog):
    path = tmp_path / "config.yaml"
    path.write_text("camera: [unclosed\n", encoding="utf-8")
    with caplog.at_level(logging.ERROR):
        config = load_config(path)
    assert config == AppConfig()


@pytest.mark.parametrize(
    "yaml_text",
    [
        "camera:\n  width: -1\n",
        "tracking:\n  detection_confidence: 1.5\n",
        "tracking:\n  max_hands: 9\n",
        "gestures:\n  min_votes: 99\n",
        "ui:\n  theme: neon\n",
    ],
)
def test_invalid_values_raise(tmp_path, yaml_text):
    path = tmp_path / "config.yaml"
    path.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError):
        load_config(path)


def test_shipped_default_config_matches_dataclass_defaults():
    """The commented YAML shipped with the app must stay in sync."""
    from pathlib import Path

    shipped = Path(__file__).parent.parent / "gesturesense/config/default_config.yaml"
    assert shipped.exists()
    assert load_config(shipped) == AppConfig()
