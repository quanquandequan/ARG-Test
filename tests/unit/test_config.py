"""配置加载器单元测试。"""

import pytest

import src.core.config as cfg_mod
from src.core.config import get_config


def test_get_config_returns_test_config(test_config):
    cfg = get_config()
    assert cfg.app.name == "test"
    assert cfg.retrieval.top_k == 5


def test_get_config_raises_when_not_loaded(monkeypatch):
    monkeypatch.setattr(cfg_mod, "_CONFIG", None, raising=False)
    with pytest.raises(RuntimeError):
        get_config()
