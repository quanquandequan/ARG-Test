"""基于 OmegaConf 的配置加载器，支持 YAML 合并和环境变量插值。"""

import os
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

_CONFIG: DictConfig | None = None


def _find_config_dir() -> Path:
    env_path = os.environ.get("RAG_CONFIG_DIR")
    if env_path:
        return Path(env_path)
    return Path(__file__).parent.parent.parent / "configs"


def load_config(env: str | None = None) -> DictConfig:
    global _CONFIG
    if _CONFIG is not None:
        return _CONFIG

    config_dir = _find_config_dir()
    env = env or os.environ.get("RAG_ENV", "development")

    base = OmegaConf.load(config_dir / "default.yaml")
    env_file = config_dir / f"{env}.yaml"
    if env_file.exists():
        env_cfg = OmegaConf.load(env_file)
        base = OmegaConf.merge(base, env_cfg)

    resolver = _EnvVarResolver()
    OmegaConf.register_new_resolver("env", resolver.resolve, replace=True)
    # 应用环境变量插值：${env:LLM_PROVIDER}
    base = OmegaConf.create(OmegaConf.to_container(base, resolve=True))

    _CONFIG = base
    return _CONFIG


def get_config() -> DictConfig:
    if _CONFIG is None:
        raise RuntimeError("Config not loaded. Call load_config() first.")
    return _CONFIG


class _EnvVarResolver:
    def resolve(self, key: str, default: str = "") -> str:
        return os.environ.get(key, default)
