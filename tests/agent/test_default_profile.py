"""默认 Agent profile 配置测试。"""

from __future__ import annotations

import pytest
from omegaconf import OmegaConf

import src.core.config as cfg_mod
from src.bootstrap import UnknownAgentProfileError, _resolve_agent_profile


def test_default_profile_resolves_to_qa_agent(monkeypatch):
    monkeypatch.setattr(
        cfg_mod,
        "_CONFIG",
        OmegaConf.create(
            {
                "agent": {
                    "profiles": {
                        "qa_agent": {
                            "tools": [
                                "search_knowledge",
                                "analyze_requirement",
                                "design_test_cases",
                                "execute_scenario",
                            ]
                        }
                    },
                }
            }
        ),
        raising=False,
    )

    profile = _resolve_agent_profile(None)

    assert profile["tools"] == [
        "search_knowledge",
        "analyze_requirement",
        "design_test_cases",
        "execute_scenario",
    ]


def test_unknown_profile_raises_config_error(monkeypatch):
    monkeypatch.setattr(
        cfg_mod,
        "_CONFIG",
        OmegaConf.create(
            {
                "agent": {
                    "profiles": {
                        "qa_agent": {"tools": ["search_knowledge"]},
                        "mobile_debug": {"tools": ["device_tool"]},
                    },
                }
            }
        ),
        raising=False,
    )

    with pytest.raises(UnknownAgentProfileError) as exc_info:
        _resolve_agent_profile("requirement_agent")

    assert exc_info.value.available_profiles == ["mobile_debug", "qa_agent"]
    assert "requirement_agent" in str(exc_info.value)
