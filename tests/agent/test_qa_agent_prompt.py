"""QA Agent profile 提示词约束测试。"""

from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


def test_qa_agent_prompt_declares_source_priority():
    cfg = OmegaConf.load(Path("configs/default.yaml"))
    profile = cfg.agent.profiles.qa_agent
    prompt = profile["system_prompt"]

    assert "Excel 测试用例作为事实来源" in prompt
    assert "Bug 记录只用于说明历史缺陷" in prompt
    assert "XMind 只作为历史测试思路或辅助参考" in prompt
    assert "网络补充" in prompt
    assert "未命中 Excel" in prompt
    assert "当前 PRD 为唯一需求事实来源" in prompt
    assert "不要再调用 search_knowledge 补全功能结论" in prompt
    assert "draft 模式" in prompt
    assert "final 模式" in prompt
    assert "analysis_json_path" in prompt
    assert "不要重新调用 analyze_requirement" in prompt
    assert "UI 文案、按钮名、入口名、页面名、状态文案必须逐字引用" in prompt
    assert "不得用同义词、美化词或产品经验改写" in prompt
