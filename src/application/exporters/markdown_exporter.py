"""Markdown exporter for TestDesignArtifact."""

from __future__ import annotations

from pathlib import Path

from src.domain.artifacts.test_design_artifact import TestDesignArtifact


class MarkdownExporter:
    """Export a lightweight human-readable test design summary."""

    def export(self, artifact: TestDesignArtifact, path: Path) -> None:
        path.write_text(self.render(artifact), encoding="utf-8")

    def render(self, artifact: TestDesignArtifact) -> str:
        lines = [
            f"# 测试设计产物：{artifact.module}",
            "",
            f"- 生成模式：{artifact.generation_mode}",
            f"- 需求功能点：{artifact.requirement_ir.feature_count()}",
            f"- 测试点：{len(artifact.test_points)}",
            f"- 测试场景：{len(artifact.scenarios)}",
            f"- 测试用例：{len(artifact.test_cases)}",
            "",
            "## 测试点",
        ]
        for point in artifact.test_points:
            lines.append(f"- {point.id} [{point.priority}] {point.title}")
        lines += ["", "## 测试场景"]
        for scenario in artifact.scenarios:
            lines.append(f"- {scenario.id} [{scenario.data_state}] {scenario.title}")
        return "\n".join(lines).strip() + "\n"
