from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json


def _display(value: Any) -> str:
    if value is None:
        return "待实验补充"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def render_report(results_path: Path, output_path: Path) -> None:
    payload = read_json(results_path)
    lines = ["# EcoSpec-KG实验结果", "", f"状态：{payload.get('status', 'unknown')}", ""]
    lines.extend(["## 主要结果", "", "| 方法 | 结果 |", "|---|---|"])
    for method, result in payload.get("main_results", {}).items():
        lines.append(f"| {method} | {_display(result)} |")
    expert = payload.get("expert_validation", {})
    lines.extend(
        [
            "",
            "## 专家验证",
            "",
            f"- 样本量：{_display(expert.get('sample_size'))}",
            f"- 专家数量：{_display(expert.get('reviewer_count'))}",
            f"- 接受率：{_display(expert.get('acceptance_rate'))}",
            f"- 一致性：{_display(expert.get('agreement'))}",
            "",
            "> 未运行指标保持“待实验补充”，不得据此形成研究结论。",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

