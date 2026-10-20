"""用构造用例验证两个指标的实现是否符合定义。

这只证明「代码算得对」，不证明「指标有意义」——后者要靠人工抽查与相关性分析，
见 README / REPORT 里的说明。

运行：python tests/test_metrics.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis import is_format_clean, repeat_ratio  # noqa: E402

CASES = [
    # (说明, 文本, 期望的复读率区间)
    ("完全循环（周期 2）", "重复重复重复重复重复重复重复重复", (0.9, 1.0)),
    ("正常中文回答", "FedProx 在本地目标函数里加了近端项，用来限制本地模型偏离全局模型。", (0.0, 0.05)),
    ("正常但含编号的列表", "1. 先看问题 2. 再看数据 3. 最后看指标", (0.0, 0.6)),
    ("长度不足 2n", "短", (0.0, 0.0)),
    ("空串", "", (0.0, 0.0)),
]


def main() -> int:
    failures = 0

    for name, text, (lo, hi) in CASES:
        value = repeat_ratio(text)
        ok = lo <= value <= hi
        failures += not ok
        print(f"[{'OK ' if ok else 'FAIL'}] 复读率 {name:16} = {value:.3f}  期望 [{lo}, {hi}]")

    format_cases = [
        ("正常回答", "QLoRA 是一种 4-bit 量化微调方法。", True),
        ("回显 Response", "### Response:\nQLoRA 是一种方法", False),
        ("回显 Instruction", "Instruction: 你好", False),
        ("空回答", "   ", False),
    ]
    for name, text, expected in format_cases:
        got = is_format_clean(text)
        ok = got is expected
        failures += not ok
        print(f"[{'OK ' if ok else 'FAIL'}] 格式合规 {name:16} = {got}  期望 {expected}")

    print(f"\n{'全部通过' if failures == 0 else f'{failures} 项失败'}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
