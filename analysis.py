"""从 results/eval.csv 里量化"微调到底改变了什么"。

三个指标（都不需要人工判断，纯文本统计）：
  1. 格式合规率：答案里不出现模板标记（### / Response:）的比例，越高越好
  2. 复读率：重复出现的 4-gram 占比，越高说明越像复读机
  3. 平均长度：微调后变短是"定向回答"最直观的指纹

人工事实对错仍看 results/eval.md 末尾的打分表。

用法：python analysis.py
"""
from __future__ import annotations

import csv
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results" / "eval.csv"
OUT_MD = ROOT / "results" / "analysis.md"

TEMPLATE_MARKS = ("###", "Response:", "Instruction:", "Input:")


def is_format_clean(answer: str) -> bool:
    """格式合规：答案里没有回显 prompt 模板的痕迹，且不是空回答。

    注意"不是空回答"这条是必需的——否则一个什么都不说的模型会拿满分。
    """
    text = answer.strip()
    if not text:
        return False
    return not any(mark in text for mark in TEMPLATE_MARKS)


def repeat_ratio(text: str, n: int = 4) -> float:
    """重复 n-gram 占比：0 表示没有复读，接近 1 表示整段在重复。

    按"字符"切 n-gram（中文没有空格，按字最稳）。分母是全部 n-gram 数，
    分子是"出现次数 >1 的那些 n-gram 的总出现次数"，所以一段完全重复的话会趋近 1。
    长度不足 2n 时返回 0（样本太短，无法判断）。
    """
    tokens = list(text)
    if len(tokens) < n * 2:
        return 0.0
    grams = Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))
    repeated = sum(count for count in grams.values() if count > 1)
    return repeated / sum(grams.values())


def main() -> int:
    if not CSV_PATH.exists():
        print(f"找不到 {CSV_PATH}，先跑 python eval.py")
        return 1

    with open(CSV_PATH, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    header = [
        "| 条件 | 题数 | 平均长度 | 格式合规率 | 复读率 | 空回答 |",
        "|---|---|---|---|---|---|",
    ]
    lines = list(header)
    for condition, items in by_condition.items():
        answers = [item["answer"].strip() for item in items]
        n = len(answers)
        avg_len = sum(len(a) for a in answers) / n
        clean = sum(1 for a in answers if is_format_clean(a))
        repeat = sum(repeat_ratio(a) for a in answers) / n
        empty = sum(1 for a in answers if not a)
        lines.append(
            f"| {condition} | {n} | {avg_len:.0f} 字 | {100 * clean / n:.0f}% | "
            f"{100 * repeat:.1f}% | {empty} |"
        )

    table = "\n".join(lines)
    print(table)

    OUT_MD.write_text(
        "# 生成质量统计（自动）\n\n"
        "由 `analysis.py` 从 `results/eval.csv` 计算，无需人工判断。\n\n"
        f"{table}\n\n"
        "说明：\n"
        "- **格式合规率**：答案中不出现 `###` / `Response:` 等模板标记的比例\n"
        "- **复读率**：重复 4-gram 占比，0.6B 底座这类模型会明显偏高\n"
        "- **平均长度**：微调后变短 = 学会了「直接回答」而不是「接着续写」\n\n"
        "事实正确率不在此表，见 `results/eval.md` 末尾的人工打分表。\n",
        encoding="utf-8",
    )
    print(f"\n已写出 {OUT_MD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
