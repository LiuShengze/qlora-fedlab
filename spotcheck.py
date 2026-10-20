"""生成人工抽查表：从 eval.csv 里抽答案，给你逐条判"有没有复读"。

产物：results/spotcheck.md
  - 上半部分：表格，每行一条，最后一列留空给你填「有」/「无」
  - 下半部分：完整答案（表格里只放前 160 字，判不准时看这里）

用法：
    python spotcheck.py                 # 每个条件抽 8 条
    python spotcheck.py --per 10        # 每个条件抽 10 条
"""
from __future__ import annotations

import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path

from analysis import repeat_ratio

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "results" / "eval.csv"
OUT = ROOT / "results" / "spotcheck.md"

CONDITIONS = ("0.6B-base", "0.6B-sft")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per", type=int, default=8, help="每个条件抽几条")
    args = parser.parse_args()

    if not CSV_PATH.exists():
        print(f"找不到 {CSV_PATH}，先跑 python eval.py")
        return 1

    rows = [r for r in csv.DictReader(open(CSV_PATH, encoding="utf-8"))
            if r["condition"] in CONDITIONS]
    by_condition: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_condition[row["condition"]].append(row)

    rng = random.Random(2026)
    picked: list[dict] = []
    for condition in CONDITIONS:
        items = by_condition.get(condition, [])
        picked.extend(rng.sample(items, min(args.per, len(items))))

    lines = [
        "# 复读抽查表",
        "",
        "对每条答案判断：**有没有整句/整段在重复**（不是指同一个词出现多次）。",
        "最后一列填 `有` 或 `无`。填完把文件发我，我算与自动指标的一致率。",
        "",
        "| # | 题号 | 条件 | 答案（前 160 字） | 指标复读率 | 人工判断（有/无） |",
        "|---|---|---|---|---|---|",
    ]
    for i, row in enumerate(picked, 1):
        excerpt = row["answer"][:160].replace("|", "｜")
        metric = repeat_ratio(row["answer"])
        lines.append(
            f"| {i} | {row['question_id']} | {row['condition']} | {excerpt} | {100 * metric:.1f}% |  |"
        )

    lines += ["", "---", "", "## 完整答案", ""]
    for i, row in enumerate(picked, 1):
        lines.append(f"**{i}. {row['question_id']} / {row['condition']}** — 指标复读率 {100 * repeat_ratio(row['answer']):.1f}%")
        lines.append("")
        lines.append("> " + row["answer"].replace("\n", " "))
        lines.append("")

    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已写出 {OUT}")
    print(f"共 {len(picked)} 条（{', '.join(f'{c} {len(by_condition.get(c, []))} 条' for c in CONDITIONS)} 中抽样）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
