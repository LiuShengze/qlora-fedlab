"""读取 spotcheck.md 里你填的人工判断，计算与自动指标的一致率。

用法：python spotcheck_agreement.py
产物：results/spotcheck_agreement.md
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "results" / "spotcheck.md"
OUT = ROOT / "results" / "spotcheck_agreement.md"

THRESHOLD = 0.5  # 指标 >= 50% 判为"有复读"


def parse() -> list[dict]:
    rows = []
    for line in SRC.read_text(encoding="utf-8").splitlines():
        if not line.startswith("| ") or line.startswith("| #") or line.startswith("|---"):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) < 6:
            continue
        judgment = cells[-1]
        if judgment not in ("有", "无"):
            continue
        metric = float(re.search(r"([\d.]+)%", cells[-2]).group(1)) / 100
        rows.append({
            "no": cells[0],
            "qid": cells[1],
            "condition": cells[2],
            "metric": metric,
            "human": judgment,
        })
    return rows


def main() -> int:
    rows = parse()
    if not rows:
        print("没有读到人工判断，检查 spotcheck.md 最后一列是否填了「有/无」")
        return 1

    for row in rows:
        row["human_pos"] = row["human"] == "有"
        row["metric_pos"] = row["metric"] >= THRESHOLD
        row["agree"] = row["human_pos"] == row["metric_pos"]

    agree = sum(r["agree"] for r in rows)
    tp = sum(r["human_pos"] and r["metric_pos"] for r in rows)
    tn = sum(not r["human_pos"] and not r["metric_pos"] for r in rows)
    fn = sum(r["human_pos"] and not r["metric_pos"] for r in rows)
    fp = sum(not r["human_pos"] and r["metric_pos"] for r in rows)

    pos = [r["metric"] for r in rows if r["human_pos"]]
    neg = [r["metric"] for r in rows if not r["human_pos"]]
    wins = sum(1 for p in pos for n in neg if p > n)
    ties = sum(1 for p in pos for n in neg if p == n)
    auc = (wins + 0.5 * ties) / (len(pos) * len(neg)) if pos and neg else float("nan")

    lines = [
        "# 指标与人工判断的一致性",
        "",
        f"样本：{len(rows)} 条 ｜ 判定阈值：指标 ≥ {THRESHOLD:.0%} 记为「有复读」",
        "",
        f"- **一致率：{agree}/{len(rows)} = {100 * agree / len(rows):.0f}%**",
        f"- AUC = **{auc:.2f}**（1.0 = 完美区分，0.5 = 随机）",
        f"- 命中：有→有 {tp} 条，无→无 {tn} 条",
        f"- 漏报（人工有、指标低）：{fn} 条 ｜ 误报（人工无、指标高）：{fp} 条",
        "",
        "| # | 题号 | 条件 | 指标 | 人工 | 指标判定 | 一致 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['no']} | {row['qid']} | {row['condition']} | {row['metric']:.1%} | "
            f"{row['human']} | {'有' if row['metric_pos'] else '无'} | {'✅' if row['agree'] else '❌'} |"
        )
    lines += [
        "",
        "## 分歧说明",
        "",
        "- **漏报**：指标基于字符 4-gram，抓不到「语义重复」——比如模型反复说同一件事但换了说法。",
        "- **误报**：正常的句式重复（如「X 是一种……的算法，用于……」出现两次）会被判成复读。",
        "",
        "结论：该指标适合做**排序与趋势**判断（哪个模型更爱复读），不适合当准确率使用。",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"一致率 {agree}/{len(rows)} = {100 * agree / len(rows):.0f}%  AUC={auc:.2f}")
    print(f"漏报 {fn} 条，误报 {fp} 条")
    print(f"已写出 {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
