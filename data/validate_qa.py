"""校验问答数据（jsonl）是否适合直接拿去训练。

检查项：
  1. 每行是合法 JSON，且字段齐全（instruction / output 必填，input / source / category 可选）
  2. instruction、output 不能为空
  3. 答案长度在合理区间（太短没信息量，超过 400 字会挤占 512 的序列窗口）
  4. 没有重复的 instruction
  5. 文件是有效 UTF-8，且没有常见乱码字符（å、ã、\ufffd 这类）

用法：
    python data/validate_qa.py                 # 校验默认的 data/lab_qa.jsonl
    python data/validate_qa.py data/xxx.jsonl  # 校验指定文件
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

DEFAULT = Path(__file__).resolve().parent / "lab_qa.jsonl"
MIN_ANSWER, MAX_ANSWER = 10, 400
MOJIBAKE = ("å", "ã", "æ", "è", "\ufffd")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    if not path.exists():
        print(f"[FAIL] 文件不存在：{path}")
        return 1

    try:
        raw = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        print(f"[FAIL] 不是有效 UTF-8：{exc}")
        return 1

    problems: list[str] = []
    rows: list[dict] = []
    for lineno, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            problems.append(f"第 {lineno} 行 JSON 解析失败：{exc}")
            continue

        for field in ("instruction", "output"):
            if not str(row.get(field, "")).strip():
                problems.append(f"第 {lineno} 行 {field} 为空")
        answer = str(row.get("output", ""))
        if len(answer) < MIN_ANSWER:
            problems.append(f"第 {lineno} 行答案过短（{len(answer)} 字）")
        if len(answer) > MAX_ANSWER:
            problems.append(f"第 {lineno} 行答案过长（{len(answer)} 字 > {MAX_ANSWER}）")
        if any(ch in answer for ch in MOJIBAKE):
            problems.append(f"第 {lineno} 行疑似乱码")
        rows.append(row)

    dupes = [q for q, c in Counter(r.get("instruction", "") for r in rows).items() if c > 1]
    for q in dupes:
        problems.append(f"重复的 instruction：{q[:30]}…")

    print(f"文件：{path}")
    print(f"条数：{len(rows)}")

    if rows:
        lengths = [len(str(r.get('output', ''))) for r in rows]
        print(f"答案长度：min={min(lengths)} 平均={sum(lengths) // len(lengths)} max={max(lengths)}")
        sources = Counter(r.get("source", "(未标注)") for r in rows)
        print("来源分布：")
        for name, count in sources.most_common():
            print(f"  {count:3d}  {name}")

    if problems:
        print(f"\n发现 {len(problems)} 个问题：")
        for p in problems[:20]:
            print(f"  - {p}")
        return 1

    print("\n[OK] 校验通过，可以直接用于训练。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
