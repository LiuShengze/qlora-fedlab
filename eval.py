"""Held-out 评测：15 道题 × 4 个条件，同一 prompt 模板、greedy 解码。

4 个条件：
    A 0.6B-base   底座，不挂 adapter
    B 0.6B-sft    底座 + 0.6B 的 LoRA adapter
    C 4B-base     底座，不挂 adapter
    D 4B-sft      底座 + 4B 的 LoRA adapter

为了可比，四个条件**统一用 4-bit 加载**（挂 adapter 时不合并、直接推理），
这样差异只来自"模型规模"和"有没有微调"，不掺入量化差异。

产物：
    results/eval.csv   机器可读，便于你后面人工打分
    results/eval.md    按题目分组，四组答案并排，直接看差异

用法：
    python eval.py                 # 跑全部可用条件
    python eval.py --only 4B       # 只跑 4B 的两组
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

MODELS_DIR = ROOT / "models"          # 仓库内；由 scripts 或手动放置

QUESTIONS = [
    ("Q01", "通用", "QQ 和微信的区别是什么？"),
    ("Q02", "通用", "王者荣耀是什么？"),
    ("Q03", "通用", "企鹅主要生活在哪里？"),
    ("Q04", "通用", "“代码复现”用英文怎么说？"),
    ("Q05", "通用", "如何取得好成绩？"),
    ("Q06", "通用", "如何安慰一个刚失恋的朋友？"),
    ("Q07", "技术", "什么是过拟合？怎么判断模型过拟合了？"),
    ("Q08", "技术", "梯度裁剪（gradient clipping）是做什么用的？"),
    ("Q09", "技术", "用一句话解释什么是 LoRA。"),
    ("Q10", "技术", "介绍你自己。"),
    ("Q11", "隐私/联邦", "FedAvg 和 FedProx 的主要区别是什么？"),
    ("Q12", "隐私/联邦", "差分隐私里的 ε（epsilon）代表什么？ε 变大意味着什么？"),
    ("Q13", "隐私/联邦", "LoRA 在联邦学习场景下为什么能显著降低通信量？"),
    ("Q14", "隐私/联邦", "DP-SGD 比普通 SGD 多了哪两步操作？"),
    ("Q15", "隐私/联邦", "联邦学习里 Non-IID 数据会带来什么问题？"),
]

CONDITIONS = [
    ("0.6B-base", MODELS_DIR / "Qwen3-0.6B", None),
    ("0.6B-sft", MODELS_DIR / "Qwen3-0.6B", ROOT / "qlora-demo-adapter"),
    ("4B-base", MODELS_DIR / "Qwen3-4B-Instruct-2507", None),
    ("4B-sft", MODELS_DIR / "Qwen3-4B-Instruct-2507", ROOT / "qlora-4b-adapter"),
]

PROMPT = "### Instruction:\n{q}\n\n### Input:\n\n\n### Response:\n"

FIELDS = ["condition", "question_id", "group", "question", "answer", "seconds", "peak_gb"]


def load(model_path: Path, adapter: Path | None):
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), quantization_config=bnb, device_map={"": 0}
    )
    if adapter is not None:
        model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    return model, tokenizer


def ask(model, tokenizer, question: str, max_new_tokens: int = 128) -> tuple[str, float]:
    inputs = tokenizer(PROMPT.format(q=question), return_tensors="pt").to(model.device)
    started = time.time()
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,               # greedy：四个条件必须用同一个解码策略
            pad_token_id=tokenizer.pad_token_id,
        )
    elapsed = time.time() - started
    answer = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
    ).strip()
    return answer, elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None, help="只跑 label 里含该子串的条件")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    args = parser.parse_args()

    rows = []
    for label, model_path, adapter in CONDITIONS:
        if args.only and args.only not in label:
            continue
        if not Path(model_path).exists():
            print(f"跳过 {label}：模型目录不存在 {model_path}", flush=True)
            continue
        if adapter is not None and not Path(adapter).exists():
            print(f"跳过 {label}：adapter 不存在 {adapter}", flush=True)
            continue

        print(f"\n=== 加载 {label} ===", flush=True)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        model, tokenizer = load(Path(model_path), Path(adapter) if adapter else None)
        base_peak = torch.cuda.memory_allocated() / 1024**3
        print(f"  加载完成，权重显存 {base_peak:.2f} GB", flush=True)

        for qid, group, text in QUESTIONS:
            answer, seconds = ask(model, tokenizer, text, args.max_new_tokens)
            peak = torch.cuda.max_memory_allocated() / 1024**3
            rows.append({
                "condition": label,
                "question_id": qid,
                "group": group,
                "question": text,
                "answer": answer.replace("\n", " ").strip(),
                "seconds": f"{seconds:.2f}",
                "peak_gb": f"{peak:.2f}",
            })
            print(f"  {qid} [{group}] {seconds:.1f}s  {answer[:40]!r}", flush=True)

        del model, tokenizer
        torch.cuda.empty_cache()

    if not rows:
        print("没有任何条件可跑。", flush=True)
        return 1

    with open(OUT / "eval.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    labels = [c[0] for c in CONDITIONS if any(r["condition"] == c[0] for r in rows)]
    lines = [
        "# Held-out 评测结果",
        "",
        "15 道题 × 4 个条件，统一 4-bit 加载、greedy 解码、max_new_tokens="
        f"{args.max_new_tokens}。答案已把换行压成空格，便于表格阅读。",
        "",
    ]
    for qid, group, text in QUESTIONS:
        lines.append(f"## {qid} [{group}] {text}")
        lines.append("")
        for label in labels:
            hit = next((r for r in rows if r["condition"] == label and r["question_id"] == qid), None)
            if hit is None:
                continue
            lines.append(f"**{label}**（{hit['seconds']}s）")
            lines.append("")
            lines.append(hit["answer"] or "（空）")
            lines.append("")
        lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 人工打分表（1–5 分）")
    lines.append("")
    lines.append("| 题号 | 组 | " + " | ".join(labels) + " | 备注 |")
    lines.append("|---|---|" + "---|" * len(labels) + "---|")
    for qid, group, _ in QUESTIONS:
        lines.append(f"| {qid} | {group} | " + " | ".join("" for _ in labels) + " |  |")
    (OUT / "eval.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\n已写出 {OUT/'eval.csv'} 和 {OUT/'eval.md'}（{len(rows)} 行）", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
