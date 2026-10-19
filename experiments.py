"""Day 3 对照实验：同一套配置换模型跑，自动生成跨规模对比表与 loss 曲线。

    python experiments.py                                   # 0.6B，5 组，约 6 分钟
    python experiments.py --model D:\\...\\Qwen3-4B-Instruct-2507 `
                          --batch-size 1 --accum 8          # 4B，显存不够放大 batch 时保持等效 batch 8

产物（results/ 下）：
    results.csv     所有模型的所有实验，带 model 列，可反复追加
    results.md      Markdown 表格（直接贴进 README）
    curves.json     每个实验的 loss 曲线（画图用，跨次运行累积）
    loss_curve.png  每个模型一个子图的 loss 对比图
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from demo_qlora import PROMPT_TEMPLATE, _default_model, load_model

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

BASE = dict(samples=500, r=16, alpha=32, lr=2e-4)
CONFIGS = [
    ("baseline (r=16, lr=2e-4, 500 samples)", dict(BASE)),
    ("A: r=64 (alpha=128)", dict(BASE, r=64, alpha=128)),
    ("B: 100 samples", dict(BASE, samples=100)),
    ("C: lr=1e-3", dict(BASE, lr=1e-3)),
    ("D: lr=2e-5", dict(BASE, lr=2e-5)),
]

FIELDS = ["model", "experiment", "r", "alpha", "lr", "samples",
          "trainable_pct", "minutes", "peak_gb", "final_loss"]


def build_dataset(limit: int):
    raw = load_dataset("yahma/alpaca-cleaned", split=f"train[:{limit}]")
    return raw.map(
        lambda ex: {
            "text": PROMPT_TEMPLATE.format(
                instruction=ex["instruction"].strip(),
                input=(ex["input"] or "").strip(),
                output=ex["output"].strip(),
            )
        },
        remove_columns=raw.column_names,
    )


def run_one(model_label: str, name: str, cfg: dict, model_path: str,
            batch_size: int, accum: int) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = load_model(model_path, quantized=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora = LoraConfig(
        r=cfg["r"],
        lora_alpha=cfg["alpha"],
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    # 4-bit 权重在显存里是打包存储的，sum(p.numel()) 会低估分母，必须用 peft 的口径
    trainable, total = model.get_nb_trainable_parameters()

    dataset = build_dataset(cfg["samples"])
    config = SFTConfig(
        output_dir=str(OUT / "_run"),
        max_length=512,
        dataset_text_field="text",
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=accum,
        learning_rate=cfg["lr"],
        num_train_epochs=1,
        logging_steps=5,
        bf16=True,
        gradient_checkpointing=True,
        packing=False,
        report_to="none",
        save_strategy="no",
        seed=42,
    )
    trainer = SFTTrainer(model=model, args=config, train_dataset=dataset, processing_class=tokenizer)

    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started
    metrics = trainer.state.log_history
    peak = torch.cuda.max_memory_allocated() / 1024**3
    curve = [(e["step"], e["loss"]) for e in metrics if "loss" in e and "step" in e]

    del trainer, model
    torch.cuda.empty_cache()

    return {
        "model": model_label,
        "experiment": name,
        "r": cfg["r"],
        "alpha": cfg["alpha"],
        "lr": f"{cfg['lr']:g}",
        "samples": cfg["samples"],
        "trainable_pct": round(100 * trainable / total, 3),
        "minutes": round(elapsed / 60, 2),
        "peak_gb": round(peak, 2),
        "final_loss": round(float(result.training_loss), 4),
        "curve": curve,
    }


def load_csv() -> list[dict]:
    path = OUT / "results.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    # 兼容旧格式（没有 model 列时，默认是 0.6B 那一轮的结果）
    for row in rows:
        row.setdefault("model", "Qwen3-0.6B")
        if not row["model"]:
            row["model"] = "Qwen3-0.6B"
    return rows


def write_outputs(rows: list[dict], curves: dict) -> None:
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in FIELDS})

    lines = ["| 模型 | 实验 | r | alpha | lr | 样本 | 可训练参数 | 训练耗时 | 峰值显存 | 最终 loss |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['experiment']} | {row['r']} | {row['alpha']} | {row['lr']} | "
            f"{row['samples']} | {row['trainable_pct']}% | {row['minutes']} min | "
            f"{row['peak_gb']} GB | {row['final_loss']} |"
        )
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (OUT / "curves.json").write_text(json.dumps(curves, ensure_ascii=False), encoding="utf-8")

    models = sorted({row["model"] for row in rows})
    fig, axes = plt.subplots(len(models), 1, figsize=(9, 4 * len(models)), squeeze=False)
    for ax, model_label in zip(axes[:, 0], models):
        for row in rows:
            if row["model"] != model_label:
                continue
            key = f"{row['model']}|{row['experiment']}"
            if key in curves and curves[key]:
                steps, losses = zip(*curves[key])
                ax.plot(steps, losses, marker="o", markersize=3, linewidth=1.5, label=row["experiment"])
        ax.set_title(f"{model_label} — QLoRA 4-bit, alpaca, 1 epoch", fontsize=10)
        ax.set_xlabel("training step")
        ax.set_ylabel("training loss")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, loc="upper right")
    fig.suptitle("QLoRA controlled experiments across model scales")
    fig.tight_layout()
    fig.savefig(OUT / "loss_curve.png", dpi=150)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=_default_model(), help="本地目录或 HuggingFace 仓库名")
    parser.add_argument("--name", default=None, help="结果表里显示的模型名，默认取目录名")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--only", default=None, help="只跑名字里含该子串的实验")
    args = parser.parse_args()

    model_label = args.name or Path(args.model).name
    configs = [c for c in CONFIGS if args.only is None or args.only in c[0]]
    print(f"模型: {model_label}\nbatch={args.batch_size} accum={args.accum} "
          f"等效 batch={args.batch_size * args.accum}\n", flush=True)

    new_rows = []
    for name, cfg in configs:
        print(f"=== {name} ===", flush=True)
        row = run_one(model_label, name, cfg, args.model, args.batch_size, args.accum)
        new_rows.append(row)
        print(
            f"  可训练 {row['trainable_pct']}% | {row['minutes']} min | "
            f"峰值 {row['peak_gb']} GB | loss {row['final_loss']}",
            flush=True,
        )

    old_rows = [r for r in load_csv() if r["model"] != model_label]
    all_rows = old_rows + [{k: str(v) for k, v in row.items() if k != "curve"} for row in new_rows]

    curves_path = OUT / "curves.json"
    curves = json.loads(curves_path.read_text(encoding="utf-8")) if curves_path.exists() else {}
    # 兼容旧格式：旧 key 没有 "模型|" 前缀，视为 0.6B
    curves = {(k if "|" in k else f"Qwen3-0.6B|{k}"): v for k, v in curves.items()}
    for row in new_rows:
        curves[f"{row['model']}|{row['experiment']}"] = row["curve"]
    for key in list(curves):
        if key.startswith(f"{model_label}|") and key.split("|", 1)[1] not in {r["experiment"] for r in new_rows}:
            del curves[key]

    write_outputs(all_rows, curves)
    print(f"\n已写出 {OUT/'results.csv'} / results.md / loss_curve.png / curves.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
