"""Day 3 对照实验：一次跑完 5 组配置，自动生成 results/ 下的表格与 loss 曲线。

每组只改一个变量，其余固定：0.6B 底座 + 4-bit NF4 + LoRA(q/k/v/o) + 1 epoch + seq 512。

    python experiments.py

产物：
    results/results.csv      原始数据
    results/results.md       Markdown 表格（直接贴进 README）
    results/loss_curve.png   loss 曲线对比图
"""
from __future__ import annotations

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


def run_one(name: str, cfg: dict, model_path: str) -> dict:
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
    # 注意：4-bit 权重在显存里是"打包"存储的，直接 sum(p.numel()) 会低估分母，
    # 必须用 peft 自己的统计（它按逻辑参数量计算，与 print_trainable_parameters 一致）。
    trainable, total = model.get_nb_trainable_parameters()

    dataset = build_dataset(cfg["samples"])
    config = SFTConfig(
        output_dir=str(OUT / "_run"),
        max_length=512,
        dataset_text_field="text",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
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
    peak = torch.cuda.max_memory_allocated() / 1024**3

    curve = [
        (entry["step"], entry["loss"])
        for entry in trainer.state.log_history
        if "loss" in entry and "step" in entry
    ]

    del trainer, model
    torch.cuda.empty_cache()

    return {
        "name": name,
        "r": cfg["r"],
        "alpha": cfg["alpha"],
        "lr": cfg["lr"],
        "samples": cfg["samples"],
        "trainable_pct": 100 * trainable / total,
        "minutes": elapsed / 60,
        "peak_gb": peak,
        "final_loss": float(result.training_loss),
        "curve": curve,
    }


def main() -> int:
    model_path = _default_model()
    print(f"底座: {model_path}\n", flush=True)
    rows = []
    for name, cfg in CONFIGS:
        print(f"=== {name} ===", flush=True)
        row = run_one(name, cfg, model_path)
        rows.append(row)
        print(
            f"  可训练 {row['trainable_pct']:.3f}% | {row['minutes']:.2f} min | "
            f"峰值 {row['peak_gb']:.2f} GB | loss {row['final_loss']:.4f}",
            flush=True,
        )

    # ---- CSV ----
    with open(OUT / "results.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["experiment", "r", "alpha", "lr", "samples", "trainable_pct", "minutes", "peak_gb", "final_loss"])
        for r in rows:
            writer.writerow([r["name"], r["r"], r["alpha"], r["lr"], r["samples"],
                             f"{r['trainable_pct']:.3f}", f"{r['minutes']:.2f}",
                             f"{r['peak_gb']:.2f}", f"{r['final_loss']:.4f}"])

    # ---- Markdown ----
    lines = [
        "| 实验 | r | alpha | lr | 样本 | 可训练参数 | 训练耗时 | 峰值显存 | 最终 loss |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['name']} | {r['r']} | {r['alpha']} | {r['lr']:g} | {r['samples']} | "
            f"{r['trainable_pct']:.3f}% | {r['minutes']:.2f} min | {r['peak_gb']:.2f} GB | {r['final_loss']:.4f} |"
        )
    (OUT / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    # ---- 曲线 ----
    plt.figure(figsize=(8, 5))
    for r in rows:
        if r["curve"]:
            steps, losses = zip(*r["curve"])
            plt.plot(steps, losses, marker="o", markersize=3, linewidth=1.5, label=r["name"])
    plt.xlabel("training step")
    plt.ylabel("training loss")
    plt.title("QLoRA on Qwen3-0.6B (4-bit NF4): controlled experiments, alpaca, 1 epoch")
    plt.grid(alpha=0.3)
    plt.legend(fontsize=7, loc="upper right")
    plt.tight_layout()
    plt.savefig(OUT / "loss_curve.png", dpi=150)
    (OUT / "curves.json").write_text(
        json.dumps({r["name"]: r["curve"] for r in rows}, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n已写出: {OUT/'results.csv'}, {OUT/'results.md'}, {OUT/'loss_curve.png'}", flush=True)
    (OUT / "_run").exists() and None
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
