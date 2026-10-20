"""QLoRA 最小可运行 demo —— 适配 2026 年的 transformers 5.x / trl 1.x。

这份代码就是《QLoRA微调Demo运行指南》里那段代码的修正版，四处必须改的地方：

  1. SFTConfig 不再有 max_seq_length，字段名是 max_length（trl >= 1.0）
  2. 指南里写了 format_prompt 却从没传给 SFTTrainer —— 这里先把数据集 map 成 text 列，
     再用 dataset_text_field="text"，训练时模型才真的看到你的模板
  3. transformers 5.x 用 dtype= 代替 torch_dtype=（旧名仍兼容但会告警）
  4. RTX 5070 是 Blackwell（sm_120），torch 必须是 cu128/cu130 版，cu121 装了也认不出卡

每个 print 都对应指南里讲的一个原理，建议开着任务管理器 → 性能 → GPU 一起看。

用法：
    python demo_qlora.py                               # 默认 0.6B，约 10 分钟冒烟
    python demo_qlora.py --model Qwen/Qwen3-4B-Instruct-2507   # 12G 显存正式版
    python demo_qlora.py --r 64 --lr 1e-3              # Day 3 对照实验
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

PROMPT_TEMPLATE = "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:\n{output}"

# 本地模型优先（免重复下载）：先找仓库内的 models/Qwen3-0.6B，再找仓库同级的 models/，
# 都没有就回落成 HuggingFace 仓库名（会自动下载）。
LOCAL_MODEL_CANDIDATES = (
    Path(__file__).resolve().parent / "models" / "Qwen3-0.6B",
    Path(__file__).resolve().parent.parent / "models" / "Qwen3-0.6B",
)


def _default_model() -> str:
    for candidate in LOCAL_MODEL_CANDIDATES:
        if candidate.exists():
            return str(candidate)
    return "Qwen/Qwen3-0.6B"


def bnb_config() -> BitsAndBytesConfig:
    """组件①：4-bit NF4 量化 + 双重量化。存 4 bit，算 bf16。"""
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",          # 按正态分布分位数取量化点
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,     # scale 再量化一次，每参数再省 ~0.37 bit
    )


def load_model(model_id: str, quantized: bool):
    """v5 用 dtype=，v4 用 torch_dtype=，这里两个都兼容。"""
    kwargs: dict = {"device_map": {"": 0}}
    if quantized:
        kwargs["quantization_config"] = bnb_config()
    try:
        return AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16, **kwargs)
    except TypeError:
        return AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, **kwargs)


def build_dataset(limit: int, data_path: str | None = None):
    """数据来源：给了 --data 就用本地 jsonl，否则用 alpaca-cleaned 前 N 条。"""
    if data_path:
        rows = [
            json.loads(line)
            for line in Path(data_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        rows = rows[:limit]
        return Dataset.from_list([
            {
                "text": PROMPT_TEMPLATE.format(
                    instruction=str(r["instruction"]).strip(),
                    input=str(r.get("input", "") or "").strip(),
                    output=str(r["output"]).strip(),
                )
            }
            for r in rows
        ])

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


def generate(model, tokenizer, question: str, max_new_tokens: int = 64) -> str:
    prompt = f"### Instruction:\n{question}\n\n### Input:\n\n### Response:\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        default=os.environ.get("QLORA_MODEL", str(_default_model())),
        help="HuggingFace 仓库名，或本地模型目录",
    )
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--r", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--out", default="qlora-demo-adapter")
    parser.add_argument("--question", default="用一句话解释什么是 QLoRA")
    parser.add_argument("--no-eval", action="store_true", help="只训练并保存 adapter，跳过前后对比推理")
    parser.add_argument("--data", default=None, help="本地问答 jsonl（instruction/input/output），不传则用 alpaca")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"模型: {args.model} | 数据: {args.samples} 条 | r={args.r} alpha={args.alpha} lr={args.lr}")

    # ---------------- 1. 加载与量化 ----------------
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = load_model(args.model, quantized=True)
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    print(f"4-bit 底座加载完成，权重显存 {torch.cuda.memory_allocated() / 1024**3:.2f} GB")

    # ---------------- 2. 注入 LoRA ----------------
    lora = LoraConfig(
        r=args.r,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()  # 这就是「为什么联邦微调必须用 LoRA」的那个百分比

    # ---------------- 3. 数据 ----------------
    dataset = build_dataset(args.samples, args.data)
    print(f"样例（截断 300 字）：\n{dataset[0]['text'][:300]}\n---")

    # ---------------- 4. 训练 ----------------
    config = SFTConfig(
        output_dir="qlora-demo-output",
        max_length=args.max_length,          # ← 不是 max_seq_length
        dataset_text_field="text",           # ← 指向上一步拼好的列
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        bf16=True,
        gradient_checkpointing=True,          # 组件③：用时间换显存
        packing=False,
        report_to="none",                     # 不装 wandb 就把日志打在屏幕上
        save_strategy="no",
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,           # trl 1.x：不再是 tokenizer=
    )

    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"\n训练完成：{elapsed / 60:.1f} 分钟 | 峰值显存 {peak:.2f} GB | 最终 loss {result.training_loss:.4f}")
    print("（把这三个数字记下来，Day 3 对照实验要用）")

    trainer.save_model(args.out)
    tokenizer.save_pretrained(args.out)
    print(f"适配器已保存到 {args.out}/（只有几十 MB，底座没被改动）")

    # ---------------- 5. 前后对比推理 ----------------
    del model, trainer
    torch.cuda.empty_cache()

    if args.no_eval:
        print("\n（--no-eval：跳过前后对比推理，适配器已保存）")
        return 0

    print("\n===== 微调前（底座，bf16）=====")
    base = load_model(args.model, quantized=False)
    print(generate(base, tokenizer, args.question))

    print("\n===== 微调后（底座 + adapter，已合并）=====")
    merged = PeftModel.from_pretrained(base, args.out).merge_and_unload()
    print(generate(merged, tokenizer, args.question))

    print(f"\n推理峰值显存 {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
