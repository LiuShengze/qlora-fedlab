"""全量微调对照：LoRA 论文的 headline 是"用极少参数达到接近全量微调的效果"。

这一组用与 LoRA 完全相同的 500 条数据、seq 512、等效 batch 8，只改两件事：
  1. 不挂 LoRA —— 600M 参数全部可训练
  2. 学习率用 2e-5（全量微调的标准量级，LoRA 用的是 2e-4，大 10 倍）

对比三个数：最终 loss、峰值显存、训练耗时。

用法：python full_ft.py
"""
from __future__ import annotations

import time

import torch
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from demo_qlora import _default_model, build_dataset, load_model

SAMPLES = 500
LR = 2e-5
BATCH = 1
ACCUM = 8


def main() -> int:
    model_path = _default_model()
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 全量微调必须用 bf16 权重（4-bit 冻结的权重没法训练）
    model = load_model(model_path, quantized=False)
    model.gradient_checkpointing_enable()
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"可训练参数: {trainable:,} / {total:,} = {100 * trainable / total:.2f}%", flush=True)

    dataset = build_dataset(SAMPLES)
    config = SFTConfig(
        output_dir="full-ft-output",
        max_length=512,
        dataset_text_field="text",
        per_device_train_batch_size=BATCH,
        gradient_accumulation_steps=ACCUM,
        learning_rate=LR,
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

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    result = trainer.train()
    elapsed = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 1024**3

    print(f"\n全量微调完成：{elapsed / 60:.2f} 分钟 | 峰值显存 {peak:.2f} GB | 最终 loss {result.training_loss:.4f}")
    print("对照：LoRA r=16 在同样数据上是 1.03 分钟 / 2.04 GB / loss 1.6765")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
