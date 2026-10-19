"""对比 bf16 与 4-bit 加载同一模型的显存占用（纯显存账，不训练）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from demo_qlora import _default_model, load_model

import torch

model = load_model(_default_model(), quantized=False)
print(f"bf16 显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
del model
torch.cuda.empty_cache()

model = load_model(_default_model(), quantized=True)
print(f"4-bit 显存: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
del model
torch.cuda.empty_cache()

print("理论值：参数量 × 0.5 byte（4-bit）+ 每 64 个权重一个 scale")
