"""逐位置看模型对下一个 token 的预测（teacher forcing）。

解读要点：
  - 高概率（0.7+）的是"背下来的搭配"，如「微」→「调」
  - 低概率（0.01~0.1）的位置，说明模型对这个词本身没有把握
  - "实际下一个"列 = 训练时的监督信号，loss 就是这些概率的 -log 平均
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from demo_qlora import _default_model

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

path = _default_model()
tok = AutoTokenizer.from_pretrained(path)
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map={"": 0})

prompt = "QLoRA 是一种高效的大模型微调技术，它的核心思想是"
inputs = tok(prompt, return_tensors="pt").to("cuda")
tokens = inputs["input_ids"][0]

with torch.no_grad():
    logits = model(**inputs).logits[0]        # 形状 [seq_len, 词表]，这次保留所有位置

# 逐位置看 top-3 预测
for pos in range(len(tokens)):
    probs = torch.softmax(logits[pos].float(), dim=-1)
    top = torch.topk(probs, 3)
    gold = tok.decode([tokens[pos + 1]]) if pos + 1 < len(tokens) else "(句子结束)"
    preds = " / ".join(f"{tok.decode([i])!r}({p.item():.2f})" for p, i in zip(top.values, top.indices))
    seen = "".join(tok.decode([t]) for t in tokens[: pos + 1])
    print(f"位置{pos} 看到 {seen!r:20} → 预测: {preds}  | 实际下一个: {gold}")
