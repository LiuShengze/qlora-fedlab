import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from demo_qlora import _default_model  # 自动找本地模型目录，找不到就回落成 HF 仓库名

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(_default_model())
print("vocab size:", len(tok))
ids = tok("QLoRA 微调")["input_ids"]
print("ids:", ids)
print("tokens:", tok.convert_ids_to_tokens(ids))
print("decode:", tok.decode(ids))
