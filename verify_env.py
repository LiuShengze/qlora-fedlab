"""QLoRA 环境自检：确认「卡能动、4-bit 能算、库版本对得上」。

用法：
    python verify_env.py                  # 基础自检（不下载模型）
    python verify_env.py --with-model Qwen/Qwen3-0.6B   # 额外做一次 4-bit 加载

全部通过时退出码为 0；任何一项失败退出码为 1，并把失败原因打印出来。
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys

OK, BAD, WARN = "[ OK ]", "[FAIL]", "[WARN]"
_failures: list[str] = []


def check(label: str, fn) -> None:
    try:
        detail = fn()
        print(f"{OK} {label}{(' — ' + str(detail)) if detail else ''}")
    except Exception as exc:  # noqa: BLE001 - 自检就是要吞掉异常继续测下一项
        _failures.append(f"{label}: {exc}")
        print(f"{BAD} {label} — {type(exc).__name__}: {exc}")


def versions() -> str:
    import accelerate
    import bitsandbytes
    import datasets
    import peft
    import torch
    import transformers
    import trl

    return (
        f"py{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro} | "
        f"torch {torch.__version__} | transformers {transformers.__version__} | "
        f"peft {peft.__version__} | trl {trl.__version__} | "
        f"datasets {datasets.__version__} | accelerate {accelerate.__version__} | "
        f"bitsandbytes {bitsandbytes.__version__}"
    )


def gpu() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() 为 False：装成 CPU 版 torch 了")
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    arch = torch.cuda.get_arch_list()
    if f"sm_{cap[0]}{cap[1]}" not in arch:
        raise RuntimeError(
            f"显卡算力 sm_{cap[0]}{cap[1]} 不在 torch 编译列表里 {arch}："
            "需要 cu128/cu130 版本的 torch（RTX 50 系是 Blackwell）"
        )
    if torch.version.cuda is None:
        raise RuntimeError("torch 是 CPU 构建")
    return f"{name} / {total:.1f} GB / sm_{cap[0]}{cap[1]} / CUDA {torch.version.cuda}"


def bnb_4bit() -> str:
    """验证 4-bit 内核能在本卡上跑前向+反向。

    注意：4-bit 权重本身是冻结的（QLoRA 里底座就不训练），所以这里在它上面挂一个
    普通可训练层，模拟 LoRA 分支——梯度要能穿过冻结的 4-bit 层传回去，这才是训练时的真实路径。
    """
    import bitsandbytes as bnb
    import torch

    linear = bnb.nn.Linear4bit(256, 128, bias=False, quant_type="nf4", compute_dtype=torch.bfloat16)
    linear = linear.to("cuda")
    head = torch.nn.Linear(128, 1).to("cuda", dtype=torch.bfloat16)
    x = torch.randn(8, 256, device="cuda", dtype=torch.bfloat16)
    out = head(linear(x))
    out.sum().backward()
    if head.weight.grad is None or not torch.isfinite(head.weight.grad).all():
        raise RuntimeError("梯度没有穿过 4-bit 层，或出现了 NaN/Inf")
    return f"NF4 前向+反向通过（梯度穿过冻结的 4-bit 层），输出 {tuple(out.shape)}"


def peak_mem() -> str:
    import torch

    torch.cuda.synchronize()
    free, total = torch.cuda.mem_get_info()
    return f"当前空余 {free / 1024**3:.2f} GB / 共 {total / 1024**3:.2f} GB"


def env_vars() -> str:
    endpoint = os.environ.get("HF_ENDPOINT", "(未设置，会走 huggingface.co，本机不可直连)")
    home = os.environ.get("HF_HOME", "(未设置，默认在 C:\\Users\\<你>\\.cache\\huggingface)")
    if not os.environ.get("HF_ENDPOINT"):
        raise RuntimeError("HF_ENDPOINT 未设置：国内网络请设为 https://hf-mirror.com")
    return f"HF_ENDPOINT={endpoint} | HF_HOME={home}"


def disk() -> str:
    target = os.environ.get("HF_HOME") or os.path.expanduser("~")
    drive = os.path.splitdrive(os.path.abspath(target))[0] or "C:"
    usage = shutil.disk_usage(drive + "\\")
    free = usage.free / 1024**3
    if free < 25:
        raise RuntimeError(f"{drive} 只剩 {free:.1f} GB，QLoRA + 模型缓存至少需要 ~25 GB")
    return f"{drive} 空闲 {free:.1f} GB"


def model_4bit(model_id: str) -> str:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb_config, device_map={"": 0}
    )
    inputs = tokenizer("你好", return_tensors="pt").to("cuda")
    with torch.no_grad():
        logits = model(**inputs).logits
    used = torch.cuda.max_memory_allocated() / 1024**3
    return f"{model_id} 4-bit 加载+前向通过，logits {tuple(logits.shape)}，峰值显存 {used:.2f} GB"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-model", default=None, help="额外加载一个小模型验证 4-bit 推理")
    args = parser.parse_args()

    print("=" * 70)
    print("QLoRA 环境自检")
    print("=" * 70)
    check("依赖版本", versions)
    check("GPU 可用性与算力匹配", gpu)
    check("bitsandbytes NF4 4-bit 计算", bnb_4bit)
    check("显存", peak_mem)
    check("HuggingFace 镜像配置", env_vars)
    check("磁盘空间", disk)
    if args.with_model:
        check(f"4-bit 加载 {args.with_model}", lambda: model_4bit(args.with_model))

    print("-" * 70)
    if _failures:
        print(f"有 {len(_failures)} 项没通过：")
        for item in _failures:
            print(f"  - {item}")
        return 1
    print("全部通过，可以跑 demo_qlora.py 了。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
