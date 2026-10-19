# QLoRA on a 12 GB Consumer GPU — Reproduction & Controlled Experiments

在单张 **RTX 5070（12 GB, Blackwell sm_120）** 上复现 QLoRA 微调，并完成四组受控对照实验。
这是「联邦 LoRA + 差分隐私」方向的第 0 步：先证明本地微调链路可复现、可度量，再把它包进联邦框架。

> **TL;DR** — 4-bit NF4 QLoRA on Qwen3-0.6B: 500 alpaca samples, 1 epoch in **~1 min**, peak VRAM **2.0 GB**,
> only **0.76%** of parameters trainable. Learning rate is the most sensitive knob (final loss 2.00 → 1.59 across lr 2e-5 → 1e-3).

---

## 结果速览

| 指标 | 数值 |
|---|---|
| 底座模型 | Qwen3-0.6B（600.6 M 参数，4-bit NF4 量化） |
| 可训练参数 | 4.59 M / 600.6 M = **0.76%** |
| 训练数据 | alpaca-cleaned，500 条，1 epoch，seq 512 |
| 单次训练耗时 | **约 1 分钟** |
| 峰值显存 | **2.04 GB**（卡有 12 GB） |
| 适配器体积 | 18.4 MB（底座权重全程未被修改） |

![loss curves](results/loss_curve.png)

---

## 环境

| 组件 | 版本 | 说明 |
|---|---|---|
| GPU | RTX 5070 12 GB | Blackwell，compute capability **sm_120** |
| 驱动 | 610.88 / CUDA 13.3 | 驱动足够新 |
| Python | 3.12.15 | 独立 venv |
| PyTorch | **2.14.1+cu130** | RTX 50 系必须 cu128/cu130，cu121 认不出这张卡 |
| transformers | 5.20.1 | |
| peft / trl | 0.21.2 / 1.16.0 | |
| bitsandbytes | 0.50.2 | Windows 有预编译轮子，不需要本地编译 |
| datasets / accelerate | 5.1.0 / 1.15.0 | |
| OS | Windows 11 22631 | 32 GB RAM |

详细的环境搭建过程（含国内镜像、踩坑与修复）见 [docs/ENVIRONMENT.md](docs/ENVIRONMENT.md)。

---

## 复现步骤

```bash
git clone https://github.com/LiuShengze/qlora-fedlab.git
cd qlora-fedlab
python -m venv .venv && . .venv/bin/activate     # Windows: .\.venv\Scripts\Activate.ps1

# CUDA 13 版 torch（Windows/Linux 都适用；RTX 50 系必须 cu128+）
pip install torch --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements.txt

# 环境自检（GPU / 4-bit 内核 / 镜像配置）
python verify_env.py

# 单次训练 + 前后对比推理
python demo_qlora.py --samples 500

# 四组受控实验（约 6 分钟，输出 results/）
python experiments.py
```

Windows 用户可以直接跑 `setup_qlora.ps1` 一键装完（见 docs）。

---

## 实验设计与结果

固定变量：Qwen3-0.6B、4-bit NF4、LoRA 注入 q/k/v/o、batch 2 × accum 4、seq 512、1 epoch、seed 42。
**每次只改一个变量。**

<!-- RESULTS_TABLE_START -->
| 实验 | r | alpha | lr | 样本 | 可训练参数 | 训练耗时 | 峰值显存 | 最终 loss |
|---|---|---|---|---|---|---|---|---|
| baseline | 16 | 32 | 2e-4 | 500 | 4.59 M（0.76%） | 1.03 min | 2.04 GB | **1.6765** |
| A 秩加倍 | 64 | 128 | 2e-4 | 500 | 18.35 M（3.06%） | 1.02 min | 2.15 GB | **1.6112** |
| B 数据减到 1/5 | 16 | 32 | 2e-4 | 100 | 4.59 M（0.76%） | 0.21 min | 1.97 GB | **1.9395** |
| C 学习率放大 5× | 16 | 32 | 1e-3 | 500 | 4.59 M（0.76%） | 0.99 min | 2.04 GB | **1.5872** |
| D 学习率缩小 10× | 16 | 32 | 2e-5 | 500 | 4.59 M（0.76%） | 1.01 min | 2.04 GB | **2.0033** |

<sub>可训练参数占比以底座 600.6 M 为分母。注：peft 的 `print_trainable_parameters()` 对 r=64 会报 2.99%，
因为它对量化权重的"总参数"口径不同（见踩坑第 3 条）。绝对参数量 18.35 M，恰为 r=16 的 4 倍。</sub>
<!-- RESULTS_TABLE_END -->

### 解读

**A. 秩 r：16 → 64（alpha 同步 32 → 128）**
可训练参数从 4.59 M 涨到 18.35 M（4 倍），但峰值显存只从 2.04 GB 涨到 2.15 GB（+5%）。
原因：LoRA 参数本身很小，显存增量主要来自它的梯度与优化器状态；
而占大头的 **4-bit 底座权重（约 0.8 GB）和激活值/梯度检查点的开销不随 r 变化**。
→ 在 12 GB 卡上，r 不是显存瓶颈，**先加 r 再考虑换更大的模型**。

**B. 数据量：500 → 100 条**
final loss 从 1.68 变成 1.94 —— **数据越少，训练集 loss 反而更高**（不是直觉里的"过拟合所以更低"）。
因为 loss 是对 batch 的平均：样本少 → 参数更新次数少（25 步 vs 63 步）→ 模型还没学会格式模板。
结论：**在"还没学会"的阶段谈过拟合没有意义**，先保证步数够。

**C / D. 学习率：2e-4 → 1e-3 和 2e-5**
2e-5 明显欠拟合（loss 2.00，曲线几乎平的）；1e-3 收敛最快（loss 1.59），
但曲线抖动变大（相邻步在 1.40–1.90 之间跳），接近不稳定边界。
→ **LoRA 的学习率确实比全量微调大一个量级**，但要卡在 1e-4 ~ 5e-4；1e-3 能跑但不稳。

---

## 目录结构

```
.
├── demo_qlora.py        单次 QLoRA 微调 + 微调前后对比推理
├── experiments.py       四组受控实验，自动出表格与曲线
├── verify_env.py        环境自检（GPU/算力匹配/4-bit 内核/镜像）
├── requirements.txt
├── setup_qlora.ps1      Windows 一键环境（可选）
├── results/             results.csv / results.md / loss_curve.png / curves.json
└── docs/
    ├── ENVIRONMENT.md   环境搭建全记录（含镜像与踩坑）
    └── PLAN.md          下一步计划
```

---

## 踩坑记录（工程细节）

复现别人的 QLoRA 教程时实际撞到的坑，都已在本仓库修好：

1. **RTX 50 系 = sm_120**，`pip install torch` 装到的默认版本没有对应内核，必须显式指定 cu128/cu130。
2. **Windows + bitsandbytes** 没问题（0.50.x 有 `win_amd64` 轮子），但要先装对 torch，否则加载 4-bit 时报 DLL 缺失。
3. **4-bit 参数量口径容易算错**：`sum(p.numel())` 在量化权重上是"打包后"的元素数（380 M），
   会得到 1.21% 这种偏差值；正确做法是用 `model.get_nb_trainable_parameters()`（600.6 M 分母，0.76%）。
4. **trl 1.x 的 API 变了**：`max_seq_length` → `max_length`，`tokenizer=` → `processing_class=`，
   并且 `formatting_func` 不传的话模板根本不会生效（不报错，静默失效）。
5. **hf-mirror 下载 1.4 GB 大文件会"假死"**（停在某个字节数不报错），用 `curl --max-time 60 -C -` 循环续传解决。
6. **hf-xet 协议在镜像上会 401**，设 `HF_HUB_DISABLE_XET=1`。
7. 反量化计算用 `bnb_4bit_compute_dtype=torch.bfloat16`，**T4 等 Turing 卡不支持 bf16**，要改 fp16。

---

## Roadmap

- [x] 本地 QLoRA 链路跑通（0.6B / 12 GB 卡）
- [x] 四组受控实验 + 结果归档
- [ ] 换成组内论文构建的领域问答数据（100–300 条），4B 模型
- [ ] 用 Flower 把 `train()` 包成 client，2–4 客户端、每轮 FedAvg 聚合 **LoRA 适配器**
- [ ] 本地训练加 Opacus DP-SGD，测 ε 与效用的权衡曲线

## 参考

- Dettmers et al., *QLoRA: Efficient Finetuning of Quantized LLMs*, 2023
- HuggingFace PEFT / TRL / bitsandbytes
