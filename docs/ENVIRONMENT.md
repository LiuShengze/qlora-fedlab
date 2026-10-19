# QLoRA 环境：开工前还差什么

对应资料：[QLoRA微调Demo运行指南](<../../Backup/Downloads/QLoRA微调Demo运行指南_Export_2026-10-19T08-35-13-949Z.md>) 的 Day 1。
结论先说：**指南 Day 1 的三条命令，在你的机器上一条都用不了**——不是你的问题，是那份指南按 2024 年的环境写的（conda + Python 3.11 + CUDA 12.1），而你手上的是一张 2026 年的 Blackwell 卡。

---

## 一、我刚在你机器上实测到的现状

| 项目 | 实测结果 | 结论 |
|---|---|---|
| 显卡 | RTX 5070 / **12227 MiB** / WDDM | 卡没问题，12G 属于「先小后大」那一档 |
| 算力 | **sm_120（Blackwell）** | 决定了只能装 cu128 / cu130 版 torch |
| 驱动 | 610.88，CUDA UMD **13.3** | 驱动足够新，不用动 |
| 系统 | Windows 11 22631，32G 内存 | 够用 |
| 磁盘 | C 132G / D 141G / E 104G | 够，但模型缓存建议放 D |
| PATH 上的 `python` | 一个 **MSYS2/MinGW 编译的 3.14** | ❌ 不能用：MinGW 编译的 Python 装不了 MSVC 编译的 torch |
| 另一个 Python | 另一个 **3.14**（pip 指向它，与 PATH 上的不是同一个） | ⚠️ 3.14 太新，生态未完全对齐 |
| conda / mamba | 未安装 | 不必装，用 uv 代替 |
| git | **未安装** | 需要（Day 5 做 repo、后面接 Flower） |
| uv / nvcc | 未安装 / 未安装 | uv 要装；nvcc **不用装** |
| torch / transformers / peft / trl / bnb | **全部未安装** | 这是主要工作 |
| 网络 | pypi.org ✅、清华 PyPI ✅、**huggingface.co ❌**、hf-mirror.com ✅ | 必须配 HF 镜像 |

> 一句话：**你现在是一台「有卡、有驱动、但 Python 生态从零开始」的机器**。之前两个项目全在 Colab 上跑，所以本地一个深度学习包都没有，这完全正常。

---

## 二、需要装什么（按优先级）

### P0 —— 不装就跑不起来

| 组件 | 版本 | 为什么是它 | 下载量 |
|---|---|---|---|
| **uv** | 最新 | 管理 Python 3.12 + 虚拟环境，比 conda 快一个数量级，不污染系统 | ~30 MB |
| **CPython 3.12** | 3.12.x | 生态最稳；3.14 太新，别当小白鼠 | ~28 MB |
| **torch（CUDA 版）** | 2.14.1+**cu130** | 唯一能认出 sm_120 的构建；**cu121 装了也会 `cuda.is_available()=False`** | **1.85 GB** |
| **transformers** | 5.20.1 | 加载 Qwen 底座、`BitsAndBytesConfig` | ~15 MB |
| **peft** | 0.21.2 | `LoraConfig` / `get_peft_model` / `prepare_model_for_kbit_training` | ~1 MB |
| **trl** | 1.16.0 | `SFTTrainer` / `SFTConfig`（API 已和指南不同，见第五节） | ~5 MB |
| **bitsandbytes** | 0.50.2 | 4-bit NF4 量化的计算内核（Windows 有预编译轮子，不用编译器） | ~60 MB |
| **datasets / accelerate** | 5.1.0 / 1.15.0 | 数据加载、`device_map="auto"` | ~20 MB |
| **git** | 最新 | `winget install --id Git.Git -e` | ~50 MB |

合计下载约 **2.2 GB**。实测速度：南大 PyTorch 镜像 **9.3 MB/s**、华为云 **10.5 MB/s**、hf-mirror 约 0.5 MB/s。

### P1 —— 强烈建议，但不影响第一次跑通

- **HuggingFace 镜像与缓存**：`HF_ENDPOINT=https://hf-mirror.com`、`HF_HOME=D:\hf-cache`、`HF_HUB_DISABLE_XET=1`（镜像不支持 xet 加速协议，不设会报 401）
- **VS Code 扩展**：`ms-python.python`、`ms-toolsai.jupyter`（你已有 VS Code）
- **tensorboard**：`SFTConfig(report_to="tensorboard")` 看 loss 曲线比盯终端舒服
- **开启 Windows 开发者模式**：让 HF 缓存能用软链接，省一份磁盘副本

### 明确不用装（省你两小时）

| 不用装 | 原因 |
|---|---|
| CUDA Toolkit / nvcc | pip 的 cu130 wheel 自带 CUDA 运行时，编内核这种事与你无关 |
| Visual Studio Build Tools | bitsandbytes 有 `win_amd64` 预编译轮子 |
| WSL2 / Linux | 你机器 `wsl --status` 报「不支持 WSL2」（虚拟化未开），别为了跑 demo 去动 BIOS |
| conda / Anaconda | 指南写它是习惯，不是必需；uv 更轻更快 |
| flash-attn / xformers | Windows 上编译地狱，QLoRA demo 用不到 |
| torch 2.4/2.5（指南时代的版本） | **不认识 RTX 50 系**，装了就是白装 |

---

## 三、怎么装（两条路，选一条）

### 路线 A：一键脚本（推荐）

```powershell
cd qlora-demo
powershell -ExecutionPolicy Bypass -File .\setup_qlora.ps1
```

脚本做六件事，跑完还会自动自检：

1. 装 uv（没有就 `winget` 装），用 uv 拉一个干净的 CPython 3.12
2. 在 `qlora-demo\.venv` 建虚拟环境（失败会自动回退到华为云的官方 3.12 安装包）
3. 从**南大镜像**装 `torch+cu130`
4. 从**清华 PyPI 镜像**装其余依赖（`requirements.txt`，版本已锁）
5. 设置 `HF_ENDPOINT` / `HF_HOME`（写入你的用户环境变量）
6. 跑 `verify_env.py` 自检

### 路线 B：手动敲（想知道每一步在干什么就用这个）

```powershell
# 1) 装 uv，然后拉 Python 3.12
winget install --id astral-sh.uv -e
uv python install 3.12

# 2) 建环境
cd qlora-demo
uv venv .venv --python 3.12 --seed
.\.venv\Scripts\Activate.ps1

# 3) 装 CUDA 版 torch（注意：不是指南里的 cu121！）
python -m pip install torch --index-url https://mirror.nju.edu.cn/pytorch/whl/cu130

# 4) 装其余依赖
python -m pip install -r requirements.txt -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# 5) 配 HF 镜像（写进用户环境变量，一次到位）
[Environment]::SetEnvironmentVariable("HF_ENDPOINT","https://hf-mirror.com","User")
[Environment]::SetEnvironmentVariable("HF_HOME","D:\hf-cache","User")
[Environment]::SetEnvironmentVariable("HF_HUB_DISABLE_XET","1","User")

# 6) 自检
python verify_env.py --with-model Qwen/Qwen3-0.6B
```

### 装完的「过关检查」

`verify_env.py` 六行全 `[ OK ]` 才算过：

```
[ OK ] 依赖版本 — py3.12.x | torch 2.14.1 | transformers 5.20.1 | ...
[ OK ] GPU 可用性与算力匹配 — NVIDIA GeForce RTX 5070 / 11.9 GB / sm_120 / CUDA 13.0
[ OK ] bitsandbytes NF4 4-bit 计算 — NF4 前向+反向通过，输出 (4, 128)
[ OK ] 显存 — 当前空余 ...
[ OK ] HuggingFace 镜像配置 — HF_ENDPOINT=https://hf-mirror.com ...
[ OK ] 磁盘空间 — D: 空闲 ...
```

这一步比指南 Day 1 的「线性回归」更有价值：它直接验证了**4-bit 内核能在 sm_120 上跑**，这才是 QLoRA 最容易死的地方。

---

## 四、12G 显存到底能跑多大（比指南更贴合你的卡）

指南里「7B 在 24G 上占 12–16G」是 3090/4090 的账。你只有 12G（桌面还占着约 0.9G），实际可用 ~11.3G：

| 模型 | 4-bit 权重 | 训练总占用（seq 512, bs1, accum16） | 你的 12G |
|---|---|---|---|
| Qwen3-0.6B | ~0.4 GB | ~2 GB | ✅ 冒烟首选 |
| Qwen3-1.7B | ~1.1 GB | ~4 GB | ✅ 从容 |
| **Qwen3-4B-Instruct-2507** | ~2.4 GB | ~8–9 GB | ✅ **主力，对应指南的「7B 正式版」** |
| Qwen3-8B / Qwen2.5-7B | ~4.8 GB | ~12–14 GB | ⚠️ 极限，多半 OOM；真要用必须 seq≤512、bs=1、开梯度检查点 |

**建议：把指南的「Day 4 上 7B」改成「上 4B」。** 原话「3B 上做出的理解和 7B 完全等价，实验结论不打折」——换成 4B 同理，而且你的卡跑得动。

---

## 五、指南代码里 5 处会让你直接报错的地方（demo_qlora.py 已改好）

| 指南写法 | 现在（2026-10）的现实 | 不改会怎样 |
|---|---|---|
| `pip install torch --index-url .../cu121` | RTX 5070 是 sm_120，需要 **cu128/cu130** | `torch.cuda.is_available()` 恒为 False，或 `no kernel image is available` |
| `conda create -n qlora python=3.11` | 你没装 conda；且环境里已有 3.14 | 第一步就卡住 |
| `SFTConfig(max_seq_length=512, ...)` | trl 1.x 的字段叫 **`max_length`** | `TypeError: unexpected keyword argument` |
| 定义了 `format_prompt` 却**没传给 SFTTrainer** | 模型收到的是原始 `instruction/input/output` 列 | 不报错，但**模板没生效**，你以为在学 `### Instruction:`，其实没有 |
| `tokenizer=` 传给 SFTTrainer | trl 1.x 用 **`processing_class=`** | 报参数错误或被忽略 |
| `torch_dtype=torch.bfloat16` | transformers 5.x 改成 **`dtype=`** | 只告警；但以后会真删 |

`demo_qlora.py` 已经按新 API 写好，并且把指南讲的每个原理（NF4、双重量化、LoRA 比例、梯度检查点）都打成了日志。

---

## 六、对应指南的 Day 1–Day 5，改成你的节奏

| 指南 | 你的版本 | 时间 |
|---|---|---|
| Day 1 装环境 + 线性回归 | `setup_qlora.ps1` + `verify_env.py`（自检比线性回归更有针对性） | 40 min（大头是等 1.85G 的 torch） |
| Day 2 跑通 0.5B | `python demo_qlora.py`（Qwen3-0.6B，约 10 分钟） | 1 h |
| Day 3 四组对照实验 | `--r 64` / `--samples 100` / `--lr 1e-3` / `--lr 2e-5` 各跑一次，记显存、耗时、loss | 2 h |
| Day 4 上「7B」换自己的数据 | `--model Qwen/Qwen3-4B-Instruct-2507`，数据换成组内论文的 100–300 条问答 | 2–3 h |
| Day 5 评估 + 打包 repo | 先 `winget install --id Git.Git -e`，再建 repo | 2 h |

---

## 七、常见报错 → 对症下药

| 报错 | 真正原因 | 处理 |
|---|---|---|
| `torch.cuda.is_available()` 是 False | 装成 CPU 版，或 PATH 上仍是 msys2/3.14 的 python | 先 `where.exe python` 确认是 `.venv\Scripts\python.exe` |
| `CUDA error: no kernel image is available` | torch 不是 cu128/cu130 构建 | 按第三节重装 torch |
| `libbitsandbytes_cuda*.dll` 找不到 | bnb 装成 CPU 轮子，或 DLL 被拦截 | `python -m pip install --force-reinstall bitsandbytes==0.50.2` |
| `OSError: We couldn't connect to huggingface.co` | 没配镜像 | `$env:HF_ENDPOINT="https://hf-mirror.com"` |
| `CUDA out of memory` | 显存不够 | 降 `--batch-size 1`、`--max-length 256`、换小模型 |
| `ValueError: Model does not support quantization` | 模型架构太老 | 换 Qwen3 / Qwen2.5 系列 |
| 训练 loss 不掉 | 数据模板不统一，或 lr 太小 | 确认 `dataset[0]['text']` 打印出来是完整模板；lr 试 1e-4~3e-4 |
| `无法加载文件 setup_qlora.ps1，因为在此系统上禁止运行脚本` | 执行策略 | `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` |

---

## 八、每天开工的第一件事

```powershell
cd qlora-demo
.\.venv\Scripts\Activate.ps1     # 提示符前出现 (.venv) 才算激活
python verify_env.py             # 10 秒，确认驱动没被系统更新搞坏
```

> 提醒一句：指南里说「每天卡住超过 40 分钟就把报错原文发我」——这条照做，尤其是 CUDA 相关报错，原文比描述有用十倍。

---

## 九、本地还是 Colab？—— 结论：**本地做主线，Colab 当备用**

先说清楚一件事：你之前两个项目在 Colab 上跑，是**因为当时本地没有环境**，不是因为 Colab 更好。这次要做的东西不一样了——它是一条要反复迭代、要出 repo、后面还要接 Flower 联邦 + Opacus 的链路。

### 两台机器摆在一起比

| 维度 | 本地 RTX 5070 | Colab 免费 T4 |
|---|---|---|
| 显存 | 12 GB | 15 GB（看着更大，但下面这些才是关键） |
| 算力 | Blackwell，bf16 原生，实测快 T4 **约 3–5 倍** | Turing（sm_75），**不支持 bf16**，只能 fp16 |
| 会话稳定性 | 你的机器，跑多久都行 | 免费版有会话时长上限，**断线 = 内存变量 + 本地文件全没** |
| 数据/产物 | 一直在本地磁盘，可 checkpoint 续训 | 每次重启都要重下模型、重跑前序 cell，除非挂 Drive |
| 可复现性 | 环境锁定在 `.venv`，半年后还能跑同一份结果 | 平台镜像会变，同一份代码下个月可能装不上 |
| 反复做对照实验 | Day 3 那 4 组实验，改个参数就重跑 | 每组都要重新排队等 GPU、重新下 1–2 GB 模型 |
| 多进程（联邦/DP） | ✅ 端口、多进程随便起 | ⚠️ 能用但很别扭，且断线就全废 |
| 成本 | 0（电费） | 0，但**免费额度/排队时间是真的成本**；Pro 约 ¥ 60–100/月 |
| 配置风险 | Blackwell 较新，需 cu130（脚本已覆盖） | 平台预装好了，风险反而低 |

你在 MNIST 那份记录里已经踩过两次「Colab 会话重启 → `mnist_cnn.pt` 不存在 → 变量全丢」。QLoRA 这个项目里同样的坑会放大：**一个 4B 模型的 4-bit 权重下载就要 8 GB，断一次线重来 10 分钟起步。**

### 所以这么用

**主线：本地。** 理由只有一条，但足够硬——**Day 3 的对照实验是这次学习的核心产出**（改 r、改 lr、改数据量，记「预测 vs 实测」）。这件事必须能在 5 分钟内改一次、跑一次；在 Colab 上，光排队和重下模型就能把学习节奏打碎。而且 Day 5 要出 repo、之后要接 Flower（多客户端进程），本地是唯一不折腾的选择。

**备用：Colab。** 只留两个用途：
1. **本地环境万一卡死超过 1 小时**——先在 Colab 上用同一份 `demo_qlora.py`（把 `bf16=True` 改成 `fp16=True`、`bnb_4bit_compute_dtype` 改 `torch.float16`）把 QLoRA 流程跑通，**把「理解 QLoRA」和「配环境」这两件事解耦**，别让环境挡住学习；
2. **想看一眼 7B/8B**——T4 的 15 GB 比你的 12 GB 宽一点，7B QLoRA 在 seq 512、bs=1 下能跑（慢，1–2 小时），适合「验证一下 7B 长什么样」，不适合当日常。

### 如果你决定两边都留着，注意这 3 处差异

| | 本地 5070 | Colab T4 |
|---|---|---|
| `bnb_4bit_compute_dtype` | `torch.bfloat16` | **`torch.float16`**（T4 无 bf16） |
| `SFTConfig` | `bf16=True` | **`fp16=True`** |
| 适配器存哪 | `./qlora-demo-adapter` | **挂 Drive**：`/content/drive/MyDrive/...`，否则会话一断就没了 |

一句话：**先花 40 分钟把本地配好，之后每一天的复习成本是 10 秒；一直用 Colab，每一天的启动成本是 10 分钟，而且随时可能清零。**

---

## 十、装完之后：实测数据（2026-10-19，本机）

### 环境

| 项 | 值 |
|---|---|
| Python / venv | 3.12.15，`.venv` 约 3.4 GB |
| torch | 2.14.1+cu130（`sm_120 in arch list: True`） |
| 自检 | `verify_env.py` 六项全 OK |

### 预下载的模型

`models/Qwen3-0.6B`（10 个文件，1.4 GB，放在仓库根或仓库同级均可）。`demo_qlora.py` 会自动优先用这个目录，
所以直接 `python demo_qlora.py` 就能秒开，不用再等下载。

### 第一次真实跑通的数字（200 条 alpaca，1 epoch，seq 256）

| 指标 | 实测 |
|---|---|
| 4-bit 底座权重显存 | **0.79 GB** |
| 可训练参数 | 4,587,520 / 600,637,440 = **0.76%** |
| 训练耗时 | **22 秒**（25 步） |
| 训练峰值显存 | **1.50 GB** |
| 最终 loss | 1.879 |
| adapter 体积 | 19.7 MB |
| 合并后推理峰值显存 | 2.41 GB |

也就是说：**12 GB 的卡跑这个 0.6B demo，显存连 1/5 都没用上**——Day 4 换 4B 完全不虚。

### 一个会反复遇到的坑：hf-mirror 的大文件会"假死"

`hf download` 下 1.4 GB 的 `model.safetensors` 时，连接会长时间停在某个字节数不动，
既不报错也不退出（本次卡过两次：1370 MB / 1274 MB 处）。**换 curl 短连接续传就稳**：

```powershell
$u   = 'https://hf-mirror.com/Qwen/Qwen3-0.6B/resolve/main/model.safetensors'
$out = 'models/Qwen3-0.6B/model.safetensors'
$want = 1503300328          # 文件的字节数，curl -sIL 可以查到
while ((Get-Item $out -ErrorAction SilentlyContinue).Length -lt $want) {
    curl.exe -sL --max-time 60 -C - -o $out $u
}
```

`--max-time 60` 让每次连接最多活 60 秒，`-C -` 从断点接着下，卡住就自动重来。
下 4B/8B 模型（8–16 GB）时这个循环基本是必备的。

另外两个已替你设好的环境变量：

- `HF_HUB_DISABLE_XET=1` —— hf-mirror 不支持 xet 加速协议，不关掉会报 `401 Unauthorized`
- `HF_HUB_DISABLE_SYMLINKS_WARNING=1`（可选）—— 没开 Windows 开发者模式时 HF 缓存用不了软链接，会刷警告
