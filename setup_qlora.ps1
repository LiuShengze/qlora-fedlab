<#
.SYNOPSIS
  为 RTX 5070（Blackwell / sm_120）搭 QLoRA 微调环境，Windows 11 一步到位。

.DESCRIPTION
  做四件事：
    1) 准备一个干净的 CPython 3.12（不用系统里现有的 msys2 / Store 版 Python）
    2) 在 .\\.venv 建独立虚拟环境
    3) 装 CUDA 版 torch（cu130，走南大镜像，比官方源快很多）+ 其余依赖（走清华 PyPI 镜像）
    4) 设置 HF 镜像/缓存目录，并跑 verify_env.py 自检

  重复执行是安全的：已有 venv / 已有包会跳过或升级。

.EXAMPLE
  pwsh -File .\setup_qlora.ps1
  pwsh -File .\setup_qlora.ps1 -HfHome D:\hf-cache
#>
[CmdletBinding()]
param(
    [string]$PythonVersion = "3.12",
    [string]$HfHome = "D:\MyDeepSeekHarness\qlora-demo\hf-cache",
    [string]$TorchIndex = "https://mirror.nju.edu.cn/pytorch/whl/cu130",
    [string]$PypiIndex = "https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple"
)

# 用 Continue 而不是 Stop：uv / pip 会把进度条写到 stderr，Windows PowerShell 5.1
# 在 Stop 模式下会把这种 stderr 当成终止性错误，脚本会莫名其妙中断。所有关键步骤
# 改为显式检查 $LASTEXITCODE。
$ErrorActionPreference = "Continue"
$root = $PSScriptRoot
$venv = Join-Path $root ".venv"
$py   = Join-Path $venv "Scripts\python.exe"

function Step($n, $msg) { Write-Host "`n[$n] $msg" -ForegroundColor Cyan }
function Refresh-Path {
    $m = [Environment]::GetEnvironmentVariable('Path','Machine')
    $u = [Environment]::GetEnvironmentVariable('Path','User')
    $env:Path = (@($m, $u) | Where-Object { $_ }) -join ';'
}
function Fail($msg) { Write-Host "`n✗ $msg" -ForegroundColor Red; exit 1 }

# ---------------------------------------------------------------- 0. 现状提示
Step 0 "环境检查"
Write-Host "  项目目录 : $root"
Write-Host "  Python   : $PythonVersion"

# ---------------------------------------------------------------- 1. 找一个干净的 Python
Step 1 "准备 Python $PythonVersion 运行时（不动系统里现有的 3.14）"
$uv = Get-Command uv -ErrorAction SilentlyContinue
if (-not $uv) {
    Write-Host "  未找到 uv，尝试用 winget 安装 ..."
    if (Get-Command winget -ErrorAction SilentlyContinue) {
        try {
            winget install --id astral-sh.uv -e --accept-package-agreements --accept-source-agreements | Out-Host
            Refresh-Path
            $uv = Get-Command uv -ErrorAction SilentlyContinue
        } catch { Write-Host "  winget 安装失败：$($_.Exception.Message)" -ForegroundColor Yellow }
    }
}

Step 2 "创建虚拟环境 .venv"
$venvReady = $false
if (Test-Path $py) {
    Write-Host "  已存在，复用。"
    $venvReady = $true
} elseif ($uv) {
    Write-Host "  uv: $($uv.Source)"
    uv python install $PythonVersion
    if ($LASTEXITCODE -eq 0) {
        uv venv $venv --python $PythonVersion --seed
        $venvReady = Test-Path $py
    } else {
        Write-Host "  uv 拉取 Python 失败，走备用通道。" -ForegroundColor Yellow
    }
}

if (-not $venvReady) {
    # 备用通道：官方 3.12 安装包（华为云镜像，实测约 10 MB/s）
    Write-Host "  改用官方 Python $PythonVersion 安装包（华为云镜像）..." -ForegroundColor Yellow
    $installer = Join-Path $env:TEMP "python-$PythonVersion-amd64.exe"
    try {
        Invoke-WebRequest -UseBasicParsing -TimeoutSec 600 `
            -Uri "https://mirrors.huaweicloud.com/python/3.12.10/python-3.12.10-amd64.exe" -OutFile $installer
    } catch {
        Fail "下载 Python 安装包失败：$($_.Exception.Message)"
    }
    $targetDir = Join-Path $env:LOCALAPPDATA "Programs\Python\Python312"
    Start-Process -FilePath $installer -Wait -ArgumentList @(
        "/quiet", "InstallAllUsers=0", "TargetDir=$targetDir",
        "Include_pip=1", "Include_launcher=0", "PrependPath=0", "Shortcuts=0"
    )
    $basePy = Join-Path $targetDir "python.exe"
    if (-not (Test-Path $basePy)) { Fail "Python 安装失败：$targetDir 下没有 python.exe" }
    & $basePy -m venv $venv
}

if (-not (Test-Path $py)) { Fail "虚拟环境创建失败：$py 不存在" }
& $py -m pip install -U pip --quiet --index-url $PypiIndex
if ($LASTEXITCODE -ne 0) { Write-Host "  pip 升级失败，继续（不影响装包）" -ForegroundColor Yellow }
& $py -c "import sys; print('  venv python:', sys.version.split()[0], sys.executable)"

# ---------------------------------------------------------------- 3. torch（CUDA 版）
Step 3 "安装 CUDA 版 PyTorch（cu130，南大镜像）"
Write-Host "  index: $TorchIndex"
& $py -m pip install torch --index-url $TorchIndex
if ($LASTEXITCODE -ne 0) {
    Write-Host "  cu130 失败，回退 cu128 ..." -ForegroundColor Yellow
    & $py -m pip install torch --index-url "https://mirror.nju.edu.cn/pytorch/whl/cu128"
    if ($LASTEXITCODE -ne 0) { Fail "torch 安装失败，检查网络或镜像可达性。" }
}
& $py -c @"
import torch
print('  torch:', torch.__version__, '| CUDA build:', torch.version.cuda)
print('  sm_120 in arch list:', 'sm_120' in torch.cuda.get_arch_list())
"@

# ---------------------------------------------------------------- 4. 其余依赖
Step 4 "安装 transformers / peft / trl / bitsandbytes 等"
& $py -m pip install -r (Join-Path $root "requirements.txt") --index-url $PypiIndex
if ($LASTEXITCODE -ne 0) {
    Write-Host "  清华镜像失败，回退官方 PyPI ..." -ForegroundColor Yellow
    & $py -m pip install -r (Join-Path $root "requirements.txt")
    if ($LASTEXITCODE -ne 0) { Fail "依赖安装失败。" }
}

# ---------------------------------------------------------------- 5. HF 镜像 + 缓存
Step 5 "配置 HuggingFace 镜像与缓存目录"
Write-Host "  huggingface.co 在本机不可直连，统一走 hf-mirror.com"
[Environment]::SetEnvironmentVariable("HF_ENDPOINT", "https://hf-mirror.com", "User")
[Environment]::SetEnvironmentVariable("HF_HOME", $HfHome, "User")
# hf-mirror 不支持 xet 加速协议，不关掉会报 401 Unauthorized
[Environment]::SetEnvironmentVariable("HF_HUB_DISABLE_XET", "1", "User")
$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:HF_HOME = $HfHome
$env:HF_HUB_DISABLE_XET = "1"
New-Item -ItemType Directory -Force -Path $HfHome | Out-Null
Write-Host "  HF_ENDPOINT = $env:HF_ENDPOINT"
Write-Host "  HF_HOME     = $env:HF_HOME"

# ---------------------------------------------------------------- 6. 自检
Step 6 "环境自检"
& $py (Join-Path $root "verify_env.py")
if ($LASTEXITCODE -ne 0) { Fail "自检未通过，把上面的输出发给我。" }

Write-Host @"

============================================================
全部就绪。以后每次开工，先激活环境：

    .\.venv\Scripts\Activate.ps1

然后：
    python demo_qlora.py                    # 0.6B 冒烟版（约 10 分钟）
    `$env:QLORA_MODEL="Qwen/Qwen3-4B-Instruct-2507"; python demo_qlora.py   # 12G 显存正式版

（如果 PowerShell 提示禁止运行脚本：
    Set-ExecutionPolicy -Scope CurrentUser RemoteSigned ）
============================================================
"@ -ForegroundColor Green
