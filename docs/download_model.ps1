# Download Qwen3-4B-Instruct-2507 (8 GB, 3 shards).
# Source priority: ModelScope (fast in CN, ~15 MB/s) then hf-mirror as fallback.
# Large files stall on long connections: use --max-time 60 with -C - in a retry loop.
param(
    [string]$Base = "https://modelscope.cn/models/Qwen/Qwen3-4B-Instruct-2507/resolve/master"
)
$ErrorActionPreference = "Continue"
$dir = 'D:\MyDeepSeekHarness\models\Qwen3-4B-Instruct-2507'
New-Item -ItemType Directory -Force $dir | Out-Null

$expect = [ordered]@{
    '.gitattributes'                    = 1570
    'LICENSE'                           = 11343
    'README.md'                         = 8168
    'config.json'                       = 727
    'generation_config.json'            = 238
    'merges.txt'                        = 1671839
    'model.safetensors.index.json'      = 32819
    'tokenizer.json'                    = 11422654
    'tokenizer_config.json'             = 9377
    'vocab.json'                        = 2776833
    'model-00001-of-00003.safetensors'  = 3957900840
    'model-00002-of-00003.safetensors'  = 3987450520
    'model-00003-of-00003.safetensors'  = 99630640
}

foreach ($f in $expect.Keys) {
    $out  = Join-Path $dir $f
    $want = [int64]$expect[$f]
    $attempt = 0
    while ($true) {
        $have = if (Test-Path $out) { (Get-Item $out).Length } else { 0 }
        if ($have -ge $want) { break }
        $attempt++
        if ($attempt -gt 120) { break }
        & curl.exe -sL --max-time 60 -C - -o $out "$Base/$f" | Out-Null
        Start-Sleep -Milliseconds 200
    }
    $now = if (Test-Path $out) { (Get-Item $out).Length } else { 0 }
    $mark = if ($now -eq $want) { "OK  " } else { "BAD " }
    Write-Output ("{0}{1,-34} {2,13} / {3,13}  attempts={4}" -f $mark, $f, $now, $want, $attempt)
}

$allOk = $true
foreach ($f in $expect.Keys) {
    $p = Join-Path $dir $f
    if (-not (Test-Path $p) -or (Get-Item $p).Length -ne [int64]$expect[$f]) { $allOk = $false }
}
if ($allOk) { Write-Output "ALL-DONE" } else { Write-Output "SOME-FILES-INCOMPLETE" }
