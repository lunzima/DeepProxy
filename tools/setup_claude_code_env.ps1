<#
.SYNOPSIS
    一键配置 Windows 用户级永久环境变量，让 Claude Code 接入本地 DeepProxy。

.DESCRIPTION
    1. 读取 config.yaml 拿到 host / coding_port / writing_port / api_key
       （host=0.0.0.0 自动改写为 127.0.0.1；缺失则用默认值）
    2. 用 [Environment]::SetEnvironmentVariable($name, $value, "User") 持久化写入
       用户级环境变量（等价 setx，无需管理员，永久生效，且能写 $null 删除）
    3. 同步写入当前 session（$env:VAR=value），让当前终端立即可用
    4. 主动删除 ANTHROPIC_API_KEY（若存在），避免与 ANTHROPIC_AUTH_TOKEN 冲突
    5. 配置 CLAUDE_CODE_ATTRIBUTION_HEADER=false 从客户端源头关掉 billing header

.PARAMETER DryRun
    只打印将写入的值，不实际写。

.PARAMETER Uninstall
    删除本脚本曾设置的 5 个变量（不动 ANTHROPIC_API_KEY）。

.PARAMETER Writing
    指向 writing_port 而非默认 coding_port。

.PARAMETER Force
    跳过已有 ANTHROPIC_API_KEY 的覆盖确认。

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File setup_claude_code_env.ps1
    powershell -ExecutionPolicy Bypass -File setup_claude_code_env.ps1 -DryRun
    powershell -ExecutionPolicy Bypass -File setup_claude_code_env.ps1 -Uninstall
#>
param(
    [switch]$DryRun,
    [switch]$Uninstall,
    [switch]$Writing,
    [switch]$Force
)

$ErrorActionPreference = 'Stop'

# 强制 UTF-8 输出，避免在系统默认非 UTF-8 codepage（如 GBK）的终端上中文乱码
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$OutputEncoding = [System.Text.Encoding]::UTF8

$MANAGED_VARS = @(
    'ANTHROPIC_BASE_URL',
    'ANTHROPIC_AUTH_TOKEN',
    'ANTHROPIC_MODEL',
    'ANTHROPIC_SMALL_FAST_MODEL',
    'CLAUDE_CODE_ATTRIBUTION_HEADER'
)

function Write-Section($title) {
    Write-Host ""
    Write-Host ("=" * 64) -ForegroundColor DarkGray
    Write-Host " $title" -ForegroundColor Cyan
    Write-Host ("=" * 64) -ForegroundColor DarkGray
}

function Mask-Token([string]$token) {
    if ([string]::IsNullOrEmpty($token)) { return '<empty>' }
    if ($token.Length -le 10) {
        $head = $token.Substring(0, [Math]::Min(2, $token.Length))
        return ($head + '***')
    }
    return ($token.Substring(0, 6) + '...' + $token.Substring($token.Length - 2) + " (len=$($token.Length))")
}

function Set-UserEnv([string]$name, $value) {
    # User 级永久写入；同时写入当前 session
    [Environment]::SetEnvironmentVariable($name, $value, 'User')
    if ($null -eq $value) {
        Remove-Item -Path "Env:$name" -ErrorAction SilentlyContinue
    } else {
        Set-Item -Path "Env:$name" -Value $value
    }
}

function Parse-ConfigYaml([string]$path) {
    # 正则提取顶层 host / coding_port / writing_port / api_key —— 不引入 YAML 模块依赖
    $defaults = @{
        host = '127.0.0.1'
        coding_port = 8000
        writing_port = 8001
        api_key = $null
    }
    if (-not (Test-Path -LiteralPath $path)) {
        Write-Warning "config.yaml 未找到: $path（使用默认值）"
        return $defaults
    }
    try {
        $lines = Get-Content -LiteralPath $path -Encoding UTF8
    } catch {
        Write-Warning "读取 config.yaml 失败: $_（使用默认值）"
        return $defaults
    }
    $result = $defaults.Clone()
    foreach ($line in $lines) {
        # 仅处理顶层字段（行首无缩进），跳过 deepseek: 嵌套层
        if ($line -match '^host:\s*["'']?([^"''\s#]+)') {
            $result.host = $matches[1]
        } elseif ($line -match '^coding_port:\s*(\d+)') {
            $result.coding_port = [int]$matches[1]
        } elseif ($line -match '^writing_port:\s*(\d+)') {
            $result.writing_port = [int]$matches[1]
        } elseif ($line -match '^api_key:\s*["'']?([^"''\s#]+)') {
            $val = $matches[1]
            if ($val -ne 'null') { $result.api_key = $val }
        }
    }
    if ($result.host -eq '0.0.0.0') { $result.host = '127.0.0.1' }
    return $result
}

# -------- 主流程 --------

$scriptDir = $PSScriptRoot
if (-not $scriptDir) { $scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition }
$repoRoot  = Split-Path -Parent $scriptDir
$configPath = Join-Path $repoRoot 'config.yaml'

if ($Uninstall) {
    Write-Section "Uninstall - 删除 Claude Code 环境变量"
    foreach ($name in $MANAGED_VARS) {
        if ($DryRun) {
            Write-Host "  [DryRun] 将删除 $name" -ForegroundColor Yellow
        } else {
            Set-UserEnv -name $name -value $null
            Write-Host "  - 已删除 $name" -ForegroundColor Green
        }
    }
    Write-Host ""
    Write-Host "ANTHROPIC_API_KEY 未被删除（脚本仅清理自己设置过的 5 个变量）" -ForegroundColor DarkGray
    Write-Host "请打开新终端使删除生效（当前 session 已同步）" -ForegroundColor Cyan
    exit 0
}

$cfg = Parse-ConfigYaml -path $configPath
$port = if ($Writing) { $cfg.writing_port } else { $cfg.coding_port }
$token = if ($cfg.api_key) { $cfg.api_key } else { 'dummy' }
$baseUrl = "http://$($cfg.host):$port"

# 冲突检测：现有 ANTHROPIC_API_KEY
$existingApiKey = [Environment]::GetEnvironmentVariable('ANTHROPIC_API_KEY', 'User')
if ($existingApiKey -and -not $Force) {
    if (-not [Environment]::UserInteractive) {
        Write-Error "检测到 ANTHROPIC_API_KEY 已设置；非交互环境请加 -Force 或先手动删除。"
        exit 3
    }
    Write-Host ""
    Write-Host "检测到已有 ANTHROPIC_API_KEY ($(Mask-Token $existingApiKey))" -ForegroundColor Yellow
    Write-Host "Claude Code 在 ANTHROPIC_API_KEY 与 ANTHROPIC_AUTH_TOKEN 同时存在时会优先用前者，"
    Write-Host "导致本脚本设置的 AUTH_TOKEN 不生效。"
    $reply = Read-Host "是否删除 ANTHROPIC_API_KEY ? (y/N)"
    if ($reply -notmatch '^[yY]') {
        Write-Error "已取消（用户拒绝覆盖）。"
        exit 3
    }
}

$plan = [ordered]@{
    'ANTHROPIC_BASE_URL'              = $baseUrl
    'ANTHROPIC_AUTH_TOKEN'            = $token
    'ANTHROPIC_MODEL'                 = 'deepseek-v4-pro[1m]'
    'ANTHROPIC_SMALL_FAST_MODEL'      = 'deepseek-v4-flash'
    'CLAUDE_CODE_ATTRIBUTION_HEADER'  = 'false'
}

Write-Section "Setup - Claude Code -> DeepProxy 环境变量"
Write-Host "  配置来源: $configPath"
Write-Host "  目标端口: $port ($(if($Writing){'writing'}else{'coding'}))"
Write-Host ""
Write-Host "  将写入用户级永久变量:"
foreach ($k in $plan.Keys) {
    $v = $plan[$k]
    $display = if ($k -eq 'ANTHROPIC_AUTH_TOKEN') { Mask-Token $v } else { $v }
    Write-Host ("    {0,-32} = {1}" -f $k, $display)
}
Write-Host ""
Write-Host "  将删除（避免与 AUTH_TOKEN 冲突）:"
Write-Host ("    {0,-32} = (deleted)" -f 'ANTHROPIC_API_KEY')

if ($DryRun) {
    Write-Host ""
    Write-Host "[DryRun] 未实际写入任何变量。" -ForegroundColor Yellow
    exit 0
}

try {
    foreach ($k in $plan.Keys) {
        Set-UserEnv -name $k -value $plan[$k]
    }
    Set-UserEnv -name 'ANTHROPIC_API_KEY' -value $null
} catch {
    Write-Error "写入失败: $_"
    exit 2
}

Write-Host ""
Write-Host "完成。" -ForegroundColor Green
Write-Host "  当前终端已同步可用；其它已打开的终端需重启后才会读到新值。" -ForegroundColor Cyan
Write-Host "  使用：1) 启动 DeepProxy (双击 start.bat)  2) 新开终端运行 'claude'" -ForegroundColor Cyan
exit 0
