@echo off
REM Double-click launcher for Claude Code env-var setup PowerShell script.
REM All args pass through to the PS script (-DryRun / -Uninstall / -Writing / -Force).
REM Usage:
REM   setup_claude_code_env.bat
REM   setup_claude_code_env.bat -DryRun
REM   setup_claude_code_env.bat -Uninstall

setlocal
cd /d "%~dp0"

REM Force console codepage to UTF-8 so the PS script's Chinese output renders correctly.
chcp 65001 >nul

REM Prefer PowerShell 7 (pwsh); fall back to Windows built-in powershell 5.1+.
where pwsh >nul 2>nul
if %ERRORLEVEL%==0 (
    set "PSEXE=pwsh"
) else (
    set "PSEXE=powershell"
)

"%PSEXE%" -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_claude_code_env.ps1" %*
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [ERROR] Setup script returned non-zero exit code: %RC%
    pause
)

endlocal ^& exit /b %RC%
