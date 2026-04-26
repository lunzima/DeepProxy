@echo off
REM DeepProxy launcher for Windows.
REM Usage: start.bat                    (default host/port from config.yaml)
REM        start.bat 0.0.0.0 8080       (override host/port via env vars)

setlocal

REM Resolve script directory and switch to it so config.yaml is found.
cd /d "%~dp0"

REM Optional CLI args override host / coding_port / writing_port.
if not "%~1"=="" set "PROXY_HOST=%~1"
if not "%~2"=="" set "PROXY_CODING_PORT=%~2"
if not "%~3"=="" set "PROXY_WRITING_PORT=%~3"

REM Pick interpreter: prefer local virtualenv, fall back to system Python.
set "PYTHON_EXE="
if exist ".venv\Scripts\python.exe" set "PYTHON_EXE=.venv\Scripts\python.exe"
if not defined PYTHON_EXE if exist "venv\Scripts\python.exe" set "PYTHON_EXE=venv\Scripts\python.exe"
if not defined PYTHON_EXE set "PYTHON_EXE=python"

REM Sanity check.
"%PYTHON_EXE%" --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python interpreter not found: %PYTHON_EXE%
    echo Please install Python 3.14+ or create a virtualenv at .venv\
    exit /b 1
)

echo ================================================================
echo  DeepProxy starting (dual-port)
echo  Interpreter : %PYTHON_EXE%
echo  Coding port : %PROXY_CODING_PORT%   (precise_sampling profile)
echo  Writing port: %PROXY_WRITING_PORT%   (creative_sampling profile)
echo  Working dir : %CD%
echo  (Empty port values mean defaults from config.yaml)
echo ================================================================

"%PYTHON_EXE%" -m deep_proxy.server
set "RC=%ERRORLEVEL%"

if not "%RC%"=="0" (
    echo.
    echo [ERROR] DeepProxy exited with code %RC%
)

endlocal & exit /b %RC%
