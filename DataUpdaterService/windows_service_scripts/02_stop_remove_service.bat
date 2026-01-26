@echo off
setlocal enabledelayedexpansion

REM ==========================================================
REM TradeTrainMode Data Updater - Stop + Remove (double click)
REM 需要「以系統管理員身分執行」才能停止/移除 Windows Service
REM ==========================================================

REM --- admin check ---
net session >nul 2>&1
if %errorlevel% neq 0 (
  echo [ERROR] Please run this .bat as Administrator.
  echo         Right-click ^> Run as administrator
  pause
  exit /b 1
)

REM --- project root: DataUpdaterService/windows_service_scripts/ -> project root ---
set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%..\.."
pushd "%PROJECT_ROOT%" >nul

REM --- pick python ---
set "PY=python"
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" set "PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"

echo [INFO] Using python: %PY%
echo [INFO] Working dir : %CD%

echo.
echo [STEP] Stop service...
%PY% -m DataUpdaterService.service --stop
echo [INFO] Stop returned code=%errorlevel%

echo.
echo [STEP] Remove service...
%PY% -m DataUpdaterService.service --remove
if %errorlevel% neq 0 (
  echo [ERROR] Remove failed.
  popd >nul
  pause
  exit /b 1
)

echo.
echo [OK] Service stopped (if running) and removed.
popd >nul
pause


