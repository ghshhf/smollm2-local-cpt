@echo off
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
"%~dp0..\venv\Scripts\python.exe" -u "%~dp0v1_01_ternary_probe.py" --arm all --steps 200 --out "%~dp0v1_01_probe_result.json" >> "%~dp0v1_01_probe.log" 2>&1
