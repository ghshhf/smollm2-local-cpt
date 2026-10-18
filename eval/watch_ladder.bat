@echo off
cd /d "%~dp0.."
set PYTHONIOENCODING=utf-8
set PYTHONUTF8=1
for /l %%i in (1,1,40) do (
  curl -s -m 20 -o "%~dp0ladder_result.txt" "http://47.99.131.12:8081/file/ladder_result.txt?key=k0u4mail"
  for %%s in ("%~dp0ladder_result.txt") do if %%~zs GTR 300 goto DONE
  ping -n 31 127.0.0.1 >nul
)
:DONE
echo RESULT_ARRIVED > "%~dp0ladder_watch.flag"
