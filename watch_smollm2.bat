@echo off
REM SmolLM2 continued-pretrain watchdog (SYSTEM scheduled task, every 5 min)
REM Capacity-level scale-up chain (train each size only to plateau, then expand):
REM   R2 135M classified novels WSD max_steps 900 cap ~3d = L1 plateau  -- running
REM   -> expand 135M to 0.2B L2 SmolLM2-200M
REM   -> R4 L2 lang foundation 1ep
REM   -> R5 L2 knowledge edu capped
REM   -> expand 0.2B to 0.3B L3 SmolLM2-300M
REM   -> R6 L3 full knowledge edu plus wiki 1ep  -- final model
REM   -> round6.done stop (no R7)
setlocal
cd /d "%~dp0"
set ROOT=D:\work\_local_ai\train_15b
set OUT135=%ROOT%\out_smollm2_135m_pt
set OUT200=%ROOT%\out_smollm2_200m_pt
set OUT300=%ROOT%\out_smollm2_300m_pt
set MODEL200=D:\work\_local_ai\models\SmolLM2-200M
set MODEL300=D:\work\_local_ai\models\SmolLM2-300M
set PY=%ROOT%\venv\Scripts\python.exe
set SCRIPT=%ROOT%\train_smollm2_pt.py

REM ===== routing =====
if exist "%OUT300%" goto :phaseC
if exist "%OUT200%" goto :phaseB
if exist "%OUT135%\round2.done" goto :phaseB
goto :phaseA

REM ===== phase A: R2 135M =====
:phaseA
set PID=
if exist "%OUT135%\train.pid" set /p PID=<"%OUT135%\train.pid"
if defined PID (
  tasklist /FI "PID eq %PID%" 2>nul | find /i "python.exe" >nul
  if not errorlevel 1 goto :eof
)
if exist "%OUT135%\done.flag" (
  move /y "%OUT135%\done.flag" "%OUT135%\round2.done" >nul
  echo [%date% %time%] === R2 DONE - expand 135M to 0.2B L2 === >> "%ROOT%\_smollm2_pt.log"
  if not exist "%MODEL200%\config.json" (
    "%PY%" -u "%ROOT%\expand_to_200m.py" >> "%ROOT%\_smollm2_pt.log" 2>&1
    if errorlevel 1 (
      echo [%date% %time%] expand failed, rollback done.flag >> "%ROOT%\_smollm2_pt.log"
      move /y "%OUT135%\round2.done" "%OUT135%\done.flag" >nul
      goto :eof
    )
  ) else ( echo [%date% %time%] 0.2B model exists, skip expand >> "%ROOT%\_smollm2_pt.log" )
  echo r4 > "%OUT200%\phase.txt"
  echo [%date% %time%] === ROUND4 START L2 - lang / max_steps 1000 / WSD === >> "%ROOT%\_smollm2_pt.log"
  "%PY%" -u "%SCRIPT%" --model_dir "%MODEL200%" --corpus "%ROOT%\pretrain_corpus\zh_text_lang.jsonl" --output_dir "%OUT200%" --seq_len 1024 --batch_size 4 --grad_accum 8 --lr 3e-4 --epochs 1 --warmup_steps 50 --sched wsd --decay_frac 0.12 --min_lr_ratio 0.05 --save_steps 10 --chunk_seconds 1800 --max_steps 1000 --resume >> "%ROOT%\_smollm2_pt.log" 2>&1
  goto :eof
)
cd /d "%ROOT%"
echo [%date% %time%] watchdog relaunch round2 >> "%ROOT%\_smollm2_pt.log"
"%PY%" -u "%SCRIPT%" --resume --chunk_seconds 1800 --save_steps 10 --epochs 2 --sched wsd --decay_frac 0.2 --min_lr_ratio 0.05 --warmup_steps 20 --max_steps 900 --corpus "%ROOT%\pretrain_corpus\zh_text_r2_cat.jsonl" --output_dir "%OUT135%" >> "%ROOT%\_smollm2_pt.log" 2>&1
goto :eof

REM ===== phase B: R4 / R5 0.2B L2 =====
:phaseB
set PID=
if exist "%OUT200%\train.pid" set /p PID=<"%OUT200%\train.pid"
if defined PID (
  tasklist /FI "PID eq %PID%" 2>nul | find /i "python.exe" >nul
  if not errorlevel 1 goto :eof
)
set PH=r4
if exist "%OUT200%\phase.txt" set /p PH=<"%OUT200%\phase.txt"
if exist "%OUT200%\done.flag" (
  if "%PH%"=="r5" (
    move /y "%OUT200%\done.flag" "%OUT200%\round5.done" >nul
    echo [%date% %time%] === R5 DONE - expand 0.2B to 0.3B L3 === >> "%ROOT%\_smollm2_pt.log"
    if not exist "%MODEL300%\config.json" (
      "%PY%" -u "%ROOT%\expand_300m.py" >> "%ROOT%\_smollm2_pt.log" 2>&1
      if errorlevel 1 (
        echo [%date% %time%] expand failed, rollback >> "%ROOT%\_smollm2_pt.log"
        move /y "%OUT200%\round5.done" "%OUT200%\done.flag" >nul
        goto :eof
      )
    ) else ( echo [%date% %time%] 0.3B model exists, skip expand >> "%ROOT%\_smollm2_pt.log" )
    echo r6 > "%OUT300%\phase.txt"
    echo [%date% %time%] === ROUND6 START L3 - full knowledge / 1ep / WSD === >> "%ROOT%\_smollm2_pt.log"
    "%PY%" -u "%SCRIPT%" --model_dir "%MODEL300%" --corpus "%ROOT%\pretrain_corpus\zh_text_know_full.jsonl" --output_dir "%OUT300%" --seq_len 1024 --batch_size 4 --grad_accum 8 --lr 3e-4 --epochs 1 --warmup_steps 60 --sched wsd --decay_frac 0.12 --min_lr_ratio 0.05 --save_steps 10 --chunk_seconds 1800 --resume >> "%ROOT%\_smollm2_pt.log" 2>&1
    goto :eof
  )
  REM R4 done -> R5 cap R4 end step plus 430 budget steps
  move /y "%OUT200%\done.flag" "%OUT200%\round4.done" >nul
  set /p CURSTEP=<"%OUT200%\checkpoint-latest\step.txt"
  set /a R5MAX=%CURSTEP% + 430
  echo %R5MAX% > "%OUT200%\r5max.txt"
  echo r5 > "%OUT200%\phase.txt"
  echo [%date% %time%] === ROUND5 START L2 - knowledge edu / max_steps=%R5MAX% / WSD === >> "%ROOT%\_smollm2_pt.log"
  "%PY%" -u "%SCRIPT%" --model_dir "%MODEL200%" --corpus "%ROOT%\pretrain_corpus\zh_text_know_edu.jsonl" --output_dir "%OUT200%" --seq_len 1024 --batch_size 4 --grad_accum 8 --lr 3e-4 --epochs 1 --warmup_steps 50 --sched wsd --decay_frac 0.12 --min_lr_ratio 0.05 --save_steps 10 --chunk_seconds 1800 --max_steps %R5MAX% --resume >> "%ROOT%\_smollm2_pt.log" 2>&1
  goto :eof
)
REM process dead, no done -> resume current phase
if "%PH%"=="r5" (
  set /p R5MAX=<"%OUT200%\r5max.txt"
  echo [%date% %time%] watchdog relaunch round5 max_steps=%R5MAX% >> "%ROOT%\_smollm2_pt.log"
  "%PY%" -u "%SCRIPT%" --model_dir "%MODEL200%" --corpus "%ROOT%\pretrain_corpus\zh_text_know_edu.jsonl" --output_dir "%OUT200%" --seq_len 1024 --batch_size 4 --grad_accum 8 --lr 3e-4 --epochs 1 --warmup_steps 50 --sched wsd --decay_frac 0.12 --min_lr_ratio 0.05 --save_steps 10 --chunk_seconds 1800 --max_steps %R5MAX% --resume >> "%ROOT%\_smollm2_pt.log" 2>&1
  goto :eof
)
echo [%date% %time%] watchdog relaunch round4 >> "%ROOT%\_smollm2_pt.log"
"%PY%" -u "%SCRIPT%" --model_dir "%MODEL200%" --corpus "%ROOT%\pretrain_corpus\zh_text_lang.jsonl" --output_dir "%OUT200%" --seq_len 1024 --batch_size 4 --grad_accum 8 --lr 3e-4 --epochs 1 --warmup_steps 50 --sched wsd --decay_frac 0.12 --min_lr_ratio 0.05 --save_steps 10 --chunk_seconds 1800 --max_steps 1000 --resume >> "%ROOT%\_smollm2_pt.log" 2>&1
goto :eof

REM ===== phase C: R6 0.3B L3 =====
:phaseC
set PID=
if exist "%OUT300%\train.pid" set /p PID=<"%OUT300%\train.pid"
if defined PID (
  tasklist /FI "PID eq %PID%" 2>nul | find /i "python.exe" >nul
  if not errorlevel 1 goto :eof
)
if exist "%OUT300%\done.flag" (
  move /y "%OUT300%\done.flag" "%OUT300%\round6.done" >nul
  echo [%date% %time%] === ROUND6 DONE - all training complete L3 - watchdog stops === >> "%ROOT%\_smollm2_pt.log"
  goto :eof
)
echo [%date% %time%] watchdog relaunch round6 >> "%ROOT%\_smollm2_pt.log"
"%PY%" -u "%SCRIPT%" --model_dir "%MODEL300%" --corpus "%ROOT%\pretrain_corpus\zh_text_know_full.jsonl" --output_dir "%OUT300%" --seq_len 1024 --batch_size 4 --grad_accum 8 --lr 3e-4 --epochs 1 --warmup_steps 60 --sched wsd --decay_frac 0.12 --min_lr_ratio 0.05 --save_steps 10 --chunk_seconds 1800 --resume >> "%ROOT%\_smollm2_pt.log" 2>&1
goto :eof
endlocal
