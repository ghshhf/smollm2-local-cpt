@echo off
REM ============================================================
REM  训练链 自提权安装器
REM  双击本文件即可：若当前非管理员，自动请求 UAC 提权后重跑；
REM  提权成功后注册 Windows(SYSTEM) 计划任务并立即启动。
REM  注册后训练由 Windows 托管，脱离 agent 会话，不再被回收。
REM ============================================================

REM --- 检测是否已管理员(标准用户 net session 会失败) ---
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo 当前非管理员，请求 UAC 提权...
    powershell -Command "Start-Process -FilePath '%~f0' -Verb RunAs"
    exit /b
)

echo [admin] 已提权，开始注册训练任务...
schtasks /Create /TN "Train15BChain" /TR "D:/work/_local_ai/train_15b/venv/Scripts/pythonw.exe D:/work/_local_ai/train_15b/train_chain.py" /SC ONSTART /RU SYSTEM /F
if %errorlevel%==0 (
    schtasks /Run /TN "Train15BChain"
    echo [OK] 任务已注册并由 Windows 托管启动。训练不再依赖本窗口。
    echo 稍后可查状态: schtasks /Query /TN Train15BChain
) else (
    echo [FAIL] 注册失败，请确认以管理员身份运行。
)
pause
