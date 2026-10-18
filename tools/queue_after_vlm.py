"""一次性排队器 queue_after_vlm：等 VLM align 训练结束后自动接棒编程续训，然后自我禁用。

纪律（2026-10-17 用户定调「本地只跑一条」）：
- 轮询到 vlm_attach.py 进程消失（VLM 让出机器）才动作；VLM 活着就安静返回
- 接棒方式：创建并触发一次性 schtasks 任务 CodeTrainV1（SYSTEM），自己不生进程
- 启动后写 flag + 禁用自身任务 QueueCodeTrain —— 一次性，绝不复活
- 日志只写 ASCII（避免 Windows 编码坑）；查不动进程时保守等待
"""
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAG = os.path.join(ROOT, "logs", "code_train_queued.flag")
LOG = os.path.join(ROOT, "logs", "queue_after_vlm.log")
TASK_SELF = "QueueCodeTrain"
TASK_TRAIN = "CodeTrainV1"
TRAIN_BAT = os.path.join(ROOT, "run_code_train.bat")


def log(msg):
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        f.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))


def vlm_running():
    ps = ("(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' "
          "-and $_.CommandLine -like '*vlm_attach*' } | Measure-Object).Count")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip()) > 0
    except Exception:
        return True  # 查不动就保守等着，绝不误启动


def main():
    if os.path.exists(FLAG):
        return
    if vlm_running():
        return
    # VLM 已让出机器 -> 接棒
    with open(FLAG, "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log("VLM gone -> queueing CodeTrainV1")
    st = time.strftime("%H:%M", time.localtime(time.time() + 120))
    r1 = subprocess.run(["schtasks", "/Create", "/TN", TASK_TRAIN, "/TR", TRAIN_BAT,
                         "/SC", "ONCE", "/ST", st, "/RU", "SYSTEM", "/F"],
                        capture_output=True, text=True)
    r2 = subprocess.run(["schtasks", "/Run", "/TN", TASK_TRAIN], capture_output=True, text=True)
    r3 = subprocess.run(["schtasks", "/Change", "/TN", TASK_SELF, "/DISABLE"],
                        capture_output=True, text=True)
    log("create=%s run=%s self_disable=%s" % (r1.returncode, r2.returncode, r3.returncode))
    log("done: CodeTrainV1 launched, QueueCodeTrain disabled (one-shot, never revives)")


if __name__ == "__main__":
    main()
