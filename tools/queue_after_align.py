"""一次性排队器 queue_after_align：等 VLM align 重训结束后自动接棒编程续训，然后自我禁用。

背景（2026-10-17）：align 重训（指令->字幕，修抄写 shortcut）优先于编程长跑——
align 约 1 小时出关键验证，编程全程约 10 小时。本排队器守第二棒。

纪律（用户定调「本地只跑一条」）：
- vlm_align_v4_redo.log 不存在 => align 还没开始，安静等
- vlm_attach.py 活着 => align 在跑，安静等；查不动就保守等，绝不误启动
- 接棒：创建并触发一次性 schtasks 任务 CodeTrainRedo（SYSTEM），自己不生进程
- 启动后写 flag + 禁用自身 QueueCodeRedo —— 一次性，绝不复活
"""
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAG = os.path.join(ROOT, "logs", "code_redo_queued.flag")
LOG = os.path.join(ROOT, "logs", "queue_after_align.log")
ALIGN_LOG = os.path.join(ROOT, "logs", "vlm_align_v4_redo.log")
TASK_SELF = "QueueCodeRedo"
TASK_TRAIN = "CodeTrainRedo"
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
    if not os.path.exists(ALIGN_LOG):
        return  # align 还没开始跑，不是我们的时点
    if vlm_running():
        return
    with open(FLAG, "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log("align redo gone -> queueing CodeTrainRedo (1-epoch protocol run)")
    st = time.strftime("%H:%M", time.localtime(time.time() + 120))
    r1 = subprocess.run(["schtasks", "/Create", "/TN", TASK_TRAIN, "/TR", TRAIN_BAT,
                         "/SC", "ONCE", "/ST", st, "/RU", "SYSTEM", "/F"],
                        capture_output=True, text=True)
    r2 = subprocess.run(["schtasks", "/Run", "/TN", TASK_TRAIN], capture_output=True, text=True)
    r3 = subprocess.run(["schtasks", "/Change", "/TN", TASK_SELF, "/DISABLE"],
                        capture_output=True, text=True)
    log("create=%s run=%s self_disable=%s" % (r1.returncode, r2.returncode, r3.returncode))
    log("done: CodeTrainRedo launched, QueueCodeRedo disabled (one-shot, never revives)")


if __name__ == "__main__":
    main()
