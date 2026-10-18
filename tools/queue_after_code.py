"""一次性排队器 queue_after_code：等编程续训结束后自动接棒 VLM align 重训，然后自我禁用。

背景（2026-10-17）：align 曾误用 (text,text) 抄写目标，val loss 0.02 是复制 shortcut，
视觉零学习（实测：训练见过的图也吐垃圾）。已修 build_vqa_prompt 为指令->字幕，
本排队器负责在编程训练让出机器后自动重跑 align（同一 3000 步配置，缓存复用）。

纪律（用户定调「本地只跑一条」）：
- 轮询到 train_smollm2_pt.py 消失才动作；活着就安静返回；查不动就保守等
- 接棒：创建并触发一次性 schtasks 任务 VLMAlignRedo（SYSTEM），自己不生进程
- 启动后写 flag + 禁用自身 QueueVLMRedo —— 一次性，绝不复活
"""
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAG = os.path.join(ROOT, "logs", "vlm_align_redo_queued.flag")
LOG = os.path.join(ROOT, "logs", "queue_after_code.log")
TASK_SELF = "QueueVLMRedo"
TASK_TRAIN = "VLMAlignRedo"
TRAIN_BAT = os.path.join(ROOT, "run_vlm_pipeline.bat")


def log(msg):
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        f.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))


def train_running():
    ps = ("(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' "
          "-and $_.CommandLine -like '*train_smollm2_pt*' } | Measure-Object).Count")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip()) > 0
    except Exception:
        return True  # 查不动就保守等着，绝不误启动


def main():
    if os.path.exists(FLAG):
        return
    if train_running():
        return
    with open(FLAG, "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log("code train gone -> queueing VLMAlignRedo (align redo, instruction->caption)")
    st = time.strftime("%H:%M", time.localtime(time.time() + 120))
    r1 = subprocess.run(["schtasks", "/Create", "/TN", TASK_TRAIN, "/TR", TRAIN_BAT,
                         "/SC", "ONCE", "/ST", st, "/RU", "SYSTEM", "/F"],
                        capture_output=True, text=True)
    r2 = subprocess.run(["schtasks", "/Run", "/TN", TASK_TRAIN], capture_output=True, text=True)
    r3 = subprocess.run(["schtasks", "/Change", "/TN", TASK_SELF, "/DISABLE"],
                        capture_output=True, text=True)
    log("create=%s run=%s self_disable=%s" % (r1.returncode, r2.returncode, r3.returncode))
    log("done: VLMAlignRedo launched, QueueVLMRedo disabled (one-shot, never revives)")


if __name__ == "__main__":
    main()
