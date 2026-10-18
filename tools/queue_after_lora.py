"""一次性排队器 queue_after_lora：等另一会话的 LoRA 微调让出机器后，自动接棒 VLM align 重训。

背景（2026-10-17 14:00 实况）：
- 另一 WorkBuddy 会话在跑 train_lora.py（r6_final + chat_sft_v1.jsonl，42/154 步，
  预计 14:40 完，日志 logs/chat_lora_v1.log）——那是别人的任务，本排队器绝不碰它。
- 本会话的 align 重训 13:33 启动，step 50 时被 LoRA 启动的内存峰值挤死（OOM 同型）。
- 本排队器守第一棒：LoRA 进程全消失 -> 触发 VLMAlignRedo 重跑 align（3000步/约1小时）
  -> 顺手建第二棒 QueueCodeRedo（等 align 完接编程续训）-> 自我禁用。

纪律（用户定调「本地只跑一条」）：
- chat_lora_v1.log 不存在 => LoRA 还没跑过，安静等
- train_lora.py 活着 => 别人在用机器，安静等；查不动就保守等，绝不误启动
- 一次性，绝不复活
"""
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAG = os.path.join(ROOT, "logs", "vlm_align_redo2_queued.flag")
LOG = os.path.join(ROOT, "logs", "queue_after_lora.log")
LORA_LOG = os.path.join(ROOT, "logs", "chat_lora_v1.log")
TASK_SELF = "QueueVLMRedo2"
TASK_ALIGN = "VLMAlignRedo"
ALIGN_BAT = os.path.join(ROOT, "run_vlm_pipeline.bat")
# 第二棒：align 完了接编程续训
TASK_CODE_WATCH = "QueueCodeRedo"
CODE_WATCH_BAT = os.path.join(ROOT, "run_queue_watch3.bat")


def log(msg):
    with open(LOG, "a", encoding="utf-8", errors="replace") as f:
        f.write("[%s] %s\n" % (time.strftime("%H:%M:%S"), msg))


def lora_running():
    ps = ("(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' "
          "-and $_.CommandLine -like '*train_lora*' } | Measure-Object).Count")
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-Command", ps],
                             capture_output=True, text=True, timeout=30)
        return int(out.stdout.strip()) > 0
    except Exception:
        return True  # 查不动就保守等着，绝不误启动


def main():
    if os.path.exists(FLAG):
        return
    if not os.path.exists(LORA_LOG):
        return  # LoRA 还没开始，不是我们的时点
    if lora_running():
        return
    with open(FLAG, "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log("lora gone -> launching VLMAlignRedo (align redo, instruction->caption)")
    st = time.strftime("%H:%M", time.localtime(time.time() + 120))
    r1 = subprocess.run(["schtasks", "/Create", "/TN", TASK_ALIGN, "/TR", ALIGN_BAT,
                         "/SC", "ONCE", "/ST", st, "/RU", "SYSTEM", "/F"],
                        capture_output=True, text=True)
    r2 = subprocess.run(["schtasks", "/Run", "/TN", TASK_ALIGN], capture_output=True, text=True)
    # 建第二棒：align 完了自动接编程续训（一次性，之后自我禁用）
    r3 = subprocess.run(["schtasks", "/Create", "/TN", TASK_CODE_WATCH, "/TR", CODE_WATCH_BAT,
                         "/SC", "MINUTE", "/MO", "3", "/RU", "SYSTEM", "/F"],
                        capture_output=True, text=True)
    r4 = subprocess.run(["schtasks", "/Change", "/TN", TASK_SELF, "/DISABLE"],
                        capture_output=True, text=True)
    log("align_create=%s align_run=%s code_watch_create=%s self_disable=%s"
        % (r1.returncode, r2.returncode, r3.returncode, r4.returncode))
    log("done: VLMAlignRedo launched + QueueCodeRedo armed; QueueVLMRedo2 disabled (one-shot)")


if __name__ == "__main__":
    main()
