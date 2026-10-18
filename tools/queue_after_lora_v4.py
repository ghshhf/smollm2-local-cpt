"""一次性排队器 queue_after_lora_v4：等另一会话的 LoRA v4 让出机器后，自动接棒 VLM align v6 续训。

背景（2026-10-17 23:57 实况）：
- 另一会话在跑 chat_lora_v4（57%，ETA 约 00:50），日志 logs/chat_lora_v4.log。
- v5 align 盲测失败（视觉条件化弱），v6 续训已就绪（run_vlm_v6.bat，3000 步）。
- 本排队器守位：train_lora 进程全消失 -> 启动 VLMAlignV6（ONCE +2min）-> 自我禁用。

纪律（用户定调「本地长进程同时只跑一条」）：
- chat_lora_v4.log 不存在 => 时点不对，安静等
- train_lora.py 活着 => 别人在用机器，安静等；查不动就保守等，绝不误启动
- 一次性，绝不复活
"""
import os
import subprocess
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FLAG = os.path.join(ROOT, "logs", "vlm_v6_queued.flag")
LOG = os.path.join(ROOT, "logs", "queue_after_lora_v4.log")
LORA_LOG = os.path.join(ROOT, "logs", "chat_lora_v4.log")
TASK_SELF = "QueueVLMRedo3"
TASK_ALIGN = "VLMAlignV6"
ALIGN_BAT = os.path.join(ROOT, "run_vlm_v6.bat")


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
        return
    if lora_running():
        return
    with open(FLAG, "w", encoding="utf-8") as f:
        f.write(time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    log("lora v4 gone -> launching VLMAlignV6 (align v6 continuation, 3000 steps)")
    st = time.strftime("%H:%M", time.localtime(time.time() + 120))
    r1 = subprocess.run(["schtasks", "/Create", "/TN", TASK_ALIGN, "/TR", ALIGN_BAT,
                         "/SC", "ONCE", "/ST", st, "/RU", "SYSTEM", "/F"],
                        capture_output=True, text=True)
    r2 = subprocess.run(["schtasks", "/Run", "/TN", TASK_ALIGN], capture_output=True, text=True)
    r3 = subprocess.run(["schtasks", "/Change", "/TN", TASK_SELF, "/DISABLE"],
                        capture_output=True, text=True)
    log("align_create=%s align_run=%s self_disable=%s"
        % (r1.returncode, r2.returncode, r3.returncode))
    log("done: VLMAlignV6 launched; QueueVLMRedo3 disabled (one-shot)")


if __name__ == "__main__":
    main()
