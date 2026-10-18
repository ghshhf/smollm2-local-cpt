#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""nightly_rsi.py — 本地夜间 RSI 自循环（双机闭环的本地腿）

node1 做梦集群每周期把银行燃料（dream_fuel.jsonl）传 ECS；本脚本（schtasks
SYSTEM 每日 00:30）负责：
  1. 拉取 ECS 燃料；比对本机 .rsi_fuel_state.json 的已消化量
  2. 新增 ≥ MIN_DELTA → corpus_curate 合并提纯 → 排「训练+自动盲测」进队列
  3. 队列里已有代码任务在排队则不重复排；训练完由 job cmd 链上的
     code_eval 自动盲测，产出 logs/code_eval_codeqwen05b_vN.json

用法: python tools/nightly_rsi.py   （无需参数，全部状态本地化）
"""
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
FUEL_URL = ECS + "/file/dream_fuel.jsonl?key=" + KEY
LOCAL_FUEL = os.path.join(ROOT, "data", "dream_fuel.jsonl")
STATE = os.path.join(ROOT, ".rsi_fuel_state.json")
LOG = os.path.join(ROOT, "logs", "nightly_rsi.log")
BASE = os.path.join(ROOT, "models", "models", "Qwen--Qwen2.5-0.5B-Instruct",
                    "snapshots", "master")
PY = os.path.join(ROOT, "venv", "Scripts", "python.exe")
MIN_DELTA = 20


def log(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    line = "[%s] %s" % (__import__("time").strftime("%F %T"), msg)
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch_fuel():
    try:
        r = urllib.request.urlopen(FUEL_URL, timeout=30)
        raw = r.read().decode("utf-8", errors="replace")
    except Exception as e:
        log("[fuel] fetch FAILED: %r" % e)
        return None
    with open(LOCAL_FUEL, "w", encoding="utf-8") as f:
        f.write(raw)
    n = sum(1 for l in raw.splitlines() if l.strip())
    log("[fuel] fetched %d triples" % n)
    return n


def queue_has_code_job():
    try:
        out = subprocess.run([PY, os.path.join(HERE, "train_queue.py"), "status"],
                             capture_output=True, text=True, cwd=ROOT)
        return "out_lora_code_qwen_v" in out.stdout
    except Exception:
        return True  # 拿不到状态就保守不排


def main():
    n = fetch_fuel()
    if n is None:
        return
    state = {"last_fuel": 0, "versions": []}
    if os.path.exists(STATE):
        try:
            state = json.load(open(STATE, encoding="utf-8"))
        except Exception:
            pass
    delta = n - state.get("last_fuel", 0)
    if delta < MIN_DELTA:
        log("[gate] 新增燃料 %d < %d，不排训练" % (delta, MIN_DELTA))
        return
    if queue_has_code_job():
        log("[gate] 队列已有代码任务在排，跳过本轮")
        return

    # 合并提纯
    r = subprocess.run([PY, os.path.join(HERE, "corpus_curate.py")],
                       capture_output=True, text=True, cwd=ROOT)
    log("[curate] " + (r.stdout.strip().replace("\n", " | ") or r.stderr[-200:]))

    # 下一版本号
    versions = state.get("versions", [])
    nxt = (max(versions) + 1) if versions else 2
    out_dir = "out_lora_code_qwen_v%d" % nxt
    train_cmd = (
        '"{py}" "{root}\\train_lora.py" '
        '--model_path "{base}" '
        '--data_path "{root}\\data\\code_sft_v2.jsonl" '
        '--output_dir "{root}\\{out}" '
        '--num_train_epochs 1 --per_device_batch_size 1 '
        '--grad_accumulation 4 --learning_rate 1e-4 '
        '--lora_r 16 --lora_alpha 32 --lora_dropout 0.05 '
        '--targets q_proj,k_proj,v_proj,o_proj --max_seq_length 768 '
        '--mask_prompt --logging_steps 10 --save_steps 500 '
        '&& "{py}" "{root}\\tools\\code_eval.py" --model "{base}" '
        '--adapter "{root}\\{out}" --tag codeqwen05b_v{v} '
        '--out "{root}\\logs\\code_eval_codeqwen05b_v{v}.json"'
    ).format(py=PY, root=ROOT, base=BASE, out=out_dir, v=nxt)

    r = subprocess.run([PY, os.path.join(HERE, "train_queue.py"),
                        "submit", "code_v%d_curated" % nxt, train_cmd],
                       capture_output=True, text=True, cwd=ROOT)
    log("[queue] " + (r.stdout.strip() or r.stderr[-200:]))

    state["last_fuel"] = n
    state["versions"] = (versions + [nxt])[-10:]
    json.dump(state, open(STATE, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    log("[done] 已排 v%d（燃料 %d 条）" % (nxt, n))


if __name__ == "__main__":
    main()
