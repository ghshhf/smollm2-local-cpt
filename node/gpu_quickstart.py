#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_quickstart.py — 魔搭 GPU 实例一键验证（在 Jupyter Python 格里跑）

一行（Jupyter 格粘贴）：
!curl -s -o /tmp/gpu_quickstart.py "http://47.99.131.12:8081/file/gpu_quickstart.py?key=k0u4mail" && exec(open("/tmp/gpu_quickstart.py").read())

做的事：
  1. 体检: NAS 可见性 / cuda / torch / 显存
  2. 组装 /root/r4work/base_next = NAS ckpt 权重 + base_tok tokenizer
  3. 从 ECS 拉 train_remote_v5_gpu.py
  4. 跑 --bench 20 步实测吞吐（不更新权重、不存盘）
  5. 把报告回传 ECS (gpu_quickstart_report.txt)

不自动开满负荷训练：拿到真实 tok/s 后再决定怎么烧额度。
"""
import os, subprocess, sys, time

W = "/mnt/workspace/node1"
CH = W + "/chain"
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
REP = []


def sh(c, timeout=1800):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)


def say(m):
    print(m, flush=True)
    REP.append(str(m))


def upload(name):
    try:
        with open("/tmp/" + name, "rb") as f:
            data = f.read()
        import urllib.request
        req = urllib.request.Request(f"{ECS}/upload?name={name}&key={KEY}", data=data, method="POST",
                                     headers={"Content-Type": "application/octet-stream"})
        urllib.request.urlopen(req, timeout=300).read()
        say(f"[upload] {name} ok")
    except Exception as e:
        say(f"[upload] FAIL {name}: {e}")


say("=== 1. 环境体检 ===")
say("time: " + time.strftime("%Y-%m-%d %H:%M:%S"))
r = sh("nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader")
say("nvidia-smi: " + (r.stdout.strip() or r.stderr.strip()[:200]))
r = sh("python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available(), torch.version.cuda)'")
say("torch: " + (r.stdout.strip() or r.stderr.strip()[:200]))
r = sh(f"ls -d {W} && ls {CH} | head -20")
say("NAS: " + (r.stdout.strip()[:600] or r.stderr.strip()[:200]))

say("=== 2. 组装 base_next (权重+tokenizer) ===")
r = sh(f"mkdir -p /root/r4work/base_next && cp -v {CH}/ckpt/*.json /root/r4work/base_next/ 2>/dev/null; "
       f"cp -v {CH}/ckpt/model.safetensors /root/r4work/base_next/ 2>/dev/null || "
       f"cp -v {CH}/ckpt/pytorch_model.bin /root/r4work/base_next/ 2>/dev/null; "
       f"cp -v {CH}/base_tok/tokenizer.json {CH}/base_tok/tokenizer_config.json /root/r4work/base_next/ 2>/dev/null; "
       f"ls -la /root/r4work/base_next/")
say(r.stdout.strip()[:1200] or r.stderr.strip()[:300])

say("=== 3. 拉 GPU 训练脚本 ===")
r = sh(f"curl -s -o /root/train_remote_v5_gpu.py '{ECS}/file/train_remote_v5_gpu.py?key={KEY}' && "
       f"wc -l /root/train_remote_v5_gpu.py && head -3 /root/train_remote_v5_gpu.py")
say(r.stdout.strip()[:400] or r.stderr.strip()[:200])

say("=== 4. bench 20 步实测吞吐 ===")
cmd = ("cd /root && timeout 1500 python3 -u /root/train_remote_v5_gpu.py "
       "--base_dir /root/r4work/base_next --tok_dir /root/r4work/base_next "
       "--run_dir /root/r4work/run_gpu --corpus_dir %s/corpus_R7 --cache /root/c_cache_R7.npy "
       "--steps 20 --bench 20 --batch 8 --accum 4 --lr 3e-4 --warmup 5 2>&1 | tail -30") % CH
say("CMD: " + cmd)
try:
    r = sh(cmd, timeout=1700)
    say(r.stdout.strip()[-3000:] or r.stderr.strip()[-1000:])
except Exception as e:
    say("bench FAIL: " + repr(e)[:300])

say("=== 5. 回传报告 ===")
open("/tmp/gpu_quickstart_report.txt", "w", encoding="utf-8").write("\n".join(REP))
upload("gpu_quickstart_report.txt")
say("DONE")
