#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_autopilot.py — 魔搭 GPU 实例「一条命令跑到底」自驾脚本（2026-10-18）

覆盖: 体检 -> 组装权重 -> 建 token 缓存 -> 自适应 bench 选 batch -> 按小时预算
      循环 chunk 训练 -> 每 chunk 落盘 NAS(断点续训) -> 每 chunk 回传报告 -> 预算耗尽或收到刹车信号则优雅退出。

Jupyter Python 格（无引号版，复制到 Jupyter 的 cell 里直接跑）:
    import urllib.request as _u; exec(_u.urlopen("http://47.99.131.12:8081/file/gpu_autopilot.py?key=k0u4mail").read().decode())

刹车: 在 node1 或任意挂载 NAS 的实例上 touch /mnt/workspace/node1/chain/STOP_GPU
      -> 当前 chunk 跑完即停（不会半途丢权重）。
调预算: 改 HOURS 变量或 touch /mnt/workspace/node1/chain/gpu_hours_XX (如 gpu_hours_8)
"""
import os, subprocess, time, glob, json, shutil, sys

W = "/mnt/workspace/node1"
CH = W + "/chain"
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
STOP = CH + "/STOP_GPU"
BASE0 = "/root/r4work/base_next"
RUN = "/root/r4work/run_gpu"
CKPT_OUT = CH + "/ckpt_gpu"
CACHE = "/root/c_cache_R7.npy"
CORPUS = CH + "/corpus_R7"
TRAIN = "/root/train_remote_v5_gpu.py"

HOURS = 30.0          # 默认预算: 35h44m 额度留 5h 余量
CHUNK_MIN = 400       # chunk 目标步数
SAVE_EVERY = 200
REP = []

T0 = time.time()


def sh(c, timeout=1800):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    REP.append(line)


def upload(name):
    try:
        with open("/tmp/" + name, "rb") as f:
            data = f.read()
        import urllib.request
        req = urllib.request.Request(f"{ECS}/upload?name={name}&key={KEY}", data=data, method="POST",
                                     headers={"Content-Type": "application/octet-stream"})
        urllib.request.urlopen(req, timeout=300).read()
    except Exception as e:
        say(f"[upload] FAIL {name}: {repr(e)[:120]}")


def report(tag=""):
    txt = "\n".join(REP)
    open("/tmp/gpu_autopilot_report.txt", "w", encoding="utf-8").write(txt)
    upload("gpu_autopilot_report.txt")
    say(f"-- 报告已回传 {tag}")


# ---------- 0. 预算 ----------
for f in glob.glob(CH + "/gpu_hours_*"):
    try:
        HOURS = float(os.path.basename(f).split("_")[-1]); break
    except Exception:
        pass
say(f"=== GPU AUTOPILOT 启动 | 预算 {HOURS}h | stop={STOP} ===")

# ---------- 1. 体检 ----------
say("--- 1. 环境体检")
r = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
say("GPU: " + (r.stdout.strip() or r.stderr.strip()[:150] or "无"))
r = sh("python3 -c 'import torch;print(torch.__version__, torch.cuda.is_available())'")
say("torch: " + (r.stdout.strip() or r.stderr.strip()[:150]))
r = sh(f"ls -d {W} 2>/dev/null && ls {CH} | head -8")
say("NAS: " + (r.stdout.strip()[:400] or "!! NAS 不可见"))

# ---------- 2. 拉训练脚本 + 组装权重 ----------
say("--- 2. 拉脚本 / 组装 base_next")
sh(f"curl -s -o {TRAIN} '{ECS}/file/train_remote_v5_gpu.py?key={KEY}'")
say("train script: " + (str(os.path.getsize(TRAIN)) if os.path.exists(TRAIN) else "!! 拉取失败") + " bytes")

if not (os.path.exists(BASE0 + "/config.json") and
        (os.path.exists(BASE0 + "/model.safetensors") or os.path.exists(BASE0 + "/pytorch_model.bin"))):
    sh(f"mkdir -p {BASE0}")
    sh(f"cp -v {CH}/ckpt/*.json {BASE0}/ 2>/dev/null; "
       f"cp -v {CH}/ckpt/model.safetensors {BASE0}/ 2>/dev/null || cp -v {CH}/ckpt/pytorch_model.bin {BASE0}/ 2>/dev/null; "
       f"cp -v {CH}/base_tok/tokenizer.json {CH}/base_tok/tokenizer_config.json {BASE0}/ 2>/dev/null")
    say("base_next 组装完成: " + str(sorted(os.listdir(BASE0))[:8]))
else:
    say("base_next 已存在, 跳过组装")

# ---------- 3. 自适应 bench: 选最大不 OOM 的 batch ----------
say("--- 3. 自适应 bench (batch 8 -> 4 -> 2)")
best = None
for b in (8, 4, 2):
    cmd = (f"cd /root && timeout 900 python3 -u {TRAIN} --base_dir {BASE0} --tok_dir {BASE0} "
           f"--run_dir {RUN} --corpus_dir {CORPUS} --cache {CACHE} --steps 10 --bench 10 "
           f"--batch {b} --accum 4 --lr 3e-4 --warmup 5 2>&1 | tail -12")
    try:
        r = sh(cmd, timeout=950)
        out = r.stdout or ""
        say(f"batch={b}: " + (out.strip()[-400:] or r.stderr.strip()[-200:]))
        if "tok/s=" in out:
            line = [l for l in out.strip().split("\n") if "tok/s=" in l][-1]
            tps = float(line.split("tok/s=")[-1].split()[0])
            best = (b, tps)
            say(f"==> batch={b} 可用, tok/s={tps:.0f}")
            break
        if "CUDA out of memory" in out or "out of memory" in out.lower():
            say(f"batch={b} OOM, 降档")
    except Exception as e:
        say(f"batch={b} 异常: {repr(e)[:150]}")
if not best:
    say("!! bench 全失败, 退出"); report("bench失败"); sys.exit(1)
BATCH, TPS = best
TOK_PER_STEP = BATCH * 4 * 512
sec_per_chunk = CHUNK_MIN * TOK_PER_STEP / max(TPS, 1e-9)
say(f"==> 选定 batch={BATCH} tok/s={TPS:.0f} => 每 chunk({CHUNK_MIN}步) 约 {sec_per_chunk/60:.1f} 分钟")

report("bench完成")

# ---------- 4. 循环 chunk 训练 ----------
say("--- 4. 开始 chunk 循环 (ckpt -> NAS ckpt_gpu, 断点续训)")
budget = HOURS * 3600
chunk_i = 0
total_tok = 0
while True:
    if os.path.exists(STOP):
        say("!! 收到刹车信号 STOP_GPU, 退出"); break
    el = time.time() - T0
    if el + sec_per_chunk > budget:
        say(f"预算不足下一个 chunk (已用 {el/3600:.2f}h / {HOURS}h), 退出")
        break
    chunk_i += 1
    cur = CKPT_OUT + "/checkpoint-latest"
    base = cur if os.path.exists(cur + "/config.json") else BASE0
    say(f"[chunk {chunk_i}] base={base}")
    cmd = (f"cd /root && timeout {int(sec_per_chunk*2.2)+600} python3 -u {TRAIN} "
           f"--base_dir {base} --tok_dir {BASE0} --run_dir {RUN} --corpus_dir {CORPUS} "
           f"--cache {CACHE} --steps {CHUNK_MIN} --batch {BATCH} --accum 4 --lr 3e-4 "
           f"--warmup 50 --save_every {SAVE_EVERY} 2>&1 | tail -40")
    try:
        r = sh(cmd, timeout=int(sec_per_chunk * 2.2) + 900)
        out = (r.stdout or "") + (r.stderr or "")
        tail = [l for l in out.strip().split("\n") if "step" in l or "DONE" in l or "ckpt" in l][-6:]
        say(f"[chunk {chunk_i}] 完成: " + " | ".join(tail)[:600])
        if "SUMMARY" in out:
            for l in out.split("\n"):
                if "SUMMARY" in l:
                    say("[chunk %d] %s" % (chunk_i, l.strip()[:200]))
    except Exception as e:
        say(f"[chunk {chunk_i}] 异常: {repr(e)[:200]}")

    src = RUN + "/checkpoint-latest"
    if os.path.exists(src + "/config.json"):
        os.makedirs(CKPT_OUT, exist_ok=True)
        tmp = CKPT_OUT + "/checkpoint-latest.tmp"
        sh(f"rm -rf {tmp} && cp -r {src} {tmp} && rm -rf {cur} && mv {tmp} {cur}")
        st = ""
        p = cur + "/step.txt"
        if os.path.exists(p):
            st = open(p).read().strip()
        say(f"[chunk {chunk_i}] ckpt -> NAS {cur} (step={st})")
        total_tok += CHUNK_MIN * TOK_PER_STEP
    else:
        say(f"[chunk {chunk_i}] !! 未产出 ckpt, 停止循环防空烧")
        break

    el = time.time() - T0
    say(f"[进度] chunk={chunk_i} 累计token={total_tok/1e6:.1f}M 已用={el/3600:.2f}h / {HOURS}h")
    report(f"chunk{chunk_i}")

say(f"=== AUTOPILOT 结束 | chunk={chunk_i} 累计token={total_tok/1e6:.1f}M 用时={(time.time()-T0)/3600:.2f}h ===")
report("结束")
