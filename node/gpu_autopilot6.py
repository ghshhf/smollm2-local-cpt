#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_autopilot6.py — v5 的续档修正版

相对 v5 的三处改动（其余一字不动）:
  1. 阶段 A (R8/R9/R10) 支持「已归档则跳过」——避免切脚本时重跑 0.3B 三档。
     判据: CK03/config.json 存在 且 autopilot5.log 里出现 "[R10] 0.3B 归档"。
  2. 04B 语料改成「准备好的全部资料」合并文件:
       ufw_zh.jsonl (通用主体) + creative_articles_novel.jsonl (创意文章 2.4亿token)
       + poetry_zh.jsonl (古诗词)
     合并到 /root/mix_04b.jsonl 再喂 —— 建库器只 tokenize 最大单文件, 分开链会漏掉后两个。
  3. cache 上限 1e9 -> 2e9, steps 取 min(预算, 实际cache) 不再超出真实 token 数。

预算来源不变: /mnt/workspace/node1/chain/ladder_budget.txt  (04B=1.5B)

用法:
  setsid nohup python3 -u /root/gpu_autopilot6.py > /root/autopilot6.log 2>&1 < /dev/null &
"""
import os, subprocess, time, sys
import numpy as np

W = "/mnt/workspace/node1"
CH = W + "/chain"
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
BASE0 = "/root/r4work/base_next"        # 0.3B 346.9M/36L
RES = CH + "/corpus_reserve"
CK03 = CH + "/ckpt_gpu/checkpoint-03b"
RUN03 = "/root/r4work/run_03b"
RUNL = CH + "/ckpt_gpu/run_live"
CKPT = CH + "/ckpt_gpu/checkpoint-latest"
STAGE_TXT = CH + "/ckpt_gpu/stage.txt"
BUDFILE = CH + "/ladder_budget.txt"
TRAIN = "/root/train_remote_v5_gpu.py"
OWNER = "/root/owner_corpus.jsonl"
TDIR = "/root/t_stage"
MIX = "/root/mix_04b.jsonl"

REP, T0 = [], time.time()


def sh(c, timeout=86400):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    REP.append(line)


def report():
    open("/tmp/autopilot6_report.txt", "w", encoding="utf-8").write("\n".join(REP))
    try:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(
            f"{ECS}/upload?name=autopilot6_report.txt&key={KEY}",
            data=open("/tmp/autopilot6_report.txt", "rb").read(), method="POST",
            headers={"Content-Type": "application/octet-stream"}), timeout=300).read()
    except Exception as e:
        say(f"[upload] FAIL {repr(e)[:80]}")


def load_budget():
    b = {}
    if not os.path.exists(BUDFILE):
        return b
    for ln in open(BUDFILE, encoding="utf-8"):
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        v = v.strip().upper().replace("_", "").replace(",", "")
        mul = 1
        if v.endswith("B"):
            mul, v = 1_000_000_000, v[:-1]
        elif v.endswith("M"):
            mul, v = 1_000_000, v[:-1]
        elif v.endswith("K"):
            mul, v = 1_000, v[:-1]
        try:
            b[k.strip()] = int(float(v) * mul)
        except Exception:
            pass
    return b


_LOCK = "/tmp/autopilot6.lock"
if os.path.exists(_LOCK):
    try:
        _pid = int(open(_LOCK).read().strip())
        os.kill(_pid, 0)
        say(f"已有 pid={_pid} 在跑, 退出"); sys.exit(0)
    except Exception:
        pass
open(_LOCK, "w").write(str(os.getpid()))

for f in ("train_remote_v5_gpu.py", "expand_to_target.py", "build_cache_par.py"):
    sh(f"curl -s -o /root/{f} '{ECS}/file/{f}?key={KEY}'")
sh(f"curl -s -o {OWNER} '{ECS}/file/owner_corpus.jsonl?key={KEY}'")
os.makedirs(RUNL, exist_ok=True)
os.makedirs(RUN03, exist_ok=True)

SMALL = []                                   # 2026-10-19: 本轮直接 04B, 跳过 0.3B 课程补完
BIG = [                                      # 阶段 B: 04B 用准备好的全部资料(合并单文件)
    ("04B", 400, [MIX] + [OWNER] * 4, 8, 4),
]


def build_stage(name, paths):
    d = f"{TDIR}/{name}"
    sh(f"mkdir -p {d} && rm -f {d}/*")
    i = 0
    for p in paths:
        if os.path.isdir(p):
            for f in sorted(os.listdir(p)):
                if f.endswith(".jsonl") or f.endswith(".txt"):
                    sh(f"ln -sf {p}/{f} {d}/m{i}.jsonl"); i += 1
        else:
            if not os.path.exists(p):
                say(f"[{name}] 跳过不存在: {p}")
                continue
            sh(f"ln -sf {p} {d}/m{i}.jsonl"); i += 1
    say(f"[{name}] 链接 {i} 个语料文件")
    return d


def ensure_cache(name, stage_dir, cap=300_000_000):
    """返回 (cache路径, token数)。检测中毒缓存(<1e6)并强制重建。"""
    cache = f"/root/c_cache_{name}.npy"
    n = 0
    if os.path.exists(cache):
        try:
            n = int(np.load(cache, mmap_mode="r").shape[0])
        except Exception:
            n = 0
        if n < 1_000_000:
            say(f"[{name}] 中毒缓存 {n} token -> 删除重建")
            os.remove(cache); n = 0
        elif n < cap * 0.50:                     # 旧的上限太小 -> 重建(阈值 0.5: 语料总量可能小于预算, 不能因此反复重建)
            say(f"[{name}] 缓存 {n/1e6:.0f}M 小于新目标 {cap/1e6:.0f}M -> 重建")
            os.remove(cache); n = 0
    if n == 0:
        r = sh(f"cd /root && timeout 5400 python3 -u /root/build_cache_par.py "
               f"{stage_dir} {cache} {cap} 8 0.02")
        out = (r.stdout or "") + (r.stderr or "")
        say(f"[{name}] 建缓存: " + out.strip()[-220:].replace("\n", " | "))
        try:
            n = int(np.load(cache, mmap_mode="r").shape[0])
        except Exception:
            n = 0
    say(f"[{name}] cache={n/1e6:.1f}M token")
    return cache, n


def train(name, stage_dir, cache, base, run_dir, batch, accum, steps):
    tok_step = batch * accum * 512
    say(f"[{name}] start base={base} steps={steps} "
        f"({steps*tok_step/1e9:.2f}B token, {batch}x{accum}={tok_step}/步)")
    cmd = (f"cd /root && timeout 172800 python3 -u {TRAIN} --base_dir {base} --tok_dir {BASE0} "
           f"--run_dir {run_dir} --corpus_dir {stage_dir} --cache {cache} --steps {steps} "
           f"--batch {batch} --accum {accum} --lr 3e-4 --warmup 50 --save_every 300 "
           f"> /root/train_{name}.log 2>&1; tail -3 /root/train_{name}.log")
    r = sh(cmd, timeout=173000)
    say(f"[{name}] 结果: " + (r.stdout or "").strip()[-300:].replace("\n", " | "))
    return os.path.exists(run_dir + "/checkpoint-latest/config.json")


BUD = load_budget()
say("=== AUTOPILOT6 启动 | 一步一个脚印, 预算驱动 ===")
say("[预算] " + (", ".join(f"{k}={v/1e9:.2f}B" for k, v in BUD.items()) or "(ladder_budget.txt 为空)"))
report()

# ---- 阶段 A: 0.3B 补完 R8/R9/R10(各吃该档语料一遍); 已归档则跳过 ----
DONE_A = False
if os.path.exists(CK03 + "/config.json"):
    try:
        DONE_A = "[R10] 0.3B 归档" in open("/root/autopilot5.log", encoding="utf-8",
                                          errors="ignore").read()
    except Exception:
        DONE_A = False

base = CK03 if DONE_A else BASE0
if DONE_A:
    say("[阶段A] R8/R9/R10 已归档, 跳过三档, 直接进 0.4B")
    report()
else:
    for name, paths, batch, accum in SMALL:
        d = build_stage(name, paths)
        cache, n = ensure_cache(name, d)
        if n < 1_000_000:
            say(f"!! {name} 语料无效, 停止"); report(); sys.exit(1)
        ok = train(name, d, cache, base, RUN03, batch, accum, max(1, n // (batch * accum * 512)))
        if not ok:
            say(f"!! {name} 未产出 ckpt, 停止"); report(); sys.exit(1)
        sh(f"rm -rf {CK03}.tmp && cp -r {RUN03}/checkpoint-latest {CK03}.tmp "
           f"&& rm -rf {CK03} && mv {CK03}.tmp {CK03}")
        base = CK03
        say(f"[{name}] 0.3B 归档 -> {CK03}")
        report()

# ---- 阶段 B: 扩容阶梯, 每档必须吃满自己的 token 预算 ----
for name, tgt, paths, batch, accum in BIG:
    bud = BUD.get(name)
    open(STAGE_TXT, "w").write(name)
    if name == "04B":
        # 准备好的资料合成单文件: 建库器只 tokenize 最大单文件, 分开会漏
        srcs = [f"{W}/corpus_04b/ufw_zh.jsonl",
                f"{W}/corpus_04b/creative_articles_novel.jsonl",
                f"{W}/corpus_04b/poetry_zh.jsonl",
                f"{W}/corpus_04b/code_pretrain.jsonl",
                f"{W}/corpus_04b/code_instruct.jsonl"]
        srcs = [s for s in srcs if os.path.exists(s) and os.path.getsize(s) > 0]
        miss = [s for s in srcs if not os.path.exists(s)]
        if miss:
            say(f"!! 04B 语料缺失: {miss}"); report(); sys.exit(1)
        need_sz = sum(os.path.getsize(s) for s in srcs)
        if not (os.path.exists(MIX) and os.path.getsize(MIX) == need_sz):
            say(f"[04B] 合并语料 -> {MIX} ({need_sz/1e9:.2f}GB)")
            sh(f"cat {' '.join(srcs)} > {MIX}.tmp && mv {MIX}.tmp {MIX}")
        say(f"[04B] 合并文件 {os.path.getsize(MIX)/1e9:.2f}GB 就绪")
        report()
    src = f"/root/r4work/base_{name}"
    # 04B 底模: CK03 不在 NAS 时, 用 NAS 上的 0.3B 本体 (models/polaris-03b) 组装
    sh(f"rm -rf {src} && cp -r {CK03 if (name == '04B' and os.path.isdir(CK03)) else BASE0} {src}")
    r = sh(f"cd /root && timeout 3600 python3 /root/expand_to_target.py {src} {tgt} 2>&1 | tail -6")
    say(f"[expand {name}] " + (r.stdout or "").strip()[-260:].replace("\n", " | "))
    report()
    o = r.stdout or ""
    if not any(k in o for k in ("OK", "跳过", "已达目标")):
        say(f"!! {name} 扩容失败, 停止"); break
    d = build_stage(name, paths)
    cap = max(300_000_000, min(int(bud or 0), 4_000_000_000))
    cache, n = ensure_cache(name, d, cap)
    if n < 1_000_000:
        say(f"!! {name} 语料无效, 停止"); break
    tok_step = batch * accum * 512
    if not bud:
        # 没给预算 => 只吃一遍就停, 绝不自动跨档(不自己决定)
        say(f"!! {name} 无 token 预算(ladder_budget.txt 缺 {name}), "
            f"只吃一遍 {n/1e6:.0f}M 后停, 不跨档")
        train(name, d, cache, src, RUNL, batch, accum, max(1, n // tok_step))
        report()
        say(f"=== 停在 {name}, 等预算 ===")
        break
    steps = max(1, min(int(bud), n) // tok_step)
    say(f"[{name}] 预算 {bud/1e9:.2f}B, 实际可吃 {min(int(bud), n)/1e9:.2f}B -> {steps} 步")
    ok = train(name, d, cache, src, RUNL, batch, accum, steps)
    if not ok:
        say(f"!! {name} 未产出 ckpt, 停止"); break
    sh(f"rm -rf {CKPT}.tmp && cp -r {RUNL}/checkpoint-latest {CKPT}.tmp "
       f"&& rm -rf {CKPT} && mv {CKPT}.tmp {CKPT}")
    say(f"[{name}] 吃满 {bud/1e9:.2f}B token, 归档 -> {CKPT}")
    report()

say(f"=== AUTOPILOT6 结束 | 用时 {(time.time()-T0)/3600:.2f}h ===")
report()
