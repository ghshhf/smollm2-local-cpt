#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_autopilot4.py — 0.3B 补完 R8/R9/R10 -> 0.4B -> 0.5B -> 0.6B

用户定调(2026-10-18):
  - 先把 0.3B 课程吃干净: R8 古籍 / R9 小说 / R10 公众号(各约 1900 万字节),
    在 0.3B(346.9M/36L) 权重上训完, archived 成 checkpoint-03b, 再扩容上 0.4B。
  - 之后吃满即自动扩: 0.4B -> 0.5B -> 0.6B, 不设预算、不设刹车。
  - 靠训练器自带自动保存续训: run_dir 放 NAS, 实例回收后原地续。

关键修复:
  - R8 以前"只读到 1 token"的真因是 /root/c_cache_R8.npy 是个 np.array([0]) 的中毒文件,
    训练器每次都"复用已有缓存"。这里 ensure_cache 会检测 token 数 < 1e6 并强制重建。
  - token cache 用 8 进程并行建(build_cache_par.py), 避免 GPU 空转等单核 tokenize。
  - 显存: 43 层在 batch8 已约 20-21G/22.5G, 53 层与 65 层必须降 micro-batch, 否则必 OOM。

用法:
  setsid nohup python3 -u /root/gpu_autopilot4.py > /root/autopilot4.log 2>&1 < /dev/null &
"""
import os, subprocess, time, re, sys
import numpy as np

W = "/mnt/workspace/node1"
CH = W + "/chain"
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
BASE0 = "/root/r4work/base_next"        # 0.3B 346.9M/36L
RES = CH + "/corpus_reserve"            # 真实文件(ZH_R_*), refined 目录同内容
CK03 = CH + "/ckpt_gpu/checkpoint-03b"  # 0.3B 三档训完的归档
RUN03 = "/root/r4work/run_03b"
RUNL = CH + "/ckpt_gpu/run_live"        # 大模型档的自动保存落点(NAS)
CKPT = CH + "/ckpt_gpu/checkpoint-latest"
STAGE_TXT = CH + "/ckpt_gpu/stage.txt"
TRAIN = "/root/train_remote_v5_gpu.py"
OWNER = "/root/owner_corpus.jsonl"
TDIR = "/root/t_stage"

REP, T0 = [], time.time()


def sh(c, timeout=86400):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    REP.append(line)


def report():
    open("/tmp/autopilot4_report.txt", "w", encoding="utf-8").write("\n".join(REP))
    try:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(
            f"{ECS}/upload?name=autopilot4_report.txt&key={KEY}",
            data=open("/tmp/autopilot4_report.txt", "rb").read(), method="POST",
            headers={"Content-Type": "application/octet-stream"}), timeout=300).read()
    except Exception as e:
        say(f"[upload] FAIL {repr(e)[:80]}")


_LOCK = "/tmp/autopilot4.lock"
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

# ---- 阶梯定义: name / 目标M / 语料 / batch / accum ----
SMALL = [                                    # 0.3B 课程补完(346.9M 上训, 不扩容)
    ("R8",  [f"{RES}/zh_r_ancient.jsonl"],            8, 4),
    ("R9",  [f"{RES}/zh_r_novels.jsonl"],             8, 4),
    ("R10", [f"{RES}/zh_r_weixin.jsonl"],             8, 4),
]
BIG = [                                      # 保函数扩容阶梯
    ("04B", 400, [f"{W}/corpus_04b/ufw_zh.jsonl"] + [OWNER] * 4, 8, 4),
    ("05B", 500, [f"{W}/corpus_04b/ufw"] + [OWNER] * 4,          4, 6),
    ("06B", 600, [f"{RES}/zh_r_knowledge.jsonl", f"{RES}/zh_r_ancient.jsonl",
                  f"{RES}/zh_r_novels.jsonl", f"{RES}/zh_r_weixin.jsonl"] + [OWNER] * 4, 3, 6),
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
            sh(f"ln -sf {p} {d}/m{i}.jsonl"); i += 1
    say(f"[{name}] 链接 {i} 个语料文件")
    return d


def ensure_cache(name, stage_dir):
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
    if n == 0:
        r = sh(f"cd /root && timeout 1800 python3 -u /root/build_cache_par.py "
               f"{stage_dir} {cache} 300000000 8 0.02")
        out = (r.stdout or "") + (r.stderr or "")
        say(f"[{name}] 建缓存: " + out.strip()[-200:].replace("\n", " | "))
        try:
            n = int(np.load(cache, mmap_mode="r").shape[0])
        except Exception:
            n = 0
    say(f"[{name}] cache={n/1e6:.1f}M token")
    return cache, n


def train(name, stage_dir, cache, n, base, run_dir, batch, accum):
    tok_step = batch * accum * 512
    steps = max(1, n // tok_step)
    say(f"[{name}] start base={base} steps={steps} {batch}x{accum}={tok_step} token/步")
    cmd = (f"cd /root && timeout 43200 python3 -u {TRAIN} --base_dir {base} --tok_dir {BASE0} "
           f"--run_dir {run_dir} --corpus_dir {stage_dir} --cache {cache} --steps {steps} "
           f"--batch {batch} --accum {accum} --lr 3e-4 --warmup 50 --save_every 300 "
           f"> /root/train_{name}.log 2>&1; tail -4 /root/train_{name}.log")
    r = sh(cmd, timeout=44000)
    say(f"[{name}] 结果: " + (r.stdout or "").strip()[-350:].replace("\n", " | "))
    return os.path.exists(run_dir + "/checkpoint-latest/config.json")


say("=== AUTOPILOT4 启动 ===")
report()

# ---- 阶段 A: 0.3B 补完 R8/R9/R10 ----
base = BASE0
for name, paths, batch, accum in SMALL:
    d = build_stage(name, paths)
    cache, n = ensure_cache(name, d)
    if n < 1_000_000:
        say(f"!! {name} 语料无效, 停止"); sys.exit(1)
    ok = train(name, d, cache, n, base, RUN03, batch, accum)
    if ok:
        sh(f"rm -rf {CK03}.tmp && cp -r {RUN03}/checkpoint-latest {CK03}.tmp "
           f"&& rm -rf {CK03} && mv {CK03}.tmp {CK03}")
        base = CK03
        say(f"[{name}] 0.3B 归档 -> {CK03}")
    else:
        say(f"!! {name} 未产出 ckpt, 停止"); sys.exit(1)
    report()

# ---- 阶段 B: 扩容阶梯 0.4B -> 0.5B -> 0.6B ----
for name, tgt, paths, batch, accum in BIG:
    open(STAGE_TXT, "w").write(name)
    src = f"/root/r4work/base_{name}"
    sh(f"rm -rf {src} && cp -r {CK03 if name == '04B' else RUNL + '/checkpoint-latest'} {src}")
    r = sh(f"cd /root && timeout 3600 python3 /root/expand_to_target.py {src} {tgt} 2>&1 | tail -6")
    say(f"[expand {name}] " + (r.stdout or "").strip()[-260:].replace("\n", " | "))
    report()
    o = r.stdout or ""
    if not any(k in o for k in ("OK", "跳过", "已达目标")):
        say(f"!! {name} 扩容失败, 停止"); break
    d = build_stage(name, paths)
    cache, n = ensure_cache(name, d)
    if n < 1_000_000:
        say(f"!! {name} 语料无效, 停止"); break
    ok = train(name, d, cache, n, src, RUNL, batch, accum)
    if not ok:
        say(f"!! {name} 未产出 ckpt, 停止"); break
    sh(f"rm -rf {CKPT}.tmp && cp -r {RUNL}/checkpoint-latest {CKPT}.tmp "
       f"&& rm -rf {CKPT} && mv {CKPT}.tmp {CKPT}")
    say(f"[{name}] 归档 -> {CKPT}")
    report()

say(f"=== AUTOPILOT4 结束 | 用时 {(time.time()-T0)/3600:.2f}h ===")
report()
