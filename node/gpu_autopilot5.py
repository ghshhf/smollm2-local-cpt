#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_autopilot5.py — 一步一个脚印: 每档吃够 token 预算才允许扩容下一档

用户定调(2026-10-18 02:13):
  「宁可真的花钱, 也要干好一件事再干一件事。整个项目逻辑就是一步一个脚印。
    跨步的话, 我直接拿 1B 的模型训练不就行了?」
  => 取消 v4 的「吃满一遍 300M cache 就自动扩」; 每档必须有**显式 token 预算**,
     吃满预算才扩; 预算没给就不扩, 就停在这一档(不自己决定)。

预算来源: /mnt/workspace/node1/chain/ladder_budget.txt
  格式(每行一条, 支持 3.5B / 800M / 纯数字):
      04B=3.5B
      05B=...
      06B=...
  改这个文件即可, 不需要改代码。

阶段 A(0.3B 课程补完)仍是「该档语料一遍」(课程原始定义), 不需要预算:
  R8 古籍 / R9 小说 / R10 公众号, 在 346.9M/36L 权重上累进训, 归档 checkpoint-03b。

用法:
  setsid nohup python3 -u /root/gpu_autopilot5.py > /root/autopilot5.log 2>&1 < /dev/null &
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

REP, T0 = [], time.time()


def sh(c, timeout=86400):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    REP.append(line)


def report():
    open("/tmp/autopilot5_report.txt", "w", encoding="utf-8").write("\n".join(REP))
    try:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(
            f"{ECS}/upload?name=autopilot5_report.txt&key={KEY}",
            data=open("/tmp/autopilot5_report.txt", "rb").read(), method="POST",
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


_LOCK = "/tmp/autopilot5.lock"
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

SMALL = [                                    # 阶段 A: 0.3B 课程补完, 各吃该档语料一遍
    ("R8",  [f"{RES}/zh_r_ancient.jsonl"], 8, 4),
    ("R9",  [f"{RES}/zh_r_novels.jsonl"],  8, 4),
    ("R10", [f"{RES}/zh_r_weixin.jsonl"],  8, 4),
]
BIG = [                                      # 阶段 B: 扩容阶梯, 必须吃满预算才扩
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
        elif n > cap * 1.5:
            say(f"[{name}] 缓存 {n/1e6:.0f}M 超过新上限, 重建")
            os.remove(cache); n = 0
    if n == 0:
        r = sh(f"cd /root && timeout 1800 python3 -u /root/build_cache_par.py "
               f"{stage_dir} {cache} {cap} 8 0.02")
        out = (r.stdout or "") + (r.stderr or "")
        say(f"[{name}] 建缓存: " + out.strip()[-160:].replace("\n", " | "))
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
say("=== AUTOPILOT5 启动 | 一步一个脚印, 预算驱动 ===")
say("[预算] " + (", ".join(f"{k}={v/1e9:.2f}B" for k, v in BUD.items()) or "(ladder_budget.txt 为空)"))
report()

# ---- 阶段 A: 0.3B 补完 R8/R9/R10(各吃该档语料一遍) ----
base = BASE0
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
    src = f"/root/r4work/base_{name}"
    sh(f"rm -rf {src} && cp -r {CK03 if name == '04B' else RUNL + '/checkpoint-latest'} {src}")
    r = sh(f"cd /root && timeout 3600 python3 /root/expand_to_target.py {src} {tgt} 2>&1 | tail -6")
    say(f"[expand {name}] " + (r.stdout or "").strip()[-260:].replace("\n", " | "))
    report()
    o = r.stdout or ""
    if not any(k in o for k in ("OK", "跳过", "已达目标")):
        say(f"!! {name} 扩容失败, 停止"); break
    d = build_stage(name, paths)
    cap = max(300_000_000, min(int(bud or 0), 1_000_000_000))
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
    steps = max(1, int(bud) // tok_step)
    ok = train(name, d, cache, src, RUNL, batch, accum, steps)
    if not ok:
        say(f"!! {name} 未产出 ckpt, 停止"); break
    sh(f"rm -rf {CKPT}.tmp && cp -r {RUNL}/checkpoint-latest {CKPT}.tmp "
       f"&& rm -rf {CKPT} && mv {CKPT}.tmp {CKPT}")
    say(f"[{name}] 吃满 {bud/1e9:.2f}B token, 归档 -> {CKPT}")
    report()

say(f"=== AUTOPILOT5 结束 | 用时 {(time.time()-T0)/3600:.2f}h ===")
report()
