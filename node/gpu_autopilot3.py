#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_autopilot3.py — 最简阶梯: 0.4B -> 0.5B -> 0.6B, 吃满就自动扩

设计原则(用户定调 2026-10-18):
  - 不设时间预算、不设刹车。一档语料吃满(一遍) -> 自动扩容进下一档。
  - 依赖训练器自带的自动保存(save_every) + NAS 上的 run_live, 实例被回收也不丢进度:
    run_dir 直接放在 NAS, 下次启动从 run_live/checkpoint-latest 原地续训。

用法(远程已就绪, 本地重启):
  setsid nohup python3 -u /root/gpu_autopilot3.py > /root/autopilot3.log 2>&1 < /dev/null &
"""
import os, subprocess, time, glob, re, sys

W = "/mnt/workspace/node1"
CH = W + "/chain"
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
BASE0 = "/root/r4work/base_next"
RUN = CH + "/ckpt_gpu/run_live"          # 🔴 放 NAS: 训练器每 save_every 步自动落盘, 实例回收可续
CKPT = CH + "/ckpt_gpu/checkpoint-latest"
STAGE_TXT = CH + "/ckpt_gpu/stage.txt"
TRAIN = "/root/train_remote_v5_gpu.py"
OW = "/root/owner_corpus.jsonl"

BATCH = 8
ACCUM = 4
TOK_STEP = BATCH * ACCUM * 512           # 16384 token/步
REP = []
T0 = time.time()


def sh(c, timeout=86400):
    return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=timeout)


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    REP.append(line)


def report():
    open("/tmp/gpu_autopilot3_report.txt", "w", encoding="utf-8").write("\n".join(REP))
    try:
        import urllib.request
        urllib.request.urlopen(urllib.request.Request(
            f"{ECS}/upload?name=gpu_autopilot3_report.txt&key={KEY}",
            data=open("/tmp/gpu_autopilot3_report.txt", "rb").read(), method="POST",
            headers={"Content-Type": "application/octet-stream"}), timeout=300).read()
    except Exception as e:
        say(f"[upload] FAIL {repr(e)[:80]}")


_LOCK = "/tmp/autopilot3.lock"
if os.path.exists(_LOCK):
    try:
        _pid = int(open(_LOCK).read().strip())
        os.kill(_pid, 0)
        say(f"已有实例 pid={_pid} 在跑, 自己退出")
        sys.exit(0)
    except Exception:
        pass
open(_LOCK, "w").write(str(os.getpid()))

sh(f"curl -s -o {TRAIN} '{ECS}/file/train_remote_v5_gpu.py?key={KEY}'")
sh(f"curl -s -o /root/expand_to_target.py '{ECS}/file/expand_to_target.py?key={KEY}'")
sh(f"curl -s -o {OW} '{ECS}/file/owner_corpus.jsonl?key={KEY}'")
os.makedirs(RUN, exist_ok=True)

# owner_corpus 占 2%, upsample 4x(docs/data-plan-0.4B: 原 50x 已废) -> 链接 4 份 = 4 倍采样
LADDER = [
    ("04B", 400, ["/mnt/workspace/node1/corpus_04b/ufw_zh.jsonl"] + [OW] * 4),
    ("05B", 500, ["/mnt/workspace/node1/corpus_04b/ufw"] + [OW] * 4),
    ("06B", 600, ["/mnt/workspace/node1/chain/corpus_reserve/zh_r_knowledge.jsonl",
                  "/mnt/workspace/node1/chain/corpus_reserve/zh_r_ancient.jsonl",
                  "/mnt/workspace/node1/chain/corpus_reserve/zh_r_novels.jsonl",
                  "/mnt/workspace/node1/chain/corpus_reserve/zh_r_weixin.jsonl"] + [OW] * 4),
]
NAMES = [x[0] for x in LADDER]


def build_stage(name, paths):
    d = f"/root/stage_{name}"
    sh(f"mkdir -p {d} && rm -f {d}/* 2>/dev/null")
    i = 0
    for p in paths:
        if os.path.isdir(p):
            try:
                for f in sorted(os.listdir(p)):
                    if f.endswith(".jsonl") or f.endswith(".txt"):
                        sh(f"ln -sf {p}/{f} {d}/m{i}.jsonl"); i += 1
            except Exception:
                pass
        else:
            sh(f"ln -sf {p} {d}/m{i}.jsonl"); i += 1
    say(f"[stage {name}] 链接 {i} 个语料文件")
    return d


def cache_and_tps(name, stage_dir):
    """跑 bench: 建 token cache(若不存在) 并测吞吐。"""
    cache = f"/root/c_cache_{name}.npy"
    cmd = (f"cd /root && timeout 3600 python3 -u {TRAIN} --base_dir {BASE0} --tok_dir {BASE0} "
           f"--run_dir /root/r4work/bench --corpus_dir {stage_dir} --cache {cache} "
           f"--steps 10 --bench 10 --batch {BATCH} --accum {ACCUM} --lr 3e-4 --warmup 5")
    r = sh(cmd, timeout=3700)
    out = (r.stdout or "") + (r.stderr or "")
    open(f"/root/bench_{name}.log", "w", encoding="utf-8").write(out)
    tps = 0.0
    for l in out.split("\n"):
        if "tok/s=" in l:
            try:
                tps = float(l.split("tok/s=")[-1].split()[0])
            except Exception:
                pass
    m = re.search(r"\[cache\].*?(\d+)\s*tokens", out)
    ntok = int(m.group(1)) if m else 0
    say(f"[bench {name}] tok/s={tps:.0f} cache={ntok/1e6:.1f}M token")
    return cache, tps, ntok


def train_stage(name, stage_dir, cache, tps, ntok):
    steps = max(1, ntok // TOK_STEP)          # 一遍, 不截断
    sec_est = steps * TOK_STEP / max(tps, 1)
    live = RUN + "/checkpoint-latest"
    base = live if os.path.exists(live + "/config.json") else (
        CKPT if os.path.exists(CKPT + "/config.json") else BASE0)
    cmd = (f"cd /root && timeout {int(sec_est * 1.8) + 3600} python3 -u {TRAIN} "
           f"--base_dir {base} --tok_dir {BASE0} --run_dir {RUN} --corpus_dir {stage_dir} "
           f"--cache {cache} --steps {steps} --batch {BATCH} --accum {ACCUM} --lr 3e-4 "
           f"--warmup 50 --save_every 300")
    say(f"[{name}] start steps={steps} base={base} 预计 {sec_est/3600:.2f}h")
    r = sh(f"{cmd} > /root/train_{name}.log 2>&1; tail -6 /root/train_{name}.log", timeout=int(sec_est * 1.8) + 4000)
    say(f"[{name}] 结果: " + (r.stdout or "")[:400].replace("\n", " | "))
    if os.path.exists(live + "/config.json"):
        sh(f"rm -rf {CKPT}.tmp && cp -r {live} {CKPT}.tmp && rm -rf {CKPT} && mv {CKPT}.tmp {CKPT}")
        say(f"[{name}] ckpt -> {CKPT}")


_done = ""
if os.path.exists(STAGE_TXT):
    _done = open(STAGE_TXT).read().strip()
_start = NAMES.index(_done) if _done in NAMES else 0
say(f"=== AUTOPILOT3 启动 | 从 {NAMES[_start]} 开始 | 吃满即扩 ===")
report()

for name, tgt, paths in LADDER[_start:]:
    src_exp = RUN + "/checkpoint-latest"
    if not os.path.exists(src_exp + "/config.json"):
        src_exp = CKPT
    open(STAGE_TXT, "w").write(name)          # 先记在训哪档, 断了下次原地续
    say(f"=== 阶梯 {name}: 目标 {tgt}M ===")
    r = sh(f"cd /root && timeout 3600 python3 /root/expand_to_target.py {src_exp} {tgt} 2>&1 | tail -8")
    say(f"[expand {name}] " + (r.stdout or "").strip()[-300:].replace("\n", " | "))
    report()
    if "OK" not in (r.stdout or "") and "跳过" not in (r.stdout or "") and "已达目标" not in (r.stdout or ""):
        say(f"!! {name} 扩容失败, 停止"); break
    d = build_stage(name, paths)
    cache, tps, ntok = cache_and_tps(name, d)
    if tps <= 0 or ntok < 1000:
        say(f"!! {name} bench 失败(tps={tps} ntok={ntok}), 停止"); break
    train_stage(name, d, cache, tps, ntok)
    report()

say(f"=== AUTOPILOT3 结束 | 用时 {(time.time()-T0)/3600:.2f}h ===")
report()
