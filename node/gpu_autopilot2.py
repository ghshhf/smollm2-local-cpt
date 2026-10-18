#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""gpu_autopilot2.py — 阶段化自驾: R7 -> R8 -> R9 -> R10 -> 扩容0.4B -> corpus_04b

与 gpu_autopilot.py 的区别: autopilot 只吃 corpus_R7 一个目录(会重复 41 遍);
本脚本按 0.3B 课程四档依次推进, 每档吃满指定遍数就换下一档, 四档封顶后
自动 Net2Net 扩到 0.4B 并切到 14GB 的 corpus_04b 继续吃。

课程(chain_v2.sh 原定义):
  R7 知识  -> corpus_reserve/zh_r_knowledge.jsonl (corpus_R7 已是它的符号链接)
  R8 古籍  -> corpus_reserve_refined/zh_r_ancient.jsonl
  R9 小说  -> corpus_reserve_refined/zh_r_novels.jsonl
  R10 公众号 -> corpus_reserve_refined/zh_r_weixin.jsonl
  (reserve_refined 是精炼版, 四档共约 98MB ≈ 3100万 token)

刹车: touch /mnt/workspace/node1/chain/STOP_GPU  (档间检查, 不丢权重)
预算: touch /mnt/workspace/node1/chain/gpu_hours_8   (默认 30h)

Jupyter(无引号):
  !curl -s -o /tmp/a2.py http://47.99.131.12:8081/file/gpu_autopilot2.py\?key=k0u4mail && python3 /tmp/a2.py
"""
import os, subprocess, time, glob, re, sys

W = "/mnt/workspace/node1"
CH = W + "/chain"
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
STOP = CH + "/STOP_GPU"
BASE0 = "/root/r4work/base_next"
RUN = "/root/r4work/run_gpu2"
CKPT = CH + "/ckpt_gpu/checkpoint-latest"
TRAIN = "/root/train_remote_v5_gpu.py"
RES = CH + "/corpus_reserve_refined"

HOURS = 5.3         # 🔴 实例单次上限约 8h(不是总额度), 阶梯必须一次跑完并提前落盘
PASSES = 2          # 每档吃几遍
BATCH = 8
ACCUM = 4
REP = []
T0 = time.time()


def sh(c, timeout=3600):
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
        urllib.request.urlopen(urllib.request.Request(
            f"{ECS}/upload?name={name}&key={KEY}", data=data, method="POST",
            headers={"Content-Type": "application/octet-stream"}), timeout=300).read()
    except Exception as e:
        say(f"[upload] FAIL {repr(e)[:100]}")


def report(tag=""):
    open("/tmp/gpu_autopilot2_report.txt", "w", encoding="utf-8").write("\n".join(REP))
    upload("gpu_autopilot2_report.txt")


def build_stage(name, src_file):
    d = f"/root/stage_{name}"
    sh(f"mkdir -p {d} && rm -f {d}/*.jsonl && ln -sf {src_file} {d}/corpus.jsonl && ls -la {d}")
    return d


def bench(stage_dir, cache):
    cmd = (f"cd /root && timeout 2400 python3 -u {TRAIN} --base_dir {BASE0} --tok_dir {BASE0} "
           f"--run_dir {RUN} --corpus_dir {stage_dir} --cache {cache} --steps 10 --bench 10 "
           f"--batch {BATCH} --accum {ACCUM} --lr 3e-4 --warmup 5 2>&1 | tail -12")
    r = sh(cmd, timeout=2500)
    out = r.stdout or ""
    say(f"[bench {stage_dir}] " + out.strip()[-300:].replace("\n", " | "))
    tps = 0.0
    if "tok/s=" in out:
        try:
            tps = float([l for l in out.strip().split("\n") if "tok/s=" in l][-1].split("tok/s=")[-1].split()[0])
        except Exception:
            pass
    m = re.search(r"(\d+)\s*tokens", out)
    ntok = int(m.group(1)) if m else 0
    return tps, ntok


def run_stage(name, stage_dir, cache, passes, budget_left):
    tps, ntok = bench(stage_dir, cache)
    if tps <= 0:
        say(f"!! {name} bench 失败, 跳过"); return False
    say(f"[{name}] tok/s={tps:.0f} 语料={ntok/1e6:.1f}M token")
    if ntok <= 0:
        ntok = 20_000_000
    tok_per_step = BATCH * ACCUM * 512
    want_steps = int(passes * ntok / tok_per_step)
    sec_est = want_steps * tok_per_step / tps
    if sec_est > budget_left:
        want_steps = int(budget_left * tps / tok_per_step)
        sec_est = budget_left          # 预算钳制后必须同步, 否则 timeout 会虚高几倍
        say(f"[{name}] 预算不足, 步数砍到 {want_steps}")
    if want_steps < 20:
        say(f"[{name}] 预算耗尽, 停止"); return False
    base = CKPT if os.path.exists(CKPT + "/config.json") else BASE0
    cmd = (f"cd /root && timeout {int(sec_est*2.2)+900} python3 -u {TRAIN} "
           f"--base_dir {base} --tok_dir {BASE0} --run_dir {RUN} --corpus_dir {stage_dir} "
           f"--cache {cache} --steps {want_steps} --batch {BATCH} --accum {ACCUM} --lr 3e-4 "
           f"--warmup 50 --save_every 150 2>&1 | tail -30")
    say(f"[{name}] start steps={want_steps} base={base} 预计 {sec_est/60:.1f} 分钟")
    try:
        r = sh(cmd, timeout=int(sec_est * 2.2) + 1200)
        out = (r.stdout or "") + (r.stderr or "")
        keep = [l for l in out.strip().split("\n") if "SUMMARY" in l or "DONE" in l or "loss=" in l][-5:]
        say(f"[{name}] 结果: " + " | ".join(keep)[:500])
    except Exception as e:
        say(f"[{name}] 异常 {repr(e)[:150]}")
    src = RUN + "/checkpoint-latest"
    if os.path.exists(src + "/config.json"):
        sh(f"rm -rf {CKPT}.tmp && cp -r {src} {CKPT}.tmp && rm -rf {CKPT} && mv {CKPT}.tmp {CKPT}")
        open(CH + "/ckpt_gpu/stage.txt", "w").write(name)
        say(f"[{name}] ckpt -> {CKPT}")
        return True
    say(f"[{name}] !! 无 ckpt 产出"); return False


for f in glob.glob(CH + "/gpu_hours_*"):
    try:
        HOURS = float(os.path.basename(f).split("_")[-1]); break
    except Exception:
        pass

say(f"=== AUTOPILOT2 启动 | 预算 {HOURS}h | 每档 {PASSES} 遍 ===")
# 自锁: 时间触发与 token 触发可能各启一次, 两个 v2 各占 15G 显存会 OOM
_LOCK = "/tmp/autopilot2.lock"
if os.path.exists(_LOCK):
    try:
        _pid = int(open(_LOCK).read().strip())
        os.kill(_pid, 0)
        say(f"已有实例 pid={_pid} 在跑, 自己退出(防双进程抢显存)")
        sys.exit(0)
    except Exception:
        pass
open(_LOCK, "w").write(str(os.getpid()))
sh(f"curl -s -o {TRAIN} '{ECS}/file/train_remote_v5_gpu.py?key={KEY}'")
sh(f"curl -s -o /root/expand_400m_node1.py '{ECS}/file/expand_400m_node1.py?key={KEY}'")

STAGES = [
    ("R7", CH + "/corpus_R7", "/root/c_cache_R7.npy"),
    ("R8", None, "/root/c_cache_R8.npy"),
    ("R9", None, "/root/c_cache_R9.npy"),
    ("R10", None, "/root/c_cache_R10.npy"),
]

# 断点: chain/ckpt_gpu/stage.txt 记录已完成的最后一档 -> 从下一档开始, 不重复吃
_names = [s[0] for s in STAGES]
_done = ""
_p = CH + "/ckpt_gpu/stage.txt"
if os.path.exists(_p):
    _done = open(_p).read().strip()
_start = _names.index(_done) + 1 if _done in _names else 0
if _start:
    say(f"断点续跑: 已完成至 {_done}, 从 {_names[_start]} 开始")

budget = HOURS * 3600
for name, srcdir, cache in STAGES[_start:]:
    if os.path.exists(STOP):
        say("!! STOP_GPU, 退出"); break
    left = budget - (time.time() - T0)
    if left < 600:
        say("预算不足 10 分钟, 退出"); break
    if srcdir is None:
        src_file = {"R8": "zh_r_ancient.jsonl", "R9": "zh_r_novels.jsonl",
                    "R10": "zh_r_weixin.jsonl"}[name]
        srcdir = build_stage(name, f"{RES}/{src_file}")
    ok = run_stage(name, srcdir, cache, PASSES, left)
    report(f"after {name}")
    if not ok:
        break

# ---- 阶梯: 0.4B -> 0.5B -> 0.6B (每级: 通用保函数扩容 -> 换语料 -> 按时长训练) ----
OW = "/root/owner_corpus.jsonl"
sh(f"curl -s -o {OW} '" + ECS + "/file/owner_corpus.jsonl?key=" + KEY + "'")
# owner_corpus 占 2%, upsample 4x (data-plan-0.4B: 原 50x 已废) -> 链接 4 份即 4 倍采样
LADDER = [
    ("04B", 400, ["/mnt/workspace/node1/corpus_04b/ufw_zh.jsonl"] + [OW] * 4, 2.2),
    ("05B", 500, ["/mnt/workspace/node1/corpus_04b/ufw"] + [OW] * 4, 1.8),
    ("06B", 600, ["/mnt/workspace/node1/chain/corpus_reserve/zh_r_knowledge.jsonl",
                  "/mnt/workspace/node1/chain/corpus_reserve/zh_r_ancient.jsonl",
                  "/mnt/workspace/node1/chain/corpus_reserve/zh_r_novels.jsonl",
                  "/mnt/workspace/node1/chain/corpus_reserve/zh_r_weixin.jsonl"] + [OW] * 4, 1.2),
]


def build_stage_multi(name, paths):
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


sh("curl -s -o /root/expand_to_target.py '" + ECS + "/file/expand_to_target.py?key=" + KEY + "'")

for lname, tgt, paths, hrs in LADDER:
    if os.path.exists(STOP):
        say("!! STOP_GPU, 退出阶梯"); break
    left = budget - (time.time() - T0)
    if left < 900:
        say(f"预算剩 {left/60:.0f} 分钟, 停止阶梯"); break
    say(f"=== 阶梯 {lname}: 目标 {tgt}M, 预算 {min(hrs, left/3600):.1f}h ===")
    src_exp = CH + "/ckpt_gpu/checkpoint-latest"
    r = sh(f"cd /root && timeout 2400 python3 /root/expand_to_target.py {src_exp} {tgt} 2>&1 | tail -12")
    say(f"[expand {lname}] " + (r.stdout or "").strip()[-400:].replace("\n", " | "))
    report(f"after expand {lname}")
    if "OK" not in (r.stdout or ""):
        say(f"!! {lname} 扩容失败, 停止"); break
    d = build_stage_multi(lname, paths)
    ok = run_stage(lname, d, f"/root/c_cache_{lname}.npy", 1, min(hrs * 3600, left))
    report(f"after {lname}")
    if not ok:
        break

say(f"=== AUTOPILOT2 结束 | 用时 {(time.time()-T0)/3600:.2f}h ===")
report("结束")
