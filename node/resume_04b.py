#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
resume_04b.py — node1 上 0.4B 断点续训（实例回收后自恢复）

背景（2026-10-18 13:20）:
  魔搭免费实例闲置被回收，/root 是容器本地盘 ⇒ 训练脚本 + 语料缓存全丢。
  NAS(/mnt/workspace) 保住：42 层 ckpt @ step 4500、语料 4.1GB、0.3B 底座、tokenizer。
  本脚本把「脚本拉回 → 合并语料 → 重建缓存 → 分段续训」串成一条自恢复链。

为什么必须分段（不是偷懒）:
  train_remote_v5_gpu.py 外层包了 timeout 172800（48h），而剩余 158401 步在
  8 核 CPU 上按实测 0.67 步/s 需 ~67h ⇒ 一次跑不完会被截断。故按 20000 步/段
  循环（每段 ~8.3h），每段完自动接着上一段的 checkpoint-latest 继续。

为什么可以丢 optimizer 状态:
  tr.py 没有 --resume，续训 = 把 --base_dir 指向 run_live/checkpoint-latest。
  CPU 小模型（400M, Adam, lr 3e-4）动量重置的影响远小于「不训」，可接受。

步数账（跨实例可用，写进 NAS 的 resume_04b_state.json）:
  offset=4500（已训）+ 本次分段累计 → total，目标 TARGET=162901
  = min(预算 3B, 实际 cache 2668M token) // (batch8 × accum4 × 512)

用法（本机投递）:
  python node/mb.py "..." 或直接看 node/resume_04b_push.py
"""
import os, sys, json, time, subprocess

W = "/mnt/workspace/node1"
CH = W + "/chain"
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
ROOT = "/root"
TRAIN = ROOT + "/train_remote_v5_gpu.py"
BUILDER = ROOT + "/build_cache_par.py"
BASE_TOK = CH + "/base_tok"
RUNL = CH + "/ckpt_gpu/run_live"
MIX = ROOT + "/mix_04b.jsonl"
CACHE = ROOT + "/c_cache_04B.npy"
LOG = ROOT + "/train_04B.log"
STATE_F = CH + "/resume_04b_state.json"
TARGET = 162901
OFFSET = 2000
SEG = 20000
SRCS = [W + "/corpus_04b/code_pretrain.jsonl",
        W + "/corpus_04b/code_instruct.jsonl",
        W + "/corpus_04b/ufw_zh.jsonl",
        W + "/corpus_04b/creative_articles_novel.jsonl",
        W + "/corpus_04b/poetry_zh.jsonl"]
BATCH, ACCUM, MAXLEN = 8, 4, 512

REP = []


def sh(c, timeout=600):
    return subprocess.run(c, shell=True, capture_output=True, text=True,
                          timeout=timeout, errors="replace")


def say(m):
    line = "[%s] %s" % (time.strftime("%m-%d %H:%M:%S"), m)
    print(line, flush=True)
    REP.append(line)
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def report(tag="resume_04b"):
    try:
        open("/tmp/%s.txt" % tag, "w", encoding="utf-8").write("\n".join(REP))
        sh('curl -s -m 60 --data-binary @/tmp/%s.txt "%s/upload?name=%s.txt&key=%s"'
           % (tag, ECS, tag, KEY), timeout=120)
    except Exception as e:
        print("[report fail] %s" % e)


def main():
    say("=== resume_04b 启动 ===")
    # 0) 单实例锁
    lock = "/tmp/resume_04b.lock"
    if os.path.exists(lock):
        try:
            pid = int(open(lock).read().strip())
            os.kill(pid, 0)
            say("已有 pid=%d 在跑, 退出" % pid)
            return
        except Exception:
            pass
    open(lock, "w").write(str(os.getpid()))

    # 1) 前提检查
    for p in SRCS + [BASE_TOK + "/tokenizer.json", RUNL + "/checkpoint-latest/config.json"]:
        if not os.path.exists(p):
            say("!! 缺关键文件: %s" % p)
            report()
            return
    say("前提 OK：语料 %d 份 / tokenizer / 42层 ckpt" % len(SRCS))

    # 2) 拉脚本
    for f in ["train_remote_v5_gpu.py", "build_cache_par.py"]:
        sh('curl -s -m 180 -o %s/%s "%s/file/%s?key=%s"' % (ROOT, f, ECS, f, KEY))
        sz = os.path.getsize(ROOT + "/" + f) if os.path.exists(ROOT + "/" + f) else 0
        say("脚本 %-26s %d B" % (f, sz))
        if sz < 1000:
            say("!! 脚本拉取失败")
            report()
            return

    # 3) 合并语料（建库器只 tokenize 最大单文件，分开会漏后两个）
    need = sum(os.path.getsize(s) for s in SRCS)
    have = os.path.getsize(MIX) if os.path.exists(MIX) else -1
    remerged = False
    if have != need:
        remerged = True
        say("合并语料 -> %s (%.2f GB)" % (MIX, need / 1e9))
        r = sh("cat %s > %s.tmp && mv %s.tmp %s" % (" ".join(SRCS), MIX, MIX, MIX), timeout=1800)
        if not os.path.exists(MIX) or os.path.getsize(MIX) != need:
            say("!! 合并失败: %s" % (r.stderr or "")[-200:])
            report()
            return
    say("语料就绪 %.2f GB" % (need / 1e9))

    # 3.5) 语料换版（2026-10-19 50%代码/50%中英文）=> 旧缓存必须作废重建
    if remerged and os.path.exists(CACHE):
        say("语料已换 -> 删旧缓存强制重建")
        os.remove(CACHE)

    # 4) 缓存（/root 易失，重建是硬成本；cap 3e9 对齐预算 04B=3B）
    import numpy as _np
    n = 0
    if os.path.exists(CACHE):
        try:
            n = int(_np.load(CACHE, mmap_mode="r").shape[0])
        except Exception:
            n = 0
        if n < 1_000_000:
            say("中毒缓存 %d -> 删除" % n)
            os.remove(CACHE)
            n = 0
    if n == 0:
        say("建缓存中（8 核 CPU，预计 40~90 分钟）...")
        t0 = time.time()
        r = sh("cd %s && timeout 7200 python3 -u %s %s %s 3000000000 8 0.02"
               % (ROOT, BUILDER, ROOT + "/mix_stage", CACHE), timeout=7400)
        out = ((r.stdout or "") + (r.stderr or "")).strip()[-300:]
        say("建缓存结果: %s" % out.replace("\n", " | "))
        try:
            n = int(_np.load(CACHE, mmap_mode="r").shape[0])
        except Exception:
            n = 0
        say("cache=%d token 用时 %.1f 分钟" % (n, (time.time() - t0) / 60))
    if n < 1_000_000:
        say("!! 缓存无效，停止")
        report()
        return

    # 5) 分段续训
    tok_step = BATCH * ACCUM * MAXLEN
    target = min(TARGET, n // tok_step)
    say("目标总步数 %d（cache %dM token / %d tok每步）" % (target, n // 10**6, tok_step))

    done = OFFSET
    if os.path.exists(STATE_F):
        try:
            st = json.load(open(STATE_F))
            done = max(OFFSET, int(st.get("total", OFFSET)))
            say("读到历史进度 total=%d" % done)
        except Exception:
            pass

    seg_i = 0
    while done < target:
        seg_i += 1
        n_seg = min(SEG, target - done)
        base = RUNL + "/checkpoint-latest"
        say("[段 %d] base=%s 从 %d 训 %d 步 -> %d" % (seg_i, base, done, n_seg, done + n_seg))
        t0 = time.time()
        r = sh("cd %s && timeout 40000 python3 -u %s --base_dir %s --tok_dir %s "
               "--run_dir %s --corpus_dir %s --cache %s --steps %d "
               "--batch %d --accum %d --max_len %d --lr 3e-4 --warmup 50 --save_every 300 "
               "--threads 8 --device auto"
               % (ROOT, TRAIN, base, BASE_TOK, RUNL, ROOT + "/mix_stage", CACHE,
                  n_seg, BATCH, ACCUM, MAXLEN), timeout=41000)
        tail = ((r.stdout or "") + (r.stderr or "")).strip()[-400:].replace("\n", " | ")
        say("[段 %d] 结束 用时 %.2fh | %s" % (seg_i, (time.time() - t0) / 3600, tail))
        if not os.path.exists(RUNL + "/checkpoint-latest/config.json"):
            say("!! 本段无产出，停止")
            break
        done += n_seg
        json.dump({"total": done, "target": target, "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                  open(STATE_F, "w"), ensure_ascii=False)
        report()
        if time.time() - t0 < 60:      # 秒退 = 出错，别死循环
            say("!! 本段异常快速退出，停止")
            break

    say("=== resume_04b 结束 total=%d / %d ===" % (done, target))
    report()


if __name__ == "__main__":
    main()
