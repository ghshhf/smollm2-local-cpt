#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""build_cache_par.py — 多进程并行 tokenize, 产出与 train_remote_v5_gpu.py 同格式的 token cache

v2 (2026-10-18 03:35) 三个致命修正:
  1) 🔴 原版用 TextIOWrapper.seek(字节偏移) 做分块 —— 文本模式 seek 会从文件头重新解码到目标位置,
     每个 worker 实际都在重读前面的数据(实测一个进程 rchar=2.43GB 而块只有 540MB), 8 进程 24 分钟只完成 3/8。
     改为 open(...,"rb") + 二进制 seek, f.tell() 也叫真实字节偏移。
  2) 🔴 原版每个 worker 把全部 token 累积在 list 里最后 concatenate, 单进程峰值 2.6GB ×8 = 内存爆。
     改为边 tokenize 边追加写 .bin(raw int32), 进程内存恒定。
  3) 支持断点复用: 已存在的 p{i}.npy(旧格式) 或 p{i}.bin(新格式) 直接跳过, 不重复算。

owner_corpus(占 2%, docs/data-plan-0.4B): 主体 98% + 尾部按比例重复插 owner, 让 owner 真正进得来
(训练器原逻辑是「按文件名顺序吃、吃满就 break」, owner 排在后面永远进不去)。

用法:
  python3 build_cache_par.py <stage_dir> <out.npy> [cap_tokens] [nproc] [owner_ratio]
"""
import os, sys, json, numpy as np
from numpy.lib.format import open_memmap
import multiprocessing as mp

BASE = "/root/r4work/base_next"
PART = "/root/cparts"


def _tok():
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(BASE)


def _parts(idx):
    return (f"{PART}/p{idx:04d}.npy", f"{PART}/p{idx:04d}.bin")


def worker(args):
    idx, path, start, end = args
    npy, raw = _parts(idx)
    # 断点复用
    if os.path.exists(npy):
        try:
            n = int(np.load(npy, mmap_mode="r").shape[0])
            if n > 0:
                return idx, npy, n, "npy"
        except Exception:
            pass
    if os.path.exists(raw) and os.path.getsize(raw) > 4096:
        return idx, raw, os.path.getsize(raw) // 4, "raw"

    tok = _tok()
    os.makedirs(PART, exist_ok=True)
    n = 0
    with open(raw, "wb") as fo, open(path, "rb") as f:
        f.seek(start)
        if start > 0:
            f.readline()                      # 丢掉跨块半行
        buf = []
        while True:
            pos = f.tell()
            if pos >= end:
                break
            line = f.readline()
            if not line:
                break
            s = line.decode("utf-8", "ignore").strip()
            if not s:
                continue
            try:
                txt = json.loads(s).get("text", "")
            except Exception:
                txt = s
            if not txt or len(txt) < 20:
                continue
            buf.append(txt)
            if len(buf) >= 400:
                ids = tok("\n".join(buf), return_tensors=None, truncation=False).input_ids
                buf = []
                if ids:
                    fo.write(np.asarray(ids, dtype=np.int32).tobytes())
                    n += len(ids)
        if buf:
            ids = tok("\n".join(buf), return_tensors=None, truncation=False).input_ids
            if ids:
                fo.write(np.asarray(ids, dtype=np.int32).tobytes())
                n += len(ids)
    return idx, raw, n, "raw"


def _open_part(fp, kind):
    if kind == "npy":
        return np.load(fp, mmap_mode="r")
    return np.memmap(fp, dtype=np.int32, mode="r")


def main():
    stage = sys.argv[1]
    outfp = sys.argv[2]
    cap = int(sys.argv[3]) if len(sys.argv) > 3 else 300_000_000
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 8
    owner_ratio = float(sys.argv[5]) if len(sys.argv) > 5 else 0.02

    files = sorted([os.path.join(stage, f) for f in os.listdir(stage)
                    if f.endswith(".jsonl") or f.endswith(".txt")])
    files = [os.path.realpath(p) for p in files]
    sizes = [(p, os.path.getsize(p)) for p in files]
    big = max(sizes, key=lambda x: x[1])
    owner = [p for p, s in sizes if s < 0.05 * big[1]]
    print(f"[plan] 主力 {big[0]} {big[1]/1e9:.2f}GB | owner 候选 {len(owner)} 份", flush=True)

    need_tok = int(cap * (1.0 - owner_ratio))
    # 主体按 ~1.7 字节/token 反推要读多少字节(实测 R8: 19.3MB->12.0M token)
    read_bytes = min(big[1], int(need_tok * 1.75))
    print(f"[plan] 目标 {need_tok/1e6:.0f}M token, 预读 {read_bytes/1e9:.2f}GB, {nproc} 进程", flush=True)

    step = read_bytes // nproc
    tasks = []
    with open(big[0], "rb") as f:
        pos = 0
        for i in range(nproc):
            start = pos
            end = start + step if i < nproc - 1 else read_bytes
            f.seek(end)
            f.readline()                      # 对齐到下一行开头
            end = f.tell()
            tasks.append((i, big[0], start, end))
            pos = end

    with mp.get_context("fork").Pool(nproc) as pool:
        res = pool.map(worker, tasks)
    res.sort()
    parts = [(fp, n, k) for _, fp, n, k in res]
    print("[done-tok] " + " ".join(f"{n/1e6:.1f}M" for _, n, _ in parts), flush=True)

    # ---- 主体: 只登记切片, 不整体载入 ----
    segs, total = [], 0
    for fp, n, kind in parts:
        take = min(int(n), need_tok - total)
        if take <= 0:
            break
        segs.append((fp, kind, take))
        total += take
    print(f"[merge] 主体 {total/1e6:.1f}M token (流式写)", flush=True)

    oa = None
    reps = 1
    if owner:
        tok = _tok()
        o_chunks, obuf = [], []
        with open(owner[0], encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    t = json.loads(line).get("text", "")
                except Exception:
                    t = line
                if t and len(t) >= 20:
                    obuf.append(t)
        if obuf:
            ids = tok("\n".join(obuf), return_tensors=None, truncation=False).input_ids
            if ids:
                o_chunks.append(np.array(ids, dtype=np.int32))
        if o_chunks:
            oa = np.concatenate(o_chunks)
            reps = max(1, int(cap * owner_ratio) // max(1, oa.shape[0]))
            reps = min(reps, 64)
            print(f"[owner] 单份 {oa.shape[0]/1e6:.2f}M token × {reps} 份 = "
                  f"{oa.shape[0]*reps/1e6:.1f}M ({oa.shape[0]*reps/cap*100:.1f}%)", flush=True)

    o_total = 0
    if oa is not None:
        o_total = min(int(oa.shape[0]) * reps, max(0, cap - total))
    final = total + o_total

    out = open_memmap(outfp, mode="w+", dtype=np.int32, shape=(final,))
    pos = 0
    for fp, kind, take in segs:
        a = _open_part(fp, kind)
        n = min(take, final - pos)
        out[pos:pos + n] = a[:n]
        pos += n
        del a
    if oa is not None:
        for _ in range(reps):
            n = min(int(oa.shape[0]), final - pos)
            if n <= 0:
                break
            out[pos:pos + n] = oa[:n]
            pos += n
    out.flush()
    del out
    print(f"[OK] {final/1e6:.1f}M tokens -> {outfp}", flush=True)


if __name__ == "__main__":
    main()
