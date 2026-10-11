#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reassemble_r4.py — 从仓库分片拼回 R4 权重与语料（幂等 + sha256 校验）
仓库内路径: backups/R4_weights/(权重分片+config四件套), backups/R4_corpus/(语料分片+manifest.txt)
用法: python reassemble_r4.py --out /mnt/workspace/node1/r4train
产物: out/r4_base/{pytorch_model.bin, config.json, ...}, out/corpus/*.jsonl
已存在且 sha256 相符的文件直接跳过 —— 重跑安全。
"""
import argparse
import hashlib
import os
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W_DIR = os.path.join(REPO, "backups", "R4_weights")
C_DIR = os.path.join(REPO, "backups", "R4_corpus")
W_SHA = "355c3754ce4f8f6a7c61604f67ef47639c1b50136e2093c8804e87a727494786"
CFG_FILES = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def cat_shards(shards, out):
    tmp = out + ".assembling"
    with open(tmp, "wb") as w:
        for s in shards:
            with open(s, "rb") as r:
                shutil.copyfileobj(r, w, 1 << 22)
    os.replace(tmp, out)


def ok_file(path, want_size, want_sha):
    if not os.path.isfile(path) or os.path.getsize(path) != want_size:
        return False
    return sha256(path) == want_sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = args.out
    base = os.path.join(out, "r4_base")
    corp = os.path.join(out, "corpus")
    os.makedirs(base, exist_ok=True)
    os.makedirs(corp, exist_ok=True)
    fails = 0

    # ---- 1) 权重分片 ----
    w_parts = sorted(
        os.path.join(W_DIR, f) for f in os.listdir(W_DIR) if f.startswith("pytorch_model.bin.part")
    )
    if not w_parts:
        print("FATAL: no weight shards in", W_DIR)
        return 2
    total = sum(os.path.getsize(p) for p in w_parts)
    dst = os.path.join(base, "pytorch_model.bin")
    if ok_file(dst, total, W_SHA):
        print("SKIP weights (ok)")
    else:
        print("assembling weights from", len(w_parts), "shards ...")
        cat_shards(w_parts, dst)
        if ok_file(dst, total, W_SHA):
            print("OK weights", total)
        else:
            print("FATAL: weight sha mismatch after assembly")
            fails += 1

    # ---- 2) config/tokenizer 四件套 ----
    for f in CFG_FILES:
        src = os.path.join(W_DIR, f)
        dstf = os.path.join(base, f)
        if os.path.isfile(src):
            if not (os.path.isfile(dstf) and os.path.getsize(dstf) == os.path.getsize(src)):
                shutil.copyfile(src, dstf)
                print("OK cfg", f)
        else:
            print("FATAL: missing", src)
            fails += 1

    # ---- 3) 语料（manifest 驱动）----
    man = os.path.join(C_DIR, "manifest.txt")
    with open(man, "r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.split()
            if len(parts) < 4:
                continue
            name, size, wsha = parts[0], int(parts[1]), parts[2]
            shards = parts[3:]
            dstc = os.path.join(corp, name)
            if ok_file(dstc, size, wsha):
                print("SKIP", name)
                continue
            spaths = [os.path.join(C_DIR, s) for s in shards]
            missing = [s for s in spaths if not os.path.isfile(s)]
            if missing:
                print("FATAL: missing shards for", name, missing)
                fails += 1
                continue
            cat_shards(spaths, dstc)
            if ok_file(dstc, size, wsha):
                print("OK", name, size)
            else:
                print("FATAL: sha mismatch", name)
                os.remove(dstc)
                fails += 1

    print("RESULT", "FAIL=%d" % fails)
    return 0 if fails == 0 else 3


if __name__ == "__main__":
    sys.exit(main())
