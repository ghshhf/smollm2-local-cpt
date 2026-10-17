#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reassemble.py — 从同目录分片拼回 0.3B 权重（幂等 + sha256 自校验）
用法: python reassemble.py [输出目录]
"""
import hashlib, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_SHA256 = "90f2f64bd01f64bd4690414d709f1291d98bf855a11958a4269dffcb33f30b70"
FINAL_SIZE = 1387470856


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    os.makedirs(out, exist_ok=True)
    parts = sorted(os.path.join(HERE, f) for f in os.listdir(HERE)
                   if f.startswith("model.safetensors.part"))
    if not parts:
        print("FATAL: no shards in", HERE)
        return 2
    dst = os.path.join(out, "model.safetensors")
    if os.path.isfile(dst) and os.path.getsize(dst) == FINAL_SIZE and sha256(dst) == FINAL_SHA256:
        print("SKIP (already ok)")
        return 0
    tmp = dst + ".assembling"
    with open(tmp, "wb") as w:
        for p in parts:
            with open(p, "rb") as r:
                shutil.copyfileobj(r, w, 1 << 22)
    sz = os.path.getsize(tmp)
    if sz != FINAL_SIZE or sha256(tmp) != FINAL_SHA256:
        print("FATAL: assemble check failed", sz)
        os.remove(tmp)
        return 3
    os.replace(tmp, dst)
    print("OK", sz, "sha256 verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
