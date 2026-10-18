# -*- coding: utf-8 -*-
"""n1_upload.py — (node1 跑) 转换 ckpt 为 float16 npz, 分块上传到 ECS /upload。
产物(远程名):
  r4_ckpt_fp16.npz.partNNN   (float16 权重, ~430MB)
  tokenizer.json.part000
  zh_text_r4.jsonl.gz.partNNN
  zh_text_r5.jsonl.gz.partNNN
"""
import json, os, sys, time, urllib.request
import numpy as np

ECS   = "http://47.99.131.12:8081"
KEY   = "k0u4mail"
CHUNK = 64 * 1024 * 1024

CKPT = "/root/r4work/r4_run/checkpoint-latest/pytorch_model.bin"
NPZ  = "/root/r4work/r4_ckpt_fp16.npz"
TOK  = "/root/r4work/r4_base/tokenizer.json"
R4   = "/root/r4work/corpus/zh_text_r4.jsonl"
R5   = "/root/r4work/corpus/zh_text_r5.jsonl"

def put(name, data):
    url = f"{ECS}/upload?name={name}&key={KEY}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url, data=data, method="POST")
            with urllib.request.urlopen(req, timeout=900) as r:
                body = r.read().decode()
                print(f"[put] {name} {len(data)/1e6:.1f}MB -> {r.status} {body}")
                sys.stdout.flush()
                return
        except Exception as e:
            print(f"[put] {name} attempt{attempt+1} FAIL: {e}"); sys.stdout.flush()
            time.sleep(5)
    raise RuntimeError(f"upload {name} failed after retries")

def put_file_chunked(path, remote_base):
    n = 0
    with open(path, "rb") as f:
        while True:
            data = f.read(CHUNK)
            if not data:
                break
            put(f"{remote_base}.part{n:03d}", data)
            n += 1
    return n

def convert_ckpt():
    import torch
    print("[load]", CKPT); sys.stdout.flush()
    try:
        sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(CKPT, map_location="cpu")
    out = {k: v.detach().cpu().float().numpy().astype(np.float16) for k, v in sd.items()}
    np.savez(NPZ, **out)
    total = sum(v.size for v in out.values())
    print(f"[npz] {len(out)} tensors, {total/1e6:.1f}M params, fp16")
    sys.stdout.flush()

def gzip_file(src, dst):
    import gzip
    with open(src, "rb") as fi, gzip.open(dst, "wb", compresslevel=6) as fo:
        while True:
            b = fi.read(8 * 1024 * 1024)
            if not b:
                break
            fo.write(b)
    print(f"[gz] {src} -> {dst} ({os.path.getsize(dst)/1e6:.1f}MB)")
    sys.stdout.flush()

def main():
    t0 = time.time()
    # 1) tokenizer (小)
    put_file_chunked(TOK, "tokenizer.json")
    # 2) 语料 gz
    for src, base in ((R4, "zh_text_r4.jsonl.gz"), (R5, "zh_text_r5.jsonl.gz")):
        gz = src + ".gz"
        if not os.path.exists(gz):
            gzip_file(src, gz)
        put_file_chunked(gz, base)
    # 3) ckpt fp16 (大, 放最后)
    if not os.path.exists(NPZ):
        convert_ckpt()
    n = put_file_chunked(NPZ, "r4_ckpt_fp16.npz")
    print(f"[done] all uploaded, ckpt {n} chunks, total {time.time()-t0:.0f}s")
    sys.stdout.flush()

if __name__ == "__main__":
    main()
