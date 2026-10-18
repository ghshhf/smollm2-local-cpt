#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""pull03.py — 把远程 0.3B（checkpoint-03b，346.9M/36L，含 R8/R9/R10）拉回本地

两段式：
  python pull03.py --dispatch   派远程把权重切 15 片 + 元文件，逐片推到 ECS 信箱
  python pull03.py --fetch      本地从 ECS 下载 -> 拼接 -> sha256 校验 -> 落到 backups/0.3B

远程源：/mnt/workspace/node1/chain/ckpt_gpu/checkpoint-03b（NAS，持久层）
tokenizer 兜底：/mnt/workspace/node1/chain/base_tok/（node1 ckpt 自带 tokenizer 缺失）
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DST = os.path.join(ROOT, "backups", "0.3B", "r10_20261018")
PARTS = os.path.join(DST, "_parts")

NPART = 15
META = ["config.json", "generation_config.json", "step.txt",
        "tokenizer.json", "tokenizer_config.json"]
NAMES = [f"model.safetensors.part{i:02d}" for i in range(NPART)] + META


def dispatch():
    cmds = []
    for n in NAMES:
        cmds.append(
            f'curl -s -m 900 -X POST "{ECS}/upload?name=ck03_{n}&key={KEY}" '
            f'--data-binary @{n} >> up.log 2>&1 && echo "UP {n}" >> up.log')
    cmds.append(f'sha256sum model.safetensors > sha.txt && '
                f'curl -s -m 180 -X POST "{ECS}/upload?name=ck03_sha.txt&key={KEY}" '
                f'--data-binary @sha.txt')
    cmds.append(f'echo DONE > DONE.txt && '
                f'curl -s -m 60 -X POST "{ECS}/upload?name=ck03_DONE.txt&key={KEY}" '
                f'--data-binary @DONE.txt')
    inner = "cd /tmp/ck03 && rm -f up.log DONE.txt && " + " && ".join(cmds)
    cmd = "( nohup bash -c " + json.dumps(inner) + " > /dev/null 2>&1 & ) echo DISPATCHED"
    body = json.dumps({"cmd": cmd, "id": "ck03-pull", "ts": int(time.time())}).encode()
    req = urllib.request.Request(ECS + "/task", data=body,
                                 headers={"Content-Type": "application/json"}, method="POST")
    print("dispatch ->", urllib.request.urlopen(req, timeout=20).status)


def get(url, timeout=900):
    return urllib.request.urlopen(url, timeout=timeout).read()


def fetch():
    os.makedirs(PARTS, exist_ok=True)
    t_all = time.time()
    for i, n in enumerate(NAMES):
        dst = os.path.join(PARTS, n)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"[{i + 1}/{len(NAMES)}] skip {n}", flush=True)
            continue
        url = f"{ECS}/file/ck03_{n}?key={KEY}"
        for attempt in range(120):
            try:
                t0 = time.time()
                data = get(url)
                if not data:
                    raise IOError("empty")
                open(dst, "wb").write(data)
                print(f"[{i + 1}/{len(NAMES)}] {n} {len(data) / 1e6:.1f}MB "
                      f"{time.time() - t0:.1f}s", flush=True)
                break
            except Exception as e:
                if attempt == 119:
                    print(f"[FATAL] {n} 下不下来: {e}", flush=True)
                    return False
                time.sleep(5)

    # 拼接
    out = os.path.join(DST, "model.safetensors")
    with open(out, "wb") as w:
        for n in NAMES[:NPART]:
            with open(os.path.join(PARTS, n), "rb") as r:
                while True:
                    b = r.read(64 << 20)
                    if not b:
                        break
                    w.write(b)
    print(f"[cat] {out} {os.path.getsize(out) / 1e9:.3f} GB", flush=True)

    # 校验
    h = hashlib.sha256()
    with open(out, "rb") as r:
        for b in iter(lambda: r.read(64 << 20), b""):
            h.update(b)
    mine = h.hexdigest()
    try:
        remote = get(f"{ECS}/file/ck03_sha.txt?key={KEY}", 120).decode("utf-8", "replace")
        rsha = remote.split()[0] if remote.strip() else ""
    except Exception:
        rsha = ""
    ok = (rsha == mine)
    print(f"[sha256] local={mine}", flush=True)
    print(f"[sha256] remote={rsha or '(未取到)'}  => {'MATCH' if ok else 'MISMATCH'}", flush=True)

    if ok:
        for n in NAMES[:NPART]:
            try:
                os.remove(os.path.join(PARTS, n))
            except Exception:
                pass
        try:
            os.rmdir(PARTS)
        except Exception:
            pass
        print(f"[DONE] {DST}  用时 {(time.time() - t_all) / 60:.1f} 分钟", flush=True)
    return ok


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    a = sys.argv[1] if len(sys.argv) > 1 else "--fetch"
    if a == "--dispatch":
        dispatch()
    elif a == "--fetch":
        sys.exit(0 if fetch() else 1)
    else:
        print("用法: pull03.py --dispatch | --fetch")
