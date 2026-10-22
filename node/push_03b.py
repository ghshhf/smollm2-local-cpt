#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""push_03b.py —— 把本机 0.3B 主干切 95MB 片推上 ECS 信箱，node1 端合并。

为什么切片：ECS 是单线程 PowerShell HttpListener，实测 >100MB 单文件会掉包
（历史 upload_base.log 里 95MB 片就有 FAIL 重试）。95MB 是稳定档。

产物：
  parts/03b_bin.00 ... 03b_bin.14      本机切片
  parts/03b.manifest.json              {sha256, total, parts}
用法：
  python node/push_03b.py split     # 只切 + 算 sha256
  python node/push_03b.py push      # 切 + 传（带校验重试）
"""
import hashlib
import json
import os
import sys
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "backups", "0.3B", "polaris-03b_20261018")
PARTS = os.path.join(HERE, "parts")
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
# 🔴 片大小不只是「ECS 单文件上限」问题，更是「控制面会不会瘫痪」问题：
#    ECS 是单线程 PowerShell HttpListener，一个 95MB 的 GET 会把它独占约 70 秒，
#    这段时间里别的机器的**心跳都传不上去**（实测 node2 因此掉线 18 分钟）。
#    切成 10MB 后单片只占 ~6 秒，请求之间留出空隙，控制面就能插进去。
CHUNK = 10 * 1000 * 1000
PREFIX = "03b_bin."

FILES = ["model.safetensors", "config.json", "generation_config.json"]


def split():
    os.makedirs(PARTS, exist_ok=True)
    # 清掉上一轮的碎片，避免旧片混进新一轮
    for f in os.listdir(PARTS):
        if f.startswith(PREFIX):
            os.remove(os.path.join(PARTS, f))
    man = {"chunk": CHUNK, "files": {}}
    for fn in FILES:
        src = os.path.join(SRC, fn)
        if not os.path.exists(src):
            print("[skip] 缺 %s" % fn)
            continue
        h = hashlib.sha256()
        size = os.path.getsize(src)
        if size <= CHUNK:
            # 小文件不切：直接原名上板（切片会让 config.json 和权重共用 03b_bin.00 撞名）
            with open(src, "rb") as f:
                b = f.read()
            h.update(b)
            with open(os.path.join(PARTS, fn), "wb") as o:
                o.write(b)
            man["files"][fn] = {"size": size, "parts": 0, "bare": True,
                                "sha256": h.hexdigest()}
            print("[split] %s  %.1f KB  单文件直传" % (fn, size / 1e3))
            continue
        n = 0
        with open(src, "rb") as f:
            while True:
                b = f.read(CHUNK)
                if not b:
                    break
                h.update(b)
                with open(os.path.join(PARTS, "%s%02d" % (PREFIX, n)), "wb") as o:
                    o.write(b)
                n += 1
        man["files"][fn] = {"size": size, "parts": n, "bare": False,
                            "sha256": h.hexdigest()}
        print("[split] %s  %.1f MB -> %d 片  sha256=%s"
              % (fn, size / 1e6, n, h.hexdigest()[:16]))
    with open(os.path.join(PARTS, "03b.manifest.json"), "w", encoding="utf-8") as f:
        json.dump(man, f, ensure_ascii=False, indent=2)
    print("[split] manifest -> parts/03b.manifest.json")
    return man


def _remote_size(url):
    """ECS 是 PowerShell HttpListener，不支持 HEAD（返回 -1）也不支持 Range。
    改用 GET 并只读首块 —— 但 95MB 全下要 57s，不如直接用 /upload 的返回值：
    它回 {"name":..,"saved":N}，N 就是服务端实际收到的字节数，比二次探测更可信。"""
    try:
        with urllib.request.urlopen(url, timeout=20) as r:
            return int(r.headers.get("Content-Length") or 0)
    except Exception:
        return -1


def push(man):
    total_n = sum(v["parts"] or 1 for v in man["files"].values())
    done = 0
    t0 = time.time()
    for fn, info in man["files"].items():
        names = [fn] if info.get("bare") else \
                ["%s%02d" % (PREFIX, i) for i in range(info["parts"])]
        for p in names:
            lp = os.path.join(PARTS, p)
            local = os.path.getsize(lp)
            ok = False
            for att in range(1, 5):
                with open(lp, "rb") as f:
                    data = f.read()
                req = urllib.request.Request(
                    "%s/upload?name=%s&key=%s" % (ECS, p, KEY), data=data, method="POST")
                try:
                    with urllib.request.urlopen(req, timeout=300) as r:
                        resp = r.read().decode("utf-8", "replace")
                    saved = json.loads(resp).get("saved", -1)
                except Exception as e:
                    print("  [err] %s att%d %s" % (p, att, str(e)[:80]))
                    time.sleep(3)
                    continue
                if int(saved) == local:
                    ok = True
                    break
                print("  [mismatch] %s att%d saved=%s want=%d" % (p, att, saved, local))
                time.sleep(3)
            done += 1
            el = time.time() - t0
            print("[%d/%d] %s %s  (%.1f MB/s)" % (
                done, total_n, p, "OK" if ok else "FAIL",
                (done * CHUNK / 1e6) / max(el, 1)))
            if not ok:
                print("[push] 中止：%s 传不上去" % p)
                return False
    # manifest 最后传（node1 看到它才开始合并）
    mp = os.path.join(PARTS, "03b.manifest.json")
    with open(mp, "rb") as f:
        data = f.read()
    req = urllib.request.Request("%s/upload?name=03b.manifest.json&key=%s" % (ECS, KEY),
                                 data=data, method="POST")
    urllib.request.urlopen(req, timeout=60).read()
    print("[push] 全部完成，manifest 已上板。耗时 %.1f 分钟" % ((time.time() - t0) / 60))
    return True


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ("split", "push"):
        print(__doc__)
        return 1
    man = split()
    if sys.argv[1] == "push":
        return 0 if push(man) else 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
