# -*- coding: utf-8 -*-
"""n2_fetch.py — (node2 跑) 自动发现 ECS /files 里的分块文件, 拼接下载到本地。
用法: python3 n2_fetch.py [pattern 过滤, 如 r4_ckpt]
"""
import json, os, re, sys, time, urllib.request

ECS = "http://47.99.131.12:80"
W = "/home/aistudio/node2/data"
os.makedirs(W, exist_ok=True)

pattern = sys.argv[1] if len(sys.argv) > 1 else ""

def get(path, tries=3):
    for a in range(tries):
        try:
            return urllib.request.urlopen(ECS + path, timeout=300).read()
        except Exception as e:
            print("retry", a + 1, path, e); time.sleep(3)
    raise RuntimeError("get fail " + path)

d = json.loads(get("/files"))
files = {f["n"]: f["sz"] for f in d["files"]}
groups = {}
for n in files:
    m = re.match(r"(.+)\.part(\d+)$", n)
    if m and pattern in m.group(1):
        groups.setdefault(m.group(1), []).append(int(m.group(2)))

print("groups:", {k: len(v) for k, v in groups.items()}); sys.stdout.flush()
for base, parts in sorted(groups.items()):
    parts.sort()
    out = os.path.join(W, base)
    expect = (max(parts) + 1) if parts else 0
    if len(parts) != expect:
        print(f"[skip] {base}: 块不齐 {len(parts)}/{expect}")
        continue
    done_mark = out + ".done"
    if os.path.exists(done_mark):
        print(f"[skip] {base} 已完成")
        continue
    t0 = time.time()
    with open(out, "wb") as fo:
        for i in parts:
            data = get(f"/file/{base}.part{i:03d}")
            fo.write(data)
            print(f"[dl] {base} part{i:03d} {len(data)/1e6:.1f}MB"); sys.stdout.flush()
    open(done_mark, "w").write(str(sum(files[f"{base}.part{i:03d}"] for i in parts)))
    print(f"[ok] {base} -> {out} ({(time.time()-t0):.0f}s)")
print("FETCH_DONE")
