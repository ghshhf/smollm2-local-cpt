#!/usr/bin/env python3
# 本机经代理拉取 公众号语料(nonamestreet/weixin_public_corpus, 分卷 zip),
# 合并 -> 流式解压 -> 切片成 zh_r_weixin.jsonl, 直接推 node1 的成品。
import os, sys, json, zipfile, urllib.request, urllib.error, shutil, time

PROXY = "http://127.0.0.1:3067"
BASE = "https://raw.githubusercontent.com/nonamestreet/weixin_public_corpus/master"
PARTS = [f"articles.z{n:02d}" for n in range(1, 15)] + ["articles.zip"]
TDIR = r"D:\work\_local_ai\tmp\weixin"
OUT = r"D:\work\_local_ai\train_15b\pretrain_corpus\zh_r_weixin.jsonl"
CAP = 320e6  # ~320MB 成品切片

os.makedirs(TDIR, exist_ok=True)

def opener():
    ph = urllib.request.ProxyHandler({"http": PROXY, "https": PROXY})
    return urllib.request.build_opener(ph)

def download():
    op = opener()
    for p in PARTS:
        fp = os.path.join(TDIR, p)
        if os.path.exists(fp) and os.path.getsize(fp) > 0:
            print(f"[skip] {p} ({os.path.getsize(fp)} B)")
            continue
        url = f"{BASE}/{p}"
        print(f"[get] {p} <- {url}")
        for attempt in range(4):
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
                with op.open(req, timeout=120) as r, open(fp, "wb") as w:
                    shutil.copyfileobj(r, w)
                print(f"      ok {os.path.getsize(fp)} B")
                break
            except Exception as e:
                print(f"      attempt {attempt+1} err: {e}")
                time.sleep(5)
        else:
            raise SystemExit(f"FAILED downloading {p}")

def concat():
    combined = os.path.join(TDIR, "combined.zip")
    if os.path.exists(combined) and os.path.getsize(combined) > 0:
        print(f"[concat] exists {os.path.getsize(combined)} B")
        return combined
    with open(combined, "wb") as out:
        for p in PARTS:
            fp = os.path.join(TDIR, p)
            with open(fp, "rb") as f:
                shutil.copyfileobj(f, out)
    print(f"[concat] -> {os.path.getsize(combined)} B")
    return combined

def extract_slice(combined):
    z = zipfile.ZipFile(combined)
    names = z.namelist()
    print("[zip members]", names[:5], "...total", len(names))
    # 找最大的 jsonl/txt
    target = None
    for n in names:
        if n.lower().endswith((".jsonl", ".txt")) and "article" in n.lower():
            target = n; break
    if target is None:
        target = max(names, key=lambda n: z.getinfo(n).file_size)
    print("[extract target]", target)
    n=0; tot=0
    with z.open(target) as fh, open(OUT, "w", encoding="utf-8") as w:
        for raw in fh:
            try:
                line = raw.decode("utf-8", "ignore")
                o = json.loads(line)
            except Exception:
                continue
            content = o.get("content") or ""
            title = o.get("title") or ""
            name = o.get("name") or "?"
            if len(content) < 80:
                continue
            rec = {"text": (title + "\n" + content).strip(),
                   "src": "weixin:" + str(name), "cat": "modern"}
            s = json.dumps(rec, ensure_ascii=False) + "\n"
            w.write(s); tot += len(s.encode("utf-8")); n += 1
            if tot >= CAP:
                break
    print(f"[done] weixin lines={n} bytes={tot} -> {OUT}")

if __name__ == "__main__":
    t0 = time.time()
    download()
    combined = concat()
    extract_slice(combined)
    print(f"[total seconds] {time.time()-t0:.1f}")
