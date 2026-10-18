#!/usr/bin/env python3
# 本机经 hf-mirror 拉 webnovel 小说语料(wdndev/webnovel-chinese, JSONL 分片),
# 只 range 拉前若干 MB -> 切片成 zh_r_novels.jsonl (纯中文叙事层).
import os, sys, json, subprocess, time

MIRROR = "https://hf-mirror.com/datasets/wdndev/webnovel-chinese/resolve/main/data/webnovel_{}.jsonl"
TDIR = r"D:\work\_local_ai\tmp\novels"
OUT = r"D:\work\_local_ai\train_15b\pretrain_corpus\zh_r_novels.jsonl"
CAP = 320e6  # 成品 ~320MB

os.makedirs(TDIR, exist_ok=True)

def fetch_shard(shard, byte_limit):
    # 用 curl -L -r 拉前 byte_limit 字节(避免拉 4GB 整片)
    url = MIRROR.format(shard)
    raw = os.path.join(TDIR, f"shard{shard}.raw")
    if os.path.exists(raw) and os.path.getsize(raw) >= byte_limit * 0.9:
        print(f"[skip] shard{shard} {os.path.getsize(raw)} B")
        return raw
    print(f"[get] shard{shard} range 0-{byte_limit} <- {url}")
    # 多试几次; 拉到 byte_limit 或文件末尾
    for attempt in range(5):
        try:
            r = subprocess.run(
                ["curl", "-sL", "-m300", "-r", f"0-{int(byte_limit)}", "-o", raw, url],
                check=False)
            sz = os.path.getsize(raw) if os.path.exists(raw) else 0
            print(f"      attempt {attempt+1}: got {sz} B")
            if sz > 1000:
                return raw
        except Exception as e:
            print("      err", e)
        time.sleep(5)
    return raw if os.path.exists(raw) else None

def process(raw, out):
    n=0; tot=0
    with open(raw, encoding="utf-8", errors="ignore") as fh, open(out, "w", encoding="utf-8") as w:
        buf=""
        for line in fh:
            buf += line
            if "\n" not in buf:
                continue
            # 取到完整行
            parts = buf.split("\n")
            buf = parts.pop()  # 最后半行留作缓冲
            for pl in parts:
                pl=pl.strip()
                if not pl: continue
                try: o=json.loads(pl)
                except: continue
                text=o.get("text") or ""
                title=o.get("title") or ""
                chapter=o.get("chapter") or ""
                if len(text) < 200: continue
                rec={"text":(title+" "+chapter+"\n"+text).strip(),"src":"webnovel","cat":"novel"}
                s=json.dumps(rec,ensure_ascii=False)+"\n"
                w.write(s); tot+=len(s.encode("utf-8")); n+=1
                if tot>=CAP: 
                    print(f"[done] novel lines={n} bytes={tot}")
                    return
    print(f"[done] novel lines={n} bytes={tot} (raw exhausted)")

if __name__=="__main__":
    t0=time.time()
    raw=None
    for shard in range(0,4):
        raw=fetch_shard(shard, 450e6)
        if raw and os.path.getsize(raw)>1000:
            process(raw, OUT)
            if os.path.getsize(OUT) >= CAP*0.9: break
    print(f"[total seconds] {time.time()-t0:.1f} out={OUT} size={os.path.getsize(OUT) if os.path.exists(OUT) else 0}")
