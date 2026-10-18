#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_wiki_knowledge.py — 在 node1 上直接抓中文维基百科，建 R7「知识 backbone」的干净源。

为什么不用现有的 zh_r_knowledge.jsonl：
  它是 build_reserve.py 从 corpus_r6 复制来的，而 R6 混池里的"教育"部分
  (zh_text_v3_edu.jsonl) 实际内容是《閱微草堂筆記》——文言笔记小说，不是教材。
  ⇒ R7 抽样出来全是水浒名单/三言二拍/孽海花，跟"知识 backbone"名不副实。
  中文维基百科才是百科型知识的标准源(数学/哲学/物理/历史/地理…全都有)。

幂等：parquet 存在且非空则跳过下载；输出已存在且非空则跳过转换。
环境变量：
  SHARDS  下载分片数(默认 3，每片约 75 万条)
  OUT     输出 jsonl 路径
"""
import os, sys, json, glob, subprocess, time

W = "/mnt/workspace/node1"
WIKI_DIR = os.path.join(W, "corpus_wiki")
OUT = os.environ.get("OUT", os.path.join(W, "chain", "corpus_reserve", "zh_wiki_all.jsonl"))
N = int(os.environ.get("SHARDS", "3"))
TOTAL = 6
_TS = "%05d" % TOTAL          # ⚠️ API 返回的是 train-00002-of-00006(5位), 不是 of-000006
_BASE = "https://hf-mirror.com/datasets/wikimedia/wikipedia/resolve/main/20231101.zh/train-%05d-of-" + _TS + ".parquet"
BASE = lambda i: _BASE % i    # noqa: E731
MIN_CHARS = 120


def L(m):
    print(m, flush=True)


def fetch():
    os.makedirs(WIKI_DIR, exist_ok=True)
    for i in range(N):
        fn = "train-%05d-of-%s.parquet" % (i, _TS)
        dst = os.path.join(WIKI_DIR, fn)
        if os.path.exists(dst) and os.path.getsize(dst) > 10_000_000:
            L("[dl] skip %s (%.0fMB 已有)" % (fn, os.path.getsize(dst) / 1e6))
            continue
        url = BASE(i)
        # 先探一次真实大小/状态, 免得白等一个 404
        r = subprocess.run(["curl", "-sIL", "-m", "60", url], capture_output=True, text=True)
        last = [l for l in (r.stdout or "").splitlines() if l.lower().startswith("http/")]
        code = last[-1].split()[1] if last else "?"
        L("[dl] HEAD %s -> %s" % (url, code))
        if code not in ("200", "206"):
            L("[dl] !! 分片 %d 不可达(HTTP %s), 跳过" % (i, code))
            continue
        L("[dl] -> %s" % fn)
        subprocess.run(["curl", "-sL", "--retry", "3", "-m", "3000", "-o", dst, url],
                       capture_output=True)
        sz = os.path.getsize(dst) if os.path.exists(dst) else 0
        magic = ""
        if sz > 4:
            with open(dst, "rb") as f:
                magic = f.read(4).decode("latin1", "replace")
        L("[dl] done %s  %.1fMB  magic=%r" % (fn, sz / 1e6, magic))
        if sz < 10_000_000 or magic != "PAR1":
            L("[dl] !! %s 校验失败(需 PAR1 开头), 丢弃" % fn)
            try:
                os.remove(dst)
            except Exception:
                pass


def convert():
    if os.path.exists(OUT) and os.path.getsize(OUT) > 1_000_000:
        L("[cv] skip (输出已存在 %.1fMB)" % (os.path.getsize(OUT) / 1e6))
        return
    try:
        import pyarrow.parquet as pq
    except Exception:
        L("[cv] 安装 pyarrow ...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow"])
        import pyarrow.parquet as pq

    ps = sorted(glob.glob(os.path.join(WIKI_DIR, "*.parquet")))
    L("[cv] 待转换 %d 个分片" % len(ps))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = skip = tot = 0
    t0 = time.time()
    with open(OUT, "w", encoding="utf-8") as f:
        for p in ps:
            try:
                pf = pq.ParquetFile(p)
            except Exception as e:
                L("[cv] SKIP_BAD %s %s" % (os.path.basename(p), str(e)[:80]))
                continue
            cnt = 0
            try:
                batches = pf.iter_batches(batch_size=2000, columns=["title", "text"])
            except Exception:
                cnt = -1
            if cnt == -1:
                continue
            for batch in batches:
                for r in batch.to_pylist():
                    t = (r.get("title") or "").strip()
                    b = (r.get("text") or "").strip()
                    if not t or len(b) < MIN_CHARS:
                        skip += 1
                        continue
                    f.write(json.dumps({"text": t + "\n" + b, "title": t},
                                       ensure_ascii=False) + "\n")
                    cnt += 1
                    tot += len(b)
            n += max(cnt, 0)
            L("[cv] %s -> +%d 条 (累计 %d, %.0fs)"
              % (os.path.basename(p), cnt, n, time.time() - t0))
    L("[cv] 完成: %d 条 (跳过 %d, 正文合计 %.0f 万字) -> %s (%.1fMB)"
      % (n, skip, tot / 1e4, OUT, os.path.getsize(OUT) / 1e6))


if __name__ == "__main__":
    t0 = time.time()
    L("===== build_wiki_knowledge start @ " + time.strftime("%Y-%m-%d %H:%M")
      + " (SHARDS=%d) =====" % N)
    fetch()
    convert()
    L("===== done in %.0fs =====" % (time.time() - t0))