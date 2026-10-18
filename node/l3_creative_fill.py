#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
l3_creative_fill.py (v3, in-memory heap) — 用 L3 zh multi_style 蒸馏数据填 0.4B「创意·文章」缺口 (精炼到极致)
位置: node1 /mnt/workspace/node1/  (由 /task 派活, nohup 后台跑)
流程: 复用已有 4.6GB jsonl -> 流式 hard_filter + 块级去重 + 密度打分 -> 内存 max-heap 保 top-K -> 预算/风格配额蒸馏 -> 直写输出
输出: /mnt/workspace/node1/corpus_04b/creative_refined/creative_articles_l3.jsonl + 回传报告
设计: 省掉 cand 写盘+回读两笔 NAS I/O；内存堆 HMAX 封顶，避免 OOM；自启动杀旧实例。
"""
import os, sys, json, time, hashlib, subprocess, glob, re, heapq

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
D = "/mnt/workspace/node1/corpus_04b"
L3_DIR = os.path.join(D, "l3_ms")
OUT_DIR = os.path.join(D, "creative_refined")
os.makedirs(L3_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ---- 自启动: 杀其它旧实例 (不杀自己) ----
me = os.getpid()
try:
    for pid in subprocess.run(["bash", "-c", "pgrep -f l3_creative_fill.py"],
                              capture_output=True, text=True, timeout=10).stdout.split():
        pid = int(pid)
        if pid != me:
            subprocess.run(["kill", "-9", str(pid)]); print("[kill] old pid %d" % pid, flush=True)
except Exception:
    pass
# 清掉旧任务残留 cand (新版内存堆, 不用 cand 文件)
_cand_left = os.path.join(OUT_DIR, ".creative.cand")
if os.path.exists(_cand_left):
    try: os.remove(_cand_left); print("[clean] removed leftover .creative.cand", flush=True)
    except Exception: pass

BASE = "https://modelscope.cn/api/v1/datasets/OpenBMB/Ultra-FineWeb-L3/repo?Revision=master&FilePath="
PARTS = [
 "data/ultrafineweb_zh_l3/multi_style/part-00000-c13afd3b-b5fb-4acd-97dc-e045a844c126-c000.snappy.parquet",
 "data/ultrafineweb_zh_l3/multi_style/part-00001-c13afd3b-b5fb-4acd-97dc-e045a844c126-c000.snappy.parquet",
 "data/ultrafineweb_zh_l3/multi_style/part-00002-c13afd3b-b5fb-4acd-97dc-e045a844c126-c000.snappy.parquet",
]
def L(m): print(m, flush=True)

# ---------- 1) 下载 (复用则跳过) ----------
L("===== [1/3] download L3 zh multi_style (skip if exists) =====")
for i, p in enumerate(PARTS):
    out = os.path.join(L3_DIR, "ms%03d.parquet" % i)
    if os.path.exists(out) and os.path.getsize(out) > 1000000:
        L("  skip existing %s (%.1fMB)" % (out, os.path.getsize(out)/1e6)); continue
    rc = subprocess.run(["curl", "-sL", "--retry", "3", "-m", "3600", "-o", out, BASE + p]).returncode
    sz = os.path.getsize(out) if os.path.exists(out) else 0
    ok = False
    if sz > 1000000:
        with open(out, "rb") as f:
            head = f.read(4); f.seek(-4, 2); tail = f.read(4)
        ok = (head[:1] == b"P" or tail == b"PAR1")
    L("  -> rc=%s size=%.1fMB parquet_ok=%s" % (rc, sz/1e6, ok))
    if not ok:
        L("  !! bad, removing"); os.remove(out)

# ---------- 2) 转 jsonl (复用则跳过) ----------
L("===== [2/3] convert parquet -> jsonl (skip if exists) =====")
try:
    import pyarrow.parquet as pq
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "pyarrow"]); import pyarrow.parquet as pq

def extract(r):
    c = r.get("content") or r.get("text") or ""
    if isinstance(c, str) and c.strip():
        return c.strip(), (r.get("style") or "")
    q = (r.get("question") or "").strip(); a = (r.get("answer") or r.get("output") or "").strip()
    if q and a: return q + "\n" + a, (r.get("style") or "")
    parts = [v.strip() for k, v in r.items()
             if isinstance(v, str) and len(v.strip()) > 40 and k not in ("style","source","id","lang")]
    return "\n".join(parts), (r.get("style") or "")

raw_jsonl = os.path.join(L3_DIR, "l3_multistyle.jsonl")
converted_rows = 0
if os.path.exists(raw_jsonl) and os.path.getsize(raw_jsonl) > 100_000_000:
    L("[cv] reuse existing %s (%.1fMB)" % (raw_jsonl, os.path.getsize(raw_jsonl)/1e6))
else:
    with open(raw_jsonl, "w", encoding="utf-8") as f:
        for p in sorted(glob.glob(os.path.join(L3_DIR, "*.parquet"))):
            try: pf = pq.ParquetFile(p)
            except Exception as e: L("  SKIP_BAD %s %s" % (os.path.basename(p), str(e)[:80])); continue
            cols = [c for c in ("content","text","question","answer","output","style","source") if c in pf.schema_arrow.names]
            for b in pf.iter_batches(batch_size=2000, columns=cols or None):
                for r in b.to_pylist():
                    t, st = extract(r)
                    if len(t) < 150: continue
                    f.write(json.dumps({"text": t, "source": "l3_ms", "style": st}, ensure_ascii=False) + "\n")
                    converted_rows += 1
    L("[cv] converted rows=%d -> %s" % (converted_rows, raw_jsonl))

# ---------- 2.5) 慢 NAS -> 本地盘 (并行分段拷贝, 加速 ~1.14MB/s 的 NAS 读) ----------
LOCAL_DIR = "/tmp/l3parts"
LOCAL_JSONL = "/tmp/l3_multistyle.jsonl"

def ensure_local(src, dst, parts=8):
    sz = os.path.getsize(src)
    if os.path.exists(dst) and os.path.getsize(dst) == sz:
        L("[cp] 本地已有完整 %s (%.1fMB), 跳过拷贝" % (dst, sz/1e6)); return True
    os.makedirs(LOCAL_DIR, exist_ok=True)
    seg = (sz + parts - 1) // parts
    pfs, procs = [], []
    for i in range(parts):
        off = i * seg; this = min(seg, sz - off)
        if this <= 0: break
        pf = "%s/p%02d" % (LOCAL_DIR, i); pfs.append(pf)
        # GNU dd: skip/count 按字节解释
        cmd = ("dd if=%s of=%s bs=1M iflag=skip_bytes,count_bytes skip=%d count=%d 2>/dev/null"
               % (src, pf, off, this))
        procs.append(subprocess.Popen(cmd, shell=True))
    L("[cp] %d 路并行拷贝启动 (源 %.1fMB)" % (len(procs), sz/1e6))
    t0 = time.time()
    rcs = [p.wait() for p in procs]
    tot = sum(os.path.getsize(p) for p in pfs if os.path.exists(p))
    L("[cp] 并行结束 rc=%s 总字节=%d/%d (%.0fs)" % (rcs, tot, sz, time.time()-t0))
    ok = all(rc == 0 for rc in rcs) and tot == sz
    if ok:
        with open(dst, "wb") as fo:
            for p in pfs:
                with open(p, "rb") as fi:
                    while True:
                        b = fi.read(1 << 22)
                        if not b: break
                        fo.write(b)
        for p in pfs:
            try: os.remove(p)
            except Exception: pass
        L("[cp] 合并完成 %s (%.1fMB, %.0fs)" % (dst, os.path.getsize(dst)/1e6, time.time()-t0))
        return os.path.getsize(dst) == sz
    L("[cp] !! 并行拷不可用, 回退单流 cp (慢)")
    subprocess.run(["cp", src, dst])
    g = os.path.getsize(dst) if os.path.exists(dst) else 0
    L("[cp] 单流 cp 完成 size=%d/%d" % (g, sz))
    return g == sz

if ensure_local(raw_jsonl, LOCAL_JSONL):
    raw_jsonl = LOCAL_JSONL
    L("[cp] ✅ 精炼输入切换为本地盘: %s" % raw_jsonl)
else:
    L("[cp] !! 本地拷贝不可用, 仍读 NAS (会很慢)")

# ---------- 3) 内存堆精炼 ----------
L("===== [3/3] in-memory heap refine (hard_filter + block_dedup + density top-K + style quota) =====")
BUDGET = 240_000_000
TOK_PER_HAN = 2.4589
CHUNK = 200
HMAX = 400_000
CJK = re.compile(r"[\u4e00-\u9fff]")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
RE_TPL = [
    re.compile(r"你是一位(?:网络)?(?:小说)?(?:专业)?(?:资深)?写手"),
    re.compile(r"你是一个(?:网络)?(?:小说)?(?:专业)?(?:资深)?写手"),
    re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|user\|>|<\|assistant\|>"),
    re.compile(r"^\s*(?:Human|Assistant|User|AI|System|GPT|ChatGPT)\s*[:：]", re.M),
    re.compile(r"^\s*#{1,4}\s*(?:指令|回答|问题|用户|助手|系统提示|prompt|Instruction|Response)\s*[:：]?\s*$", re.M),
    re.compile(r"^\s*\[INST\]|^\s*<<SYS>>", re.M),
    re.compile(r"作为一(?:个|名)(?:AI|人工智能|语言模型|助手)"),
    re.compile(r"我的(?:训练|知识截止)(?:数据|日期|时间)"),
    re.compile(r"^\s*继续写下去\s*[:：]", re.M),
    re.compile(r"^\s*续写\s*[:：]", re.M),
    re.compile(r"^\s*接上文\s*[:：]", re.M),
    re.compile(r"^\s*(?:上文|下集|未完)接续\s*[:：]", re.M),
]
RE_NOISE = [
    re.compile(r"扫码|长按识别|关注我们|点击关注|订阅号|公眾號|公众号ID|商务合作|转载请注明|版权归原作者"),
    re.compile(r"<!--|-->|<div|<p>|<br\s*/?>|&nbsp;|&lt;|&gt;|\[img\]|!\["),
    re.compile(r"<\s*/?\s*span\b|<\s*/?\s*style\b|font-family|font-size|</?(?:b|i|u|em|strong)\s*>", re.I),
    re.compile(r"https?://\S{6,}"),
    re.compile(r"(?:上一篇|下一篇|上一条|下一条|返回顶部|相关推荐|延伸阅读)\s*[:：]"),
]
RE_BOOK = re.compile(r"第[0-9一二三四五六七八九十百千零〇两]+[章節节回卷篇]|chapter\s*\d+", re.I)
SENT_END = "。！？…”』）】"
PUNC = frozenset("，。、！？；：""''（）《》—…·")
def book_key(t):
    first = RE_BOOK.split(t.split("\n")[0].strip())[0]
    first = re.sub(r"[（(].*?[)）]", "", first).strip()
    return first[:24] or "?"

def hard_filter(t):
    if not t: return "empty"
    if len(t) < 20: return "too_short_abs"
    if CTRL.search(t): return "ctrl_char"
    if "�" in t: return "mojibake"
    for r in RE_TPL:
        if r.search(t): return "template_residue"
    zh = len(CJK.findall(t))
    if zh / max(1, len(t)) < 0.35: return "not_chinese"
    return None

def strip_noise(t):
    keep = []
    for ln in t.split("\n"):
        s = ln.strip()
        if not s: continue
        if any(r.search(s) for r in RE_NOISE): continue
        if len(s) <= 3 and not CJK.search(s): continue
        keep.append(s)
    return "\n".join(keep)

def sample_chars(t, n=2000):
    if len(t) <= n*3: return t
    a=t[:n]; b=t[len(t)//2-n//2:len(t)//2+n//2]; c=t[-n:]
    return a+b+c

def score(t, pen_narr=0.0):
    n=len(t); s=sample_chars(t); ls=len(s)
    if ls==0: return 0.0
    zh=len(CJK.findall(s)); zh_r=zh/ls
    uniq=len(set(s))/ls
    uniq_s=1.0-min(1.0,max(0.0,(0.12-uniq)/0.12)) if uniq<0.12 else 1.0
    punc=sum(1 for c in s if c in PUNC)
    digit=sum(1 for c in s if c.isdigit() or c.isspace())
    noise=(punc+digit)/ls
    noise_s=1.0-min(1.0,max(0.0,(noise-0.12)/0.30))
    paras=t.count("\n")+1
    para_s=min(1.0,paras/6.0) if n>800 else 1.0
    tail=t.rstrip()[-1] if t.rstrip() else ""
    end_s=1.0 if tail in SENT_END else 0.45
    if 200 <= n <= 60000: len_s=1.0
    elif n<200: len_s=max(0.15,n/200)
    else: len_s=max(0.25,60000/n)
    top1=max((s.count(c) for c in set(s)),default=0)/ls
    rep_s=1.0-min(1.0,max(0.0,(top1-0.25)/0.35))
    base=(0.26*zh_r+0.22*uniq_s+0.16*noise_s+0.14*para_s+0.10*end_s+0.06*len_s+0.06*rep_s)
    if pen_narr>0:
        q=s.count("“")+s.count("”")+s.count('"')+s.count("'"); qr=q/ls
        if qr>0.02: base*=max(1.0-pen_narr,1.0-min(1.0,(qr-0.02)/0.10)*pen_narr)
    return base

def tok_of(t):
    return len(CJK.findall(t))*TOK_PER_HAN + (len(t)-len(CJK.findall(t)))*0.6

# 流式: 堆里只保留 top-HMAX (score 最小在堆顶)
heap = []                 # (score, seq, text, style)  min-heap by score
seen = set()              # 块级去重 digest
st = {"lines":0,"drop":{},"kept":0,"heap_max":0,"t0":time.time()}
seq = 0
with open(raw_jsonl, "r", encoding="utf-8", errors="replace") as fin:
    for line in fin:
        st["lines"] += 1
        line = line.strip()
        if not line: continue
        try:
            o = json.loads(line)
            t = (o.get("text") or "").strip() if isinstance(o, dict) else str(o).strip()
            sty = o.get("style", "") if isinstance(o, dict) else ""
        except Exception:
            t = line.strip().strip('"'); sty = ""
        why = hard_filter(t)
        if why:
            st["drop"][why] = st["drop"].get(why,0)+1; continue
        t = strip_noise(t)
        if len(t) < 200 or len(t) > 200000:
            st["drop"]["len_out_of_range"] = st["drop"].get("len_out_of_range",0)+1; continue
        n = len(t); blocks = [t[i:i+CHUNK] for i in range(0,n,CHUNK)]
        if len(blocks) >= 3:
            dup = 0
            digh = []
            for b in blocks:
                h = hashlib.sha1(b.encode("utf-8")).digest()[:8]
                digh.append(h)
                if h in seen: dup += 1
            if dup/len(blocks) > 0.6:
                st["drop"]["near_dup_doc"] = st["drop"].get("near_dup_doc",0)+1; continue
            if dup > 0:
                kept = []
                for b, h in zip(blocks, digh):
                    if h not in seen: seen.add(h); kept.append(b)
                t = "".join(kept)
                if len(t) < 200:
                    st["drop"]["emptied_by_dedup"] = st["drop"].get("emptied_by_dedup",0)+1; continue
            else:
                for h in digh: seen.add(h)
        else:
            for b in blocks: seen.add(hashlib.sha1(b.encode("utf-8")).digest()[:8])
        sc = score(t, 0.0)
        sty = sty or "unknown"
        entry = (sc, seq, t, sty); seq += 1
        if len(heap) < HMAX:
            heapq.heappush(heap, entry)
        else:
            heapq.heappushpop(heap, entry)
        st["kept"] += 1
        st["heap_max"] = max(st["heap_max"], len(heap))
        if st["lines"] % 500000 == 0:
            L("  ...%dk 行, 入堆 %d, 堆=%d, %.0fs" % (st["lines"]//1000, st["kept"], len(heap), time.time()-st["t0"]))
L("[cv] Pass1 完成: 读 %d 行, 入堆 %d, 堆峰值 %d (%.0fs)" % (st["lines"], st["kept"], st["heap_max"], time.time()-st["t0"]))

# top-K 蒸馏 + 按风格配额
st["kept"] = len(heap)
order = sorted(range(len(heap)), key=lambda i: heap[i][0], reverse=True)
picked = []; tok = 0.0; by = 0; per_style = {}; STYLE_CAP = BUDGET * 0.4
for i in order:
    if tok >= BUDGET: break
    sc, _, t, sty = heap[i]
    if per_style.get(sty, 0.0) >= STYLE_CAP: continue
    picked.append((t, sty)); tk = tok_of(t)
    tok += tk; by += len(t.encode("utf-8"))+1
    per_style[sty] = per_style.get(sty, 0.0) + tk

outf = os.path.join(OUT_DIR, "creative_articles_l3.jsonl")
with open(outf, "w", encoding="utf-8") as fo:
    for t, sty in picked:
        fo.write(json.dumps({"text": t, "src": "l3_ms", "style": sty}, ensure_ascii=False) + "\n")
L("[cv] 蒸馏: %d/%d 条入选, %.1fMB, ≈%.2fM token (预算 %.0fM)" % (len(picked), st["kept"], by/1e6, tok/1e6, BUDGET/1e6))
L("[cv] 风格分布: " + ", ".join("%s≈%.0fM" % (k, v/1e6) for k, v in sorted(per_style.items(), key=lambda x:-x[1])))
L("[cv] 丢弃原因(前8): " + (", ".join("%s=%d" % (k, v) for k, v in sorted(st["drop"].items(), key=lambda x:-x[1])[:8]) or "无"))
L("===== l3_creative_fill DONE -> %s (%.1fMB) =====" % (outf, os.path.getsize(outf)/1e6))

report = ("l3_creative_fill v3 (in-memory heap) 报告\n"
          "输入: L3 zh multi_style 复用 jsonl %.1fMB / %d 行\n"
          "硬过滤后入堆: %d 条\n"
          "蒸馏入选: %d 条, 输出 %.1fMB\n"
          "≈%.2fM token (目标 %.0fM)\n"
          "风格分布: %s\n"
          "丢弃(前8): %s\n"
          "输出: %s\n"
          % (os.path.getsize(raw_jsonl)/1e6, st["lines"], st["kept"], len(picked),
             os.path.getsize(outf)/1e6, tok/1e6, BUDGET/1e6,
             ", ".join("%s≈%.0fM" % (k, v/1e6) for k, v in sorted(per_style.items(), key=lambda x:-x[1])),
             ", ".join("%s=%d" % (k, v) for k, v in sorted(st["drop"].items(), key=lambda x:-x[1])[:8]),
             outf))
try:
    subprocess.run(["curl","-s","-X","POST","%s/upload?name=creative_fill_report_v3.txt&key=%s"%(ECS,KEY),
                    "--data-binary",report.encode("utf-8")],timeout=20)
    L("[upload] report_v3 ok")
except Exception as e:
    L("[upload] report_v3 fail: %s"%e)
