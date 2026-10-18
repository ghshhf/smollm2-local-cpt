#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
refine_reserve.py — 0.3B 储备语料的「数据蒸馏」精炼层

定位：build_reserve.py 只做原始拼装（零清洗/零去重/零筛选），本脚本是它下游的精炼层。
思路 = 数据蒸馏(data distillation)：语料相对每档 budget 严重过剩(十几倍)，
      与其让训练器顺序吃掉前面随机的一截，不如按「信息密度」蒸馏出高密子集喂进去。

原则（宪法级，勿改）：
  ★ 只洗「污染」与「水分」，绝不洗「内容」——不删成人/敏感内容
    （用户 2026-10-14：模型退化源于知识太少，解法是加覆盖不是审查）
  ★ 不用自家小模型给语料打分（自举偏见），只用规则+统计

流程(每档)：
  Pass0 硬过滤(行级)   : prompt 模板残留 / HTML / 广告导航 / Gutenberg 许可证样板 / 语言不符 / 长度越界
  Pass1 块级去重       : 200字块 sha1，>60% 块已见则整篇丢，否则只保留独有块（去转载，不浪费独有部分）
  Pass2 信息密度打分   : 汉字占比 / 唯一字率(采样) / 噪声率 / 结构完整度 / 长度贴合度
  Pass3 蒸馏取 top N   : 按分降序累计 token 到预算(步数×4096×冗余)为止
  Pass4 出报告         : 各阶段条数/字节/丢弃原因分布/样例 diff

用法:
  python3 refine_reserve.py                # 精炼全部四档
  python3 refine_reserve.py ancient        # 只精炼指定档(knowledge/ancient/novels/weixin)
环境变量:
  TOKEN_BUDGET  每档目标 token(默认 18_400_000 = 1500步×4096×3 冗余)
  RAW_DIR       输入目录(默认 /mnt/workspace/node1/chain/corpus_reserve)
  OUT_DIR       输出目录(默认 .../corpus_reserve_refined)
  PUBLISH       1=验证后覆盖回 RAW_DIR 并备份 .raw(默认 0)
"""
import os, re, sys, json, time, hashlib, shutil
from array import array

RAW_DIR = os.environ.get("RAW_DIR", "/mnt/workspace/node1/chain/corpus_reserve")
OUT_DIR = os.environ.get("OUT_DIR", "/mnt/workspace/node1/chain/corpus_reserve_refined")
TOKEN_BUDGET = int(os.environ.get("TOKEN_BUDGET", "18400000"))   # 1500步×4096×3
PUBLISH = os.environ.get("PUBLISH", "0") == "1"
CHUNK = 200                      # 块级去重窗口(字)
TOK_PER_HAN = 2.4589             # 实测 token/汉字 (eval/vocab_add_gain.py)

# 每档的输入/输出文件名与长度门槛
#   lo/hi      : 通过硬过滤的字符数区间
#   best_lo/hi : 打分时长贴合度满分区间
#   quota      : ★多样性配额——同一"书名/来源"最多入选条数。
#                 不加的话实测 novels 1887 条只覆盖 186 本书、前三本占 54%，模型会被几本书灌满。
#   pen_narr   : 叙事惩罚强度(0=不惩罚)。knowledge 档开 0.35，让百科型压过写手续写流水账。
#   budget     : 该档单独的 token 预算(缺省用全局 TOKEN_BUDGET)。
SPEC = {
    # R7 知识 backbone —— 2026-10-15 重建: 源换成中文维基百科(干净百科型知识)。
    # 旧源 zh_r_knowledge.jsonl 是 R6 复制, 其"教育"部分实为《閱微草堂筆記》(文言笔记小说),
    # 导致抽样全是水浒名单/三言二拍。新源 build_wiki_knowledge.py 从 hf-mirror 直抓。
    # 知识是 backbone, 预算从 24.6M 提到 55M token("更加填充知识")。
    "knowledge": {"out": "zh_r_knowledge.jsonl", "in": ["zh_wiki_all.jsonl", "zh_r_knowledge.jsonl"],
                  "lo": 40, "hi": 200000, "best_lo": 800, "best_hi": 60000,
                  "quota": 40, "pen_narr": 0.35, "budget": 55000000},
    "ancient":   {"out": "zh_r_ancient.jsonl",   "in": ["zh_r_ancient.jsonl"],
                  "lo": 30, "hi": 200000, "best_lo": 200, "best_hi": 40000,
                  "quota": 60, "pen_narr": 0.0, "budget": 24600000},
    "novels":    {"out": "zh_r_novels.jsonl",    "in": ["zh_r_novels.jsonl"],
                  "lo": 200, "hi": 400000, "best_lo": 1500, "best_hi": 100000,
                  "quota": 30, "pen_narr": 0.0, "budget": 24600000},
    # R10: build_reserve 写的是 zh_r_modern.jsonl, chain_v2.sh 要的是 zh_r_weixin.jsonl -> 两名字都收
    "weixin":    {"out": "zh_r_weixin.jsonl",    "in": ["zh_r_weixin.jsonl", "zh_r_modern.jsonl"],
                  "lo": 60, "hi": 200000, "best_lo": 600, "best_hi": 30000,
                  "quota": 30, "pen_narr": 0.0, "budget": 24600000},
}

RE_BOOKSPLIT = re.compile(r"第[0-9一二三四五六七八九十百千零〇两]+[章節节回卷篇]|"
                          r"chapter\s*\d+", re.I)


def book_key(t):
    """书名指纹——用于多样性配额(按首行去章节号提取, 廉价启发式)"""
    first = t.split("\n")[0].strip()
    first = RE_BOOKSPLIT.split(first)[0]
    first = re.sub(r"[（(].*?[)）]", "", first).strip()
    return first[:24] or "?"

# ---------- 硬过滤规则(行级/篇级) ----------
# 1) 数据集指令模板残留 —— 对话/SFT 数据混进预训练池，会让模型输出"你是一位小说写手…"
RE_TEMPLATE = [
    re.compile(r"你是一位(?:网络)?(?:小说)?(?:专业)?(?:资深)?写手"),
    re.compile(r"你是一个(?:网络)?(?:小说)?(?:专业)?(?:资深)?写手"),
    re.compile(r"<\|im_start\|>|<\|im_end\|>|<\|endoftext\|>|<\|user\|>|<\|assistant\|>"),
    re.compile(r"^\s*(?:Human|Assistant|User|AI|System|GPT|ChatGPT)\s*[:：]", re.M),
    re.compile(r"^\s*#{1,4}\s*(?:指令|回答|问题|用户|助手|系统提示|prompt|Instruction|Response)\s*[:：]?\s*$", re.M),
    re.compile(r"^\s*\[INST\]|^\s*<<SYS>>", re.M),
    re.compile(r"作为一(?:个|名)(?:AI|人工智能|语言模型|助手)"),
    re.compile(r"我的(?:训练|知识截止)(?:数据|日期|时间)"),
    # 写手续写流水账的指令痕迹(实测 knowledge 档被"继续写下去：王杰…"这类灌满)
    re.compile(r"^\s*继续写下去\s*[:：]", re.M),
    re.compile(r"^\s*续写\s*[:：]", re.M),
    re.compile(r"^\s*接上文\s*[:：]", re.M),
    re.compile(r"^\s*(?:上文|下集|未完)接续\s*[:：]", re.M),
]
# 2) 网页/广告/运营噪声
RE_NOISE = [
    re.compile(r"扫码|长按识别|关注我们|点击关注|订阅号|公眾號|公众号ID|商务合作|转载请注明|版权归原作者"),
    re.compile(r"<!--|-->|<div|<p>|<br\s*/?>|&nbsp;|&lt;|&gt;|\[img\]|!\["),
    # 实测 ancient 档有 "<span style= font-family:" 这类连写残留
    re.compile(r"<\s*/?\s*span\b|<\s*/?\s*style\b|font-family|font-size|</?(?:b|i|u|em|strong)\s*>", re.I),
    re.compile(r"https?://\S{6,}"),
    re.compile(r"(?:上一篇|下一篇|上一条|下一条|返回顶部|相关推荐|延伸阅读)\s*[:：]"),
]
# 3) Project Gutenberg 许可证样板(整段)
RE_GUTEN = [
    re.compile(r"The Project Gutenberg eBook of"),
    re.compile(r"\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG EBOOK"),
    re.compile(r"\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG EBOOK"),
    re.compile(r"www\.gutenberg\.org"),
    re.compile(r"Project Gutenberg License"),
    re.compile(r"Most people start at our Web site"),
]
CJK = re.compile(r"[\u4e00-\u9fff]")
CTRL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SENT_END = "。！？…”』）】"


def L(m):
    print(m, flush=True)


def hard_filter(t):
    """返回 None 表示通过, 否则返回丢弃原因"""
    if not t:
        return "empty"
    if len(t) < 20:
        return "too_short_abs"
    if CTRL.search(t):
        return "ctrl_char"
    if "�" in t:
        return "mojibake"
    for r in RE_TEMPLATE:
        if r.search(t):
            return "template_residue"
    for r in RE_GUTEN:
        if r.search(t):
            return "gutenberg_boilerplate"
    zh = len(CJK.findall(t))
    if zh / max(1, len(t)) < 0.35:
        return "not_chinese"
    return None


def strip_noise_lines(t):
    """行级外科切除：只删污染行，保留其余正文(不整篇丢弃)"""
    keep = []
    for ln in t.split("\n"):
        s = ln.strip()
        if not s:
            continue
        bad = False
        for r in RE_NOISE:
            if r.search(s):
                bad = True
                break
        if bad:
            continue
        # 纯数字/纯符号/极短碎片行
        if len(s) <= 3 and not CJK.search(s):
            continue
        keep.append(s)
    return "\n".join(keep)


def sample_chars(t, n=2000):
    """三段采样，避免对超长文本全扫(性能)"""
    if len(t) <= n * 3:
        return t
    a = t[:n]
    b = t[len(t) // 2 - n // 2: len(t) // 2 + n // 2]
    c = t[-n:]
    return a + b + c


def score(t, best_lo, best_hi, pen_narr=0.0):
    """信息密度打分 0~1 —— 蒸馏的核心排序依据(纯统计, 不用模型)"""
    n = len(t)
    s = sample_chars(t)
    ls = len(s)
    if ls == 0:
        return 0.0

    zh = len(CJK.findall(s))
    zh_r = zh / ls

    # 唯一字率 = 词汇丰富度/信息密度代理(重复刷屏的文本此项极低)
    uniq = len(set(s)) / ls
    uniq_s = 1.0 - min(1.0, max(0.0, (0.12 - uniq) / 0.12)) if uniq < 0.12 else 1.0

    # 噪声：标点+数字+空白占比过高 => 结构化垃圾/表格残留
    punc = sum(1 for c in s if c in "，。、！？；：""''（）《》—…·")
    digit = sum(1 for c in s if c.isdigit() or c.isspace())
    noise = (punc + digit) / ls
    noise_s = 1.0 - min(1.0, max(0.0, (noise - 0.12) / 0.30))

    # 结构完整度：分段落 + 句子有终止符
    paras = t.count("\n") + 1
    para_s = min(1.0, paras / 6.0) if n > 800 else 1.0
    tail = t.rstrip()[-1] if t.rstrip() else ""
    end_s = 1.0 if tail in SENT_END else 0.45

    # 长度贴合度：落在 best 区间满分, 越界衰减
    if best_lo <= n <= best_hi:
        len_s = 1.0
    elif n < best_lo:
        len_s = max(0.15, n / best_lo)
    else:
        len_s = max(0.25, best_hi / n)

    # 单字刷屏检测(如"哈哈哈…"或整段同一字符)
    top1 = max((s.count(c) for c in set(s)), default=0) / ls
    rep_s = 1.0 - min(1.0, max(0.0, (top1 - 0.25) / 0.35))

    base = (0.26 * zh_r + 0.22 * uniq_s + 0.16 * noise_s
            + 0.14 * para_s + 0.10 * end_s + 0.06 * len_s + 0.06 * rep_s)

    # 叙事/对话体惩罚：引号密度越高越像小说对话, 知识档要压它让百科型优先。
    # 注意——这是"体例"判断不是"内容"判断, 不涉及任何主题审查。
    if pen_narr > 0:
        q = s.count("\u201c") + s.count("\u201d") + s.count('"') + s.count("'")
        qr = q / ls
        if qr > 0.02:
            base *= max(1.0 - pen_narr, 1.0 - min(1.0, (qr - 0.02) / 0.10) * pen_narr)
    return base


def est_tokens(t):
    return len(CJK.findall(t)) * TOK_PER_HAN + (len(t) - len(CJK.findall(t))) * 0.6


def refine(key, spec):
    src = None
    for cand in spec["in"]:
        p = os.path.join(RAW_DIR, cand)
        if os.path.exists(p) and os.path.getsize(p) > 0:
            src = p
            break
    if src is None:
        L(f"[{key}] !! 无源文件 (试过 {spec['in']}) —— 跳过")
        return None
    os.makedirs(OUT_DIR, exist_ok=True)
    candf = os.path.join(OUT_DIR, "." + spec["out"] + ".cand")
    outf = os.path.join(OUT_DIR, spec["out"])

    L(f"[{key}] 源={src} ({os.path.getsize(src)/1e6:.1f}MB)")

    st = {"lines": 0, "bytes_in": 0, "drop": {},
          "chunk_dup_doc": 0, "chunk_trim": 0, "cand": 0, "bytes_cand": 0}
    seen = set()          # 块哈希
    lens = array("i"); scos = array("f"); chrs = array("i"); bks = array("q")
    quota = int(spec.get("quota", 0))
    pen_narr = float(spec.get("pen_narr", 0.0))
    budget = int(spec.get("budget", TOKEN_BUDGET))
    t0 = time.time()

    with open(src, "r", encoding="utf-8", errors="replace") as fin, \
         open(candf, "w", encoding="utf-8") as fc:
        for line in fin:
            st["lines"] += 1
            st["bytes_in"] += len(line)
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
                t = (o.get("text") or "").strip() if isinstance(o, dict) else str(o).strip()
            except Exception:
                # 纯文本行也认(容错)
                t = line.strip().strip('"')

            why = hard_filter(t)
            if why:
                st["drop"][why] = st["drop"].get(why, 0) + 1
                continue

            t = strip_noise_lines(t)
            if len(t) < spec["lo"] or len(t) > spec["hi"]:
                st["drop"]["len_out_of_range"] = st["drop"].get("len_out_of_range", 0) + 1
                continue

            # ---- 块级去重：去掉转载重复块, 保留独有块 ----
            n = len(t)
            blocks = [t[i:i + CHUNK] for i in range(0, n, CHUNK)]
            if len(blocks) >= 3:
                dup = 0
                for b in blocks:
                    if hashlib.sha1(b.encode("utf-8")).digest()[:8] in seen:
                        dup += 1
                if dup / len(blocks) > 0.6:
                    st["chunk_dup_doc"] += 1
                    st["drop"]["near_dup_doc"] = st["drop"].get("near_dup_doc", 0) + 1
                    continue
                if dup > 0:
                    st["chunk_trim"] += 1
                    kept = []
                    for b in blocks:
                        h = hashlib.sha1(b.encode("utf-8")).digest()[:8]
                        if h not in seen:
                            seen.add(h)
                            kept.append(b)
                    t = "".join(kept)
                    if len(t) < spec["lo"]:
                        st["drop"]["emptied_by_dedup"] = st["drop"].get("emptied_by_dedup", 0) + 1
                        continue
            else:
                for b in blocks:
                    seen.add(hashlib.sha1(b.encode("utf-8")).digest()[:8])

            sc = score(t, spec["best_lo"], spec["best_hi"], pen_narr)
            b = t.encode("utf-8")
            fc.write(json.dumps({"text": t, "src": "refined", "cat": key},
                                ensure_ascii=False) + "\n")
            lens.append(len(b)); scos.append(sc); chrs.append(len(CJK.findall(t)))
            bks.append(hash(book_key(t)) & 0x7FFFFFFFFFFFFFFF)
            st["cand"] += 1
            st["bytes_cand"] += len(b) + 1

            if st["lines"] % 200000 == 0:
                L(f"  [{key}] ...{st['lines']//1000}k 行, 候选 {st['cand']}, "
                  f"{time.time()-t0:.0f}s")

    L(f"[{key}] Pass1 完成: 候选 {st['cand']} 条 / {st['bytes_cand']/1e6:.1f}MB "
      f"({time.time()-t0:.0f}s), 块去重整篇丢 {st['chunk_dup_doc']}, 部分裁剪 {st['chunk_trim']}")

    if st["cand"] == 0:
        L(f"[{key}] !! 候选为空, 跳过")
        return None

    # ---- Pass3 蒸馏：按分降序取 top 直到 token 预算 ----
    order = sorted(range(st["cand"]), key=lambda i: scos[i], reverse=True)
    picked, tok, by = [], 0, 0
    per, skipped_q = {}, 0
    for i in order:
        if tok >= budget:
            break
        if quota:                       # ★多样性配额: 同一本书不灌满整档
            b = bks[i]
            if per.get(b, 0) >= quota:
                skipped_q += 1
                continue
            per[b] = per.get(b, 0) + 1
        picked.append(i)
        tok += chrs[i] * TOK_PER_HAN + lens[i] * 0.6
        by += lens[i] + 1
    if quota:
        st["books"] = len(per)
        st["quota_skipped"] = skipped_q
    global picked_set
    picked_set = set(picked)

    with open(candf, "r", encoding="utf-8") as fc, \
         open(outf, "w", encoding="utf-8") as fo:
        cur = 0
        for line in fc:
            if cur in picked_set:
                fo.write(line)
            cur += 1
    os.remove(candf)

    L(f"[{key}] 蒸馏: {len(picked)}/{st['cand']} 条入选, "
      f"{by/1e6:.1f}MB, ≈{tok/1e6:.2f}M token (预算 {budget/1e6:.0f}M)")
    if quota:
        L(f"[{key}] 多样性: 覆盖 {st.get('books',0)} 个来源, 配额={quota}, "
          f"因超额跳过 {st.get('quota_skipped',0)} 条")
    L(f"[{key}] 丢弃原因: " + (", ".join(f"{k}={v}" for k, v in
                                        sorted(st["drop"].items(), key=lambda x: -x[1])) or "无"))
    return {"key": key, "out": outf, "bytes": os.path.getsize(outf),
            "picked": len(picked), "cand": st["cand"],
            "bytes_in": st["bytes_in"], "drop": st["drop"], "tok": tok}


def main():
    keys = sys.argv[1:] or list(SPEC.keys())
    L("===== refine_reserve start @ " + time.strftime("%Y-%m-%d %H:%M")
      + f" (budget={TOKEN_BUDGET/1e6:.0f}M token/档, PUBLISH={int(PUBLISH)}) =====")
    res = []
    for k in keys:
        if k not in SPEC:
            L(f"[skip] 未知档位 {k}")
            continue
        try:
            r = refine(k, SPEC[k])
            if r:
                res.append(r)
        except Exception as e:
            L(f"[{k}] !! 异常: {type(e).__name__}: {e}")

    L("===== 精炼结果 =====")
    for r in res:
        L(f"  {r['out']}: {r['bytes']/1e6:.1f}MB  {r['picked']} 条 "
          f"(候选 {r['cand']}, 源 {r['bytes_in']/1e6:.1f}MB)  ≈{r['tok']/1e6:.2f}M token")

    if PUBLISH and res:
        L("===== PUBLISH: 覆盖回 corpus_reserve (原文件备份为 .raw) =====")
        for r in res:
            dst = os.path.join(RAW_DIR, os.path.basename(r["out"]))
            if os.path.exists(dst) and not os.path.exists(dst + ".raw"):
                shutil.move(dst, dst + ".raw")
            shutil.copy2(r["out"], dst)
            L(f"  -> {dst} ({os.path.getsize(dst)/1e6:.1f}MB)")

    # 报告落盘
    rp = os.path.join(OUT_DIR, "refine_report.txt")
    with open(rp, "w", encoding="utf-8") as f:
        for r in res:
            f.write(f"{r['key']}: {os.path.basename(r['out'])} {r['bytes']/1e6:.1f}MB "
                    f"{r['picked']}/{r['cand']} 条 ≈{r['tok']/1e6:.2f}M token\n")
            for k, v in sorted(r["drop"].items(), key=lambda x: -x[1]):
                f.write(f"    drop {k}={v}\n")
    L("===== refine_reserve done =====")


picked_set = set()      # membership 加速(在 refine 内赋值)

if __name__ == "__main__":
    main()
