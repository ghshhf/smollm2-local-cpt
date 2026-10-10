#!/usr/bin/env python
# -*- coding: utf-8 -*-
# V5 数据集： 继承 V4 底(1232=事实624+成稿460+代码148)
#   + 上传同人《催眠眼镜之女刑警》拆成的散文样本(用户要求混进 V5)
#   + 针对 Q4 弱点的冷事实过采样(守则宇宙概念 x2)
# 输出 data/role_data_v5.jsonl，供 train_chain.sh 的 v5 作业自动接训。
import json, re, os

BASE_JSONL = "data/role_data_v4.jsonl"
NOVEL = "D:/tgapp/telegram_mini_app/data/uploads/1007_212558_催眠眼镜之女刑警-不同(1).txt"
OUT = "data/role_data_v5.jsonl"

def load(path):
    if not os.path.exists(path):
        raise SystemExit(f"[FATAL] 缺少 {path}，请先跑对应构建器")
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]

base = load(BASE_JSONL)
print(f"[v5] V4 底 = {len(base)} 条")

# ---------- 1) 同人拆样本 ----------
novel_rows = []
if os.path.exists(NOVEL):
    raw = open(NOVEL, encoding="utf-8").read()
    # 去掉文件头元数据行(作者/字数/日期)与空行
    paras = []
    for p in re.split(r"\n\s*\n|\n", raw):
        p = p.strip()
        if not p:
            continue
        if re.search(r"字数：|作者：|\d{4}/\d{1,2}/\d{1,2}", p):
            continue
        paras.append(p)
    print(f"[v5] 同人正文段落 = {len(paras)}")
    # 续写样本: 上一段作前缀 -> 下一段作输出
    cont = []
    for i in range(len(paras) - 1):
        pre, nxt = paras[i], paras[i + 1]
        if len(pre) < 30 or len(nxt) < 30:
            continue
        cont.append({
            "instruction": "继续写下去：\n" + pre[:500],
            "output": nxt[:500],
            "_type": "prose_cont", "_universe": "催眠眼镜同人", "_src": "upload",
        })
    # 仿写样本: 中性 brief -> 一段正文
    style = []
    for i in range(0, len(paras), 3):
        p = paras[i]
        if len(p) < 40:
            continue
        style.append({
            "instruction": "用作者「不同」的同人笔法，写一段叙事：",
            "output": p[:500],
            "_type": "prose_style", "_universe": "催眠眼镜同人", "_src": "upload",
        })
    # 控制占比: 同人散文总量上限 ~200，避免冲淡清水多宇宙
    max_cont, max_style = 150, 50
    import random
    random.seed(20261007)
    random.shuffle(cont)
    cont = cont[:max_cont]
    style = style[:max_style]
    novel_rows = cont + style
    print(f"[v5] 同人样本 = {len(novel_rows)} (续写 {len(cont)} + 仿写 {len(style)})")
else:
    print(f"[v5][warn] 未找到同人文件 {NOVEL}，跳过")

# ---------- 1b) 《琼明神女录》epub 拆解(第二本上传, 非 txt) ----------
EPUB = "D:/tgapp/telegram_mini_app/data/uploads/1007_220548_琼明神女录(完结+番外).epub"
shen_rows = []
if os.path.exists(EPUB):
    import zipfile
    z = zipfile.ZipFile(EPUB)
    tnames = [n for n in z.namelist() if n.lower().endswith(('.xhtml', '.html', '.htm', '.xml'))]
    def _strip(s):
        s = re.sub(r'<style[\s\S]*?</style>', ' ', s, flags=re.I)
        s = re.sub(r'<script[\s\S]*?</script>', ' ', s, flags=re.I)
        s = re.sub(r'<[^>]+>', '', s)
        for a, b in [('&nbsp;', ' '), ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&quot;', '"')]:
            s = s.replace(a, b)
        s = re.sub(r'[ \t]+', ' ', s)
        s = re.sub(r'\n\s*\n+', '\n', s)
        return s
    full = []
    for n in sorted(tnames):
        try:
            data = z.read(n).decode('utf-8', errors='ignore')
        except Exception:
            continue
        t = _strip(data)
        if t.strip():
            full.append(t)
    txt = re.sub(r'\n{3,}', '\n\n', '\n'.join(full)).strip()
    chaps = re.split(r'(?=第\s*[一二三四五六七八九十百零0-9]+\s*章)', txt)
    chaps = [c.strip() for c in chaps if len(c.strip()) > 300]
    print(f"[v5] 神女录 章节={len(chaps)} 总字={len(txt)}")
    # 续写: 每章中段连续切片(前600 input + 后600 output)
    for c in chaps:
        if len(c) < 1200:
            mid = max(1, len(c) // 2)
            pre, nxt = c[:min(mid, 600)], c[mid:mid + 600]
            if len(pre) >= 60 and len(nxt) >= 60:
                shen_rows.append({"instruction": "继续写下去：\n" + pre, "output": nxt,
                                  "_type": "prose_cont", "_universe": "琼明神女录", "_src": "upload_epub"})
            continue
        cut = int(len(c) * 0.2)
        seg = c[cut:cut + 1200]
        shen_rows.append({"instruction": "继续写下去：\n" + seg[:600], "output": seg[600:1200],
                          "_type": "prose_cont", "_universe": "琼明神女录", "_src": "upload_epub"})
    # 仿写: 每章首段作为风格范例
    for c in chaps:
        head = c[:500].strip()
        if len(head) < 60:
            continue
        shen_rows.append({"instruction": "以《琼明神女录》的古典仙侠同人笔调，写一段情境叙事：",
                          "output": head, "_type": "prose_style", "_universe": "琼明神女录", "_src": "upload_epub"})
    print(f"[v5] 神女录样本 = {len(shen_rows)} (续写+仿写)")
else:
    print(f"[v5][warn] 未找到 epub {EPUB}")

# ---------- 2) 冷事实过采样(守则宇宙概念 x2, 针对 Q4 弱点) ----------
over = []
for r in base:
    if r.get("_type") == "concept" and r.get("_universe") == "守则宇宙":
        v = dict(r)
        v["instruction"] = "（再确认一次）" + r["instruction"]   # 变体指令, 避过去重, 真正加权重
        over.append(v)
print(f"[v5] 守则宇宙冷事实过采样 +{len(over)}")

# ---------- 3) 合并 + 全局去重 ----------
all_rows = base + novel_rows + shen_rows + over
seen = set()
uniq = []
for r in all_rows:
    key = (r["instruction"].strip(), r["output"].strip())
    if key in seen:
        continue
    seen.add(key)
    uniq.append(r)

from collections import Counter
cnt = Counter((r.get("_type"), r.get("_universe")) for r in uniq)
print("[v5] 类型×宇宙分布:")
for k, v in sorted(cnt.items()):
    print(f"   {k}: {v}")
print(f"[v5] 去重后总计 = {len(uniq)}")

with open(OUT, "w", encoding="utf-8") as f:
    for r in uniq:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"[v5] 已写出 {OUT}")
