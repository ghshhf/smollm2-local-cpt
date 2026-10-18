#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""paper_digest.py — 前沿论文「AI 预消化」：英文摘要 -> 中文四行卡。

背景: 爬虫每轮新增 200+ 篇(arXiv 12 类/HF/OpenAlex 中英), 全部烂在
by_category 仓库里没人读。本工具用免 key 端点(pollinations)当阅读引擎,
把每篇消化成中文卡: 问题/方法/结论/对本项目的借鉴点, 落
data/frontier/digest/<桶>.md。增量处理(记账 _digested.json), 可反复跑。

用法:
  python tools/paper_digest.py                 # 处理最多 25 篇新论文
  python tools/paper_digest.py --limit 50      # 加大批量
  python tools/paper_digest.py --retry-fail    # 重试往日失败的
  python tools/paper_digest.py --dry-run       # 只列待消化清单

schtasks: PaperDigest 30min/轮(与爬虫错开 10 分钟)。
端点失联时不写记账(下轮自动重试), 不阻塞其他自动腿。
"""

import argparse
import glob
import json
import os
import sys
import time
import urllib.parse
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_GLOB = os.path.join(ROOT, "data", "frontier", "by_category", "*.jsonl")
DST_DIR = os.path.join(ROOT, "data", "frontier", "digest")
LEDGER = os.path.join(DST_DIR, "_digested.json")

ENDPOINT = "https://text.pollinations.ai/"
TIMEOUT = 40
SLEEP_BETWEEN = 1.2          # 礼貌限速, 免费端点
MIN_ABSTRACT = 120           # 摘要太短的(通知/占位)不消化

PROMPT_TMPL = (
    "下面是一篇AI论文的标题和摘要。请用简体中文严格输出4行, 不要任何额外解释:\n"
    "问题: <它解决什么问题, 一句话>\n"
    "方法: <核心方法, 一句话>\n"
    "结论: <关键结果/结论, 一句话>\n"
    "借鉴: <对一个'本地小模型自进化(做梦闭环/三元组银行/适配器自训)/记忆层/"
    "模块化挂载/测试时学习'项目的可借鉴点, 一句话; 没有就写'无'>\n"
    "标题: {title}\n"
    "摘要: {abstract}"
)


def load_ledger():
    if os.path.exists(LEDGER):
        try:
            return json.load(open(LEDGER, encoding="utf-8"))
        except Exception:
            pass
    return {"ids": [], "fails": []}


def save_ledger(lg):
    os.makedirs(DST_DIR, exist_ok=True)
    json.dump(lg, open(LEDGER, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


def ask(prompt, retries=2):
    """调 pollinations 免 key 端点(GET 明文)。返回 None=失败。同篇内重试。"""
    url = ENDPOINT + urllib.parse.quote(prompt[:3000])
    for k in range(retries + 1):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "paper-digest/1.0"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                body = r.read().decode("utf-8", "replace").strip()
            if len(body) >= 20 and "error" not in body[:80].lower():
                return body
        except Exception:
            pass
        if k < retries:
            time.sleep(2 + k * 3)     # 限流退避
    return None


def fmt_card(paper, zh):
    lines = [
        "### %s" % paper.get("title", "?"),
        "- %s | %s | %s" % (paper.get("date", ""), paper.get("src", ""),
                            paper.get("url", "")),
    ]
    for ln in zh.splitlines():
        ln = ln.strip()
        if ln:
            lines.append("- **%s**" % ln)
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--retry-fail", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    lg = load_ledger()
    done = set(lg["ids"])

    # 收集候选(按桶分组保持输出顺序)
    todo = []
    for f in sorted(glob.glob(SRC_GLOB)):
        if "_index" in f:
            continue
        cat = os.path.basename(f).replace(".jsonl", "")
        for ln in open(f, encoding="utf-8"):
            ln = ln.strip()
            if not ln:
                continue
            try:
                p = json.loads(ln)
            except Exception:
                continue
            pid = "%s:%s" % (p.get("src", "?"), p.get("id", "?"))
            if pid in done:
                continue
            if len(p.get("abstract") or "") < MIN_ABSTRACT:
                done.add(pid)          # 太薄, 标记跳过不重试
                continue
            todo.append((cat, pid, p))

    if a.dry_run:
        print("[digest] 待消化 %d 篇(已消化 %d)" % (len(todo), len(done)))
        from collections import Counter
        for c, n in Counter(c for c, _, _ in todo).most_common():
            print("  %-24s %d" % (c, n))
        save_ledger(lg)
        return 0

    if not todo:
        print("[digest] 无新论文(已消化 %d)" % len(done))
        return 0

    todo = todo[:a.limit]
    print("[digest] 本轮处理 %d 篇" % len(todo))
    ok, fail = 0, 0
    for cat, pid, p in todo:
        zh = ask(PROMPT_TMPL.format(title=p.get("title", ""),
                                    abstract=p.get("abstract", "")[:2200]))
        if zh is None:
            fail += 1
            lg["fails"].append(pid)
            print("  [fail] %s" % pid)
            time.sleep(3)              # 端点可能限流, 退避
            continue
        out = os.path.join(DST_DIR, "%s.md" % cat)
        os.makedirs(DST_DIR, exist_ok=True)
        with open(out, "a", encoding="utf-8") as f:
            f.write(fmt_card(p, zh))
        done.add(pid)
        ok += 1
        print("  [ok] %s | %s" % (pid, p.get("title", "")[:60]))
        time.sleep(SLEEP_BETWEEN)

    lg["ids"] = sorted(done)
    lg["fails"] = sorted(set(lg["fails"]) - done)
    save_ledger(lg)
    print("[digest] 完成 ok=%d fail=%d 累计=%d" % (ok, fail, len(done)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
