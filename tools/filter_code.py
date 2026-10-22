#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/filter_code.py — 代码预训练资源筛选器。

输入 data/code_raw/*（原始指令对 + 仓库原始代码），输出两份**能直接喂训练**的语料：

    corpus_04b/code_instruct.jsonl   {"text": "问题\n\n```lang\n代码\n```"}   指令→代码
    corpus_04b/code_raw.jsonl        {"text": "原始代码块"}                   纯代码

筛选规则（全部可复核，不靠主观判断）：
  1. 长度：指令 4~800 字，代码 60~6000 字
  2. 必须真有代码：围栏块 或 语言关键字（def/class/function/import/#include/SELECT/fn/public…）
  3. 结构：代码行占比 >= 0.45（挡掉"我建议你……"这种散文）
  4. 反常：拒绝话术、单行复读、乱码（替换字符）、超长单行
  5. 去重：md5 精确去重 + 64bit simhash 近似去重（汉明距离 <= 3）
  6. 语言：识别主语言，统计分布（Python/JS/TS/Java/Go/C++/Rust/SQL/Shell/其他）

用法：
    python tools/filter_code.py --list        # 只看原始文件与预估
    python tools/filter_code.py               # 全量筛选
    python tools/filter_code.py --quick       # 每源只读 2 万条，验证规则
"""
import argparse
import ast
import hashlib
import io
import json
import os
import re
import sys
import time
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RAW = os.path.join(ROOT, "data", "code_raw")
OUT = os.path.join(ROOT, "corpus_04b")

MIN_Q, MAX_Q = 4, 800
MIN_C, MAX_C = 60, 6000

FENCE = re.compile(r"```([A-Za-z0-9+#._-]*)\r?\n(.*?)```", re.S)
KEYW = re.compile(
    r"\b(def|class|function|import|from|const|let|var|public|private|static|void|"
    r"return|#include|using\s+namespace|fn|impl|struct|SELECT|INSERT|UPDATE|"
    r"CREATE\s+TABLE|package|interface|async|await|try|except|catch|elif)\b")
REFUSE = re.compile(r"(抱歉|对不起|我无法|我不能|作为(一个)?(AI|人工智能|语言模型)|"
                    r"I'?m sorry|I cannot|As an AI)", re.I)
BADCH = re.compile(r"[\ufffd\x00]")

LANGS = [
    ("python", re.compile(r"^\s*(def |class |import |from \w+ import |@\w+)", re.M)),
    ("js", re.compile(r"\b(const|let|var|function|=>|console\.log|require\()", re.M)),
    ("ts", re.compile(r"\b(interface|type \w+ =|: (string|number|boolean)\b)", re.M)),
    ("java", re.compile(r"\b(public class|private static|System\.out|@Override)", re.M)),
    ("go", re.compile(r"^\s*(package |func |import \()", re.M)),
    ("cpp", re.compile(r"(#include|std::|template\s*<|->\w+\()", re.M)),
    ("rust", re.compile(r"\b(fn \w+|let mut |impl |pub struct|::new\()", re.M)),
    ("sql", re.compile(r"\b(SELECT|INSERT INTO|UPDATE \w+ SET|CREATE TABLE)\b", re.I)),
    ("shell", re.compile(r"^\s*(#!|\$ |sudo |apt |pip |npm |export )", re.M)),
    ("csharp", re.compile(r"\b(using System|namespace \w+|Console\.WriteLine)", re.M)),
]

CODE_LINE = re.compile(
    r"^\s*(def |class |import |from |#|//|/\*|\*|@|\}|\{|\)|"
    r"(const|let|var|function|return|if|for|while|try|catch|public|private|static|"
    r"void|int|float|string|bool|fn|impl|struct|package|func|SELECT|INSERT|UPDATE)\b|"
    r"[A-Za-z_][\w.\[\]]*\s*[=:(]|->|=>|;)")


# ---------------- 基础判定 ----------------
def norm(s):
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def code_line_ratio(code):
    lines = [l for l in code.split("\n") if l.strip()]
    if not lines:
        return 0.0
    hit = sum(1 for l in lines if CODE_LINE.match(l))
    return hit / len(lines)


def detect_lang(code, hint=""):
    h = (hint or "").lower()
    for name, rx in LANGS:
        if h and (name in h or (name == "js" and h in ("javascript", "node"))):
            return name
    for name, rx in LANGS:
        if len(rx.findall(code)) >= 2:
            return name
    return "other"


def extract_code(output):
    """从 answer 里抠出代码主体：有围栏取围栏，否则整段当代码。"""
    blocks = FENCE.findall(output)
    if blocks:
        # 取最长的块
        lang, code = max(blocks, key=lambda x: len(x[1]))
        return code, lang
    return output, ""


def code_ok(code):
    if not (MIN_C <= len(code) <= MAX_C):
        return False, "len"
    if BADCH.search(code):
        return False, "badchar"
    if not KEYW.search(code) and code_line_ratio(code) < 0.5:
        return False, "notcode"
    if code_line_ratio(code) < 0.45:
        return False, "prose"
    lines = [l.strip() for l in code.split("\n") if l.strip()]
    if not lines:
        return False, "empty"
    # 单行复读：出现次数最多的行占比过高
    c = Counter(lines)
    if len(lines) >= 6 and c.most_common(1)[0][1] / len(lines) > 0.4:
        return False, "repeat"
    # 超长单行（压缩代码/minified，学不到结构）
    if max(len(l) for l in lines) > 600:
        return False, "longline"
    return True, ""


def simhash(text, bits=64):
    toks = re.findall(r"[A-Za-z_]\w*|[^\sA-Za-z_]", text[:3000])
    if not toks:
        return 0
    v = [0] * bits
    for t in toks:
        h = int(hashlib.md5(t.encode("utf-8", "ignore")).hexdigest()[:16], 16)
        for i in range(bits):
            v[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if v[i] > 0:
            out |= (1 << i)
    return out


class Deduper:
    def __init__(self):
        self.exact = set()
        self.bands = {}

    def seen(self, key_text, code):
        h = hashlib.md5(key_text.encode("utf-8", "ignore")).hexdigest()
        if h in self.exact:
            return True
        self.exact.add(h)
        sh = simhash(code)
        for b in range(4):
            band = (sh >> (b * 16)) & 0xFFFF
            k = (b, band)
            bucket = self.bands.setdefault(k, [])
            for other in bucket:
                if bin(other ^ sh).count("1") <= 3:
                    return True
            bucket.append(sh)
            if len(bucket) > 4000:      # 防内存爆
                self.bands[k] = bucket[-2000:]
        return False


# ---------------- 各源读取 ----------------
def _rows_jsonl(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def _rows_json_array(path):
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        try:
            data = json.load(f)
        except Exception:
            return
        if isinstance(data, list):
            for r in data:
                yield r


def _pick(d, *keys):
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return ""


def iter_instruct(path):
    name = os.path.basename(path)
    gen = _rows_jsonl(path) if path.endswith(".jsonl") else _rows_json_array(path)
    for r in gen:
        if not isinstance(r, dict):
            continue
        q = _pick(r, "problem", "instruction", "query", "question", "prompt")
        a = _pick(r, "solution", "output", "answer", "response", "completion")
        if q and a:
            yield q, a, name


def iter_codeparrot(path):
    """原始仓库代码：抽 docstring→函数体 对；抽不到就切块做纯代码。"""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=2000, columns=["code", "path"]):
        codes = batch.column("code").to_pylist()
        paths = batch.column("path").to_pylist()
        for code, pth in zip(codes, paths):
            if not code:
                continue
            yield code, pth or ""


def docstring_pairs(code):
    """ast 解析，取 (docstring, 函数源码) 对。解析失败返回空。"""
    try:
        tree = ast.parse(code)
    except Exception:
        return []
    out = []
    src_lines = code.split("\n")
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        doc = ast.get_docstring(node)
        if not doc or len(doc) < 20:
            continue
        start = min([d.lineno for d in node.decorator_list] + [node.lineno]) - 1
        end = getattr(node, "end_lineno", None)
        if not end:
            continue
        body = "\n".join(src_lines[start:end])
        if MIN_C <= len(body) <= MAX_C:
            out.append((doc.strip(), body))
    return out


def chunks(code, size=2400):
    """按行切成 ~size 字块，只在函数/类边界附近切不好判，就按行累加。"""
    lines = code.split("\n")
    buf, n = [], 0
    for l in lines:
        buf.append(l)
        n += len(l) + 1
        if n >= size:
            t = "\n".join(buf).strip()
            if len(t) >= MIN_C:
                yield t
            buf, n = [], 0
    t = "\n".join(buf).strip()
    if len(t) >= MIN_C:
        yield t


# ---------------- 主流程 ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--quick", action="store_true", help="每源只读 2 万条")
    ap.add_argument("--raw-cap", type=int, default=0, help="原始代码最多处理多少个文件(0=全部)")
    a = ap.parse_args()

    if not os.path.isdir(RAW):
        print("没有 data/code_raw，先跑 tools/fetch_code_raw.py")
        return 1
    files = sorted(os.listdir(RAW))
    if a.list:
        for f in files:
            p = os.path.join(RAW, f)
            print("  %-46s %8.0f MB" % (f, os.path.getsize(p) / 1e6))
        return 0

    os.makedirs(OUT, exist_ok=True)
    fi = open(os.path.join(OUT, "code_instruct.jsonl"), "w", encoding="utf-8")
    fr = open(os.path.join(OUT, "code_raw.jsonl"), "w", encoding="utf-8")
    dd = Deduper()
    st = {"src": Counter(), "drop": Counter(), "lang": Counter(), "lang_raw": Counter()}
    n_in_q = n_in_tok = n_raw_q = n_raw_tok = 0
    t0 = time.time()

    for f in files:
        p = os.path.join(RAW, f)
        if not os.path.isfile(p) or f.endswith(".part"):
            continue
        if not f.endswith((".jsonl", ".json", ".parquet")):
            continue
        print("[源] %s  %.0f MB" % (f, os.path.getsize(p) / 1e6), flush=True)
        limit = 20000 if a.quick else 10 ** 12

        if f.startswith("codeparrot"):
            kept = 0
            for i, (code, pth) in enumerate(iter_codeparrot(p)):
                if i >= limit or (a.raw_cap and kept >= a.raw_cap):
                    break
                # A. docstring → 函数
                for doc, body in docstring_pairs(code):
                    if not (MIN_Q <= len(doc) <= MAX_Q):
                        continue
                    ok, why = code_ok(body)
                    if not ok:
                        st["drop"]["cp:" + why] += 1
                        continue
                    lang = detect_lang(body, os.path.splitext(pth)[1])
                    if dd.seen(doc + body, body):
                        st["drop"]["dup"] += 1
                        continue
                    fi.write(json.dumps({"text": doc + "\n\n" + body, "_src": "codeparrot",
                                         "_lang": lang}, ensure_ascii=False) + "\n")
                    n_in_q += 1
                    n_in_tok += (len(doc) + len(body)) // 2
                    st["lang"][lang] += 1
                    kept += 1
                # B. 纯代码块
                for ch in chunks(code):
                    if not code_ok(ch)[0]:
                        continue
                    lang = detect_lang(ch, os.path.splitext(pth)[1])
                    if dd.seen(ch[:400] + str(len(ch)), ch):
                        st["drop"]["dup_raw"] += 1
                        continue
                    fr.write(json.dumps({"text": ch, "_src": "codeparrot",
                                         "_lang": lang}, ensure_ascii=False) + "\n")
                    n_raw_q += 1
                    n_raw_tok += len(ch) // 2
                    st["lang_raw"][lang] += 1
                    kept += 1
            print("     -> 指令对 %d / 原始块 %d" % (n_in_q, n_raw_q), flush=True)
            continue

        kept0 = n_in_q
        for i, (q, ans, name) in enumerate(iter_instruct(p)):
            if i >= limit:
                break
            q = norm(q)
            if not (MIN_Q <= len(q) <= MAX_Q):
                st["drop"]["q_len"] += 1
                continue
            if REFUSE.search(ans):
                st["drop"]["refuse"] += 1
                continue
            code, hint = extract_code(ans)
            code = code.strip()
            ok, why = code_ok(code)
            if not ok:
                st["drop"][why] += 1
                continue
            lang = detect_lang(code, hint)
            if dd.seen(q + code, code):
                st["drop"]["dup"] += 1
                continue
            txt = q + "\n\n" + ("```%s\n%s\n```" % (lang if lang != "other" else "", code))
            fi.write(json.dumps({"text": txt, "_src": os.path.splitext(f)[0],
                                 "_lang": lang}, ensure_ascii=False) + "\n")
            n_in_q += 1
            n_in_tok += (len(q) + len(code)) // 2
            st["lang"][lang] += 1
            st["src"][f] += 1
        print("     -> 收 %d 条" % (n_in_q - kept0), flush=True)

    fi.close()
    fr.close()

    stats = {
        "code_instruct": {"条": n_in_q, "约_tok": n_in_tok},
        "code_raw": {"条": n_raw_q, "约_tok": n_raw_tok},
        "语言分布": dict(st["lang"].most_common()),
        "原始码语言分布": dict(st["lang_raw"].most_common()),
        "各源收录": dict(st["src"].most_common()),
        "丢弃原因": dict(st["drop"].most_common(20)),
        "耗时秒": round(time.time() - t0, 1),
    }
    with open(os.path.join(OUT, "code_filter_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("\n=== 筛选完成 ===")
    print(json.dumps(stats, ensure_ascii=False, indent=2)[:1800])
    return 0


if __name__ == "__main__":
    sys.exit(main())
