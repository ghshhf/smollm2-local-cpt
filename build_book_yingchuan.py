"""把《影视大世界之美女如云》整本书切成训练样本，喂给 prose 卡。

规模：1298 万字 / 217885 行 / 2349 章 / 37MB。这是目前语料里最大的一本，
补的正是「文章太少」这个缺口。

切分规则（按章，不按固定长度硬切）：
  - 章标题：    第N章 + 章名
  - 每章切成若干【续写样本】= 前文片段 + "继续写下去：" + 后文片段
  - 样本长度按 token 预算截断（下游 max_seq_length=1024，超了会被静默截断）

为什么要切而不整章灌：整章普遍 2000+ 字，单条远超 1024 token 上限，
训练时会触发 trainer 的静默截断（见 check_token_len.py）。所以必须切。

输出：data/card_prose_urban.jsonl 之外的独立文件 data/book_yingchuan.jsonl
用法：python build_book_yingchuan.py [--max-chars 900] [--sample-per-chapter 3]
"""
import argparse
import codecs
import json
import os
import re

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = r"D:/tgapp/telegram_mini_app/data/uploads/1008_125126_uaa_ 影视大世界之美女如云 1-2349.txt"
OUT = os.path.join(ROOT, "data", "book_yingchuan.jsonl")

CH_TITLE = re.compile(r"^\s*第\s*\d+\s*章\s*(.*)$")


def read_all(path):
    # 🔴 必须用增量解码器：文件是 UTF-8，但中文 3 字节序列会横跨 1MB 读块边界，
    #直接 read().decode() 或固定分块都会在边界炸 UnicodeDecodeError（已实测33 处）。
    dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
    out = []
    with open(path, "rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            out.append(dec.decode(b))
    return "".join(out)


def parse_chapters(text):
    """切成 [(章标题, 章正文)]，去掉站点头尾信息。"""
    lines = text.split("\n")
    chapters = []
    cur_title, cur = None, []
    for ln in lines:
        m = CH_TITLE.match(ln)
        if m:
            if cur_title is not None and cur:
                chapters.append((cur_title, "\n".join(cur).strip()))
            cur_title = m.group(0).strip()
            cur = []
        else:
            if cur_title is not None:
                cur.append(ln)
    if cur_title is not None and cur:
        chapters.append((cur_title, "\n".join(cur).strip()))
    return chapters


def split_chapter(body, max_chars, per_chapter):
    """章内按段落边界切成样本，凑成 [前文 -> 续写] 对。
    做法：按 max_chars 滑动取前后两段，中间留一小段作为"待续写"的提示范围。"""
    paras = [p.strip() for p in body.split("\n") if p.strip()]
    if not paras:
        return []
    samples = []
    # 从不同起点采样，避免每章都取开头（那样 2349 章全是同一段开头）
    step = max(1, len(paras) // per_chapter)
    for k in range(per_chapter):
        start = (k * step + k * 7) % max(1, len(paras))   # 加个错位避免重叠雷同
        buf, total = [], 0
        for j in range(start, len(paras)):
            p = paras[j]
            if total + len(p) > max_chars:
                break
            buf.append(p)
            total += len(p)
        if len(buf) < 2:
            continue
        cut = max(1, len(buf) // 2)
        head = "".join(buf[:cut])
        tail = "".join(buf[cut:])
        if len(tail) < 50 or len(head) < 50:
            continue
        samples.append((head, tail))
    return samples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=760,
                    help="样本正文字符上限；约 760 汉字 ≈ 700-800 token，留足指令与标签空间")
    ap.add_argument("--sample-per-chapter", type=int, default=3)
    ap.add_argument("--limit-chapters", type=int, default=0, help="0=全书；调试时可限章数")
    args = ap.parse_args()

    text = read_all(SRC)
    print(f"读入: {len(text)} 字符")
    chapters = parse_chapters(text)
    print(f"识别章节: {len(chapters)} 章")
    if args.limit_chapters:
        chapters = chapters[:args.limit_chapters]

    rows, seen, skipped = [], set(), 0
    for title, body in chapters:
        for head, tail in split_chapter(body, args.max_chars, args.sample_per_chapter):
            instr = f"继续写下去： {head}"
            if instr in seen:
                continue
            seen.add(instr)
            rows.append({
                "instruction": instr,
                "output": tail,
                "_type": "prose_cont",
                "_universe": "yingchuan_dashijie",
                "_src": "upload:影视大世界之美女如云",
                "_chapter": title,
            })
    print(f"生成样本: {len(rows)} 条（去重跳过 {skipped}）")
    if not rows:
        print("!! 没生成样本，检查切分规则")
        return

    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"已写出 -> {OUT}")
    lens = [len(r["instruction"]) + len(r["output"]) for r in rows]
    print(f"样本总长: min={min(lens)} median={sorted(lens)[len(lens)//2]} max={max(lens)} 字符")


if __name__ == "__main__":
    main()