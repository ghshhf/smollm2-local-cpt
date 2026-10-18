#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm_data.py — 构建 VLM 中文图文对数据集
数据源:
  1. COCO val2017 图片 (detection-datasets/coco parquet, 5000张)
  2. COCO-CN 人工中文 caption (imageid.human-written-caption.txt, 22218条)
  3. COCO-CN 机翻增强 (imageid.machine-translated-caption.bosonseg.txt, 61.7万条, 需清洗)
输出:
  data/vlm/images/{image_id}.jpg     图片
  data/vlm/vlm_pairs.jsonl           {"image": "...", "text": "中文描述"}
用法:
  python tools/vlm_data.py            # 全流程: probe -> 提图 -> 配对 -> 抽样
"""
import os, sys, json, re, random

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VLM_DIR = os.path.join(ROOT, "data", "vlm")
IMG_DIR = os.path.join(VLM_DIR, "images")
COCO_CN = os.path.join(VLM_DIR, "coco-cn-version1805v1.1")
OUT_JSONL = os.path.join(VLM_DIR, "vlm_pairs.jsonl")


def probe_schema():
    """探测 parquet 列名"""
    import pyarrow.parquet as pq
    for f in ("val0.parquet", "val1.parquet"):
        p = os.path.join(VLM_DIR, f)
        if not os.path.exists(p):
            continue
        pf = pq.ParquetFile(p)
        print("[schema] %s  rows=%d" % (f, pf.metadata.num_rows))
        for i in range(pf.metadata.num_columns):
            print("   col[%d] %-20s %s" % (i, pf.schema_arrow.field(i).name,
                                           pf.schema_arrow.field(i).type))
        # 读1行看值
        t = pf.read_row_group(0)
        d = t.slice(0, 1).to_pylist()[0]
        for k, v in d.items():
            if isinstance(v, dict):
                print("   sample %-14s = dict keys=%s" % (k, list(v.keys())[:5]))
            elif isinstance(v, bytes):
                print("   sample %-14s = bytes len=%d" % (k, len(v)))
            else:
                print("   sample %-14s = %r" % (k, str(v)[:60]))
        break  # 看一片够


def extract_images():
    """从 parquet 提取图片到 images/{image_id}.jpg"""
    import pyarrow.parquet as pq
    os.makedirs(IMG_DIR, exist_ok=True)
    n_ok = n_skip = 0
    for f in ("val0.parquet", "val1.parquet"):
        p = os.path.join(VLM_DIR, f)
        if not os.path.exists(p):
            print("[img] 缺文件", f); continue
        pf = pq.ParquetFile(p)
        for rg in range(pf.metadata.num_row_groups):
            t = pf.read_row_group(rg).to_pylist()
            for row in t:
                # 找 image bytes 列
                img = row.get("image")
                if isinstance(img, dict):
                    img = img.get("bytes") or img.get("path")
                if not isinstance(img, bytes):
                    n_skip += 1; continue
                # 找 image_id
                iid = row.get("image_id")
                if iid is None:
                    iid = row.get("imageid")
                if iid is None:
                    n_skip += 1; continue
                out = os.path.join(IMG_DIR, "%012d.jpg" % int(iid))
                if os.path.exists(out):
                    n_skip += 1; continue
                with open(out, "wb") as fh:
                    fh.write(img)
                n_ok += 1
            print("[img] %s rg%d done  ok=%d skip=%d" % (f, rg, n_ok, n_skip), flush=True)
    print("[img] 提取完成 ok=%d skip=%d" % (n_ok, n_skip))
    return n_ok


# ---------- COCO-CN 标注解析 ----------
def clean_mt(text):
    """清洗机翻 boson 分词: '一个:m 男人:n' -> '一个男人'"""
    text = re.sub(r":[a-z]+", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def load_coco_cn():
    """返回 {image_id(int): {"human":[..], "mt":[..], "tags":[..]}}"""
    caps = {}

    # 人工 caption
    p = os.path.join(COCO_CN, "imageid.human-written-caption.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip() or "\t" not in line:
                    continue
                key, _, cap = line.partition("\t")
                cap = cap.strip()
                if not cap:
                    continue
                m = re.search(r"(\d{12})", key)
                if not m:
                    continue
                iid = int(m.group(1))
                caps.setdefault(iid, {"human": [], "mt": []})
                caps[iid]["human"].append(cap)

    # 机翻增强
    p = os.path.join(COCO_CN, "imageid.machine-translated-caption.bosonseg.txt")
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split("\t") if "\t" in line else line.split(" ", 1)
                key, cap = parts[0], parts[-1]
                m = re.search(r"(\d{12})", key)
                if not m:
                    continue
                c = clean_mt(cap)
                if len(c) < 4:
                    continue
                iid = int(m.group(1))
                caps.setdefault(iid, {"human": [], "mt": []})
                caps[iid]["mt"].append(c)
    return caps


def build_pairs():
    """图片存在 × 中文caption => vlm_pairs.jsonl"""
    caps = load_coco_cn()
    have = set()
    if os.path.isdir(IMG_DIR):
        for fn in os.listdir(IMG_DIR):
            if fn.endswith(".jpg"):
                have.add(int(fn[:-4]))
    print("[pair] 图片 %d 张, 标注覆盖 %d 张" % (len(have), len(caps)))

    pairs = []
    for iid in sorted(have & set(caps.keys())):
        c = caps[iid]
        texts = list(c["human"])
        # 人工不足的用机翻补（每图最多取 8 条机翻，防偏斜）
        if len(texts) < 3:
            texts += c["mt"][:8]
        for t in texts:
            if 4 <= len(t) <= 120:
                pairs.append({"image": os.path.join(IMG_DIR, "%012d.jpg" % iid),
                              "text": t})
    random.seed(42)
    random.shuffle(pairs)
    n_val = min(800, len(pairs) // 10)
    val, train = pairs[:n_val], pairs[n_val:]
    for name, data in (("train", train), ("val", val)):
        out = os.path.join(VLM_DIR, "vlm_%s.jsonl" % name)
        with open(out, "w", encoding="utf-8") as f:
            for r in data:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print("[pair] %s: %d 条 -> %s" % (name, len(data), out))
    # 抽样（铁律: 先看数据）
    print("\n[sample] 随机 5 条:")
    for r in random.sample(train, min(5, len(train))):
        print("   %s | %s" % (os.path.basename(r["image"]), r["text"]))


if __name__ == "__main__":
    if "--probe" in sys.argv:
        probe_schema()
    elif "--extract" in sys.argv:
        extract_images()
    elif "--pair" in sys.argv:
        build_pairs()
    else:
        probe_schema()
        extract_images()
        build_pairs()
