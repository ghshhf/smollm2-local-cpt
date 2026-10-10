#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
把 0.1B 这一站(135M 底座 + R2 训完权重 + R2 语料)打包成单个 zip,
再切成分片(默认 95MB, 绕过 GitHub 100MB/文件硬限), 输出到 backups/0.1B/。

产物(全部进 git):
  backups/0.1B/0.1B_part_00.zip ... 0.1B_part_0N.zip   <- 分片(每个 <100MB)
  backups/0.1B/MANIFEST.json                              <- 分片清单 + 合并后 sha256
  backups/0.1B/reassemble.py                              <- 纯标准库重组脚本(跨平台)
  backups/0.1B/README.md                                  <- 本里程碑说明

重组: python reassemble.py   ->  还原 0.1B.zip 并解压到 ./0.1B_extracted/

不含 optimizer.pt / scheduler.pt(续训态, 里程碑快照不需要, 省 ~1GB)。
"""
import os, sys, io, json, hashlib, zipfile, shutil, datetime

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # train_15b
LOCAL_AI = os.path.dirname(REPO)                                     # _local_ai
OUT_DIR = os.path.join(REPO, "backups", "0.1B")
CHUNK = 95 * 1024 * 1024                                             # 95MB

# (源绝对路径, 分片内相对路径)
SPEC = [
    (os.path.join(LOCAL_AI, "models", "SmolLM2-135M"), "base"),
    (os.path.join(REPO, "out_smollm2_135m_pt", "checkpoint-latest", "pytorch_model.bin"), "model/pytorch_model.bin"),
    (os.path.join(REPO, "out_smollm2_135m_pt", "checkpoint-latest", "step.txt"), "model/step.txt"),
    (os.path.join(REPO, "pretrain_corpus", "zh_text_r2_cat.jsonl"), "corpus/zh_text_r2_cat.jsonl"),
]

README_MD = """# 0.1B 里程碑备份 (SmolLM2-135M + R2 小说续训)

本目录是 **0.1B 这一站** 的离线快照, 用于防止本地盘损坏 / 实验改崩后丢失已训成果。

## 包含内容
| 路径 | 说明 | 约体积 |
|---|---|---|
| `base/` | SmolLM2-135M 底座 (config + tokenizer + safetensors) | 260MB |
| `model/pytorch_model.bin` | **R2 训完的 0.1B 权重** (全量 CPT, 小说分类语料) | 538MB |
| `model/step.txt` | 落盘步数 | 3B |
| `corpus/zh_text_r2_cat.jsonl` | R2 训练语料 (37721 篇小说, 带 cat 题材标签) | 76MB |

## 不含
- `optimizer.pt` / `scheduler.pt` (续训态, 1GB+, 里程碑快照不需要; 如需从权重续训, 训练脚本会重建 optimizer)

## 重组
```bash
cd backups/0.1B
python reassemble.py
```
会校验 sha256, 还原 `0.1B.zip` 并解压到 `./0.1B_extracted/`。
解压后: `base/` 可直接 `from_pretrained`, `model/pytorch_model.bin` 是全模型权重(配 `base/` 的 config+tokenizer 加载)。

## 升级链上下文
0.1B(R2) -> Net2Net 保函数扩到 0.2B(214M, R4/R5) -> 0.3B(350M, R6)。
扩容宪法: 绝不改 注意力/隐藏维/GQA头数/RoPE, 只加宽 MLP + 零输出恒等层。
"""


def sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_of_file(path: str, buf=1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(buf), b""):
            h.update(blk)
    return h.hexdigest()


def stage_to_zip(spec, readme, tmp_zip):
    """把 SPEC 里的文件/目录打成一个 zip (ZIP_DEFLATED)。"""
    written = []
    with zipfile.ZipFile(tmp_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, rel in spec:
            if os.path.isdir(src):
                for root, _, files in os.walk(src):
                    for fn in sorted(files):
                        full = os.path.join(root, fn)
                        arc = os.path.join(rel, os.path.relpath(full, src)).replace("\\", "/")
                        z.write(full, arc)
                        written.append(arc)
            else:
                arc = rel.replace("\\", "/")
                z.write(src, arc)
                written.append(arc)
        # README
        z.writestr("README.md", readme)
        written.append("README.md")
    return written


def split_file(path, out_dir, chunk):
    names = []
    with open(path, "rb") as f:
        idx = 0
        while True:
            blk = f.read(chunk)
            if not blk:
                break
            name = "0.1B_part_%02d.zip" % idx
            with open(os.path.join(out_dir, name), "wb") as o:
                o.write(blk)
            names.append(name)
            idx += 1
    return names


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    tmp = os.path.join(OUT_DIR, "_tmp_0.1B.zip")
    print("[1/4] 打包 zip ...")
    contents = stage_to_zip(SPEC, README_MD, tmp)
    print("      压缩后体积: %.1f MB, 文件数: %d" % (os.path.getsize(tmp) / 1e6, len(contents)))
    print("[2/4] 计算合并 sha256 并切分 ...")
    combined_sha = sha256_of_file(tmp)
    combined_size = os.path.getsize(tmp)
    chunk_names = split_file(tmp, OUT_DIR, CHUNK)
    print("      分片数: %d, 合并 sha256=%s" % (len(chunk_names), combined_sha[:16]))
    # 每个分片的 sha + size
    chunks = []
    for n in chunk_names:
        p = os.path.join(OUT_DIR, n)
        chunks.append({"name": n, "size": os.path.getsize(p), "sha256": sha256_of_file(p)})
    # 写清单
    manifest = {
        "name": "0.1B",
        "desc": "SmolLM2-135M base + R2 trained weights + R2 corpus",
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "combined_file": "0.1B.zip",
        "combined_size": combined_size,
        "combined_sha256": combined_sha,
        "chunk_size": CHUNK,
        "chunks": chunks,
        "contents": contents,
    }
    with open(os.path.join(OUT_DIR, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    # 写重组脚本
    write_reassembler(OUT_DIR)
    # 删临时完整 zip(避免被误提交 >100MB)
    os.remove(tmp)
    total = sum(c["size"] for c in chunks)
    print("[3/4] 分片已写出: %d 片, 合计 %.1f MB" % (len(chunks), total / 1e6))
    for c in chunks:
        print("      %s  %.1f MB" % (c["name"], c["size"] / 1e6))
    print("[4/4] 完成. 下一步: git add backups/0.1B && commit && push")


def write_reassembler(out_dir):
    code = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""0.1B 分片重组: 按 MANIFEST.json 顺序拼接分片 -> 0.1B.zip -> 解压到 ./0.1B_extracted/"""
import os, json, hashlib, zipfile

HERE = os.path.dirname(os.path.abspath(__file__))

def main():
    m = json.load(open(os.path.join(HERE, "MANIFEST.json"), encoding="utf-8"))
    out_zip = os.path.join(HERE, m["combined_file"])
    # 拼接
    with open(out_zip, "wb") as o:
        for c in m["chunks"]:
            p = os.path.join(HERE, c["name"])
            data = open(p, "rb").read()
            assert hashlib.sha256(data).hexdigest() == c["sha256"], "分片损坏: " + c["name"]
            o.write(data)
    # 校验合并 sha
    h = hashlib.sha256()
    for blk in iter(lambda: open(out_zip, "rb").read(1024 * 1024), b""):
        h.update(blk)
    assert h.hexdigest() == m["combined_sha256"], "合并 sha256 不匹配!"
    print("合并校验通过: %s (%d bytes)" % (m["combined_file"], m["combined_size"]))
    # 解压
    dst = os.path.join(HERE, "0.1B_extracted")
    if os.path.exists(dst):
        import shutil as _s
        _s.rmtree(dst)
    with zipfile.ZipFile(out_zip, "r") as z:
        z.extractall(dst)
    print("已解压到:", dst)
    print("内容:", ", ".join(m["contents"][:8]), "..." if len(m["contents"]) > 8 else "")

if __name__ == "__main__":
    main()
'''
    with open(os.path.join(out_dir, "reassemble.py"), "w", encoding="utf-8") as f:
        f.write(code)


if __name__ == "__main__":
    main()
