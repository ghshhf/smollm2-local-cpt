#!/usr/bin/env python3
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
