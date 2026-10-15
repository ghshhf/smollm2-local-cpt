#!/usr/bin/env python3
# reassemble.py — 把本目录 9 个 pytorch_model.bin 分片拼回（纯标准库，跨平台）
# 用法：python reassemble.py [输出目录]
import hashlib, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "restored")
PARTS = 9
FINAL_SIZE = 856674211   # 8 x 99614720 + 59756451

os.makedirs(OUT_DIR, exist_ok=True)
dst = os.path.join(OUT_DIR, "pytorch_model.bin")
with open(dst, "wb") as out:
    for i in range(PARTS):
        p = os.path.join(HERE, "pytorch_model.bin.part%02d" % i)
        with open(p, "rb") as f:
            while True:
                buf = f.read(8 << 20)
                if not buf:
                    break
                out.write(buf)
size = os.path.getsize(dst)
print("restored -> %s (%d bytes)" % (dst, size))
assert size == FINAL_SIZE, "size mismatch: %d != %d" % (size, FINAL_SIZE)
# config 四件套一并带过去（权重 + base config/tokenizer 才能 from_pretrained）
for name in ("config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json"):
    src = os.path.join(HERE, name)
    if os.path.exists(src):
        with open(src, "rb") as f, open(os.path.join(OUT_DIR, name), "wb") as g:
            g.write(f.read())
print("OK. load with: AutoModelForCausalLM.from_pretrained('%s')" % OUT_DIR)
