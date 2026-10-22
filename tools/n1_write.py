#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/n1_write.py — 把本地文件可靠写到 node1 的绝对路径（base64 直写，绕开 rc.py put 的相对路径坑）

用法:
    python tools/n1_write.py tools/fetch_04b_data.py /mnt/workspace/node1/tools/fetch_04b_data.py
"""
import base64
import hashlib
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
RC = os.path.join(ROOT, "node", "rc.py")


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    data = open(src, "rb").read()
    b64 = base64.b64encode(data).decode()
    md5 = hashlib.md5(data).hexdigest()
    cmd = ("echo '%s' | base64 -d > %s && wc -c %s && md5sum %s"
           % (b64, dst, dst, dst))
    py = sys.executable
    r = subprocess.run([py, RC, "sh", cmd], capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    ok = md5 in out
    print("[write] %s -> %s  %dB  md5=%s  %s"
          % (src, dst, len(data), md5[:8], "OK" if ok else "MISMATCH"))
    if not ok:
        print(out[-500:])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
