#!/usr/bin/env python
# 本地 -> node1 指令投递器：把一条 bash 指令内容上传为 ECS cmd_<seq>.txt（递增序号，绝不重名）
# 用法: python node/ecs_cmd.py <cmd文件路径> [显式序号]
import sys, os, urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
SEQF = os.path.join(os.path.dirname(__file__), "ecs_cmd_seq.txt")


def next_seq():
    try:
        return int(open(SEQF).read().strip()) + 1
    except Exception:
        return 1


def main():
    path = sys.argv[1]
    content = open(path, "rb").read()
    n = int(sys.argv[2]) if len(sys.argv) > 2 else next_seq()
    name = "cmd_%06d.txt" % n
    req = urllib.request.Request(
        f"{ECS}/upload?name={name}&key={KEY}", data=content, method="POST"
    )
    resp = urllib.request.urlopen(req, timeout=30)
    print(f"OK name={name} status={resp.status} bytes={len(content)}")
    if len(sys.argv) <= 2:
        open(SEQF, "w").write(str(n))


if __name__ == "__main__":
    main()
