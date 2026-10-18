#!/usr/bin/env python
# 本地 -> 读取 node1 信箱：按 seq 顺序拉取未读的 mail_<seq>.txt
# 用法: python node/ecs_mail.py [一次最多读几条，默认 5]
import sys, os, urllib.request, urllib.error

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
SEQF = os.path.join(os.path.dirname(__file__), "ecs_mail_seq.txt")


def last_read():
    try:
        return int(open(SEQF).read().strip())
    except Exception:
        return 0


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    start = last_read()
    got = 0
    for n in range(start + 1, start + 1 + limit):
        name = "mail_%06d.txt" % n
        try:
            with urllib.request.urlopen(
                f"{ECS}/file/{name}?key={KEY}", timeout=20
            ) as r:
                body = r.read().decode("utf-8", "replace")
        except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
            break
        if '"error"' in body[:40]:
            break
        print(f"\n########## {name} ##########")
        print(body)
        open(SEQF, "w").write(str(n))
        got += 1
    if got == 0:
        print(f"(no new mail since #{start})")


if __name__ == "__main__":
    main()
