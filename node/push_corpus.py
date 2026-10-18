#!/usr/bin/env python3
# 把本机已处理好的成品 jsonl 切片 -> 传 ECS /file/ -> 派 node1 拼回 NAS corpus_reserve/
# 用法: push_corpus.py <本地文件> <片前缀> <node1目标名> [每片MB]
import sys, os, json, time, urllib.request, subprocess

LOCAL = sys.argv[1]
PREFIX = sys.argv[2]          # 例如 nv / wx
DEST = sys.argv[3]            # 例如 zh_r_novels.jsonl
CHUNK_MB = int(sys.argv[4]) if len(sys.argv) > 4 else 90
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
W = "/mnt/workspace/node1/chain/corpus_reserve"
TMP = r"D:\work\_local_ai\tmp\push"

def split_file(path, chunk_bytes):
    os.makedirs(TMP, exist_ok=True)
    parts = []
    i = 0
    with open(path, "rb") as f:
        while True:
            data = f.read(chunk_bytes)
            if not data:
                break
            # 避免在多字节 UTF-8 字符中间截断: 回退到上一个换行
            if len(data) == chunk_bytes:
                last_nl = data.rfind(b"\n")
                if last_nl != -1 and last_nl < chunk_bytes - 1:
                    tail = data[last_nl+1:]
                    data = data[:last_nl+1]
                    p = os.path.join(TMP, f"{PREFIX}_{i:02d}")
                    with open(p, "wb") as o: o.write(data)
                    parts.append(p)
                    # 把 tail 放回文件指针前(通过 re-open 续读太复杂, 简单处理: 写入独立 part 续接)
                    i += 1
                    # 把 tail 作为下一片开头: 写临时
                    p2 = os.path.join(TMP, f"{PREFIX}_{i:02d}")
                    with open(p2, "wb") as o: o.write(tail)
                    parts.append(p2)
                    i += 1
                    # 后续从 tail 之后继续: 但 f 已前进 chunk_bytes, 会重复读 tail 部分
                    # 为避免复杂, 采用更简单稳妥方案 -> 见下方 fallback
                    break
            else:
                p = os.path.join(TMP, f"{PREFIX}_{i:02d}")
                with open(p, "wb") as o: o.write(data)
                parts.append(p)
                i += 1
                break
    return parts

# 上面回退逻辑易错, 改用稳健的按行分片(行不会跨片截断)
def split_by_lines(path, chunk_bytes):
    os.makedirs(TMP, exist_ok=True)
    parts = []
    i = 0
    buf = b""
    p = os.path.join(TMP, f"{PREFIX}_{i:02d}")
    pf = open(p, "wb")
    with open(path, "rb") as f:
        for line in f:
            if pf.tell() + len(line) > chunk_bytes and pf.tell() > 0:
                pf.close()
                parts.append(p)
                i += 1
                p = os.path.join(TMP, f"{PREFIX}_{i:02d}")
                pf = open(p, "wb")
            pf.write(line)
    pf.close()
    if os.path.getsize(p) > 0:
        parts.append(p)
    else:
        os.remove(p)
    return parts

def upload(part):
    name = os.path.basename(part)
    for attempt in range(4):
        try:
            r = subprocess.run(["curl","-s","-m180","-X","POST",
                                f"{ECS}/upload?name={name}&key={KEY}",
                                "--data-binary", f"@{part}"], check=True)
            return True
        except Exception as e:
            print("  upload retry", name, e); time.sleep(3)
    return False

def main():
    print(f"[split] {LOCAL}")
    parts = split_by_lines(LOCAL, CHUNK_MB*1_000_000)
    print(f"  parts: {len(parts)} x ~{CHUNK_MB}MB")
    for p in parts:
        ok = upload(p)
        print(f"  upload {os.path.basename(p)} ({os.path.getsize(p)}B) -> {ok}")
        if not ok:
            raise SystemExit("upload failed")
    # node1 拼回
    names = [os.path.basename(p) for p in parts]
    cat_cmd = "cd /tmp && "
    for nm in names:
        cat_cmd += f"curl -s -m180 -o {nm} {ECS}/file/{nm} && "
    cat_cmd += f"cat {' '.join(names)} > {W}/{DEST} && "
    cat_cmd += f"echo SIZE=$(wc -c < {W}/{DEST}) && ls -la {W}/{DEST}"
    CMD = (cat_cmd + f" | curl -s -X POST '{ECS}/upload?name=status_push_{PREFIX}.txt&key={KEY}' --data-binary @-")
    body = json.dumps({"cmd": CMD, "id": f"push-{PREFIX}-{int(time.time())}", "ts": int(time.time())}).encode()
    req = urllib.request.Request(ECS+"/task", data=body, headers={"Content-Type":"application/json"}, method="POST")
    r = urllib.request.urlopen(req, timeout=15)
    print("POST /task ->", r.status)

if __name__ == "__main__":
    main()
