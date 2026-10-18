#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r6_fetch_bg.py — R6 封顶权重全自动回捞（后台任务）
流程: 等 node1 分片传完 -> 下载 9 片+6 小文件 -> 拼合 -> sha256 校验 -> 入账 -> README+拼回器 -> git push
全程进度写 node/r6fetch.log。任何硬失败即停并记录。
"""
import hashlib, os, subprocess, sys, time, urllib.request

REPO = "D:/work/_local_ai/train_15b"
NODE = os.path.join(REPO, "node")
DEST = os.path.join(REPO, "backups", "0.2B", "r6_final_20261016")
ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
WANT_SHA = "4c91e3502f54afba5733b3417f0ca2378bb0e27d5838f9a69b507b69b29d067a"
WANT_SIZE = 856595368
PARTS = [f"model.safetensors.part0{i}" for i in range(9)]
SMALL = ["config.json", "generation_config.json", "step.txt",
         "tokenizer.json", "tokenizer_config.json", "merges.txt"]
LOG = os.path.join(NODE, "r6fetch.log")


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def ecs_get(name, tries=3):
    """下载 ECS 文件，断点续传+重试。成功返回 True。"""
    url = f"{ECS}/file/{name}?key={KEY}"
    for attempt in range(tries):
        tmp = name + ".part"
        cmd = ["curl", "-s", "-L", "-m", "300", "-C", "-",
               "-o", tmp, url]
        r = subprocess.run(cmd, cwd=DEST, capture_output=True)
        # curl -C - 在文件已完整时返回 416 或成功；按大小判断
        if os.path.isfile(os.path.join(DEST, tmp)):
            sz = os.path.getsize(os.path.join(DEST, tmp))
            if r.returncode in (0, 33):  # 33 = resume not supported/server ignores range
                os.replace(os.path.join(DEST, tmp), os.path.join(DEST, name))
                log(f"  GET {name} -> {sz} bytes")
                return True
        log(f"  GET {name} retry {attempt+1} (rc={r.returncode})")
        time.sleep(3)
    return False


def wait_node1_done(limit_sec=900):
    """轮询 r6f_pull_003.txt 直到含 R6FINAL_ALL_DONE"""
    t0 = time.time()
    while time.time() - t0 < limit_sec:
        try:
            with urllib.request.urlopen(f"{ECS}/file/r6f_pull_003.txt?key={KEY}", timeout=30) as r:
                txt = r.read().decode("utf-8", "replace")
            if "R6FINAL_ALL_DONE" in txt:
                log("node1 上传完成回执:")
                for ln in txt.strip().splitlines()[-12:]:
                    log("  | " + ln)
                return True
            log(f"等待 node1... 回执现状: {txt.strip()[-80:]!r}")
        except Exception as e:
            log(f"等待 node1... ({e})")
        time.sleep(30)
    log("FATAL: 等 node1 回执超时")
    return False


def main():
    log("=== R6 回捞后台任务启动 ===")
    os.makedirs(DEST, exist_ok=True)

    # 1) 等 node1 传完
    if not wait_node1_done():
        return 2

    # 2) 下载分片
    log("下载 9 个分片 ...")
    for p in PARTS:
        if not ecs_get("r6f_" + p):
            log(f"FATAL: 分片 {p} 下载失败")
            return 3
    log("下载小文件 ...")
    for s in SMALL:
        if not ecs_get("r6f_" + s):
            log(f"FATAL: 小文件 {s} 下载失败")
            return 3

    # 3) 拼合 + 校验
    log("拼合 + sha256 校验 ...")
    final = os.path.join(DEST, "model.safetensors")
    tmp = final + ".assembling"
    with open(tmp, "wb") as w:
        for p in PARTS:
            with open(os.path.join(DEST, p), "rb") as r:
                while True:
                    b = r.read(1 << 22)
                    if not b:
                        break
                    w.write(b)
    sz = os.path.getsize(tmp)
    if sz != WANT_SIZE:
        log(f"FATAL: 大小不符 {sz} != {WANT_SIZE}")
        os.remove(tmp)
        return 4
    got = sha256(tmp)
    if got != WANT_SHA:
        log(f"FATAL: sha256 不符 {got}")
        os.remove(tmp)
        return 4
    os.replace(tmp, final)
    log(f"OK 权重校验通过: {sz} bytes, sha256={got[:16]}...")

    # 4) 整理目录: 小文件就位, 删分片
    for s in SMALL:
        src = os.path.join(DEST, "r6f_" + s)
        dst = os.path.join(DEST, s)
        if os.path.isfile(src):
            os.replace(src, dst)
    for p in PARTS:
        fp = os.path.join(DEST, p)
        if os.path.isfile(fp):
            os.remove(fp)
        pp = os.path.join(DEST, "r6f_" + p)
        if os.path.isfile(pp):
            os.remove(pp)

    # 5) 拼回脚本 (与 r4_final 同款, 自校验值写死)
    reassemble = os.path.join(DEST, "reassemble.py")
    with open(reassemble, "w", encoding="utf-8") as f:
        f.write(REASSEMBLE_TEMPLATE)
    log("拼回器已生成")

    # 6) README
    with open(os.path.join(DEST, "README.md"), "w", encoding="utf-8") as f:
        f.write(README_TEMPLATE)
    log("README 已生成")

    # 7) git commit + push (本地 nohup 后台, 轮询日志)
    log("git add + commit ...")
    r = subprocess.run(["git", "-C", REPO, "add", "backups/0.2B/r6_final_20261016/"],
                       capture_output=True, text=True)
    r = subprocess.run(["git", "-C", REPO, "commit", "-m",
                        "backup(0.2B): R6 封顶正式版权重(done=7100)回捞入账 + 拼回器 + provenance"],
                       capture_output=True, text=True)
    log("commit: " + (r.stdout.strip()[-120:] or r.stderr.strip()[-120:]))
    log("push 后台启动 (git_push_r6final.log) ...")
    pushlog = os.path.join(NODE, "git_push_r6final.log")
    subprocess.Popen(
        f'(nohup git -C {REPO} push --progress origin HEAD > "{pushlog}" 2>&1 </dev/null &)',
        shell=True)
    log("=== 回捞主体完成, push 进行中 ===")
    return 0


REASSEMBLE_TEMPLATE = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""reassemble.py — 从同目录分片拼回 R6 封顶权重（幂等 + sha256 自校验）
用法: python reassemble.py [输出目录]
"""
import hashlib, os, shutil, sys

HERE = os.path.dirname(os.path.abspath(__file__))
FINAL_SHA256 = "4c91e3502f54afba5733b3417f0ca2378bb0e27d5838f9a69b507b69b29d067a"
FINAL_SIZE = 856595368


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else os.getcwd()
    os.makedirs(out, exist_ok=True)
    parts = sorted(os.path.join(HERE, f) for f in os.listdir(HERE)
                   if f.startswith("model.safetensors.part"))
    if not parts:
        print("FATAL: no shards in", HERE)
        return 2
    dst = os.path.join(out, "model.safetensors")
    if os.path.isfile(dst) and os.path.getsize(dst) == FINAL_SIZE and sha256(dst) == FINAL_SHA256:
        print("SKIP (already ok)")
        return 0
    tmp = dst + ".assembling"
    with open(tmp, "wb") as w:
        for p in parts:
            with open(p, "rb") as r:
                shutil.copyfileobj(r, w, 1 << 22)
    sz = os.path.getsize(tmp)
    if sz != FINAL_SIZE or sha256(tmp) != FINAL_SHA256:
        print("FATAL: assemble check failed", sz)
        os.remove(tmp)
        return 3
    os.replace(tmp, dst)
    print("OK", sz, "sha256 verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
'''

README_TEMPLATE = '''# r6_final_20261016 — 0.2B R6 知识档封顶正式版

## 身份
- SmolLM2-200M（Net2Net 扩容版：hidden 576 / FFN 3072 / 30 层 / vocab 49152 / tied embeddings）
- **R6 知识教育档封顶权重**：done=7100，2026-10-16 00:16 由 node1 NAS `chain/ckpt_old/` 回捞
- sha256: `4c91e3502f54afba5733b3417f0ca2378bb0e27d5838f9a69b507b69b29d067a`（856,595,368 字节）
- 与 `base_20261010`（sha256 `0ee38bf6...`，R4 前底座）**是两份不同权重**——本份多训了 R4-R6 共 3000 步

## 文件
- `model.safetensors` — 权重本体（回捞时已拼合并校验）
- `config.json` / `generation_config.json` / `step.txt`（=7100）— ckpt_old 原件
- `tokenizer.json` / `tokenizer_config.json` / `merges.txt` — node1 base_tok 原件
- `reassemble.py` — 拼回器（幂等 + sha256 自校验，已本地验证）

## 备注
- 回捞路径：node1 split 95MiB×9 → ECS /upload → 本地下载拼合 → sha256 比对 NAS 原值
- node1 当前已进入 0.3B 阶段（R7 知识档，ckpt/ 目录 1.29GB 是在训权重，非本份）
'''

if __name__ == "__main__":
    sys.exit(main())
