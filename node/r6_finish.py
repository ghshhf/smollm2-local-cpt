#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""r6_finish.py — 修复 r6_fetch_bg.py 的前缀 bug 后接力: 拼合->校验->整理->入账->push
前置: 9 分片 + 小文件已全部下载到 DEST (文件名带 r6f_ 前缀)
"""
import hashlib, os, subprocess, sys, time

REPO = "D:/work/_local_ai/train_15b"
DEST = os.path.join(REPO, "backups", "0.2B", "r6_final_20261016")
BASE = os.path.join(REPO, "backups", "0.2B", "base_20261010")
WANT_SHA = "4c91e3502f54afba5733b3417f0ca2378bb0e27d5838f9a69b507b69b29d067a"
WANT_SIZE = 856595368
PARTS = [f"model.safetensors.part0{i}" for i in range(9)]


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    # 1) 分片去前缀就位 (model.safetensors.partNN)
    for p in PARTS:
        src = os.path.join(DEST, "r6f_" + p)
        dst = os.path.join(DEST, p)
        if not os.path.isfile(src):
            log(f"FATAL: 缺分片 {src}")
            return 2
        if os.path.isfile(dst):
            os.remove(dst)
        os.replace(src, dst)
    log("分片就位 (去前缀)")

    # 2) 小文件就位: config/step/tokenizer 四件套用新下的, merges 用本地 base 版(ECS 那份 0 字节)
    for s in ["config.json", "generation_config.json", "step.txt",
              "tokenizer.json", "tokenizer_config.json"]:
        src = os.path.join(DEST, "r6f_" + s)
        dst = os.path.join(DEST, s)
        if os.path.isfile(src):
            if os.path.isfile(dst):
                os.remove(dst)
            os.replace(src, dst)
            log(f"  {s} <- r6f_ 版 ({os.path.getsize(dst)}B)")
        else:
            log(f"  !! 缺 {s}")
    src_m = os.path.join(BASE, "merges.txt")
    if os.path.isfile(src_m) and os.path.getsize(src_m) > 0:
        subprocess.run(["cp", src_m, os.path.join(DEST, "merges.txt")])
        log(f"  merges.txt <- base_20261010 版 ({os.path.getsize(src_m)}B)")
    # 清掉 0 字节的坏 merges
    bad = os.path.join(DEST, "r6f_merges.txt")
    if os.path.isfile(bad):
        os.remove(bad)

    # 3) 拼合 + 校验 (完整版本地留存, 不入 git)
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
        return 3
    got = sha256(tmp)
    if got != WANT_SHA:
        log(f"FATAL: sha256 不符 {got}")
        os.remove(tmp)
        return 3
    os.replace(tmp, final)
    log(f"OK 权重拼合+校验通过: {sz}B sha256={got[:16]}...")

    # 4) .gitignore: 完整版不入库 (超 100MB)
    gi = os.path.join(DEST, ".gitignore")
    with open(gi, "w") as f:
        f.write("model.safetensors\n")
    log(".gitignore 写好 (完整版不入库)")

    # 5) 拼回器 + README
    import importlib.util
    spec = importlib.util.spec_from_file_location("bg", os.path.join(REPO, "node", "r6_fetch_bg.py"))
    # 不执行主流程, 只取模板字符串
    src_txt = open(os.path.join(REPO, "node", "r6_fetch_bg.py"), encoding="utf-8").read()
    ns = {}
    exec(src_txt.split('if __name__')[0], ns)
    with open(os.path.join(DEST, "reassemble.py"), "w", encoding="utf-8") as f:
        f.write(ns["REASSEMBLE_TEMPLATE"])
    readme = ns["README_TEMPLATE"].replace(
        "sha256: `4c91e3502f54afba5733b3417f0ca2378bb0e27d5838f9a69b507b69b29d067a`（856,595,368 字节）",
        "sha256: `4c91e3502f54afba5733b3417f0ca2378bb0e27d5838f9a69b507b69b29d067a`（856,595,368 字节）\n- 注意: ckpt_old/step.txt 内容为 `400`（链上局部步数计数），全局进度以 chain.log `R6 complete @ 7100` 为准")
    with open(os.path.join(DEST, "README.md"), "w", encoding="utf-8") as f:
        f.write(readme)
    log("拼回器 + README 写好")

    # 6) git add + commit
    r = subprocess.run(["git", "-C", REPO, "add", "backups/0.2B/r6_final_20261016/"],
                       capture_output=True, text=True)
    r = subprocess.run(["git", "-C", REPO, "commit", "-m",
                        "backup(0.2B): R6 封顶正式版权重(done=7100)回捞入账 + 9分片 + 拼回器 + provenance"],
                       capture_output=True, text=True)
    log("commit: " + ((r.stdout or r.stderr).strip()[-150:]))

    # 7) push 后台
    pushlog = os.path.join(REPO, "node", "git_push_r6final.log")
    subprocess.Popen(f'(nohup git -C {REPO} push --progress origin HEAD > "{pushlog}" 2>&1 </dev/null &)',
                     shell=True)
    log("push 后台启动 (git_push_r6final.log)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
