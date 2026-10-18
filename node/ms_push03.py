#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ms_push03.py — 把 0.3B checkpoint 推到魔搭模型仓库(阿里云 OSS 底座)

为什么走这条路(2026-10-18 用户定调):
  ECS 只是 2C4G 信箱, 走它中转 1.3GB 是浪费; 魔搭模型仓库是 OSS 底座,
  远程内网上传、本地 SDK 拉取, 两端都直连阿里的存储, 不经过任何中转。

用法(远程):
  python3 -u /root/ms_push03.py            # 上传
  python3 -u /root/ms_push03.py --probe    # 只探测/列仓库, 不传
"""
import os, sys, time, json

SRC = "/mnt/workspace/node1/chain/ckpt_gpu/checkpoint-03b"
TOK = "/mnt/workspace/node1/chain/base_tok"
REPO = "hhhlhuk/polaris-03b"
LOG = "/root/ms_push03.log"

T0 = time.time()


def say(m):
    line = f"[{time.strftime('%H:%M:%S')}] {m}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main():
    probe = "--probe" in sys.argv
    from modelscope.hub.api import HubApi

    api = HubApi()
    tk = os.environ.get("MODELSCOPE_ACCESS_TOKEN", "").strip()
    if tk:
        api.login(tk)
        say(f"login ok, token len={len(tk)}")
    else:
        say("[WARN] 无 MODELSCOPE_ACCESS_TOKEN, 按已登录态尝试")

    # 1) 建仓库(私有: visibility=1；已存在就忽略)
    if not probe:
        try:
            api.create_model(REPO, visibility=1, license="Apache License 2.0",
                             chinese_name="北极星 0.3B")
            say(f"create_model ok: {REPO}")
        except Exception as e:
            say(f"create_model skip: {repr(e)[:160]}")

    # 2) 组装文件清单: 权重 + config + tokenizer(ckpt 里没有 tokenizer, 从 base_tok 兜底)
    files = []
    for fn in ("config.json", "generation_config.json", "model.safetensors"):
        p = os.path.join(SRC, fn)
        if os.path.exists(p):
            files.append((p, fn))
    for fn in sorted(os.listdir(TOK)):
        p = os.path.join(TOK, fn)
        if os.path.isfile(p):
            files.append((p, fn))
    # step.txt 也带上, 便于本地核对全局步数
    sp = os.path.join(SRC, "step.txt")
    if os.path.exists(sp):
        files.append((sp, "step.txt"))

    say(f"待传 {len(files)} 个文件, 合计 {sum(os.path.getsize(p) for p, _ in files)/1e9:.2f}GB")
    for p, n in files:
        say(f"  - {n}  {os.path.getsize(p)/1e6:.1f}MB")

    if probe:
        say("PROBE_ONLY")
        return

    # 3) 逐文件上传(单文件 1.3GB, 失败可单独重试, 不用整包重来)
    ok, fail = [], []
    for p, n in files:
        t = time.time()
        try:
            api.upload_file(repo_id=REPO, path_or_fileobj=p, path_in_repo=n)
            dt = time.time() - t
            mb = os.path.getsize(p) / 1e6
            say(f"[OK] {n}  {mb:.1f}MB  {dt:.0f}s  ({mb/max(dt,0.1):.1f}MB/s)")
            ok.append(n)
        except Exception as e:
            say(f"[FAIL] {n}  {repr(e)[:200]}")
            fail.append(n)
    say(f"完成 ok={len(ok)} fail={len(fail)} 总耗时 {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
