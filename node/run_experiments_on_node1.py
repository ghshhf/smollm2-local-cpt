#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""run_experiments_on_node1.py — 在 node1(云端 Linux) 上跑 V1-02 + E1 两条实验。

流程:
  0. cd 到仓库根(git pull 拉取最新 eval 脚本)
  1. 若 135M 权重缺失 -> 重组 0.1B(本地已有分片) -> 摆到 eval 脚本默认期望的路径
  2. 设 SMOL_BASE / SMOL_CKPT 环境变量, 跑 v1_02_alpha_ladder.py 与 e1_subliminal_proxy.py
  3. 把结果 JSON + 状态文件 curl POST 回 ECS(文件名带时间戳轮换, 绕开大文件锁)

用法(由 n1_cmd.py 派发的 shell 命令调用):
  cd /mnt/workspace/node1/gh_repo && git pull && python node/run_experiments_on_node1.py
"""
import os, sys, time, glob, json, subprocess, shutil, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # gh_repo
ECS = "http://47.99.131.12:8081"
TS = int(time.time())

CKPT = os.path.join(ROOT, "out_smollm2_135m_pt", "checkpoint-latest", "pytorch_model.bin")
BASE = os.path.join(ROOT, "models", "SmolLM2-200M")
R1_JSON = os.path.join(ROOT, "eval", "v1_02_alpha_ladder.json")
E1_JSON = os.path.join(ROOT, "eval", "e1_result.json")

status = {"ts": TS, "stages": [], "errors": []}


def log(stage, info):
    status["stages"].append({"stage": stage, "info": info, "t": int(time.time())})
    print(f"[{stage}] {info}", flush=True)


def ecs_upload(local, name):
    try:
        with open(local, "rb") as f:
            data = f.read()
        req = urllib.request.Request(
            f"{ECS}/upload?name={name}&key=k0u4mail", data=data, method="POST")
        r = urllib.request.urlopen(req, timeout=120)
        log("upload", f"{name} -> {r.status} ({len(data)}B)")
        return name
    except Exception as e:
        status["errors"].append(f"upload {name}: {e}")
        log("upload-err", f"{name}: {e}")
        return None


def shell(cmd, env=None):
    e = dict(os.environ)
    if env:
        e.update(env)
    p = subprocess.run(cmd, shell=True, cwd=ROOT, env=e,
                       capture_output=True, text=True)
    return p


def main():
    os.chdir(ROOT)
    # 0. git pull
    try:
        p = shell("git pull --ff-only")
        log("git-pull", f"rc={p.returncode} out={p.stdout[-300:]}")
    except Exception as e:
        status["errors"].append(f"git pull: {e}")
        log("git-pull-err", str(e))

    # 1. 摆权重 + tokenizer
    if not os.path.exists(CKPT):
        log("ckpt-missing", "135M 权重缺失, 准备重组 0.1B")
        reassemble = os.path.join(ROOT, "backups", "0.1B", "reassemble.py")
        if os.path.exists(reassemble):
            p = shell(f"python {reassemble}")
            log("reassemble", f"rc={p.returncode} out={p.stdout[-400:]}")
        else:
            status["errors"].append("reassemble.py 不存在")
            log("reassemble-missing", "backups/0.1B/reassemble.py 不在(需 git pull)")
        # 找重组出来的 ckpt
        cands = glob.glob(os.path.join(ROOT, "backups", "0.1B", "**", "pytorch_model.bin"), recursive=True)
        if cands:
            os.makedirs(os.path.dirname(CKPT), exist_ok=True)
            shutil.copy(cands[0], CKPT)
            log("ckpt-placed", f"{cands[0]} -> {CKPT}")
        else:
            status["errors"].append("找不到重组后的 pytorch_model.bin")
            log("ckpt-notfound", "重组未产出 pytorch_model.bin")
    else:
        log("ckpt-ok", CKPT)

    # 找 tokenizer 目录(base/) 摆到 models/SmolLM2-200M
    if not os.path.exists(os.path.join(BASE, "tokenizer.json")):
        tdirs = glob.glob(os.path.join(ROOT, "backups", "0.1B", "**", "tokenizer.json"), recursive=True)
        if tdirs:
            src = os.path.dirname(tdirs[0])
            os.makedirs(BASE, exist_ok=True)
            for fn in os.listdir(src):
                shutil.copy(os.path.join(src, fn), os.path.join(BASE, fn))
            log("tokenizer-placed", f"{src} -> {BASE}")
        else:
            status["errors"].append("找不到 tokenizer")
            log("tokenizer-notfound", "0.1B 包里无 tokenizer.json")
    else:
        log("tokenizer-ok", BASE)

    env = {"SMOL_BASE": BASE, "SMOL_CKPT": CKPT,
           "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}

    # 2a. V1-02
    if os.path.exists(CKPT) and os.path.exists(os.path.join(BASE, "tokenizer.json")):
        log("v1_02-start", "running v1_02_alpha_ladder (threads=4)")
        p = shell(f"python eval/v1_02_alpha_ladder.py --threads 4 --out eval/v1_02_alpha_ladder.json", env=env)
        log("v1_02-done", f"rc={p.returncode} out={p.stdout[-400:]} err={p.stderr[-300:]}")
        if os.path.exists(R1_JSON):
            ecs_upload(R1_JSON, f"n1_v1_02_{TS}.json")
        else:
            status["errors"].append("v1_02 未产出 json")
    else:
        status["errors"].append("权重/tokenizer 未就绪, 跳过 v1_02")

    # 2b. E1
    if os.path.exists(CKPT) and os.path.exists(os.path.join(BASE, "tokenizer.json")):
        log("e1-start", "running e1_subliminal_proxy")
        p = shell(f"python eval/e1_subliminal_proxy.py", env=env)
        log("e1-done", f"rc={p.returncode} out={p.stdout[-400:]} err={p.stderr[-300:]}")
        if os.path.exists(E1_JSON):
            ecs_upload(E1_JSON, f"n1_e1_{TS}.json")
        else:
            status["errors"].append("e1 未产出 json")
    else:
        status["errors"].append("权重/tokenizer 未就绪, 跳过 e1")

    # 3. 上传状态文件
    status_path = os.path.join(ROOT, f"node_exp_status_{TS}.json")
    with open(status_path, "w", encoding="utf-8") as f:
        json.dump(status, f, ensure_ascii=False, indent=2)
    ecs_upload(status_path, f"n1_exp_status_{TS}.json")
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
