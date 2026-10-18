#!/usr/bin/env python3
# fuel_relay.py — 本地「进化档案回收」腿(全自动, schtasks 15min/轮)
#
# 背景: node1 自进化闭环每周期把银行导出的 dream_fuel.jsonl 外送 ECS 信箱
# (evolve_cycle.sh 的 [fuel] 段)。这台机器负责把它拉回本地档案仓、按
# output 定码去重合并、有增量就自动 git commit —— 实现「node1 半夜进化,
# 本地档案自动长见识」, 全程无人值守。
#
# 用法:
#   python tools/fuel_relay.py            # 跑一轮(拉取+合并+有变更则归档)
#   python tools/fuel_relay.py --no-git   # 只合并不提交(调试用)
#
# 去重口径: output(完整代码)逐字节相同判重; 与 triple_bank 入库去重同口径。
# 信箱语义: /file/X 可重复读(非单次消费), 拉失败不删本地旧档。

import argparse
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # train_15b/
ECS = "http://47.99.131.12:8081/file"
KEY = "k0u4mail"
SRC = "dream_fuel.jsonl"
DST_DIR = os.path.join(HERE, "data", "fuel_relay")
DST = os.path.join(DST_DIR, "dream_fuel.jsonl")
LOG = os.path.join(HERE, "logs", "fuel_relay.log")


def log(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    line = "%s %s" % (__import__("time").strftime("%F %T"), msg)
    print(line)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def fetch(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "fuel-relay/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def git(*args):
    subprocess.run(["git", "-C", HERE] + list(args),
                   capture_output=True, timeout=120)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-git", action="store_true")
    a = ap.parse_args()

    # 1) 拉 ECS 信箱上的新燃料
    try:
        raw = fetch("%s/%s?key=%s" % (ECS, SRC, KEY)).decode("utf-8")
    except Exception as e:
        log("[relay] 拉取失败(%s)，保留本地旧档不动" % e)
        return 0
    new_rows = []
    for ln in raw.splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            new_rows.append(json.loads(ln))
        except Exception:
            continue
    if not new_rows:
        log("[relay] 信箱文件为空，无操作")
        return 0

    # 2) 与本地合并(按 output 定码去重)
    old = {}
    if os.path.exists(DST):
        with open(DST, encoding="utf-8") as f:
            for ln in f:
                ln = ln.strip()
                if ln:
                    try:
                        r = json.loads(ln)
                        old[r.get("output", "")] = r
                    except Exception:
                        pass
    added = 0
    for r in new_rows:
        k = r.get("output", "")
        if k and k not in old:
            old[k] = r
            added += 1
    if added == 0:
        log("[relay] 无新增(信箱 %d 条, 本地已有 %d 条)" % (len(new_rows), len(old)))
        return 0

    # 3) 落盘(按 instruction 排序，稳定 diff)
    os.makedirs(DST_DIR, exist_ok=True)
    rows = sorted(old.values(), key=lambda r: r.get("instruction", ""))
    with open(DST, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log("[relay] 新增 %d 条 => 本地共 %d 条" % (added, len(rows)))

    # 4) 有增量就自动归档(显式路径 + -f: fuel 是 node1 进化史的不可重建
    #    物证——ECS 信箱里同名文件会被逐周期覆盖——按"换台机器能否重建"
    #    标准必须入库; 绝不 git add -a)
    if not a.no_git:
        git("add", "-f", "data/fuel_relay/dream_fuel.jsonl")
        r = subprocess.run(["git", "-C", HERE, "diff", "--cached", "--quiet"],
                           capture_output=True)
        if r.returncode != 0:
            git("commit", "-m",
                "fuel_relay: +%d 条 node1 自进化燃料(自动归档)" % added)
            p = subprocess.run(["git", "-C", HERE, "push", "origin", "main"],
                               capture_output=True, timeout=180)
            log("[relay] 已归档并推送(rc=%d)" % p.returncode)
        else:
            log("[relay] 无差异，跳过提交")
    return 0


if __name__ == "__main__":
    sys.exit(main())
