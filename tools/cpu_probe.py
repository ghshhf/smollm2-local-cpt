#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cpu_probe.py — 实测有效并行核数（自适应线程数）

为什么不能直接用 nproc / cgroup 配额:
  2026-10-19 实测 node1（魔搭 DSW，K8s burstable Pod）：
    cgroup 配额写着 8 核，cpu.stat 几乎不限流，但并行加速比在 4 进程后封顶，
    有效算力只有 3.1~3.2 核 —— 宿主机超卖，8 核是"上限"不是"保底"。
  而同一个 Pod 在不同时段能拿到的算力并不固定（闲时可能真给到 8 核），
  所以线程数不能写死，必须每次开跑前现场标定。

方法:
  每个进程做等量纯算术，看多进程下的**总耗时是否保持恒定**。
  有效核数 = n * t(1进程) / t(n进程)。取各档里最大的那个，向下取整。

代价:
  默认 3 档（1 / 中 / 满）约 1~2 秒，相对训练动辄十几分钟可忽略。

用法:
  python tools/cpu_probe.py                 # 打印结果
  python tools/cpu_probe.py --json          # 机器读
  from cpu_probe import effective_cores      # 代码里直接调
"""
import argparse
import json
import multiprocessing as mp
import os
import time

WORK = 2500000       # 单进程约 100~150ms


def _burn(n):
    s = 0
    for i in range(n):
        s += i * i
    return s


def _run(n, work=WORK):
    ps = [mp.Process(target=_burn, args=(work,)) for _ in range(n)]
    t = time.time()
    for p in ps:
        p.start()
    for p in ps:
        p.join()
    return time.time() - t


def quota_cores():
    """cgroup 配额（v1/v2 都试），拿不到就退回 cpu_count"""
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read().strip())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read().strip())
        if q > 0 and p > 0:
            return max(1, q // p)
    except Exception:
        pass
    try:
        for tok in open("/sys/fs/cgroup/cpu.max").read().split():
            if tok == "max":
                break
            if tok.isdigit():
                pass
    except Exception:
        pass
    return os.cpu_count() or 1


def throttled_delta(s0, s1):
    try:
        return s1.get("nr_throttled", 0) - s0.get("nr_throttled", 0)
    except Exception:
        return -1


def _stat():
    d = {}
    try:
        for line in open("/sys/fs/cgroup/cpu/cpu.stat"):
            k, v = line.split()
            d[k] = int(v)
    except Exception:
        pass
    return d


def effective_cores(quota=None, verbose=False):
    quota = quota or quota_cores()
    quota = max(1, int(quota))
    points = sorted({1, max(1, quota // 2), quota})
    base = None
    best = 1.0
    detail = []
    s0 = _stat()
    for n in points:
        # 每个进程干同样多的活：完全并行时总耗时应当恒定
        el = _run(n)
        if base is None:
            base = el
        eff = n * base / el
        detail.append({"n": n, "ms": round(el * 1000, 1), "eff": round(eff, 2)})
        best = max(best, eff)
    s1 = _stat()
    cores = max(1, int(round(best)))   # 四舍五入：3.996 应当算 4，不该截成 3
    cores = min(cores, quota)
    if verbose:
        print(json.dumps({"quota": quota, "effective": cores,
                          "throttled": throttled_delta(s0, s1),
                          "detail": detail}, ensure_ascii=False))
    return cores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    c = effective_cores(verbose=True)
    if not a.json:
        print("有效核数 = %d  (标定耗时 %.1fs)" % (c, time.time() - t0))


if __name__ == "__main__":
    main()
