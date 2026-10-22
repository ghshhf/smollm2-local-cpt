#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""tools/dev_pick.py — 设备与线程的**唯一**自适应入口。

用户 2026-10-19 要求：以后所有训练都自适应，"就一行代码的事"。
理由不是省事，是可移植 —— 同一份代码换机器（本机 8C12T / 远程 4 核 burstable /
别的电脑 / 将来带 GPU 的实例）都不需要改参数，自己认。

用法（训练脚本入口处两行）：
    from dev_pick import setup
    device, n_threads = setup()

约定：
  * 有 CUDA 就用 CUDA（GPU 实例上跑得快一个量级），没有就 CPU；
  * CPU 的线程数不写死，走 cpu_probe 现场标定（远程实测 8 核配额只给到 4 核，
    线程开多反而慢 16%）；
  * CUDA 上不动 torch 线程数（那是 CPU 侧的事，改了没意义）。
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)


def device():
    """cuda > mps > cpu，不解释参数，直接认。"""
    try:
        import torch
    except Exception:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def threads(quota_default=4):
    """现场标定有效核数；标不出来就退到 quota_default，绝不崩。"""
    for v in (os.environ.get("POLARIS_THREADS"), os.environ.get("SUP_THREADS")):
        if v and str(v).strip().isdigit() and int(v) > 0:
            return int(v)
    try:
        from cpu_probe import effective_cores
        return max(1, int(effective_cores()))
    except Exception as e:
        print("[dev] 线程标定失败(%r)，退用 %d" % (e, quota_default))
        return quota_default


def setup(verbose=True, quota_default=4):
    """设置好一切并返回 (device, n_threads)。幂等，重复调用无害。"""
    dev = device()
    n = threads(quota_default)
    try:
        import torch
        if dev == "cpu":
            torch.set_num_threads(n)
            for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                os.environ.setdefault(k, str(n))
        if verbose:
            if dev == "cuda":
                name = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory / 1e9
                print("[dev] GPU  %s  %.1fGB  (CPU 线程仍为 %d)" % (name, mem, n))
            else:
                print("[dev] CPU  %d 线程(现场标定)" % n)
    except Exception:
        pass
    return dev, n


if __name__ == "__main__":
    d, n = setup()
    try:
        import torch
        print("  torch=%s  threads=%d  cuda=%s" % (torch.__version__, torch.get_num_threads(),
                                                   torch.cuda.is_available()))
    except Exception:
        print("  (torch 未安装)")
