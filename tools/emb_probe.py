# -*- coding: utf-8 -*-
"""量 LLM embedding 的真实尺度，用来给 projector 定归一化目标。"""
import argparse

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", required=True)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(a.llm, torch_dtype=torch.float32)
    e = m.get_input_embeddings().weight.detach()
    n = e.norm(dim=1)
    print("[emb] shape=%s" % (tuple(e.shape),))
    print("[emb] per-dim std = %.5f" % float(e.std()))
    print("[emb] 行 norm: mean=%.3f  std=%.3f  p50=%.3f  p95=%.3f  max=%.3f"
          % (float(n.mean()), float(n.std()),
             float(n.median()), float(n.quantile(0.95)), float(n.max())))
    # 每层 RMSNorm 的 weight
    try:
        rm = m.model.layers[0].input_layernorm.weight.detach()
        print("[rms] layer0 input_layernorm weight mean=%.4f min=%.4f max=%.4f"
              % (float(rm.mean()), float(rm.min()), float(rm.max())))
    except Exception as ex:
        print("[rms] %s" % ex)


if __name__ == "__main__":
    main()
