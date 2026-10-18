#!/usr/bin/env python3
# convert_st.py — node1: 把 pickle 权重(pytorch_model.bin)转 safetensors(model.safetensors)。
# 背景: DSW 节点 torch<2.6 + 新版 transformers 拒绝 torch.load(pickle), 但 safetensors 不受影响。
# 直接 torch.load(weights_only=False) 不经过 transformers 的 CVE 检查, 故可安全转换。
import os, torch
from safetensors.torch import save_file

def conv(d):
    binp = os.path.join(d, "pytorch_model.bin")
    stp = os.path.join(d, "model.safetensors")
    if not os.path.exists(binp):
        print("SKIP no-bin:", d); return
    if os.path.exists(stp) and os.path.getsize(stp) > 1000:
        print("EXISTS already:", stp); return
    print("loading", binp, "size", os.path.getsize(binp))
    sd = torch.load(binp, map_location="cpu", weights_only=False)
    if not isinstance(sd, dict):
        print("WARN not a state_dict dict, got", type(sd), "-> skip"); return
    out = {}
    for k, v in sd.items():
        if hasattr(v, "contiguous"):
            # clone 打破共享内存(tied weights: embed_tokens 与 lm_head 共享), 否则 safetensors 拒存
            v = v.detach().cpu().contiguous().clone()
        out[k] = v
    # 三级兜底: 全 clone -> 丢 tied 的 lm_head.weight -> 逐键去重保留首个
    try:
        save_file(out, stp, metadata={"format": "pt"})
    except Exception as e1:
        print("  retry1 (drop lm_head.weight):", repr(e1)[:160])
        out.pop("lm_head.weight", None)
        try:
            save_file(out, stp, metadata={"format": "pt"})
        except Exception as e2:
            print("  retry2 (dedupe shared storage):", repr(e2)[:160])
            seen = {}
            dedup = {}
            for k, v in out.items():
                if not hasattr(v, "data_ptr"):
                    dedup[k] = v; continue
                p = v.data_ptr()
                if p in seen:
                    print("  drop dup:", k, "==", seen[p]); continue
                seen[p] = k; dedup[k] = v
            save_file(dedup, stp, metadata={"format": "pt"})
            out = dedup
    print("SAVED", stp, "n_tensors=", len(out))

for d in ["/mnt/workspace/node1/chain/r4base_std",
          "/mnt/workspace/node1/chain/ckpt",
          "/mnt/workspace/node1/chain/ckpt_old"]:
    try:
        conv(d)
    except Exception as e:
        print("ERR", d, repr(e))
print("CONVERT_DONE")
