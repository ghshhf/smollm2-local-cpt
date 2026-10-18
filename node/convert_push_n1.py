# -*- coding: utf-8 -*-
"""convert_push_n1.py — (node1 上跑) torch ckpt -> npz, 并把三件套 PUT 到 OSS。
生成: /root/r4work/r4_ckpt.npz (fp32, 键=HF 参数名)
上传: relay/r4_ckpt.npz  relay/tok_cache.npy  relay/tokenizer.json
"""
import json, sys, time, urllib.request

CKPT = "/root/r4work/r4_run/checkpoint-latest/pytorch_model.bin"
NPZ  = "/root/r4work/r4_ckpt.npz"
CACHE = "/root/r4work/tok_cache.npy"
TOK   = "/root/r4work/r4_base/tokenizer.json"

PUT_NPZ  = sys.argv[1]   # 签名 PUT URL (公网签名, 但 node1 换 internal host 用)
PUT_CACHE = sys.argv[2]
PUT_TOK   = sys.argv[3]

def to_internal(url):
    return url.replace("oss-cn-hangzhou.aliyuncs.com", "oss-cn-hangzhou-internal.aliyuncs.com")

def put(url, path):
    t0 = time.time()
    data = open(path, "rb").read()
    req = urllib.request.Request(url, data=data, method="PUT")
    with urllib.request.urlopen(req, timeout=7200) as r:
        print(f"[put] {path} ({len(data)/1e6:.0f}MB) -> {r.status} in {time.time()-t0:.0f}s")
    sys.stdout.flush()

def main():
    import torch, numpy as np
    print("[load]", CKPT)
    try:
        sd = torch.load(CKPT, map_location="cpu", weights_only=True)
    except Exception:
        sd = torch.load(CKPT, map_location="cpu")
    out = {k: v.detach().cpu().float().numpy() for k, v in sd.items()}
    np.savez(NPZ, **out)
    meta = {"keys": list(out.keys()), "shapes": {k: list(v.shape) for k, v in out.items()},
            "dtypes": {k: str(v.dtype) for k, v in out.items()}}
    json.dump(meta, open("/root/r4work/r4_ckpt_meta.json", "w"))
    print(f"[npz] {len(out)} tensors, total {sum(v.size for v in out.values())/1e6:.1f}M params")
    sys.stdout.flush()

    put(to_internal(PUT_NPZ), NPZ)
    put(to_internal(PUT_CACHE), CACHE)
    put(to_internal(PUT_TOK), TOK)
    print("[done] all uploaded")

if __name__ == "__main__":
    main()
