"""批量下载经过兼容性筛选的公开 LoRA 卡（同基座 Qwen2.5-1.5B-Instruct + 全 7 层）。

筛选标准由 fetch_hf_cards.py 自动核对 adapter_config.json 保证：
  1. base_model_name_or_path 含 Qwen/Qwen2.5-1.5B-Instruct（不能是 Coder 版或 4bit 量化版）
  2. target_modules 恰好是全部 7 个线性层（q/k/v/o/gate/up/down）——少层就挂不上
  3. 仓库里必须有 adapter_config.json（有则说明是未合并的 adapter，merged 模型挂不了）

用法：python fetch_hf_cards.py [--only r16] [--skip-existing]
"""
import argparse
import json
import os
import re

from huggingface_hub import HfApi, snapshot_download

ROOT = os.path.dirname(os.path.abspath(__file__))
DST = os.path.join(ROOT, "hf_cards")
NEED = {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}

# 关键词：只要像微调卡的
KW = re.compile(r"lora|adapter|sft|chat|legal|med|domain|expert|finetune|code|reason|sql|extract|classif", re.I)
# 量化/去量化产物：直接排除
QUANT = re.compile(r"GGUF|AWQ|MLX|BNB|4bit|8bit|Q\d|imatrix|i1-|GPTQ|NVFP4|MXFP4|exl|UD-|abliterated|heretic", re.I)


def discover(api):
    res = list(api.list_models(search="Qwen2.5-1.5B-Instruct", limit=1500, sort="downloads"))
    cand = [m.id for m in res if not QUANT.search(m.id) and KW.search(m.id)]
    ok = []
    for r in cand:
        try:
            p = api.hf_hub_download(r, "adapter_config.json")
            c = json.load(open(p, encoding="utf-8"))
        except Exception:
            continue          # 没有 adapter_config.json => merged 模型，跳过
        base = c.get("base_model_name_or_path", "")
        tm = set(c.get("target_modules") or [])
        if "Qwen/Qwen2.5-1.5B-Instruct" in base and tm == NEED:
            ok.append((r, c.get("r"), c.get("lora_alpha")))
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="只下某种 rank，如 r16；空=全部")
    ap.add_argument("--skip-existing", action="store_true", default=True)
    args = ap.parse_args()

    api = HfApi()
    cards = discover(api)
    if args.only:
        want = args.only.lstrip("r")
        cards = [c for c in cards if str(c[1]) == want]
    print(f"兼容性筛选通过: {len(cards)} 张")

    os.makedirs(DST, exist_ok=True)
    for repo, r, alpha in cards:
        name = repo.split("/")[-1]
        out = os.path.join(DST, name)
        if args.skip_existing and os.path.isfile(os.path.join(out, "adapter_model.safetensors")):
            print(f"  [已有] {name}")
            continue
        try:
            p = snapshot_download(repo, local_dir=out,
                                  allow_patterns=["adapter_config.json",
                                                  "adapter_model.safetensors", "README.md"])
            sz = os.path.getsize(os.path.join(p, "adapter_model.safetensors")) / 1e6
            print(f"  [OK] {name}  r={r} alpha={alpha}  {sz:.0f}MB")
        except Exception as e:
            print(f"  [FAIL] {name}: {type(e).__name__} {str(e)[:70]}")


if __name__ == "__main__":
    main()