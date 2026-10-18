# -*- coding: utf-8 -*-
"""
module_port_test.py — 模块跨代移植的原理实测

回答三个问题(全部给数字, 不给结论):
  Q1 形状: 0.2B 的 LoRA 各模块放进 0.3B, 哪些真的对不上? 逐键列出。
  Q2 保函数: 零填充移植后, 有效增量 dW 是否逐位不变? (数学上应完全相等)
  Q3 表示漂移: 同一输入, 两代的残差流方向差多少? (这是移植失败的真原因)

用法:
  python tools/module_port_test.py --do shape
  python tools/module_port_test.py --do dw
  python tools/module_port_test.py --do drift
  python tools/module_port_test.py --do gen
"""
import argparse, json, os, struct, sys, time

import torch

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
M02 = os.path.join(BASE, "backups/0.2B/r6_final_20261016")
M03 = os.path.join(BASE, "backups/0.3B/polaris-03b_20261018")
LORA = os.path.join(BASE, "out_lora_chat_v4")

PROMPTS = [
    ("寒暄", "你好，你是谁？\n答："),
    ("常识", "中国的首都是哪里？\n答："),
    ("知识", "什么是比特币？\n答："),
]


def lora_header():
    p = os.path.join(LORA, "adapter_model.safetensors")
    with open(p, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def cfg(p):
    return json.load(open(os.path.join(p, "config.json"), encoding="utf-8"))


# ---------------------------------------------------------------- Q1 形状
def do_shape():
    h = lora_header()
    c2, c3 = cfg(M02), cfg(M03)
    H2, H3 = c2["hidden_size"], c3["hidden_size"]
    I2, I3 = c2["intermediate_size"], c3["intermediate_size"]
    L2, L3 = c2["num_hidden_layers"], c3["num_hidden_layers"]
    kv2 = c2["num_key_value_heads"] * (H2 // c2["num_attention_heads"])
    kv3 = c3["num_key_value_heads"] * (H3 // c3["num_attention_heads"])

    print(f"0.2B: layers={L2} hidden={H2} inter={I2} kv_dim={kv2}")
    print(f"0.3B: layers={L3} hidden={H3} inter={I3} kv_dim={kv3}\n")

    # 0.3B 的基座权重形状
    from safetensors.torch import load_file
    w3 = load_file(os.path.join(M03, "model.safetensors"))

    def base_of(key):
        """把 lora_A/lora_B 键还原成基座权重键"""
        k = key.replace(".lora_A.weight", ".weight").replace(".lora_B.weight", ".weight")
        k = k.replace("base_model.model.", "")
        return k

    rows = []
    for name in sorted(h):
        if name == "__metadata__":
            continue
        bkey = base_of(name)
        if bkey not in w3:
            rows.append((name, tuple(h[name]["shape"]), None, "基座无此键"))
            continue
        want = tuple(h[name]["shape"])
        got = tuple(w3[bkey].shape)
        # lora_A 期望 [r, d_in]; lora_B 期望 [d_out, r]
        if "lora_A" in name:
            ok = want[1] == got[1]
            need = f"[{want[0]}, {got[1]}]"
        else:
            ok = want[0] == got[0]
            need = f"[{got[0]}, {want[1]}]"
        rows.append((name, want, got, "OK" if ok else f"需改-> {need}"))

    mod_stat = {}
    for n, w, g, s in rows:
        mod = n.split(".layers.")[1].split(".", 1)[1].rsplit(".", 1)[0] if ".layers." in n else n
        d = mod_stat.setdefault(mod, {"ok": 0, "bad": 0})
        if s == "OK":
            d["ok"] += 1
        else:
            d["bad"] += 1
    print("按模块汇总(跨全部层):")
    for m, d in mod_stat.items():
        print(f"  {m:<28s} 兼容 {d['ok']:>3d}  需调整 {d['bad']:>3d}")
    print()
    print("逐键明细(只显示第 0 层, 其余层同形):")
    for n, w, g, s in rows:
        if ".layers.0." in n or ".layers.1." not in n and ".layers." not in n:
            if ".layers." in n and ".layers.0." not in n:
                continue
            print(f"  {n.split('.layers.')[-1] if '.layers.' in n else n:<50s}")
            print(f"      adapter {w}  基座 {g}  {s}")
    print()
    extra = [n for n, w, g, s in rows if s != "OK"]
    print(f"总张量 {len(rows)} | 直接兼容 {len(rows)-len(extra)} | 需调整 {len(extra)}")
    print(f"层覆盖: adapter 有 {L2} 层, 目标模型有 {L3} 层 -> 缺 {L3-L2} 层无模块")


# ---------------------------------------------------------------- Q2 保函数
def do_dw():
    """验证: 零填充移植后, dW = B@A 在旧子空间上逐位不变"""
    from safetensors.torch import load_file
    w = load_file(os.path.join(LORA, "adapter_model.safetensors"))
    c2, c3 = cfg(M02), cfg(M03)
    I2, I3 = c2["intermediate_size"], c3["intermediate_size"]
    r, alpha = 16, 32
    scale = alpha / r

    print(f"inter: {I2} -> {I3}   新增宽度 {I3-I2}\n")

    def dw(a, b):
        return (b.float() @ a.float()) * scale

    # up_proj: A[16,576] B[3072,16] -> 需要 B[4608,16], 零填充新行
    k = "base_model.model.model.layers.0.mlp.up_proj"
    A, B = w[k + ".lora_A.weight"], w[k + ".lora_B.weight"]
    dw_old = dw(A, B)
    B_pad = torch.zeros(I3, B.shape[1], dtype=B.dtype)
    B_pad[:B.shape[0]] = B
    dw_new = dw(A, B_pad)
    d = (dw_new[:dw_old.shape[0]] - dw_old).abs().max().item()
    tail = dw_new[dw_old.shape[0]:].abs().max().item()
    print(f"[up_proj]   dW {tuple(dw_old.shape)} -> {tuple(dw_new.shape)}")
    print(f"   旧子空间 max|diff| = {d:.3e}   新增行 max|.| = {tail:.3e}")

    # down_proj: A[16,3072] B[576,16] -> 需要 A[16,4608], 零填充新列
    k = "base_model.model.model.layers.0.mlp.down_proj"
    A, B = w[k + ".lora_A.weight"], w[k + ".lora_B.weight"]
    dw_old = dw(A, B)
    A_pad = torch.zeros(A.shape[0], I3, dtype=A.dtype)
    A_pad[:, :A.shape[1]] = A
    dw_new = dw(A_pad, B)
    d = (dw_new[:, :dw_old.shape[1]] - dw_old).abs().max().item()
    tail = dw_new[:, dw_old.shape[1]:].abs().max().item()
    print(f"[down_proj] dW {tuple(dw_old.shape)} -> {tuple(dw_new.shape)}")
    print(f"   旧子空间 max|diff| = {d:.3e}   新增列 max|.| = {tail:.3e}")
    print("\n=> 零填充是严格保函数的: 旧子空间逐位相等, 新增维度贡献恰为 0")


# ---------------------------------------------------------------- Q3 表示漂移
def do_drift():
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(8)
    tok = AutoTokenizer.from_pretrained(M02)

    ids = tok("你好，你是谁？\n答：", return_tensors="pt").input_ids
    hs = {}
    for tag, path in [("02B", M02), ("03B", M03)]:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
        with torch.no_grad():
            o = m(ids, output_hidden_states=True)
        hs[tag] = [x[0].float() for x in o.hidden_states]   # [seq, hidden]
        del m
    print(f"输入: {ids.shape[1]} token, 隐藏维 {hs['02B'][0].shape[1]}\n")
    print(f"{'相对深度':<10s} {'绝对层':<8s} {'cos(02B,03B)':>14s}")
    n2, n3 = len(hs["02B"]) - 1, len(hs["03B"]) - 1
    for frac in [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0]:
        i2 = int(round(frac * n2)); i3 = int(round(frac * n3))
        a, b = hs["02B"][i2], hs["03B"][i3]
        cos = torch.nn.functional.cosine_similarity(a, b, dim=-1).mean().item()
        print(f"{frac*100:>6.0f}%    {i2:>2d} vs {i3:<3d}   {cos:>14.4f}")
    # 同模型自比, 作 sanity
    a = hs["02B"][n2]
    print(f"\nsanity 02B vs 02B = {torch.nn.functional.cosine_similarity(a, a, dim=-1).mean():.4f}")


# ---------------------------------------------------------------- Q4 生成
def port_lora(out_path):
    """零填充移植: 生成一个 0.3B 可用的 adapter 目录"""
    from safetensors.torch import load_file, save_file
    w = load_file(os.path.join(LORA, "adapter_model.safetensors"))
    c2, c3 = cfg(M02), cfg(M03)
    I2, I3 = c2["intermediate_size"], c3["intermediate_size"]
    L3 = c3["num_hidden_layers"]

    new = {}
    for k, v in w.items():
        layer = int(k.split(".layers.")[1].split(".")[0])
        if layer >= L3:
            continue
        if "up_proj" in k or "gate_proj" in k:
            if "lora_B" in k and v.shape[0] == I2:
                p = torch.zeros(I3, v.shape[1], dtype=v.dtype); p[:I2] = v; v = p
        if "down_proj" in k:
            if "lora_A" in k and v.shape[1] == I2:
                p = torch.zeros(v.shape[0], I3, dtype=v.dtype); p[:, :I2] = v; v = p
        new[k] = v
    os.makedirs(out_path, exist_ok=True)
    save_file(new, os.path.join(out_path, "adapter_model.safetensors"))
    c = json.load(open(os.path.join(LORA, "adapter_config.json"), encoding="utf-8"))
    c["base_model_name_or_path"] = M03
    json.dump(c, open(os.path.join(out_path, "adapter_config.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(f"[port] {len(new)} 个张量 -> {out_path}")
    return out_path


def do_gen(n=40):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.set_num_threads(8)

    def run(tag, path, adapter=None):
        tok = AutoTokenizer.from_pretrained(path)
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32).eval()
        if adapter:
            from peft import PeftModel
            try:
                m = PeftModel.from_pretrained(m, adapter)
                m = m.merge_and_unload() if hasattr(m, "merge_and_unload") else m
            except Exception as e:
                print(f"  !! {tag} 装载失败: {type(e).__name__}: {str(e)[:160]}")
                return
        print(f"\n===== {tag} =====")
        for name, p in PROMPTS:
            ids = tok(p, return_tensors="pt").input_ids
            with torch.no_grad():
                o = m.generate(ids, max_new_tokens=n, do_sample=False,
                               repetition_penalty=1.05, pad_token_id=tok.eos_token_id)
            txt = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=True)
            print(f"[{name}] {txt.strip()[:70]!r}")
        del m

    run("A. 0.2B + LoRA(chat_v4)  [原生]", M02, LORA)
    run("B. 0.3B 裸模型", M03)
    p = port_lora(os.path.join(BASE, "models/chat_lora_v4_port03b"))
    run("C. 0.3B + 零填充移植的 LoRA [不重训]", M03, p)
    run("D. 0.3B + LoRA 原样直装 [应失败]", M03, LORA)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--do", required=True, choices=["shape", "dw", "drift", "gen"])
    ap.add_argument("--n", type=int, default=40)
    a = ap.parse_args()
    {"shape": do_shape, "dw": do_dw, "drift": do_drift,
     "gen": lambda: do_gen(a.n)}[a.do]()
