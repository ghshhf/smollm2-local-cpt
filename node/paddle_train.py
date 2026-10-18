# -*- coding: utf-8 -*-
"""paddle_train.py — (node2=AI Studio Iluvatar GPU 上跑) R4 续训, Paddle 版。

与 node1(torch CPU) 对齐:
  - 数据: flat int32 token 流 mmap, 随机窗口 max_len+1
  - 优化: AdamW(lr, wd=0.01), warmup 20 线性 + cosine
  - 模型: Llama 架构(576/30L/9Q3KV/3072ffn/vocab49152/tied), fp32
权重从 r4_ckpt.npz (HF 参数名) 加载, Linear 自动转置。

用法:
  python paddle_train.py --npz r4_ckpt.npz --cache tok_cache.npy \
      --steps 30 --bs 16 --accum 2 --put_url "<签名PUT URL>"
"""
import argparse, os, sys, json, time, gzip
import numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/aistudio/node2/data/r4_ckpt_fp16.npz")
    ap.add_argument("--cache", default="/home/aistudio/node2/data/tok_cache.npy")
    ap.add_argument("--jsonl", default="", help="语料(jsonl 或 .gz), cache 不存在时现场 tokenize")
    ap.add_argument("--tok_dir", default="/home/aistudio/node2/data/tok", help="tokenizer.json 所在目录")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--max_len", type=int, default=512)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--save_every", type=int, default=200)
    ap.add_argument("--out_dir", default="/home/aistudio/node2/run")
    ap.add_argument("--put_url", default="", help="checkpoint 签名 PUT URL (弃用)")
    ap.add_argument("--upload", default="", help="ECS /upload 回传: http://host:port/upload key=KEY name=prefix (分块)")
    args = ap.parse_args()

    import paddle
    import paddle.nn as nn
    import paddle.nn.functional as F

    dev = "iluvatar_gpu"
    try:
        paddle.set_device(dev)
        print(f"[dev] {dev} ok")
    except Exception as e:
        print(f"[dev] {dev} FAIL -> cpu: {e}")
        dev = "cpu"
        paddle.set_device(dev)
    sys.stdout.flush()

    # ---------------- config (与 r4_base/config.json 一致) ----------------
    H, NL, NH, NKV, FF, V = 576, 30, 9, 3, 3072, 49152
    HD = 64                      # head_dim
    EPS, THETA = 1e-5, 100000.0
    G = NH // NKV                # 3

    # ---------------- model ----------------
    class RMSNorm(nn.Layer):
        def __init__(self, dim):
            super().__init__()
            self.weight = self.create_parameter([dim], default_initializer=nn.initializer.Constant(1.0))
        def forward(self, x):
            var = x.cast("float32").pow(2).mean(-1, keepdim=True)
            xn = x.cast("float32") * paddle.rsqrt(var + EPS)
            return (xn.cast(x.dtype) * self.weight).cast(x.dtype)

    class Attn(nn.Layer):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(H, NH * HD, bias_attr=False)
            self.k_proj = nn.Linear(H, NKV * HD, bias_attr=False)
            self.v_proj = nn.Linear(H, NKV * HD, bias_attr=False)
            self.o_proj = nn.Linear(NH * HD, H, bias_attr=False)
        def forward(self, x, cos, sin, mask):
            B, T, _ = x.shape
            q = self.q_proj(x).reshape([B, T, NH, HD]).transpose([0, 2, 1, 3])
            k = self.k_proj(x).reshape([B, T, NKV, HD]).transpose([0, 2, 1, 3])
            v = self.v_proj(x).reshape([B, T, NKV, HD]).transpose([0, 2, 1, 3])
            # RoPE (逐 head_dim 旋转)
            qr = q * cos + paddle.concat([-q[:, :, :, HD//2:], q[:, :, :, :HD//2]], axis=-1) * sin
            kr = k * cos + paddle.concat([-k[:, :, :, HD//2:], k[:, :, :, :HD//2]], axis=-1) * sin
            # GQA: KV 重复 G 组
            kr = paddle.repeat_interleave(kr, G, axis=1)
            vr = paddle.repeat_interleave(v, G, axis=1)
            sc = paddle.matmul(qr, kr.transpose([0, 1, 3, 2])) / (HD ** 0.5) + mask
            p = F.softmax(sc, axis=-1)
            o = paddle.matmul(p, vr)                      # (B,NH,T,HD)
            o = o.transpose([0, 2, 1, 3]).reshape([B, T, NH * HD])
            return self.o_proj(o)

    class Mlp(nn.Layer):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(H, FF, bias_attr=False)
            self.up_proj = nn.Linear(H, FF, bias_attr=False)
            self.down_proj = nn.Linear(FF, H, bias_attr=False)
        def forward(self, x):
            return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    class Block(nn.Layer):
        def __init__(self):
            super().__init__()
            self.self_attn = Attn()
            self.mlp = Mlp()
            self.input_layernorm = RMSNorm(H)
            self.post_attention_layernorm = RMSNorm(H)
        def forward(self, x, cos, sin, mask):
            x = x + self.self_attn(self.input_layernorm(x), cos, sin, mask)
            x = x + self.mlp(self.post_attention_layernorm(x))
            return x

    class Llama(nn.Layer):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(V, H)
            self.layers = nn.LayerList([Block() for _ in range(NL)])
            self.final_norm = RMSNorm(H)
            self.lm_head = nn.Linear(H, V, bias_attr=False)  # weight 与 embed 共享(tied)
        def forward(self, ids):
            B, T = ids.shape
            # RoPE cache
            pos = paddle.arange(T, dtype="float32")
            inv = 1.0 / (THETA ** (paddle.arange(0, HD, 2, dtype="float32") / HD))
            fr = paddle.outer(pos, inv)                          # (T, HD/2)
            emb = paddle.concat([fr, fr], axis=-1)               # (T, HD)
            cos = emb.cos()[None, None, :, :]                    # (1,1,T,HD)
            sin = emb.sin()[None, None, :, :]
            mask = paddle.triu(paddle.full([T, T], float("-inf")), 1)[None, None, :, :]
            h = self.embed(ids)
            for blk in self.layers:
                h = blk(h, cos, sin, mask)
            h = self.final_norm(h)
            return self.lm_head(h)                              # (B,T,V)

    model = Llama()

    # ---------------- load npz (HF names -> paddle) ----------------
    def resolve(key):
        """'model.layers.3.self_attn.q_proj.weight' -> (layer_obj, param_attr)"""
        parts = key.split(".")
        if key == "model.embed_tokens.weight":
            return model.embed.weight
        if key == "model.norm.weight":
            return model.final_norm.weight
        if key == "lm_head.weight":
            return model.lm_head.weight  # untied on GPU: 独立参数(布局必须 (in,out))
        assert parts[0] == "model" and parts[1] == "layers", key
        blk = model.layers[int(parts[2])]
        rest = parts[3:]
        obj = blk
        for p in rest[:-1]:
            obj = getattr(obj, p)
        return getattr(obj, rest[-1])

    print("[load] npz:", args.npz)
    z = np.load(args.npz)
    n_loaded = 0
    for key in z.files:
        arr = z[key].astype("float32")   # fp16 存储 -> fp32 参数
        param = resolve(key)
        if param is None:
            continue
        if param.ndim == 2 and param.shape[0] != arr.shape[0] and param.shape[1] == arr.shape[0]:
            arr = arr.T          # torch Linear (out,in) -> paddle (in,out)
        assert tuple(param.shape) == tuple(arr.shape), f"{key}: {param.shape} vs {arr.shape}"
        param.set_value(paddle.to_tensor(arr))
        n_loaded += 1
    total = sum(p.size for p in model.parameters()) / 1e6
    print(f"[load] {n_loaded} tensors, {total:.1f}M params on {dev} (lm_head untied)")
    sys.stdout.flush()

    # ---------------- data ----------------
    if not os.path.exists(args.cache):
        assert args.jsonl, "无 cache 且未给 --jsonl"
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(args.tok_dir)
        print("[cache] 现场 tokenize:", args.jsonl); sys.stdout.flush()
        opener = gzip.open if args.jsonl.endswith(".gz") else open
        chunks, total, buf = [], 0, []
        with opener(args.jsonl, "rt", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    txt = json.loads(line).get("text", "")
                except Exception:
                    txt = line
                if not txt or len(txt) < 20:
                    continue
                buf.append(txt)
                if len(buf) >= 400:
                    ids = tk("\n".join(buf), return_tensors=None, truncation=False).input_ids
                    chunks.append(np.array(ids, dtype=np.int32)); total += len(ids); buf = []
                    if total % 5_000_000 < 3000:
                        print(f"[cache] {total/1e6:.0f}M tokens"); sys.stdout.flush()
        if buf:
            ids = tk("\n".join(buf), return_tensors=None, truncation=False).input_ids
            chunks.append(np.array(ids, dtype=np.int32)); total += len(ids)
        arr = np.concatenate(chunks)
        np.save(args.cache, arr)
        print(f"[cache] 完成: {total} tokens -> {args.cache}"); sys.stdout.flush()
    arr = np.load(args.cache, mmap_mode="r")
    N = arr.shape[0]
    print(f"[cache] {N} tokens")

    def sample_batch():
        idx = np.random.randint(0, N - args.max_len - 1, size=args.bs)
        batch = np.stack([arr[s:s + args.max_len + 1].astype(np.int64) for s in idx])
        return paddle.to_tensor(batch)

    # ---------------- optimizer ----------------
    opt = paddle.optimizer.AdamW(learning_rate=args.lr,
                                 parameters=list({id(p): p for p in model.parameters()}.values()),
                                 weight_decay=0.01)

    def lr_at(step):
        if step <= args.warmup:
            return args.lr * 0.1 + 0.9 * args.lr * step / max(1, args.warmup)
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        import math
        return args.lr * (0.05 + 0.95 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))

    os.makedirs(args.out_dir, exist_ok=True)

    def save_ckpt(step):
        out = {}
        for key in z.files:
            param = resolve(key)
            if param is None:
                continue
            a = param.numpy()
            if param.ndim == 2 and param.shape[0] != z[key].shape[0] and param.shape[1] == z[key].shape[0]:
                a = a.T
            out[key] = a.astype(np.float16)   # fp16 回传, 体积减半; 加载端 astype(float32) 兼容
        path = os.path.join(args.out_dir, f"ckpt_{step}.npz")
        np.savez(path, **out)
        open(os.path.join(args.out_dir, "step.txt"), "w").write(str(step))
        print(f"[ckpt] saved @ {step} -> {path}")
        if args.upload:
            try:
                import urllib.request as _ur
                ecs, key, prefix = args.upload.split(" ")
                CHUNK = 64 * 1024 * 1024
                data_all = open(path, "rb").read()
                nch = (len(data_all) + CHUNK - 1) // CHUNK
                for i in range(nch):
                    part = data_all[i*CHUNK:(i+1)*CHUNK]
                    url = f"{ecs}/upload?name={prefix}_{step}.part{i:03d}&key={key}"
                    req = _ur.Request(url, data=part, method="POST")
                    _ur.urlopen(req, timeout=900).read()
                print(f"[ckpt] upload {nch} chunks OK")
            except Exception as e:
                print(f"[ckpt] upload FAIL: {e}")
        sys.stdout.flush()

    # ---------------- train ----------------
    print(f"[train] steps={args.steps} bs={args.bs} accum={args.accum} lr={args.lr} len={args.max_len}")
    sys.stdout.flush()
    t0 = time.time(); loss_acc = 0.0
    for step in range(1, args.steps + 1):
        opt.set_lr(lr_at(step))
        opt.clear_grad()
        for _ in range(args.accum):
            batch = sample_batch()
            logits = model(batch[:, :-1])
            loss = F.cross_entropy(logits.reshape([-1, V]), batch[:, 1:].reshape([-1]))
            loss = loss / args.accum
            loss.backward()
            loss_acc += float(loss) * args.accum
        opt.step()
        if step % 10 == 0:
            mem = ""
            try:
                mem = f" gpu_mem={paddle.device.cuda.memory_allocated()/1e9:.1f}G"
            except Exception:
                pass
            print(f"[train] step {step}/{args.steps} loss={loss_acc/10:.4f} lr={lr_at(step):.2e} t={time.time()-t0:.0f}s{mem}")
            sys.stdout.flush(); loss_acc = 0.0
        if step % args.save_every == 0:
            save_ckpt(step)
    save_ckpt(args.steps)
    print(f"[train] DONE total_t={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
