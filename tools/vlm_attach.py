#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vlm_attach.py v2 — 给 SmolLM2 主线挂视觉（识图能力）
架构: SigLIP vision tower + 2层MLP投影 + SmolLM2 主干, early fusion

v2 修复与增强（2026-10-17）:
  [nan 修复] v1 在 Flickr8k 上 loss=nan 发散。根因: lr=1e-3 偏大 + 投影输出未约束。
    修复: ①fc2 零初始化(初始输出=0, LLaVA 同款) ②lr 默认 3e-4 ③warmup 200步
         ④输入 sanity check(视觉特征 nan/inf 直接报错退出) ⑤梯度裁剪 1.0
  [数据] 中文图文对 (COCO val2017 + COCO-CN 人工中文caption + 机翻增强)
  [阶段] align: 只训投影层 | joint: 放开视觉塔(0.05x)+LLM(0.1x) | vqa: 指令格式
  [推理] --infer 模式: 加载 projector.pt, 中文问答

用法:
  python tools/vlm_attach.py --stage align --steps 3000
  python tools/vlm_attach.py --stage joint --steps 1500 --init models/vlm_02b_align/projector.pt
  python tools/vlm_attach.py --infer --image xxx.jpg --prompt "请描述这张图片"
"""
import os, sys, json, time, math, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


class MLPProjector(nn.Module):
    """2层MLP: vision_hidden -> llm_hidden

    ⚠️ fc2 必须用微小随机初始化(std=0.01)，绝不能用零初始化：
    vtok 精确为 0 时，Llama RMSNorm 反向要除 sqrt(eps)≈316，30 层复合后
    fp32 梯度溢出 => loss 有限但梯度全 nan（2026-10-17 实测复现）。
    LLaVA "small random init" 即此意；零初始化只适用于有残差直连的结构。
    """
    def __init__(self, v_dim, l_dim):
        super().__init__()
        self.fc1 = nn.Linear(v_dim, l_dim, bias=True)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(l_dim, l_dim, bias=True)
        nn.init.normal_(self.fc1.weight, std=0.02)
        nn.init.normal_(self.fc2.weight, std=0.01)  # 微小随机: 初始输出≈0 但非精确0
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))


def pixel_shuffle_features(x, ps):
    """SmolVLM2 式空间压缩: (B, S*S, D) -> (B, (S/ps)^2, D*ps^2)
    把 ps×ps 相邻 patch 的特征拼进通道维，token 数降 ps^2 倍、信息不丢。"""
    if ps <= 1:
        return x
    B, N, D = x.shape
    S = int(round(N ** 0.5))
    assert S * S == N and S % ps == 0, "patch 数 %d 不能被 ps=%d 整除" % (N, ps)
    x = x.view(B, S, S, D)
    x = x.view(B, S // ps, ps, S // ps, ps, D)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(B, (S // ps) ** 2, D * ps * ps)


def preprocess(img, size=224):
    from PIL import Image
    import numpy as np
    img = img.convert("RGB")
    w, h = img.size
    s = min(w, h)
    img = img.resize((size, size), Image.BICUBIC) if (w == h) else \
        img.crop(((w - s) // 2, (h - s) // 2, (w - s) // 2 + s, (h - s) // 2 + s)).resize((size, size), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - 0.5) / 0.5
    return torch.from_numpy(a).permute(2, 0, 1).unsqueeze(0)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def load_pairs(split):
    """读 vlm_train.jsonl / vlm_val.jsonl -> [(img_path, text)]"""
    p = os.path.join(ROOT, "data", "vlm", "vlm_%s.jsonl" % split)
    if not os.path.exists(p):
        raise FileNotFoundError("先跑 tools/vlm_data.py --pair 生成 %s" % p)
    pairs = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                pairs.append((r["image"], r["text"]))
    return pairs


def build_vqa_prompt(stage, text):
    """把描述包装成指令对。
    align 与 vqa 用同一指令前缀（LLaVA 式：align=只训投影层学[视觉]->字幕，
    vqa=全放开学指令遵循）；2026-10-17 修复：align 曾误用 (text,text) 抄写目标，
    loss 被复制 shortcut 压到 0.02 但视觉零学习——验收实测（见过的图也吐垃圾）抓出。"""
    return "请描述这张图片的内容。", text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", default="backups/0.2B/r6_final_20261016")
    ap.add_argument("--vision", default=os.path.join(ROOT, "..", "models", "siglip2"))
    ap.add_argument("--stage", default="align", choices=["align", "joint", "vqa"])
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--save", default=os.path.join(ROOT, "models", "vlm_02b_align"))
    ap.add_argument("--init", default=None, help="加载已有 projector.pt 续训")
    ap.add_argument("--dummy", action="store_true")
    ap.add_argument("--infer", action="store_true", help="推理模式")
    ap.add_argument("--image", default=None)
    ap.add_argument("--prompt", default="请描述这张图片的内容。")
    ap.add_argument("--max-new", type=int, default=60)
    ap.add_argument("--precompute", action="store_true", help="预计算全部图片的视觉特征缓存")
    ap.add_argument("--cache", default=os.path.join(ROOT, "data", "vlm", "vis_feats.pt"))
    ap.add_argument("--use-cache", action="store_true", help="训练用缓存视觉特征(跳过vision前向)")
    ap.add_argument("--vis-tokens", type=int, default=196, help="视觉token数(196=全量, 64=池化提速)")
    ap.add_argument("--ps", type=int, default=1,
                    help="pixel shuffle 空间压缩系数(SmolVLM2 配方): 196 patch -> (14/ps)^2 个 token, "
                         "投影层输入维=v_dim*ps^2。小模型上压缩反而提性能(Arxiv 2504.05299)")
    ap.add_argument("--train-llm", action="store_true",
                    help="SmolVLM2 式全参训练 LLM(lr=--lr*0.1, 视觉仍冻结)。小 LLM 只训投影层学不会读视觉token——"
                         "投影层对齐配方只对 7B+ 大 LLM 成立")
    ap.add_argument("--text-mix", type=float, default=0.14,
                    help="纯文本样本混比(SmolVLM2 混 14%% 纯文本保语言能力)，从 chat_sft_v3.jsonl 采样")
    ap.add_argument("--rept-penalty", type=float, default=1.3, help="推理重复惩罚")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, SiglipVisionModel, SiglipVisionConfig

    # ---- 1. 语言主干 ----
    llm_path = args.llm if os.path.isabs(args.llm) else os.path.join(ROOT, args.llm)
    tok = AutoTokenizer.from_pretrained(llm_path)
    llm = AutoModelForCausalLM.from_pretrained(llm_path, dtype=torch.float32)
    llm.eval()
    l_dim = llm.config.hidden_size
    print("[LLM] %s  hidden=%d layers=%d  params=%.1fM"
          % (os.path.basename(llm_path.rstrip('/')), l_dim,
             llm.config.num_hidden_layers, count_params(llm) / 1e6))

    # ---- 2. 视觉编码器（从 checkpoint 推断维度，不看名字）----
    vis_path = os.path.abspath(args.vision)
    from safetensors import safe_open
    ck = os.path.join(vis_path, "model.safetensors")
    with safe_open(ck, framework="pt") as f:
        keys = list(f.keys())
        pos = f.get_tensor("vision_model.embeddings.position_embedding.weight")
        n_pos, v_dim = pos.shape[0], pos.shape[1]
        n_layer = max(int(k.split(".")[3]) for k in keys
                      if k.startswith("vision_model.encoder.layers.")) + 1
        inter = f.get_tensor("vision_model.encoder.layers.0.mlp.fc1.weight").shape[0]
    patch = 224 // int(math.sqrt(n_pos))
    vcfg = SiglipVisionConfig(image_size=224, patch_size=patch, hidden_size=v_dim,
                              num_hidden_layers=n_layer,
                              num_attention_heads=max(1, v_dim // 64),
                              intermediate_size=inter)
    vision = SiglipVisionModel(vcfg)
    sd = {}
    with safe_open(ck, framework="pt") as f:
        for k in keys:
            if k.startswith("vision_model."):
                sd[k[len("vision_model."):]] = f.get_tensor(k)
    vision.load_state_dict(sd, strict=False)
    vision.eval()
    print("[VISION] patch=%d hidden=%d layers=%d 每图%d token  params=%.1fM"
          % (patch, v_dim, n_layer, n_pos, count_params(vision) / 1e6))

    # ---- 3. 投影层（输入维随 pixel shuffle 放大）----
    proj_in = v_dim * args.ps * args.ps
    proj = MLPProjector(proj_in, l_dim)
    if not args.init and args.infer and os.path.exists(os.path.join(args.save, "projector.pt")):
        args.init = os.path.join(args.save, "projector.pt")
        print("[PROJ] 推理未指定 --init，自动加载 %s" % args.init)
    if args.init and os.path.exists(args.init):
        st = torch.load(args.init, map_location="cpu")
        sd = st["proj"] if "proj" in st else st
        if "proj" in st and "cfg" in st and st["cfg"].get("ps") and st["cfg"]["ps"] != args.ps:
            print("[PROJ] checkpoint 用 ps=%d，自动跟随" % st["cfg"]["ps"])
            args.ps = st["cfg"]["ps"]
            proj = MLPProjector(v_dim * args.ps * args.ps, l_dim)
        proj.load_state_dict(sd)
        print("[PROJ] 加载 %s" % args.init)
    print("[PROJ] %d -> %d (ps=%d, %d tok/图)  params=%.2fM"
          % (proj_in, l_dim, args.ps, (14 // args.ps) ** 2, count_params(proj) / 1e6))

    # ---- 4. 差速 LR ----
    params = [{"params": proj.parameters(), "lr": args.lr}]
    if args.train_llm or args.stage in ("joint", "vqa"):
        # SmolVLM2 配方: 小 LLM 必须一起训，否则学不会读视觉 token
        for p in llm.parameters():
            p.requires_grad = True
        params.append({"params": llm.parameters(), "lr": args.lr * 0.1})
        if args.stage in ("joint", "vqa"):
            for p in vision.parameters():
                p.requires_grad = True
            params.append({"params": vision.parameters(), "lr": args.lr * 0.05})
        else:
            for p in vision.parameters():
                p.requires_grad = False
    else:
        for p in llm.parameters():
            p.requires_grad = False
        for p in vision.parameters():
            p.requires_grad = False
    opt = torch.optim.AdamW(params, weight_decay=0.0)
    trainable = sum(p.numel() for p in proj.parameters() if p.requires_grad)
    if args.stage != "align":
        trainable += sum(p.numel() for p in llm.parameters() if p.requires_grad)
        trainable += sum(p.numel() for p in vision.parameters() if p.requires_grad)
    print("[TRAIN] stage=%s 可训练参数=%.2fM" % (args.stage, trainable / 1e6))

    # ---- 5. 前向 ----
    _feat_cache = None
    if args.use_cache and os.path.exists(args.cache):
        print("[cache] mmap 加载视觉特征缓存 %s ..." % args.cache)
        # mmap: 3GB 缓存走文件页，内存压力下可被回收，防止与本地其他训练进程
        # 争内存导致 torch C++ 层分配失败直接 abort（2026-10-17 实测 16GB 机器被杀）
        _feat_cache = torch.load(args.cache, map_location="cpu", mmap=True)
        print("[cache] %d 张图特征就绪" % len(_feat_cache))

    def pool_vis(vfeat, n_tok):
        """视觉 token 池化: (B,N,D) -> (B,n_tok,D)"""
        if n_tok >= vfeat.size(1):
            return vfeat
        idx = torch.linspace(0, vfeat.size(1) - 1, n_tok).round().long()
        return vfeat[:, idx, :]

    def get_vtok(pixel_values=None, img_path=None):
        """取(pixel shuffle 压缩后的)视觉 token；优先缓存"""
        if _feat_cache is not None and img_path is not None:
            key = os.path.basename(img_path)
            vfeat = _feat_cache[key]                       # (N, v_dim) 已缓存
            vf = pixel_shuffle_features(vfeat.unsqueeze(0), args.ps)
            vtok = proj(vf)
            return pool_vis(vtok, args.vis_tokens)
        with torch.no_grad():
            vout = vision(pixel_values=pixel_values)
        vfeat = vout.last_hidden_state
        if not torch.isfinite(vfeat).all():
            raise RuntimeError("视觉特征含 nan/inf，SigLIP 权重可能损坏")
        vtok = proj(pixel_shuffle_features(vfeat, args.ps))
        return pool_vis(vtok, args.vis_tokens)

    def forward_text(prompt, answer):
        """纯文本样本前向（无图，混入训练保语言能力，SmolVLM2 14% 配方）"""
        pids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        aids = tok(answer, return_tensors="pt", add_special_tokens=False).input_ids
        ids = torch.cat([pids, aids], dim=1)
        out = llm(inputs_embeds=llm.get_input_embeddings()(ids))
        logits = out.logits[:, pids.size(1) - 1:-1, :]
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), aids.reshape(-1))

    def forward_step(pixel_values, prompt, answer, img_path=None):
        vtok = get_vtok(pixel_values, img_path)
        pids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
        aids = tok(answer, return_tensors="pt", add_special_tokens=False).input_ids
        ids = torch.cat([pids, aids], dim=1)
        temb = llm.get_input_embeddings()(ids)
        inputs = torch.cat([vtok, temb], dim=1)          # early fusion
        out = llm(inputs_embeds=inputs)
        np_ = vtok.size(1) + pids.size(1)                 # 只对 answer 段算 loss
        logits = out.logits[:, np_ - 1:-1, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), aids.reshape(-1))
        return loss

    # ---- 6. 预计算视觉特征（vision 冻结 + 图集固定 => 特征可缓存）----
    if args.precompute:
        from PIL import Image
        feats = {}
        if os.path.exists(args.cache):
            feats = torch.load(args.cache, map_location="cpu")
            print("[pre] 已有缓存 %d 张，续算" % len(feats))
        todo = []
        for split in ("train", "val"):
            for ip, _ in load_pairs(split):
                if os.path.basename(ip) not in feats:
                    todo.append(ip)
        todo = sorted(set(todo))
        print("[pre] 待计算 %d 张" % len(todo))
        t0 = time.time()
        B = 16
        with torch.no_grad():
            for i in range(0, len(todo), B):
                batch = todo[i:i + B]
                imgs = []
                for ip in batch:
                    try:
                        imgs.append(preprocess(Image.open(ip)))
                    except Exception:
                        pass
                if not imgs:
                    continue
                pv = torch.cat(imgs, dim=0)
                vout = vision(pixel_values=pv)
                for j, ip in enumerate(batch[:len(imgs)]):
                    feats[os.path.basename(ip)] = vout.last_hidden_state[j].clone()
                if (i // B) % 20 == 0:
                    el = time.time() - t0
                    print("  [pre] %d/%d  %.0fs" % (i + len(batch), len(todo), el), flush=True)
        torch.save(feats, args.cache)
        print("[pre] 完成 %d 张 -> %s  (%.0fs, %.1fMB)"
              % (len(feats), args.cache, time.time() - t0,
                 os.path.getsize(args.cache) / 1e6))
        return

    # ---- 7. 推理模式 ----
    if args.infer:
        from PIL import Image
        assert args.image and os.path.exists(args.image), "--infer 需要 --image"
        img = Image.open(args.image)
        pv = preprocess(img)
        with torch.no_grad():
            vout = vision(pixel_values=pv)
            vtok = proj(pixel_shuffle_features(vout.last_hidden_state, args.ps))
            vtok = pool_vis(vtok, args.vis_tokens)   # 必须与训练一致(默认64)，否则196全量token分布没学过
        pids = tok(args.prompt, return_tensors="pt", add_special_tokens=False).input_ids
        gen_ids = pids.clone()
        emb = llm.get_input_embeddings()
        cur = torch.cat([vtok, emb(pids)], dim=1)
        gen = []                                       # 已生成 token（用于重复惩罚/三连块）
        for _ in range(args.max_new):
            out = llm(inputs_embeds=cur)
            logits = out.logits[0, -1].clone()
            # 重复惩罚：已生成 token 的 logit 压制（小模型贪心解码防复读）
            if args.rept_penalty > 1.0 and gen:
                for t in set(gen):
                    logits[t] = logits[t] / args.rept_penalty if logits[t] > 0 \
                        else logits[t] * args.rept_penalty
                # no_repeat_ngram(3)：若 (t1,t2,x) 三元组已出现过，则屏蔽 x
                if len(gen) >= 2:
                    for i in range(len(gen) - 2):
                        if gen[i] == gen[-2] and gen[i + 1] == gen[-1]:
                            logits[gen[i + 2]] = -float("inf")
            nxt = logits.argmax()
            gen_ids = torch.cat([gen_ids, nxt.view(1, 1)], dim=1)
            gen.append(nxt.item())
            cur = torch.cat([cur, emb(nxt.view(1, 1))], dim=1)
            if nxt.item() == tok.eos_token_id:
                break
        print("[生成] %s" % tok.decode(gen_ids[0], skip_special_tokens=True))
        return

    # ---- 8. dummy 验证 ----
    if args.dummy:
        print("\n[dummy] 合成数据验证通路")
        pixel = torch.randn(1, 3, 224, 224)
        loss = forward_step(pixel, "一张图片：", "画面中有一个红色的圆形。")
        loss.backward()
        print("  loss=%.4f (fc2零初始化初始loss应≈ln(vocab)≈10.6)" % loss.item())
        return

    # ---- 9. 训练 ----
    train_pairs = load_pairs("train")
    try:
        val_pairs = load_pairs("val")
    except Exception:
        val_pairs = []
    # 纯文本混样（SmolVLM2 14% 配方，保语言能力）
    text_pairs = []
    if args.train_llm and args.text_mix > 0:
        chat_p = os.path.join(ROOT, "data", "chat_sft_v3.jsonl")
        if os.path.exists(chat_p):
            import re as _re
            for line in open(chat_p, encoding="utf-8"):
                t = json.loads(line)["text"]
                mu = _re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", t, _re.S)
                ma = _re.search(r"<\|im_start\|>assistant\n(.*?)<\|im_end\|>", t, _re.S)
                if mu and ma:
                    text_pairs.append((mu.group(1), ma.group(1)))
        print("[data] 纯文本混样 %d 条 (mix=%.2f)" % (len(text_pairs), args.text_mix))
    print("[data] train=%d val=%d  train_llm=%s ps=%d" % (len(train_pairs), len(val_pairs), args.train_llm, args.ps))
    random.seed(42)
    opt.zero_grad()
    t0 = time.time()
    running, best = 0.0, float("inf")
    hist = []
    for step in range(args.steps):
        # warmup 线性升 LR（组0=投影层 lr，组1=LLM lr×0.1）
        if step < args.warmup:
            base = args.lr * (step + 1) / max(1, args.warmup)
            for gi, g in enumerate(opt.param_groups):
                g["lr"] = base if gi == 0 else base * 0.1
        # 14% 概率走纯文本样本（--train-llm 时）
        if text_pairs and random.random() < args.text_mix:
            tp, ta = random.choice(text_pairs)
            loss = forward_text(tp, ta)
        else:
            ip, text = train_pairs[step % len(train_pairs)]
            prompt, answer = build_vqa_prompt(args.stage, text)
            if _feat_cache is not None:
                loss = forward_step(None, prompt, answer, img_path=ip)
            else:
                from PIL import Image
                pv = preprocess(Image.open(ip))
                loss = forward_step(pv, prompt, answer)
        (loss / args.accum).backward()
        running += loss.item()
        # 早期每步查有限性: nan 早发现早存现场，别等 50 步
        if step < 200 and not math.isfinite(loss.item()):
            print("[FATAL] step %d loss=%.4f 非有限值，保存现场退出" % (step + 1, loss.item()), flush=True)
            os.makedirs(args.save, exist_ok=True)
            torch.save({"proj": proj.state_dict(),
                        "cfg": {"v_dim": v_dim, "l_dim": l_dim, "stage": args.stage}},
                       os.path.join(args.save, "projector_crash.pt"))
            return
        if (step + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in opt.param_groups for p in g["params"] if p.requires_grad], 1.0)
            opt.step()
            opt.zero_grad()
        if (step + 1) % 25 == 0:
            el = time.time() - t0
            avg = running / min(25, step + 1)
            print("  step %4d  loss=%.4f  (%.0fs, %.2f s/step)"
                  % (step + 1, avg, el, el / (step + 1)), flush=True)
            hist.append({"step": step + 1, "loss": avg})
            running = 0.0
            if not math.isfinite(avg):
                print("[FATAL] loss 非有限值，训练发散，保存现场退出")
                os.makedirs(args.save, exist_ok=True)
                torch.save({"proj": proj.state_dict(),
                            "cfg": {"v_dim": v_dim, "l_dim": l_dim, "stage": args.stage}},
                           os.path.join(args.save, "projector.pt"))
                return

    def save_all(tag="projector.pt"):
        os.makedirs(args.save, exist_ok=True)
        torch.save({"proj": proj.state_dict(),
                    "cfg": {"v_dim": v_dim * args.ps * args.ps, "l_dim": l_dim,
                            "stage": args.stage, "ps": args.ps}},
                   os.path.join(args.save, tag))
        if args.train_llm:
            llm.save_pretrained(args.save)
            print("  [save] LLM 全参权重已存 %s" % args.save, flush=True)

    # 每 500 步验证 + 保存最优
    if val_pairs and (step + 1) % 500 == 0:
            proj.eval()
            vloss = 0.0
            with torch.no_grad():
                for ip, text in random.sample(val_pairs, min(50, len(val_pairs))):
                    prompt, answer = build_vqa_prompt(args.stage, text)
                    if _feat_cache is not None:
                        vloss += forward_step(None, prompt, answer, img_path=ip).item()
                    else:
                        from PIL import Image
                        pv = preprocess(Image.open(ip))
                        vloss += forward_step(pv, prompt, answer).item()
            vloss /= min(50, len(val_pairs))
            proj.train()
            print("  [val] step %d  val_loss=%.4f" % (step + 1, vloss), flush=True)
            if vloss < best:
                best = vloss
                save_all()
                print("  [save] 最优 (val=%.4f)" % best, flush=True)

    save_all()
    with open(os.path.join(args.save, "history.json"), "w") as f:
        json.dump(hist, f)
    print("\n[save] -> %s/projector.pt  (best val=%.4f)" % (args.save, best))
    print("[done] %.0fs" % (time.time() - t0))


if __name__ == "__main__":
    main()
