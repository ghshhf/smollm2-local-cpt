# -*- coding: utf-8 -*-
"""
北极星启动脚本 —— 上下文 loader（薄胶水层）。
记忆存取全部使用 TinyContext（github.com/TinySuiteHQ/TinyContext）的成熟实现：
  - save_memories / recall_memories / update_memory / delete_memory
  - 身份事实用 kind="profile"（其设计：全局常驻附加、不参与语义排序、自带去重/更新）
本脚本只保留两样必要胶水：ChatML prompt 拼装 + transformers generate。
运行解释器：venv_ttt（含 torch cpu + transformers 4.57 + peft + tinycontext）
  D:/work/_local_ai/train_15b/venv_ttt/Scripts/python.exe tools/northstar_chat.py ...
⚠️ 训练 venv（venv/）与此不兼容（tinysuite-context 死钉 huggingface_hub==0.36），勿混用。
"""
import argparse, os, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = os.path.join(ROOT, "backups", "0.2B", "r6_final_20261016")

# ---------- 记忆：全走 TinyContext ----------

def tc_save(text, kind="episodic"):
    from tinycontext import MemoryInput, save_memories
    return save_memories([MemoryInput(content=text, kind=kind)])

def tc_recall(query, max_tokens=300):
    from tinycontext import recall_memories
    return recall_memories(query=query, max_tokens=max_tokens)

def tc_seed_profile():
    """首次初始化：身份事实以 profile 类记忆写入（幂等，靠其去重/更新语义）"""
    from tinycontext import MemoryInput, save_memories
    facts = [
        "模型名叫北极星，是主人起的名字",
        "称呼用户为主人",
        "北极星是主人从0.2B底子一手训练出来的私有模型，跑在主人自己的电脑上",
        "北极星绝口不提'助手'二字",
        "主人追求模型自进化与保函数扩容路线",
    ]
    return save_memories([MemoryInput(content=f, kind="profile") for f in facts])

def tc_context_block(query):
    """取 TinyContext 的 prompt 就绪上下文块（profile 常驻 + 语义召回，它自己拼好边界）"""
    out = tc_recall(query)
    ctx = out.get("context") if isinstance(out, dict) else str(out)
    return ctx or ""

# ---------- 生成：ChatML + transformers ----------

def build_prompt(profile_block, memories_block, user_msg):
    parts = []
    if profile_block.strip():
        parts.append(f"<|im_start|>system\n{profile_block.strip()}<|im_end|>")
    if memories_block.strip():
        parts.append(f"<|im_start|>system\n历史记忆参考：\n{memories_block.strip()}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n")
    return "".join(parts)

def load_llm(ckpt):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    tok = AutoTokenizer.from_pretrained(BASE)
    llm = AutoModelForCausalLM.from_pretrained(BASE, dtype=torch.float32)
    ck = os.path.join(ROOT, ckpt) if not os.path.isabs(ckpt) else ckpt
    if os.path.exists(ck):
        llm = PeftModel.from_pretrained(llm, ck)
        print("[北极星] 适配器 %s 已挂载" % ckpt)
    else:
        print("[北极星] 适配器 %s 不存在，裸基底待命" % ckpt)
    llm.eval()
    return tok, llm

def respond(tok, llm, user_msg, max_tokens=300):
    ctx = tc_context_block(user_msg)
    # TinyContext 返回的 context 含 profile 块+召回块；拆开只是展示，注入不拆
    prompt = build_prompt("", ctx, user_msg)
    ids = tok(prompt, return_tensors="pt").input_ids
    import torch
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    with torch.no_grad():
        out = llm.generate(ids, max_new_tokens=80, do_sample=False,
                           pad_token_id=tok.eos_token_id, eos_token_id=im_end,
                           repetition_penalty=1.2)
    ans = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).split("<|im_end|>")[0].strip()
    print("北极星：", ans)
    return ans

def main():
    ap = argparse.ArgumentParser(description="北极星上下文启动器")
    ap.add_argument("--ask", type=str, help="带上下文答一次")
    ap.add_argument("--remember", type=str, help="存一条情景记忆 (episodic)")
    ap.add_argument("--profile", type=str, help="存一条身份事实 (profile)")
    ap.add_argument("--seed", action="store_true", help="初始化默认身份事实")
    ap.add_argument("--chat", action="store_true", help="交互循环")
    ap.add_argument("--ckpt", default="out_lora_chat_v3")
    args = ap.parse_args()

    if args.remember:
        print(tc_save(args.remember)); return
    if args.profile:
        print(tc_save(args.profile, kind="profile")); return
    if args.seed:
        print(tc_seed_profile()); return

    if not (args.ask or args.chat):
        ap.print_help(); return

    tok, llm = load_llm(args.ckpt)
    if args.ask:
        respond(tok, llm, args.ask)
        return
    print("北极星已上线（Ctrl+C 退出）")
    while True:
        try:
            q = input("主人> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n北极星：下次见，主人。"); break
        if q:
            respond(tok, llm, q)

if __name__ == "__main__":
    main()
