# -*- coding: utf-8 -*-
"""0.3B 底座纯文本 NLL 测试：判定 checkpoint / 词表是否正常。

判定标准（中文 LM）：
  loss 3~6  => 底座正常，问题在音频侧
  loss ~10.8 => 等于随机猜（ln49152=10.80），底座没学到语言
  loss >11   => 词表或权重错位
"""
import argparse
import sys

import torch
import torch.nn.functional as F


TEXTS = [
    "家中接连有了四位重病患者",
    "全国将新建多座城镇污水处理厂",
    "五十九岁的方女士拨打本报新闻热线",
    "今天是星期一，天气很好，我们一起去公园散步。",
    "import os\nprint(os.path.abspath(__file__))",
]


def run(path, tok_override=None, n=3):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("=" * 60)
    print("[load] %s" % path, flush=True)
    tok = AutoTokenizer.from_pretrained(tok_override or path)
    llm = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.float32)
    try:
        llm = llm.float()
    except Exception:
        pass
    llm.eval()
    c = llm.config
    print("[cfg] hidden=%s layers=%s vocab=%s  lm_head=%s tie=%s"
          % (c.hidden_size, c.num_hidden_layers, c.vocab_size,
             tuple(llm.get_output_embeddings().weight.shape),
             getattr(c, "tie_word_embeddings", None)))
    print("[tok] vocab_size=%d  len=%d" % (tok.vocab_size, len(tok)))

    # 关键：特殊 token 是否真的存在
    for t in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"]:
        try:
            i = tok.convert_tokens_to_ids(t)
            ok = i is not None and i != tok.unk_token_id
            print("[tok] %-14s id=%s %s" % (t, i, "OK" if ok else "MISSING"))
        except Exception as e:
            print("[tok] %-14s ERR %s" % (t, e))

    ln_v = torch.log(torch.tensor(float(c.vocab_size)))
    print("[ref] 随机基线 ln(vocab)=%.3f" % float(ln_v))

    for t in TEXTS[:n]:
        ids = tok(t, return_tensors="pt", add_special_tokens=False).input_ids
        if ids.size(1) < 2:
            print("[skip] %s" % t)
            continue
        with torch.no_grad():
            out = llm(input_ids=ids)
            logits = out.logits[:, :-1, :]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   ids[:, 1:].reshape(-1))
        print("[text] nll=%.3f ppl=%.1f  toks=%d  | %s"
              % (float(loss), float(torch.exp(loss)), ids.size(1), t.replace("\n", "\\n")[:34]),
              flush=True)

    # 用 chat 模板再测一次（对齐训练时的格式）
    prompt = "<|im_start|>user\n请把这段语音转成文字。<|im_end|>\n<|im_start|>assistant\n"
    tgt = "家中接连有了四位重病患者<|im_end|>"
    p = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    g = tok(tgt, return_tensors="pt", add_special_tokens=False).input_ids
    ids = torch.cat([p, g], dim=1)
    with torch.no_grad():
        out = llm(input_ids=ids)
        logits = out.logits[:, p.size(1) - 1:-1, :]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), g.reshape(-1))
    print("[chat] nll=%.3f ppl=%.1f  (prompt=%d tgt=%d)"
          % (float(loss), float(torch.exp(loss)), p.size(1), g.size(1)), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="append", required=True)
    ap.add_argument("--tok", default=None)
    ap.add_argument("--n", type=int, default=3)
    a = ap.parse_args()
    torch.set_num_threads(3)
    for p in a.llm:
        run(p, a.tok, a.n)


if __name__ == "__main__":
    main()
