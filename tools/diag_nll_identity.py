# -*- coding: utf-8 -*-
"""Teacher-forced NLL on identity samples vs casual samples: is the knowledge in the weights?"""
import torch, json, re
from transformers import AutoModelForCausalLM, AutoTokenizer

M = r'D:/work/_local_ai/train_15b/models/northstar_chat_v3_merged'
tok = AutoTokenizer.from_pretrained(M)
m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32)
m.eval()

IM_S = '<|' + 'im_start|' + '>'
IM_E = '<|' + 'im_end|' + '>'

def nll(prompt, answer):
    p_ids = tok(prompt, return_tensors='pt').input_ids
    a_ids = tok(answer, return_tensors='pt').input_ids
    ids = torch.cat([p_ids, a_ids], dim=1)
    with torch.no_grad():
        out = m(ids).logits
    # predict answer tokens only
    lp = torch.log_softmax(out[0, p_ids.shape[1]-1:-1], dim=-1)
    tgt = ids[0, p_ids.shape[1]:]
    return -lp[range(len(tgt)), tgt].mean().item()

print('== identity samples (teacher-forced NLL, lower=remembered) ==')
cases = [
    ('\u4f60\u662f\u8c01', '\u6211\u662f\u5317\u6781\u661f'),                       # 你是谁 / 我是北极星
    ('\u4f60\u53eb\u4ec0\u4e48\u540d\u5b57', '\u6211\u53eb\u5317\u6781\u661f'),           # 你叫什么名字 / 我叫北极星
    ('\u4f60\u53eb\u5565', '\u5317\u6781\u661f'),                                # 你叫啥 / 北极星
    ('\u4f60\u662f\u4e2a\u4ec0\u4e48\u4e1c\u897f', '\u5317\u6781\u661f'),              # 你是个什么东西 / 北极星
]
for q, a in cases:
    p = IM_S + 'user\n' + q + IM_E + '\n' + IM_S + 'assistant\n'
    print('  %-14s -> %.3f' % (q, nll(p, a)))

print('== casual samples ==')
for q, a in [('\u665a\u5b89', '\u665a\u5b89\uff0c\u4e3b\u4eba'), ('\u4f60\u4f1a\u5199\u4ee3\u7801\u5417', '\u4f1a')]:
    p = IM_S + 'user\n' + q + IM_E + '\n' + IM_S + 'assistant\n'
    print('  %-14s -> %.3f' % (q, nll(p, a)))

print('== top-3 next token after 你是谁 prompt ==')
p = IM_S + 'user\n' + IM_S[0:0] + '\u4f60\u662f\u8c01' + IM_E + '\n' + IM_S + 'assistant\n'
ids = tok(p, return_tensors='pt').input_ids
with torch.no_grad():
    lg = m(ids).logits[0, -1]
top = torch.topk(lg, 5)
for v, i in zip(top.values.tolist(), top.indices.tolist()):
    print('  %-8s %.3f' % (repr(tok.decode([i])), v))