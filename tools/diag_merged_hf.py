# -*- coding: utf-8 -*-
import torch, sys
from transformers import AutoModelForCausalLM, AutoTokenizer

M = r'D:/work/_local_ai/train_15b/models/northstar_chat_v3_merged'
tok = AutoTokenizer.from_pretrained(M)
IM_S, IM_E = chr(60) + '|im_start|>' + chr(0), chr(0)  # placeholder, replaced below
IM_S = '<|' + 'im_start|' + '>'
IM_E = '<|' + 'im_end|' + '>'
print('special ids:', tok.convert_tokens_to_ids([IM_S, IM_E]), 'vocab', len(tok))

m = AutoModelForCausalLM.from_pretrained(M, dtype=torch.float32)
m.eval()
print('model loaded')

for q in ['\u4f60\u662f\u8c01', '\u4f60\u53eb\u4ec0\u4e48\u540d\u5b57', '\u665a\u5b89', '\u4f60\u4f1a\u5199\u4ee3\u7801\u5417']:
    p = IM_S + 'user\n' + q + IM_E + '\n' + IM_S + 'assistant\n'
    ids = tok(p, return_tensors='pt').input_ids
    with torch.no_grad():
        o = m.generate(ids, max_new_tokens=40, do_sample=False, pad_token_id=tok.eos_token_id)
    txt = tok.decode(o[0][ids.shape[1]:], skip_special_tokens=False)
    print('[Q]', q, '=>', repr(txt)[:160])