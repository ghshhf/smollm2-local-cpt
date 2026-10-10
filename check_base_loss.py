import torch, json, math
from transformers import AutoModelForCausalLM, AutoTokenizer
md = "D:/work/_local_ai/models/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(md)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
m = AutoModelForCausalLM.from_pretrained(md, torch_dtype=torch.float32, low_cpu_mem_usage=True)
m.eval()
print("config.vocab_size =", m.config.vocab_size)
print("lm_head weight shape =", tuple(m.lm_head.weight.shape))
print("embed_tokens weight shape =", tuple(m.model.embed_tokens.weight.shape))
# 单句
line = open("pretrain_corpus/zh_text.jsonl", encoding="utf-8").readline()
txt = json.loads(line)["text"][:400]
ids = tok.encode(txt, return_tensors="pt")
print("seq_len =", ids.shape[1], "| max token id =", int(ids.max()))
with torch.no_grad():
    out = m(input_ids=ids, labels=ids)
    print("BASE eval loss =", float(out.loss))
    # 手动算 CE 上限
    logits = out.logits
    print("logits shape =", tuple(logits.shape), "| logits max =", float(logits.max()), "| min =", float(logits.min()))
print("ln(vocab) =", round(math.log(m.config.vocab_size), 3))
