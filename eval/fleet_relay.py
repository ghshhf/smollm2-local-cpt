#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fleet relay — 统一免费 AI 舰队交叉问答 + 收敛投票。
- 注册器内置 krisphyn(38模型) + zen-gate(64, 但免费档被 OpenCode 锁，仅作本地常驻占位) + ai_hub 免 key 外站。
- 每个问题并行扇出到所有"此刻在线"的舰船（流式 SSE，规避 60s 反向代理枪毙）。
- 死舰自动跳过；活舰答案落盘 + 调用一艘强舰做"收敛合成"（交叉答案）。
- 红线：机密(owner_corpus/持仓/密钥/私人文档)绝不进任何中转站——本脚本只发"问题文本"，不发任何本地机密。
用法：python fleet_relay.py <questions.json> <out_dir> [--ships kris|zen|free|all]
"""
import json, sys, os, time, ssl, threading
import urllib.request, urllib.error

CT = "application/json"
_CTX = ssl.create_default_context(); _CTX.check_hostname=False; _CTX.verify_mode=ssl.CERT_NONE

# ── 舰队注册表（privacy: public=公益中转/外部免费, local=本机网关）──────────────
KRIS_KEY = "sk-go28BcSeT4Ig14N30NVLfDq6fZKquKRLRW0S3zdhosXGwa9U"
FLEET = [
    # krisphyn 公益中转（38 模型，多厂商，上游真通道）
    ("kris.gemini","https://api.krisphyn.cc/v1/chat/completions",KRIS_KEY,"gemini-3.8-flash-high","google","public"),
    ("kris.deepseek","https://api.krisphyn.cc/v1/chat/completions",KRIS_KEY,"deepseek-v4-pro-0813","deepseek","public"),
    ("kris.qwen","https://api.krisphyn.cc/v1/chat/completions",KRIS_KEY,"Qwen3.8-27B","qwen","public"),
    ("kris.glm","https://api.krisphyn.cc/v1/chat/completions",KRIS_KEY,"glm-5.2","zhipu","public"),
    ("kris.minimax","https://api.krisphyn.cc/v1/chat/completions",KRIS_KEY,"MiniMax-M3","minimax","public"),
    ("kris.step","https://api.krisphyn.cc/v1/chat/completions",KRIS_KEY,"step-5-preview","step","public"),
    ("kris.longcat","https://api.krisphyn.cc/v1/chat/completions",KRIS_KEY,"longcat-2.5","longcat","public"),
    ("kris.nemotron","https://api.krisphyn.cc/v1/chat/completions",KRIS_KEY,"nemotron-3-ultra","nvidia","public"),
    ("kris.north","https://api.krisphyn.cc/v1/chat/completions",KRIS_KEY,"north-mini-code","cohere","public"),
    ("kris.grok","https://api.krisphyn.cc/v1/chat/completions",KRIS_KEY,"grok-4.7","xai","public"),
    # ai_hub 免 key 外站（波动大，活则参战）
    ("pollinations","https://text.pollinations.ai/openai","any","gpt-oss-20b","openai-oss","public"),
    ("unclose","https://hermes.ai.unturf.com/v1/chat/completions","any","turboderp/Qwen3.8-27B-exl3","qwen","public"),
    ("llm7","https://api.llm7.io/v1/chat/completions",None,"GLM-5.3-Flash","zhipu","public"),
    ("kilo","https://api.kilo.ai/api/gateway/chat/completions","any","kilo-auto/free","kilo","public"),
    ("dgai","https://dg-ai.scriptsnsenses.workers.dev/v1/chat/completions","any","llama-4-scout","meta","public"),
    ("ovh","https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions","any","Meta-Llama-3_3-70B-Instruct","meta","public"),
    # zen-gate 本机网关（免费档被 OpenCode 锁，仅占位；若日后解锁自动生效）
    ("zen.deepseek","http://127.0.0.1:8787/v1/chat/completions","ofm-Ew_gPxhHHGNBpX9PcU-TKqqB1WXuuxmf","deepseek-v4-flash-free","deepseek","local"),
]

# 系统约束（注入每个问题，防止模型乱给脱离路线的方案）
CONSTRAINT = (
    "背景约束（回答必须遵守）：我们在做一条本地小模型进化路线——0.1B→0.4B 逐级 Net2Net 保函数扩容，"
    "0.4B 加中文词表(只加不换)+灌自有资料+认主对比头；0.5B 融合版=架构层从头(BitNet b1.58 原生三值 QAT,深窄)"
    "+能力层从 0.4B 蒸馏继承+外置可寻址记忆层(每人10M,删除权0/1硬需求)+端侧常驻服务(看门狗内嵌/开机自启/联网自学习)→RSI闭环。"
    "算力=免费CPU(魔搭node1 64核)+未来GPU；端侧目标CPU≤15ms/token@0.5B；语料人写纯净(0.5B前主干预路径禁模型生成数据)；"
    "owner_corpus 等机密不出本地。已拍板：embedding保fp16(b_eff≈5.5,N*≈200M@3.5e9)；深度优先扩容；"
    "0.1B–0.4B默认单遍；记忆层排除Titans式(写参数不可删)；L1压缩准入/L2信息天花板/L3通道分离三规则入宪法。"
    "回答格式：①机制/最佳理论 ②可执行方案或伪代码 ③可证伪判据(什么结果算你错) ④若需实验标注node1 CPU最小可跑版。"
    "若你认为某前提错了，直接推翻并说明理由。"
)

def stream_chat(base, key, model, messages, timeout=120):
    body = json.dumps({"model":model,"messages":messages,"stream":True,"max_tokens":1500}).encode()
    req = urllib.request.Request(base, data=body, method="POST")
    req.add_header("Content-Type", CT)
    if key: req.add_header("Authorization","Bearer "+key)
    else:   req.add_header("Authorization","Bearer sk-any")
    req.add_header("Accept","text/event-stream")
    out=[]; served=None; err=None
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=_CTX) as r:
            buf=""
            for raw in r:
                buf+=raw.decode("utf-8","ignore")
                while "\n" in buf:
                    line, buf = buf.split("\n",1)
                    line=line.strip()
                    if not line or not line.startswith("data:"): continue
                    payload=line[5:].strip()
                    if payload=="[DONE]": break
                    try:
                        j=json.loads(payload)
                        if served is None and j.get("model"): served=j["model"]
                        for c in j.get("choices",[]):
                            d=c.get("delta",{})
                            if d.get("content"): out.append(d["content"])
                    except: pass
    except urllib.error.HTTPError as e:
        err=f"HTTP {e.code}: {e.read().decode('utf-8','ignore')[:160]}"
    except Exception as e:
        err=f"{type(e).__name__}: {str(e)[:120]}"
    return "".join(out), served, err

def ask_one(ship, q_msg, res):
    sid,base,key,model,vendor,privacy = ship
    t=time.time()
    text,served,err = stream_chat(base,key,model,[{"role":"system","content":CONSTRAINT},{"role":"user","content":q_msg}])
    res[sid]={"vendor":vendor,"privacy":privacy,"model":model,"served":served,
              "ok":bool(text and not err),"text":text,"err":err,"sec":round(time.time()-t,1)}

def run_question(q, ships, outdir, workers=12):
    os.makedirs(outdir, exist_ok=True)
    res={}
    threads=[]
    for s in ships:
        t=threading.Thread(target=ask_one,args=(s,q["body"],res)); threads.append(t)
    # 简单并发控制
    for i in range(0,len(threads),workers):
        batch=threads[i:i+workers]
        for t in batch: t.start()
        for t in batch: t.join(timeout=150)
    # 落盘
    meta={"id":q["id"],"title":q["title"],"n_ships":len(ships),
          "alive":[k for k,v in res.items() if v["ok"]],
          "dead":[k for k,v in res.items() if not v["ok"]]}
    for sid,v in res.items():
        if v["ok"]:
            open(f"{outdir}/ans_{sid}.md","w",encoding="utf-8").write(
                f"# {sid} (vendor={v['vendor']}, served={v['served']}, {v['sec']}s)\n\n"+v["text"])
    json.dump(res, open(f"{outdir}/raw.json","w"), ensure_ascii=False, indent=1)
    json.dump(meta, open(f"{outdir}/meta.json","w"), ensure_ascii=False, indent=1)
    print(f"  [{q['id']}] {q['title'][:30]} -> alive={len(meta['alive'])}/{len(ships)}: {meta['alive']}")
    return res, meta

def main():
    qfile, outroot = sys.argv[1], sys.argv[2]
    sel = sys.argv[3] if len(sys.argv)>3 else "all"
    questions=json.load(open(qfile,encoding="utf-8"))
    if sel=="kris": ships=[s for s in FLEET if s[0].startswith("kris")]
    elif sel=="zen": ships=[s for s in FLEET if s[0].startswith("zen")]
    elif sel=="free": ships=[s for s in FLEET if s[0] in ("pollinations","unclose","llm7","kilo","dgai","ovh")]
    else: ships=FLEET
    print(f"FLEET ships registered: {len(ships)}  ({sel})")
    for q in questions:
        od=f"{outroot}/{q['id']}"
        run_question(q, ships, od)

if __name__=="__main__":
    main()
