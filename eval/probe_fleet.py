#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Probe the live free-AI fleet: which ships are alive + latency + served model.
Sandbox must be disabled (direct egress). No secrets leave local: every endpoint here is a free/public relay.
"""
import json, time, urllib.request, urllib.error, ssl

CT = "application/json"
def post(url, key=None, body=None, timeout=25, proxy=None):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method="POST")
    req.add_header("Content-Type", CT)
    if key: req.add_header("Authorization", "Bearer "+key)
    else: req.add_header("Authorization", "Bearer sk-any")
    opener = urllib.request.build_opener()
    if proxy:
        req.set_proxy(proxy, "http"); req.set_proxy(proxy, "https")
    t=time.time()
    with opener.open(req, timeout=timeout) as r:
        raw=r.read().decode("utf-8","ignore")
    return time.time()-t, raw

SHIPS = [
    # (name, url, key, model)
    ("krisphyn.gemini","https://api.krisphyn.cc/v1/chat/completions","sk-go28BcSeT4Ig14N30NVLfDq6fZKquKRLRW0S3zdhosXGwa9U","gemini-3.8-flash-high"),
    ("krisphyn.deepseek","https://api.krisphyn.cc/v1/chat/completions","sk-go28BcSeT4Ig14N30NVLfDq6fZKquKRLRW0S3zdhosXGwa9U","deepseek-v4-pro-0813"),
    ("krisphyn.qwen","https://api.krisphyn.cc/v1/chat/completions","sk-go28BcSeT4Ig14N30NVLfDq6fZKquKRLRW0S3zdhosXGwa9U","Qwen3.8-27B"),
    ("krisphyn.glm","https://api.krisphyn.cc/v1/chat/completions","sk-go28BcSeT4Ig14N30NVLfDq6fZKquKRLRW0S3zdhosXGwa9U","glm-5.2"),
    ("zen-gate","http://127.0.0.1:8787/v1/chat/completions",None,"deepseek-v4-flash-free"),
    ("unclose","https://hermes.ai.unturf.com/v1/chat/completions","any","turboderp/Qwen3.8-27B-exl3"),
    ("pollinations","https://text.pollinations.ai/openai","any","gpt-oss-20b"),
    ("llm7","https://api.llm7.io/v1/chat/completions",None,"GLM-5.3-Flash"),
    ("kilo","https://api.kilo.ai/api/gateway/chat/completions","any","kilo-auto/free"),
    ("dgai","https://dg-ai.scriptsnsenses.workers.dev/v1/chat/completions","any","llama-4-scout"),
    ("ovh","https://oai.endpoints.kepler.ai.cloud.ovh.net/v1/chat/completions","any","Meta-Llama-3_3-70B-Instruct"),
]

print(f"{'SHIP':<18}{'ALIVE':<6}{'LAT':>8}  served/err")
alive=[]
for name,url,key,model in SHIPS:
    try:
        dt,raw = post(url,key,{"model":model,"messages":[{"role":"user","content":"reply with exactly: OK"}],"max_tokens":5},timeout=25)
        try: j=json.loads(raw)
        except: j={}
        served=""
        if "choices" in j:
            served=(j["model"] or model)
            txt=j["choices"][0]["message"]["content"][:6]
            if "ok" in txt.lower(): 
                print(f"{name:<18}{'YES':<6}{dt:>7.2f}s  {served}")
                alive.append((name,url,key,model)); continue
        # not clean OK -> report anyway with truncation
        print(f"{name:<18}{'??':<6}{dt:>7.2f}s  {str(raw)[:60]}")
    except Exception as e:
        print(f"{name:<18}{'NO':<6}{'':>8}  {type(e).__name__}: {str(e)[:50]}")

print("\nALIVE:",[a[0] for a in alive])
json.dump(alive, open("fleet_alive.json","w"), ensure_ascii=False, indent=1)
print("wrote fleet_alive.json")
