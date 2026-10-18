#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fleet Relay 进度速览：每个问题的 done/失败/待重试 统计 + 各舰船存活率。"""
import json, os, glob, sys
HERE = os.path.dirname(os.path.abspath(__file__))
LEDGER = os.path.join(HERE, "fleet_answers", "ledger.json")
QDIR = os.path.join(HERE, "..", "docs", "netarch", "questions")

def load_qs():
    qs={}
    for fp in glob.glob(os.path.join(QDIR,"*.json")):
        try:
            arr=json.load(open(fp,encoding="utf-8"))
            if isinstance(arr,list):
                for q in arr:
                    if isinstance(q,dict) and q.get("id"): qs[q["id"]]=q.get("title","")
        except: pass
    return qs

def main():
    try: L=json.load(open(LEDGER,encoding="utf-8"))
    except: L={}
    qs=load_qs()
    print(f"舰队 ledger 记录数: {len(L)}")
    print(f"问题总数: {len(qs)}")
    print("-"*70)
    # 按问题聚合
    byq={}
    for k,v in L.items():
        qid=k.split("|")[0]
        byq.setdefault(qid,{"done":0,"fail":0})
        if v.get("status")=="done": byq[qid]["done"]+=1
        elif v.get("status")=="fail": byq[qid]["fail"]+=1
    for qid in sorted(qs):
        st=byq.get(qid,{"done":0,"fail":0})
        print(f"  {qid:6s} done={st['done']:2d} fail={st['fail']:2d}  {qs[qid][:36]}")
    print("-"*70)
    # 按舰船聚合
    bys={}
    for k,v in L.items():
        sid=k.split("|")[1]
        bys.setdefault(sid,{"done":0,"fail":0})
        if v.get("status")=="done": bys[sid]["done"]+=1
        elif v.get("status")=="fail": bys[sid]["fail"]+=1
    print("舰船存活 (done/fail):")
    for sid in sorted(bys):
        st=bys[sid]
        flag="OK" if st["done"]>0 else ("dead" if st["fail"]>=3 else "...")
        print(f"  {sid:18s} done={st['done']:2d} fail={st['fail']:2d}  [{flag}]")

if __name__=="__main__":
    main()
