#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fleet Relay Daemon — 统一免费 AI 舰队常驻轮询 + 交叉问答累积。
- 舰队从 fleet.json 加载（enabled=false 不参与）。
- 问题从 QUESTIONS_DIR 下所有 *.json（数组 [{id,title,body}]）动态加载——随时丢新文件即参战。
- ledger 记录每个 (问题×舰船) 状态：pending / done / fail(带退避)。
- 每个 tick：挑 done 数最少的问题，向"到期可重试"的舰船扇出（流式 SSE，规避 60s 代理枪毙），
  成功答案落盘 fleet_answers/<qid>/ans_<ship>.md，并标记 done。
- 死舰指数退避（30s→60s→…→上限 600s）后重试；活舰持续累积。
- 进程守护：pidfile 防重复；stop.flag 文件优雅停止。
- 日志：fleet_relay_daemon.log（追加，自带时间戳）。

红线：本守护只发"问题文本 + 系统约束"，绝不发任何本地机密（owner_corpus/持仓/密钥/私人文档）。
启动：fleet_relay_start.bat  （由计划任务 schtasks 拥有，脱离 agent 任务树常驻）
"""
import json, sys, os, time, ssl, threading, glob, traceback
import urllib.request, urllib.error

HERE = os.path.dirname(os.path.abspath(__file__))
FLEET_JSON = os.path.join(HERE, "fleet.json")
QUESTIONS_DIR = os.path.join(HERE, "..", "docs", "netarch", "questions")
ANS_ROOT = os.path.join(HERE, "fleet_answers")
LEDGER = os.path.join(ANS_ROOT, "ledger.json")
PIDFILE = os.path.join(HERE, "fleet_relay.pid")
STOPFLAG = os.path.join(HERE, "fleet_relay.stop")
LOG = os.path.join(HERE, "fleet_relay_daemon.log")

TICK_INTERVAL = 90          # 每个 tick 间隔（秒）
TARGET_ANSWERS = 6          # 每个问题目标累积答案数（达到后不再为该问题发问）
MAX_CONCURRENT = 16         # 单 tick 最大并发流
TIMEOUT = 150               # 单舰流式超时（秒）
MAX_TOKENS = 1500
FAIL_BACKOFF_BASE = 30      # 退避基数
FAIL_BACKOFF_MAX = 600      # 退避上限

CT = "application/json"
_CTX = ssl.create_default_context(); _CTX.check_hostname=False; _CTX.verify_mode=ssl.CERT_NONE

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

_lock = threading.Lock()
_logf = None
def log(msg, level="INFO"):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}][{level}] {msg}"
    try:
        with open(LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass
    print(line, flush=True)

def load_fleet():
    try:
        d = json.load(open(FLEET_JSON, encoding="utf-8"))
        ships = [s for s in d.get("ships", []) if s.get("enabled", True)]
        return ships
    except Exception as e:
        log(f"load_fleet failed: {e}", "ERROR")
        return []

def load_questions():
    qs = {}
    try:
        for fp in sorted(glob.glob(os.path.join(QUESTIONS_DIR, "*.json"))):
            try:
                arr = json.load(open(fp, encoding="utf-8"))
                if not isinstance(arr, list):
                    continue
                for q in arr:
                    if isinstance(q, dict) and q.get("id") and q.get("body"):
                        qs[q["id"]] = q
            except Exception as e:
                log(f"load_questions skip {fp}: {e}", "WARN")
    except Exception as e:
        log(f"load_questions dir err: {e}", "ERROR")
    return qs

def load_ledger():
    try:
        return json.load(open(LEDGER, encoding="utf-8"))
    except Exception:
        return {}

def save_ledger(L):
    tmp = LEDGER + ".tmp"
    json.dump(L, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    os.replace(tmp, LEDGER)

def stream_chat(ship, q_body):
    base, key, model = ship["base"], ship.get("key"), ship["model"]
    body = json.dumps({"model":model,"messages":[
        {"role":"system","content":CONSTRAINT},
        {"role":"user","content":q_body}],
        "stream":True,"max_tokens":MAX_TOKENS}).encode()
    req = urllib.request.Request(base, data=body, method="POST")
    req.add_header("Content-Type", CT)
    if key: req.add_header("Authorization","Bearer "+key)
    else:   req.add_header("Authorization","Bearer sk-any")
    req.add_header("Accept","text/event-stream")
    out=[]; served=None; err=None
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT, context=_CTX) as r:
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
                    except Exception: pass
    except urllib.error.HTTPError as e:
        err=f"HTTP {e.code}: {e.read().decode('utf-8','ignore')[:140]}"
    except Exception as e:
        err=f"{type(e).__name__}: {str(e)[:100]}"
    return "".join(out), served, err

def ask_one(ship, q_body, L, qid):
    sid = ship["id"]
    key_l = f"{qid}|{sid}"
    t=time.time()
    text,served,err = stream_chat(ship, q_body)
    ok = bool(text and not err)
    with _lock:
        if ok:
            od = os.path.join(ANS_ROOT, qid)
            os.makedirs(od, exist_ok=True)
            open(os.path.join(od, f"ans_{sid}.md"),"w",encoding="utf-8").write(
                f"# {sid} (vendor={ship['vendor']}, served={served}, {round(time.time()-t,1)}s)\n\n"+text)
            L[key_l] = {"status":"done","tries":L.get(key_l,{}).get("tries",0)+1,
                        "last":t,"served":served,"sec":round(time.time()-t,1)}
            log(f"  OK  {qid}/{sid} ({ship['vendor']}) {round(time.time()-t,1)}s")
        else:
            rec = L.get(key_l, {})
            tries = rec.get("tries",0)+1
            L[key_l] = {"status":"fail","tries":tries,"last":t,"err":err}
            log(f"  XX  {qid}/{sid} -> {err}", "WARN")
        save_ledger(L)

def due_pairs(qs, ships, L):
    """返回 [(qid, ship)] 中当前到期可重试的（pending 或退避已过期）且未达 done 的。"""
    now=time.time()
    pairs=[]
    for qid in qs:
        done_cnt=sum(1 for k,v in L.items() if k.startswith(qid+"|") and v.get("status")=="done")
        if done_cnt >= TARGET_ANSWERS:
            continue
        for s in ships:
            kl=f"{qid}|{s['id']}"
            rec=L.get(kl,{})
            st=rec.get("status")
            if st=="done":
                continue
            last=rec.get("last",0)
            tries=rec.get("tries",0)
            backoff=min(FAIL_BACKOFF_MAX, FAIL_BACKOFF_BASE*(2**max(0,tries-1)))
            if st is None or (now-last) >= backoff:
                pairs.append((qid,s))
    return pairs

def pick_starving_question(qs, L):
    """挑 done 数最少且未达标的问题（round-robin 偏饿）。"""
    best=None; best_cnt=10**9
    for qid in qs:
        done_cnt=sum(1 for k,v in L.items() if k.startswith(qid+"|") and v.get("status")=="done")
        if done_cnt>=TARGET_ANSWERS: continue
        if done_cnt<best_cnt:
            best_cnt=done_cnt; best=qid
    return best

def tick():
    ships=load_fleet()
    qs=load_questions()
    if not ships or not qs:
        log(f"tick: ships={len(ships)} qs={len(qs)} (skip)", "WARN")
        return
    L=load_ledger()
    qid=pick_starving_question(qs, L)
    if not qid:
        log("tick: all questions reached target answers", "INFO")
        return
    pairs=due_pairs({qid:qs[qid]}, ships, L)
    if not pairs:
        log(f"tick: {qid} no due ships now (all backing off)", "INFO")
        return
    pairs=pairs[:MAX_CONCURRENT]
    log(f"tick: question={qid} ({qs[qid]['title'][:28]}) -> fan-out to {len(pairs)} ships")
    threads=[]
    for qid2,s in pairs:
        t=threading.Thread(target=ask_one,args=(s,qs[qid2]["body"],L,qid2),daemon=True)
        threads.append(t)
    for t in threads: t.start()
    for t in threads: t.join(timeout=TIMEOUT+15)
    # 刷新 ledger 中本问题计数（ask_one 已各自保存，这里无额外动作）

def daemon_main():
    log("=== Fleet Relay Daemon started ===")
    log(f"fleet.json ships(enabled) loaded at start; questions dir={QUESTIONS_DIR}")
    while True:
        if os.path.exists(STOPFLAG):
            log("stop flag detected, exiting.", "INFO")
            try: os.remove(STOPFLAG)
            except: pass
            break
        try:
            tick()
        except Exception as e:
            log(f"tick exception: {e}\n{traceback.format_exc()}", "ERROR")
        time.sleep(TICK_INTERVAL)
    log("=== Fleet Relay Daemon stopped ===")

def write_pid():
    try:
        with open(PIDFILE,"w") as f: f.write(str(os.getpid()))
    except Exception:
        pass

def already_running():
    try:
        if not os.path.exists(PIDFILE): return False
        pid=int(open(PIDFILE).read().strip())
        # 检查进程是否存活（Windows：tasklist）
        import subprocess
        out=subprocess.run(["tasklist","/fi",f"PID eq {pid}"],capture_output=True,text=True).stdout
        return str(pid) in out
    except Exception:
        return False

if __name__=="__main__":
    if already_running():
        log("another instance running, exit.", "WARN")
        sys.exit(0)
    write_pid()
    try:
        daemon_main()
    finally:
        try: os.remove(PIDFILE)
        except: pass
