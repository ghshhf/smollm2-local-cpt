#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""node/ag.py —— 一条命令调远程角色智能体（不用登录、不用 SSH、不用填任何东西）

背景：node2 那台免费云主机上已经跑着 agent_roles.py（5 个角色智能体：researcher /
critic / editor / coder / validator），每个角色背后挂的是免 key 云脑（pollinations /
llm7 / kilo / unclose / ovh）。它没有公网入口，所以本机通过广播协议派活：
  本机一条命令 -> board.json -> node2 领取 -> 本地 curl 127.0.0.1:18092 -> 云脑作答
  -> out_node2_<seq>.txt -> 本机打印

用法：
  python node/ag.py list                          # node2 上有哪些角色
  python node/ag.py ask coder "写个 LRU 缓存"      # 问单个角色
  python node/ag.py ask critic "这个方案有什么漏洞" -n node1 -p 18091
  python node/ag.py pipe "给北极星设计一个语料去重方案"   # 探索->挑刺->统稿 三连
  python node/ag.py up                            # 服务掉了就重新拉起（幂等）

为什么 prompt 走 base64：中文 + 引号 + 换行直接塞进单行 sh 命令会被 shell 吃掉。
base64 只有 [A-Za-z0-9+/=]，跳过了所有转义地狱。
"""
import argparse
import base64
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from board import C, get_text, load_reg  # noqa: E402
from polaris_node import http_put  # noqa: E402

PKG_URL = "http://47.99.131.12:8081/file/ng2.tgz?key=k0u4mail"
BOOT_CMD = ("mkdir -p $HOME/agents && cd $HOME/agents && "
            "[ -f agent_roles.py ] || curl -sS -m 30 -o ng2.tgz '%s' && tar xzf ng2.tgz; "
            "chmod +x ng2_boot.sh 2>/dev/null; sh ng2_boot.sh" % PKG_URL)


def _next_seq():
    old = 0
    prev = get_text("board.json")
    if prev:
        try:
            old = int(json.loads(prev).get("seq") or 0)
        except ValueError:
            old = 0
    return max(int(time.time() * 1000), old + 1)


def send(node, cmd, timeout=120, wait=None):
    """上一块白板，等这一个节点的回传。返回 (ok, 输出文本)。"""
    if wait is None:
        wait = timeout + 30
    seq = _next_seq()
    body = {"seq": seq, "cmd": cmd, "target": node,
            "timeout": int(timeout), "ts": int(time.time())}
    http_put(C, "board.json", json.dumps(body, ensure_ascii=False).encode("utf-8"))
    deadline = time.time() + wait
    while time.time() < deadline:
        txt = get_text("out_%s_%d.txt" % (node, seq))
        if txt is not None:
            return True, txt
        time.sleep(0.15)
    return False, ""


def _role_req(node, port, role, prompt, max_tokens=700, timeout=120):
    body = json.dumps({"role": role, "q": prompt, "max_tokens": max_tokens},
                      ensure_ascii=False).encode("utf-8")
    b64 = base64.b64encode(body).decode("ascii")
    cmd = ("printf %%s '%s' | base64 -d > /tmp/ag_q.json && "
           "curl -sS -m %d -X POST -H 'Content-Type: application/json' "
           "--data-binary @/tmp/ag_q.json http://127.0.0.1:%d/role/ask"
           % (b64, timeout - 10, port))
    return send(node, cmd, timeout=timeout)


def _parse_answer(raw):
    """out_*.txt 的格式是：
        node=node2 seq=... elapsed=1.2s
        ---
        rc=0
        <正文>

    '---' 之后紧跟一行 rc=N，要一起剥掉（否则 json.loads 会被 rc=0 卡死）。"""
    lines = raw.splitlines()
    for i, ln in enumerate(lines):
        if ln.strip() == "---":
            body = lines[i + 1:]
            if body and body[0].startswith("rc="):
                body = body[1:]
            raw = "\n".join(body).strip()
            break
    else:
        raw = raw.strip()
    try:
        d = json.loads(raw)
    except ValueError:
        return None, raw
    return d, d.get("answer") or ""


def ask(node, port, role, prompt, timeout=120, quiet=False, retries=2):
    """调一个角色。连不上/回传里有 connect refused 时自动重试 —— 实测会撞上服务
    看护循环那 5 秒重启窗口，重试一次基本必成，不需要人工介入。"""
    last = ""
    for i in range(retries + 1):
        ok, raw = _role_req(node, port, role, prompt, timeout=timeout)
        if not ok:
            last = "（%s 在 %ds 内没回传，可能掉线或在重连）" % (node, timeout + 30)
            continue
        d, ans = _parse_answer(raw)
        _down = (not d) and ("Could not connect" in ans or "connect" in ans and "refused" in ans)
        if _down and i < retries:
            if not quiet:
                print("[ag] 服务正在重启窗口，%.0fs 后重试（%d/%d）" % (4, i + 1, retries))
            time.sleep(4)
            continue
        if not quiet:
            via = (d or {}).get("via", "?")
            print("[ag] %s/%s via=%s" % (node, role, via))
        return True, ans
    return False, last


def cmd_list(a):
    ok, raw = send(a.node, "curl -sS -m 10 http://127.0.0.1:%d/role/list" % a.port, timeout=30)
    if not ok:
        print("[ag] %s 无回传。试试：python node/ag.py up" % a.node)
        return 2
    d, _ = _parse_answer(raw)
    if not d:
        print("[ag] 角色服务没在跑。回传正文：%s" % raw[:300])
        print("[ag] 拉起：python node/ag.py up")
        return 2
    print("[ag] %s:%d  共 %d 个角色" % (a.node, a.port, len(d.get("roles") or [])))
    for r in d.get("roles") or []:
        print("  %-12s %s" % (r.get("id"), str(r.get("desc") or r.get("name") or "")[:78]))
    return 0


def cmd_ask(a):
    prompt = " ".join(a.prompt) if isinstance(a.prompt, list) else a.prompt
    if not prompt.strip():
        print("[ag] 缺 prompt"); return 1
    t0 = time.time()
    ok, ans = ask(a.node, a.port, a.role, prompt, timeout=a.timeout)
    print(ans)
    print("\n[ag] %.1fs" % (time.time() - t0))
    return 0 if ok else 2


def cmd_pipe(a):
    topic = " ".join(a.prompt) if isinstance(a.prompt, list) else a.prompt
    chain = [x.strip() for x in a.roles.split(",") if x.strip()]
    ctx = topic
    got = []                      # 已成功产出的片段，失败轮次用它兜底
    for i, role in enumerate(chain, 1):
        print("\n%s\n=== [%d/%d] %s ===" % ("-" * 58, i, len(chain), role))
        t0 = time.time()
        ok, ans = ask(a.node, a.port, role, ctx, timeout=a.timeout, quiet=True)
        if not ok:
            # 实测 pollinations 偶发 >90s 超时。单轮失败不该毁掉整条链：
            # 记一笔，把已有产出原样递给下一个角色，让它自己在缺料下前进。
            print("[跳过] %s —— %s" % (role, ans))
            print("       （链路继续，用前 %d 轮的产出兜底）" % len(got))
            if not got:
                return 2
            ctx = "已知材料（上一轮缺料，仅供你判断）：\n\n" + got[-1]
            continue
        print(ans)
        print("--- %.1fs" % (time.time() - t0))
        got.append(ans)
        nxt = {"researcher": "以下是调研结果，请你挑出漏洞与反方证据，不要重写：\n\n",
               "critic": "以下是原始调研与批评意见，请综合成结构化简报：\n\n",
               "validator": "以下内容请核对事实一致性，只标冲突与不确定项：\n\n"}.get(
                   chain[i] if i < len(chain) else "", "")
        ctx = (nxt or "以下是上一步产出，请在此基础上继续：\n\n") + ans[:4000]
        if i == 1 and topic.strip():
            ctx = "原始需求：" + topic.strip() + "\n\n" + ctx
    print("\n[ag] 完成 %d/%d 轮" % (len(got), len(chain)))
    return 0 if got else 2


def cmd_up(a):
    print("[ag] 检查 %s:%d ..." % (a.node, a.port))
    ok, raw = send(a.node, "curl -sS -m 8 -o /dev/null -w %%{http_code} "
                           "http://127.0.0.1:%d/role/list || echo DOWN" % a.port, timeout=30)
    if ok and "200" in raw:
        print("[ag] 服务在跑，不用动。")
        return 0
    print("[ag] 没在跑，下发拉起命令 ...")
    ok, raw = send(a.node, BOOT_CMD, timeout=120)
    print(raw.strip()[-400:] if ok else "[ag] 拉起命令无回传（节点可能掉线）")
    time.sleep(3)
    ok, raw = send(a.node, "curl -sS -m 8 http://127.0.0.1:%d/role/list" % a.port, timeout=30)
    if ok and "roles" in raw:
        print("[ag] 已在跑 ✅")
        return 0
    print("[ag] 仍未就绪，看日志：python node/board.py send -t %s 'tail -20 ~/agents/roles.log'" % a.node)
    return 2


def main():
    ap = argparse.ArgumentParser(description="一条命令调远程角色智能体")
    sub = ap.add_subparsers(dest="fn")

    def common(p):
        p.add_argument("-n", "--node", default="node2")
        p.add_argument("-p", "--port", type=int, default=18092)
        # 默认 180s：pollinations 在长 prompt 上实测会到 90s+，120s 仍会偶发超时
        p.add_argument("-t", "--timeout", type=int, default=180)

    p = sub.add_parser("list", help="列角色")
    common(p); p.set_defaults(fn=cmd_list)

    p = sub.add_parser("ask", help="问一个角色")
    common(p)
    p.add_argument("role")
    p.add_argument("prompt", nargs="+")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("pipe", help="多角色串成流水线")
    common(p)
    p.add_argument("prompt", nargs="+")
    p.add_argument("-r", "--roles", default="researcher,critic,editor")
    p.set_defaults(fn=cmd_pipe)

    p = sub.add_parser("up", help="服务掉了就拉起")
    common(p); p.set_defaults(fn=cmd_up)

    a = ap.parse_args()
    if not getattr(a, "fn", None):
        ap.print_help(); return 1
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
