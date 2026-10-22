#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""rc.py — 远程 node1 命令行（Remote CLI）+ 本地镜像索引

为什么要有它:
  过去看远程只能靠"贴一行命令 / 临时拼 shell"，等于每次都在盲摸。
  这里把远程当成一个可以 ls / tree / du / cat / ps 的文件系统 + 服务面板，
  并且把目录结构**同步一份到本地**（data/remote_index.json + docs/remote-index.md），
  所以远程关机时也能查"那里有什么、哪个目录是什么"。

两个速度档:
  在线(走信箱): 投递 ~1ms，远程执行 1~3s。命令: ls tree du cat stat ps svc find sh
  离线(读本地索引): 0ms，完全不碰远程。命令: idx sync(刷新时才联网)

用法:
  python node/rc.py ls                 # 列根目录
  python node/rc.py ls models -l       # 列 models 并带大小
  python node/rc.py tree -d 2          # 两层目录树
  python node/rc.py du                 # 各目录占用
  python node/rc.py cat data/seed_code_200.jsonl -n 3
  python node/rc.py find '*.safetensors'
  python node/rc.py ps                 # 谁在跑
  python node/rc.py svc                # 服务面板: 通道/隧道/训练/jupyter
  python node/rc.py sh 'df -h /mnt/workspace'
  python node/rc.py sync               # 全量扫描 → 本地索引
  python node/rc.py idx                # 看索引目录树(离线)
  python node/rc.py idx qwen           # 在索引里搜关键词(离线)

🔴 关键坑: mb.py 的实现在命令末尾拼 " | curl"，裸写 "a; b; c" 只有最后一条进管道，
   前面的输出全丢。所以本工具统一把命令包成 "{ ...; } 2>&1" 再发。
"""
import argparse
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from mb import Conn  # noqa: E402

ECS_HOST = "47.99.131.12"
ECS_PORT = 8081
KEY = "k0u4mail"
RROOT = "/mnt/workspace/node1"

IDX_JSON = os.path.join(ROOT, "data", "remote_index.json")
IDX_MD = os.path.join(ROOT, "docs", "remote-index.md")

# ---------------------------------------------------------------- 传输层


UPLOAD = (" | curl -s -m 900 -X POST 'http://%s:%d/upload?name=%%s&key=%s' --data-binary @-"
          % (ECS_HOST, ECS_PORT, KEY))


def oneline(cmd):
    """远程只吃单行命令：多行会被截断，表现为"(无回传)"。统一折叠成 '; ' 连接。"""
    parts = []
    for x in cmd.splitlines():
        x = x.strip()
        if x and not x.startswith("#"):
            parts.append(x)
    return " ; ".join(parts)


def remote(cmd, timeout=60, quiet=False):
    """发一条命令到 node1，等到回传，返回文本。

    优先走直连守护(SSH 长连接, ~240ms)；不可用时回落信箱(mailbox, ~2.5s)。
    """
    if os.environ.get("RC_NO_DAEMON") != "1":
        txt = via_daemon(cmd, timeout)
        if txt is not None:
            if not quiet:
                print("[rc] 直连", file=sys.stderr)
            return txt
    return via_mailbox(cmd, timeout, quiet)


def via_daemon(cmd, timeout):
    """走 chd.py 守护进程。返回 None 表示通道不可用（让上层回落信箱）。"""
    try:
        import chd
    except Exception:
        return None
    for attempt in (0, 1):
        try:
            r = chd.client(cmd, timeout)
            out = (r.get("out") or "").rstrip()
            err = (r.get("err") or "").rstrip()
            if r.get("rc") not in (0, None) and err and not out:
                return "[rc=%s] %s" % (r.get("rc"), err)
            if err and ("daemon:" in err):
                raise IOError(err)
            return out if out else "(空输出)"
        except Exception:
            if attempt:
                return None
            try:
                chd.spawn()          # 本地守护没起就拉起来
            except Exception:
                return None
    return None


def via_mailbox(cmd, timeout=60, quiet=False):
    tid = "rc%d" % int(time.time() * 1000)
    name = "rc_%s.txt" % tid
    # 🔴 结束标记：命令没输出时 curl 会传空 body，客户端分不清"执行完了没输出"和"没执行"，
    #    只能干等到超时（实测白等 26 秒）。所以恒定附一行 __RC_END__ 兜底。
    full = ("{ { %s ; } 2>&1 ; echo __RC_END__ ; } 2>&1" % oneline(cmd)) + (UPLOAD % name)
    payload = json.dumps({"cmd": full, "id": tid, "ts": int(time.time())}).encode()

    c = Conn()
    t0 = time.time()
    c.req("POST", "/task", payload)
    if not quiet:
        print("[rc] 投递 %.0fms" % ((time.time() - t0) * 1000), file=sys.stderr)

    deadline = time.time() + timeout
    text = ""
    while time.time() < deadline:
        try:
            text = c.req("GET", "/file/%s?key=%s" % (name, KEY), timeout=20).decode("utf-8", "replace")
        except Exception:
            text = ""
        if text.strip():
            break
        time.sleep(0.25)
    if not quiet:
        print("[rc] 回传 %.2fs" % (time.time() - t0), file=sys.stderr)
    text = text.replace("__RC_END__", "").strip()
    return text if text else "(空输出)"


def shq(s):
    """单引号安全包裹（路径里可能有空格/冒号/感叹号）"""
    return "'" + str(s).replace("'", "'\\''") + "'"


def abspath(p):
    if not p:
        return RROOT
    if p.startswith("/"):
        return p
    return RROOT + "/" + p


# ---------------------------------------------------------------- 在线命令


def cmd_ls(a):
    p = abspath(a.path)
    if a.long:
        print(remote("ls -lh --time-style=+%%Y-%%m-%%d\\ %%H:%%M %s | tail -n +2" % shq(p), a.timeout))
    else:
        print(remote("ls -p %s" % shq(p), a.timeout))


def cmd_tree(a):
    p = abspath(a.path)
    cmd = ("cd %s && find . -maxdepth %d -type d -printf '%%p\\n' 2>/dev/null | sort "
           "| head -%d" % (shq(p), a.depth, a.limit))
    print(remote(cmd, a.timeout))


def cmd_du(a):
    p = abspath(a.path)
    print(remote("du -sh %s/* 2>/dev/null | sort -h | tail -%d" % (shq(p), a.limit), a.timeout))


def cmd_cat(a):
    p = abspath(a.path)
    print(remote("head -n %d %s" % (a.n, shq(p)), a.timeout))


def cmd_stat(a):
    p = abspath(a.path)
    cmd = ("stat -c '%%n  size=%%s  mtime=%%y' %s; echo '--- head ---'; head -c 400 %s"
           % (shq(p), shq(p)))
    print(remote(cmd, a.timeout))


def cmd_find(a):
    print(remote("find %s -maxdepth %d -name %s -printf '%%s\\t%%TY-%%Tm-%%Td %%TH:%%TM\\t%%p\\n' "
                 "2>/dev/null | sort -rn | head -%d"
                 % (RROOT, a.depth, shq(a.pattern), a.limit), a.timeout))


def cmd_ps(a):
    print(remote("ps -eo pid,etime,pcpu,pmem,args --sort=-pcpu | head -n %d" % a.limit, a.timeout))


SVC_PROBE = r"""
FILT="grep -v 'sh -c' | grep -v 'curl -s -m 900' | grep -v pgrep"
echo '=== 通道/信箱 ==='
eval "pgrep -af 'node_v6|node\.py' | $FILT | head -3" || echo '(无)'
echo '=== 隧道 ==='
eval "pgrep -af 'ssh -N|frpc|cloudflared' | $FILT | head -3" || echo '(无)'
echo '=== 训练 ==='
eval "pgrep -af 'sft_full|dream_code|evolve_lora' | $FILT | head -3" || echo '(无)'
echo '=== jupyter ==='
eval "pgrep -af jupyter | $FILT | head -2" || echo '(无)'
echo '=== 端口 ==='
(ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) | head -8
echo '=== 负载/内存 ==='
uptime
free -m | head -2
"""


def cmd_svc(a):
    print(remote(SVC_PROBE, a.timeout))


def cmd_sh(a):
    print(remote(a.remote_cmd, a.timeout))


# ---------------------------------------------------------------- 索引

KIND_RULES = [
    ("models", "权重：北极星 0.3B(sft_code_03b / sft_seed200_remote) / siglip2 视觉 / whisper 听觉"),
    ("data", "种子、银行、holdout、训练用 jsonl"),
    ("corpus", "预训练/续训语料"),
    ("chain", "旧续训链（0.2B 时代）剩余件"),
    ("archive", "归档区：历史代资产，只归类不删除"),
    ("tools", "在用的脚本（sft_full / dream_code / seed_to_bank…）"),
    ("logs", "运行日志"),
    ("gh_repo", "档案仓 ghshhf/smollm2-local-cpt"),
    ("node", "信箱/自检/隧道/命令行客户端"),
    ("audio_parts", "听觉线音频分片（Whisper 训练素材）"),
    ("dream", "做梦器产物：衍生题、生成数据、旧 evolve 适配器"),
    ("bench_dl", "下载基准对照目录（旧）"),
    ("cli_dl", "命令行下载测试目录（旧）"),
    ("smoke_ckpt", "10 月 11 日冒烟产物（step30 / final）"),
    ("sftboot", "引导包：boot.sh + 数据 + tools（base 已归档）"),
    ("sftrun", "远程训练运行目录"),
    ("mdl", "模型小工具/中间件"),
    ("relay_dl", "relay 传输下载缓存"),
    ("frp", "frp 内网穿透二进制"),
    (".ipynb_checkpoints", "Jupyter 检查点"),
    ("__pycache__", "Python 字节码缓存"),
]


def guess_kind(path):
    path = path.replace("\\", "/")          # Windows 上 os.path.join 会拼出反斜杠
    rel = path.replace(RROOT + "/", "").strip("/")
    if not rel:
        return "root"
    top = rel.split("/")[0]
    for k, _ in KIND_RULES:
        if top.startswith(k):
            return k
    return "other"


def cmd_sync(a):
    """全量浅扫（深度可控）→ 本地 JSON + Markdown 索引"""
    probe = r'''
echo '@@SYS'
date '+%Y-%m-%dT%H:%M:%S%z'
echo '@@DISK'
df -h /mnt/workspace | tail -1
echo '@@CPU'
nproc; free -m | awk 'NR==2{print $2" "$3" "$7}'
echo '@@PY'
python3 -c "import torch,transformers;print('torch',torch.__version__,'tf',transformers.__version__)" 2>/dev/null || echo 'py?'
echo '@@REPO'
cd /mnt/workspace/node1/gh_repo 2>/dev/null && (git remote get-url origin 2>/dev/null; git rev-parse --abbrev-ref HEAD 2>/dev/null; git log -1 --format='%h %ad %s' --date=short 2>/dev/null) || echo '(no repo)'
echo '@@TREE'
find /mnt/workspace/node1 -maxdepth __DEPTH__ -printf '%y\t%s\t%TY-%Tm-%Td %TH:%TM\t%p\n' 2>/dev/null | head -__LIMIT__
echo '@@DU'
du -sh /mnt/workspace/node1/* 2>/dev/null | sort -h | tail -25
echo '@@END'
'''.replace("__DEPTH__", str(a.depth)).replace("__LIMIT__", str(a.limit))
    raw = remote(probe, a.timeout)
    if a.verbose:
        print(raw)
        return

    blocks, cur = {}, None
    for line in raw.splitlines():
        if line.startswith("@@"):
            cur = line[2:].strip()
            blocks[cur] = []
            continue
        if cur:
            blocks[cur].append(line)

    entries = []
    for line in blocks.get("TREE", []):
        parts = line.split("\t", 3)
        if len(parts) < 4:
            continue
        typ, size, mtime, path = parts
        path = path.strip()
        if path == RROOT:
            continue
        rel = path[len(RROOT) + 1:]
        entries.append({
            "path": rel,
            "type": "d" if typ == "d" else "f",
            "size": int(size) if size.isdigit() else 0,
            "mtime": mtime,
            "kind": guess_kind(path),
        })

    du_rows = []
    for line in blocks.get("DU", []):
        if "\t" in line:
            sz, p = line.split("\t", 1)
            du_rows.append([sz.strip(), p.strip().split("/")[-1]])

    repo_parts = [x.strip() for x in blocks.get("REPO", []) if x.strip() and not x.startswith("(")]
    idx = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "host": "node1",
        "root": RROOT,
        "sys": {"date": (blocks.get("SYS") or [""])[0],
                "disk": (blocks.get("DISK") or [""])[0],
                "cpu_mem": (blocks.get("CPU") or [""]),
                "python": (blocks.get("PY") or [""])[0]},
        "repo": {"url": repo_parts[0] if len(repo_parts) > 0 else "",
                 "branch": repo_parts[1] if len(repo_parts) > 1 else "",
                 "head": repo_parts[2] if len(repo_parts) > 2 else ""},
        "du": du_rows,
        "entries": entries,
    }
    os.makedirs(os.path.dirname(IDX_JSON), exist_ok=True)
    with open(IDX_JSON, "w", encoding="utf-8") as f:
        json.dump(idx, f, ensure_ascii=False, indent=1)
    write_md(idx)
    print("[idx] %d 条记录 → %s" % (len(entries), IDX_JSON))
    print("[idx] 目录说明 → %s" % IDX_MD)


def write_md(idx):
    L = []
    L.append("# 远程 node1 索引（离线可查）\n")
    L.append("> 由 `python node/rc.py sync` 生成，本地镜像。远程关机也能看。\n")
    L.append("| 项 | 值 |\n|---|---|")
    L.append("| 生成时间 | %s |" % idx["generated"])
    L.append("| 主机 | node1（阿里云 NAS 挂载，8 核，无 GPU） |")
    L.append("| 根目录 | `%s` |" % idx["root"])
    L.append("| 磁盘 | %s |" % idx["sys"].get("disk", ""))
    L.append("| 仓库 | %s |" % (idx["repo"].get("url") or "(无)"))
    L.append("| 分支/HEAD | %s / %s |" % (idx["repo"].get("branch", ""), idx["repo"].get("head", "")))
    L.append("| Python | %s |" % idx["sys"].get("python", ""))
    L.append("")

    L.append("## 一、一级目录：叫什么、是什么\n")
    L.append("| 目录 | 大小 | 用途 |\n|---|---|---|")
    du_map = {n: s for s, n in idx["du"]}
    seen = set()
    for e in idx["entries"]:
        if "/" in e["path"] or e["type"] != "d":
            continue
        n = e["path"]
        if n in seen:
            continue
        seen.add(n)
        desc = dict(KIND_RULES).get(guess_kind(os.path.join(RROOT, n)), "")
        L.append("| `%s/` | %s | %s |" % (n, du_map.get(n, ""), desc))
    L.append("")

    L.append("## 二、二级目录（前 %d 条）\n" % min(80, len(idx["entries"])))
    L.append("| 路径 | 类型 | 大小 | 时间 |\n|---|---|---|---|")
    n2 = 0
    for e in idx["entries"]:
        if e["path"].count("/") != 1:
            continue
        n2 += 1
        if n2 > 80:
            L.append("| … | | | |")
            break
        sz = "-" if e["type"] == "d" else human(e["size"])
        L.append("| `%s` | %s | %s | %s |" % (e["path"], "目录" if e["type"] == "d" else "文件",
                                             sz, e["mtime"]))
    L.append("")
    L.append("## 三、怎么用\n")
    L.append("```bash")
    L.append("python node/rc.py idx            # 目录树(离线, 0ms)")
    L.append("python node/rc.py idx qwen       # 关键词搜(离线)")
    L.append("python node/rc.py ls models -l   # 在线看(1~3s)")
    L.append("python node/rc.py svc            # 服务面板")
    L.append("python node/rc.py sync           # 重新扫描刷新本索引")
    L.append("```")
    with open(IDX_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L))


def human(n):
    for u in ("B", "K", "M", "G", "T"):
        if n < 1024:
            return "%.0f%s" % (n, u) if u == "B" else "%.1f%s" % (n, u)
        n /= 1024.0
    return "%.1fP" % n


def load_idx():
    if not os.path.exists(IDX_JSON):
        print("[idx] 还没有本地索引，先跑: python node/rc.py sync", file=sys.stderr)
        sys.exit(1)
    with open(IDX_JSON, encoding="utf-8") as f:
        return json.load(f)


def cmd_md(a):
    """不联网，只按现有 JSON 重生成人看的 markdown"""
    idx = load_idx()
    write_md(idx)
    print("[idx] %s 已重建（基于 %s 的扫描结果）" % (IDX_MD, idx["generated"]))


def cmd_shell(a):
    """常驻会话：一次启动，连续执行多条命令。

    为什么要有：单条命令的 0.25s 里约 0.15s 是 Python 解释器启动，
    这个开销每条都要付一次。在这里只付一次，之后每条只剩链路 ~90ms。
    用法:  echo -e 'ls\npwd' | python node/rc.py shell
    """
    import chd
    for line in sys.stdin:
        c = line.rstrip("\r\n")
        if c.strip() in ("exit", "quit"):
            break
        if not c.strip():
            continue
        t0 = time.time()
        try:
            r = chd.client(c, a.timeout)
            out, err, rc = r.get("out", ""), r.get("err", ""), r.get("rc")
        except Exception as e:
            out, err, rc = "", "会话失败: %r" % (e,), 255
        if out:
            sys.stdout.write(out if out.endswith("\n") else out + "\n")
        if err:
            sys.stdout.write(err if err.endswith("\n") else err + "\n")
        print("[%.0fms rc=%s]" % ((time.time() - t0) * 1000, rc), file=sys.stderr)


def cmd_idx(a):
    idx = load_idx()
    if a.key:
        kw = a.key.lower()
        hits = [e for e in idx["entries"] if kw in e["path"].lower()]
        print("[idx] 生成于 %s，命中 %d 条" % (idx["generated"], len(hits)))
        for e in sorted(hits, key=lambda x: -x["size"])[:60]:
            print("  %s %10s  %s  %s" % ("d" if e["type"] == "d" else "-",
                                         "" if e["type"] == "d" else human(e["size"]),
                                         e["mtime"], e["path"]))
        return

    print("[idx] node1 @ %s   生成于 %s" % (idx["root"], idx["generated"]))
    print("      磁盘 %s" % idx["sys"].get("disk", ""))
    print("      仓库 %s" % (idx["repo"].get("url") or "(无)"))
    du_map = {n: s for s, n in idx["du"]}
    print("")
    for e in sorted([x for x in idx["entries"] if "/" not in x["path"] and x["type"] == "d"],
                    key=lambda x: -x["size"]):
        print("  %-6s %-8s %s/" % (dict(KIND_RULES).get(guess_kind(os.path.join(RROOT, e["path"])), "·")[:6],
                                   du_map.get(e["path"], ""), e["path"]))
    print("\n  提示: rc.py idx <关键词>  再细查；rc.py sync 刷新；docs/remote-index.md 是给人看的版本")


# ---------------------------------------------------------------- 文件传输


def cmd_put(a):
    """上传本地文件到 node1（分块 base64，走常驻通道，不经 ECS 中转）。

    为什么不用魔搭/ECS 中转：那条路下行只有 947KB/s 且会断；常驻通道虽然分块，
    但每块都是一次往返，脚本级文件(几十 KB)一秒内就到。
    🔴 上限意识：512KB 块 = 2600 块/G，1GB 级文件请走 NAS 或 ECS 中转，别用这个。
    """
    src = a.local
    if not os.path.exists(src):
        raise SystemExit("本地文件不存在: %s" % src)
    dst = a.remote or (RROOT + "/" + os.path.basename(src))
    data = open(src, "rb").read()
    ch = max(64, a.chunk) * 1024
    parts = [data[i:i + ch] for i in range(0, len(data), ch)] or [b""]
    tmp = "/tmp/_rcput_%d" % int(time.time() * 1000)
    remote("mkdir -p %s" % tmp, a.timeout)
    t0 = time.time()
    for i, p in enumerate(parts):
        b64 = __import__("base64").b64encode(p).decode()
        remote("printf '%%s' '%s' | base64 -d > %s/p%04d" % (b64, tmp, i), a.timeout, quiet=True)
        if a.verbose:
            print("  [%d/%d] %.0fKB" % (i + 1, len(parts), len(p) / 1024), file=sys.stderr)
    remote("mkdir -p %s && cat %s/p* > %s && rm -rf %s && md5sum %s && wc -c < %s"
           % (shq(os.path.dirname(dst)), tmp, shq(dst), tmp, shq(dst), shq(dst)), a.timeout)
    el = time.time() - t0
    print("[put] %s -> node1:%s  %.1fKB  %.1fs  (%.0fKB/s)"
          % (src, dst, len(data) / 1024, el, len(data) / 1024 / max(el, 0.01)))


def cmd_get(a):
    """从 node1 取文件到本地（小文件用；大文件走 ECS 中转）。"""
    dst = a.local
    txt = remote("base64 -w0 %s" % shq(abspath(a.remote)), max(a.timeout, 120))
    txt = (txt or "").strip()
    if not txt or txt.startswith("(空输出)") or txt.startswith("[rc="):
        raise SystemExit("[get] 远端读取失败: %s" % txt[:200])
    import base64 as _b64
    data = _b64.b64decode(txt)
    os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
    with open(dst, "wb") as f:
        f.write(data)
    print("[get] node1:%s -> %s  %.1fKB" % (a.remote, dst, len(data) / 1024))


# ---------------------------------------------------------------- entry


def main():
    ap = argparse.ArgumentParser(prog="rc", description="远程 node1 命令行 + 本地镜像索引")
    ap.add_argument("-t", "--timeout", type=float, default=60)
    sub = ap.add_subparsers(dest="op")

    p = sub.add_parser("ls", help="列目录"); p.add_argument("path", nargs="?", default="")
    p.add_argument("-l", "--long", action="store_true"); p.set_defaults(fn=cmd_ls)

    p = sub.add_parser("tree", help="目录树"); p.add_argument("path", nargs="?", default="")
    p.add_argument("-d", "--depth", type=int, default=2)
    p.add_argument("--limit", type=int, default=200); p.set_defaults(fn=cmd_tree)

    p = sub.add_parser("du", help="占用"); p.add_argument("path", nargs="?", default="")
    p.add_argument("--limit", type=int, default=25); p.set_defaults(fn=cmd_du)

    p = sub.add_parser("cat", help="看文件头"); p.add_argument("path")
    p.add_argument("-n", type=int, default=40); p.set_defaults(fn=cmd_cat)

    p = sub.add_parser("stat", help="单文件详情"); p.add_argument("path"); p.set_defaults(fn=cmd_stat)

    p = sub.add_parser("find", help="找文件"); p.add_argument("pattern")
    p.add_argument("-d", "--depth", type=int, default=4)
    p.add_argument("--limit", type=int, default=30); p.set_defaults(fn=cmd_find)

    p = sub.add_parser("ps", help="进程"); p.add_argument("--limit", type=int, default=15)
    p.set_defaults(fn=cmd_ps)

    p = sub.add_parser("svc", help="服务面板"); p.set_defaults(fn=cmd_svc)

    p = sub.add_parser("sh", help="任意命令"); p.add_argument("remote_cmd"); p.set_defaults(fn=cmd_sh)

    p = sub.add_parser("sync", help="扫描并刷新本地索引")
    p.add_argument("-d", "--depth", type=int, default=3)
    p.add_argument("--limit", type=int, default=4000)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("-t", "--timeout", type=float, default=180)
    p.set_defaults(fn=cmd_sync)

    p = sub.add_parser("idx", help="查本地索引(离线)")
    p.add_argument("key", nargs="?", default=""); p.set_defaults(fn=cmd_idx)

    p = sub.add_parser("md", help="按现有索引重生成 markdown(离线)")
    p.set_defaults(fn=cmd_md)

    p = sub.add_parser("shell", help="常驻会话(连续执行, 每条 ~90ms)")
    p.set_defaults(fn=cmd_shell)

    p = sub.add_parser("put", help="上传文件到 node1（分块, 适合脚本级文件）")
    p.add_argument("local"); p.add_argument("remote", nargs="?", default="")
    p.add_argument("--chunk", type=int, default=256, help="每块 KB（默认 256）")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=cmd_put)

    p = sub.add_parser("get", help="从 node1 取文件到本地（小文件）")
    p.add_argument("remote"); p.add_argument("local")
    p.set_defaults(fn=cmd_get)

    a = ap.parse_args()
    if not getattr(a, "fn", None):
        ap.print_help()
        return
    a.fn(a)


if __name__ == "__main__":
    main()
