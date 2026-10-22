# -*- coding: utf-8 -*-
"""boot_resume.py — node1 开机一键：自检 + 自适应续训(参数增量, 非LoRA) + 回传  [v2]

v2 修复（针对 2026-10-19 12:28 首次实跑暴露的三个问题）：
  1. **根选偏**：v1 只选一个项目根，结果选了 /mnt/workspace/model（只有 models/），
     代码在别处 → tools 全判缺失。v2 把 **代码根 / 数据根 / 模型** 分开各自发现。
  2. **curl 静默失败**：v1 用 curl -s -o，失败也留不下文件。v2 改 urllib 下载并校验大小。
  3. **数据找不到**：v1 只认 seed_code_200.jsonl。v2 先按 sft_meta.data 的文件名匹配
     （card_code.jsonl / card_code_long.jsonl），再退种子，再退最新 jsonl。
  另：续训命令全部传**绝对路径**，彻底绕开"脚本 ROOT ≠ 数据所在根"的错位。

适配（用户定调）：自适应，不指定"上次训练是哪个"；远程打分器不动，只把 LoRA 换成参数增量。
用法（魔搭终端贴一行）：
  mkdir -p /mnt/workspace/node1 && curl -s -o /mnt/workspace/node1/boot_resume.py \
    "http://47.99.131.12:8081/file/boot_resume.py?key=k0u4mail" && cd /mnt/workspace/node1 && python3 boot_resume.py
"""
import json
import os
import subprocess
import time
import urllib.request

ECS = "http://47.99.131.12:8081"
KEY = "k0u4mail"
TS = time.strftime("%Y%m%d_%H%M%S")
WS = "/mnt/workspace"
REP = []


def say(m):
    print(m, flush=True)
    REP.append(str(m))


def sh(c, t=60):
    try:
        return subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t)
    except Exception as e:
        class R:
            stdout = ""
            stderr = str(e)
            returncode = -1
        return R()


def ls(d):
    try:
        return os.listdir(d)
    except OSError:
        return []


# ---------------- 环境 ----------------
def cgroup_cpu():
    try:
        q = open("/sys/fs/cgroup/cpu.max").read().split()
        if q[0] != "max":
            return max(1, int(int(q[0]) / int(q[1])))
    except Exception:
        pass
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return max(1, (os.cpu_count() or 4))


def mem_gb():
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemTotal"):
                return int(ln.split()[1]) / 1048576.0
    except Exception:
        pass
    return 0.0


def disk_gb(p):
    try:
        s = os.statvfs(p)
        return s.f_bavail * s.f_frsize / 1073741824.0
    except Exception:
        return 0.0


def gpu_info():
    out = []
    r = sh("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null", t=15)
    if r.stdout.strip():
        out.append("nvidia: " + r.stdout.strip().replace("\n", " | "))
    r = sh("rocm-smi --showmeminfo vram 2>/dev/null | head -3", t=15)
    if r.stdout.strip() and "GPU" in r.stdout:
        out.append("rocm: 有")
    return out or ["无 GPU"]


# ---------------- 分开发现：代码根 / 数据根 ----------------
def discover(ws=WS):
    """返回 (code_roots, data_roots)。扫 ws 一级目录 + 它们本身。纯 listdir/stat。"""
    code, data = [], []
    tops = [ws]
    for n in ls(ws):
        p = os.path.join(ws, n)
        if os.path.isdir(p):
            tops.append(p)
    for top in tops:
        ts = os.path.join(top, "tools")
        if any(os.path.isfile(os.path.join(ts, f))
               for f in ("sft_full.py", "train_lora.py", "dream_code.py", "train_smollm2_pt.py")):
            code.append(top)
        dd = os.path.join(top, "data")
        if os.path.isdir(dd) and any(f.endswith(".jsonl") for f in ls(dd)):
            data.append(dd)
    return code, data


def probe_model(d):
    cfg = os.path.join(d, "config.json")
    if not os.path.isfile(cfg):
        return None
    wts, size = [], 0
    for fn in ls(d):
        if fn.endswith(".safetensors") or (fn.startswith("pytorch_model") and fn.endswith(".bin")):
            fp = os.path.join(d, fn)
            try:
                sz = os.path.getsize(fp)
            except OSError:
                continue
            if sz > 50 * 1048576:
                wts.append(fn)
                size += sz
    if not wts:
        return None
    try:
        c = json.load(open(cfg, encoding="utf-8"))
    except Exception:
        c = {}
    meta = None
    mp = os.path.join(d, "sft_meta.json")
    if os.path.isfile(mp):
        try:
            meta = json.load(open(mp, encoding="utf-8"))
        except Exception:
            meta = {}
    try:
        mt = max(os.path.getmtime(os.path.join(d, fn)) for fn in wts)
    except Exception:
        mt = os.path.getmtime(d)
    return {"dir": d, "size": size, "mtime": mt,
            "hidden": c.get("hidden_size"), "layers": c.get("num_hidden_layers"),
            "vocab": c.get("vocab_size"), "meta": meta}


def scan_models(ws=WS):
    """扫 ws 一级目录、其 models/、models/*/、以及 ws 本身的两层。排除纯 LoRA adapter。"""
    hits, seen = [], set()

    def add(d):
        if d in seen or not os.path.isdir(d):
            return
        seen.add(d)
        if os.path.isfile(os.path.join(d, "adapter_config.json")) and \
           not os.path.isfile(os.path.join(d, "config.json")):
            return
        h = probe_model(d)
        if h:
            hits.append(h)

    bases = [ws]
    for n in ls(ws):
        p = os.path.join(ws, n)
        if os.path.isdir(p):
            bases.append(p)
            bases.append(os.path.join(p, "models"))
            bases.append(os.path.join(p, "backups"))
    for b in bases:
        add(b)
        for n2 in ls(b):
            add(os.path.join(b, n2))
    return hits


# ---------------- 资产兜底（urllib，校验大小） ----------------
def ensure_file(dst, name, min_bytes=300):
    if os.path.isfile(dst) and os.path.getsize(dst) >= min_bytes:
        return True, "已有 %.1fKB" % (os.path.getsize(dst) / 1024.0)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = "%s/file/%s?key=%s" % (ECS, name, KEY)
    for attempt in (0, 1):
        try:
            data = urllib.request.urlopen(url, timeout=60).read()
            if len(data) >= min_bytes:
                open(dst, "wb").write(data)
                return True, "已从ECS补 %dB" % len(data)
        except Exception as e:
            if attempt:
                return False, "下载失败 %r" % (e,)
    return False, "下载过小"


# ---------------- 数据选择 ----------------
def all_jsonl(data_roots):
    out = []
    for d in data_roots:
        for f in ls(d):
            if f.endswith(".jsonl"):
                fp = os.path.join(d, f)
                try:
                    if os.path.getsize(fp) > 200:
                        out.append(fp)
                except OSError:
                    pass
    return out


def pick_data(meta, data_roots):
    """按 sft_meta.data 的文件名匹配 → seed_code_200 → 最新 jsonl。返回绝对路径列表。"""
    files = all_jsonl(data_roots)
    names = {os.path.basename(p): p for p in files}
    if meta and isinstance(meta.get("data"), list):
        got = [names[os.path.basename(p)] for p in meta["data"]
               if os.path.basename(p) in names]
        if got:
            return got, "sft_meta.data 文件名匹配"
    if "seed_code_200.jsonl" in names:
        return [names["seed_code_200.jsonl"]], "seed_code_200.jsonl"
    if files:
        files.sort(key=lambda p: -os.path.getmtime(p))
        return files[:2], "最新 jsonl"
    return [], "无"


def rows(p):
    n = 0
    try:
        for _ in open(p, encoding="utf-8"):
            n += 1
    except Exception:
        pass
    return n


# ---------------- 主流程 ----------------
def main():
    say("=== boot_resume v2 自适应自检 @ %s ===" % time.strftime("%Y-%m-%d %H:%M:%S"))
    say("host=%s  uptime=%s" % (sh("hostname").stdout.strip() or "?",
                                " ".join(sh("uptime").stdout.split()[-3:])))
    cores = cgroup_cpu()
    say("CPU: cgroup配额=%d (nproc=%s)" % (cores, sh("nproc").stdout.strip()))
    # 🔴 2026-10-19 实测（docs/remote-server-plan-2026-10-19.md §8.7）:
    #    Pod 是 burstable QoS，cgroup 配额写着 8 核，但并行加速比在 4 进程后就封顶，
    #    有效算力只有 3.1~3.2 核（cpu.stat 几乎不 throttled，是宿主机超卖）。
    #    所以训练线程按 3 开，而不是按配额 8 开 —— 开 7~8 只会互相抢。
    train_threads = int(os.environ.get("POLARIS_THREADS", "3") or 3)
    train_threads = max(1, min(train_threads, cores))
    say("训练线程=%d (配额 %d，有效算力实测约 3.2 核)" % (train_threads, cores))
    say("MEM: %.1f GB  DISK(%s): %.1f GB" % (mem_gb(), WS, disk_gb(WS)))
    say("GPU: " + " ; ".join(gpu_info()))
    say("python: " + sh("python3 -V").stdout.strip())
    for mod in ("torch", "transformers", "sqlite3"):
        r = sh("python3 -c \"import %s as m;print(getattr(m,'__version__','ok'))\"" % mod, t=30)
        say("  %-13s %s" % (mod, (r.stdout or r.stderr).strip()[:80]))

    # 分开发现根
    code_roots, data_roots = discover()
    say("--- 发现 ---")
    say("代码根(%d): %s" % (len(code_roots), code_roots or "无"))
    say("数据根(%d): %s" % (len(data_roots), data_roots or "无"))
    for d in data_roots:
        js = [f for f in ls(d) if f.endswith(".jsonl")]
        say("  %s: %s" % (d, ", ".join("%s(%d)" % (f, rows(os.path.join(d, f))) for f in js[:6])))

    # 模型
    models = scan_models()
    models.sort(key=lambda h: -h["mtime"])
    say("--- 模型/训练产物 top12 ---")
    for h in models[:12]:
        tag = ""
        if h["meta"]:
            tag = " [sft_meta step=%s/%s rows=%s lr=%s data=%s]" % (
                h["meta"].get("step"), h["meta"].get("total"), h["meta"].get("rows"),
                h["meta"].get("lr"), h["meta"].get("data"))
        say("  %-56s %5.0fMB h=%s L=%s v=%s %s%s" % (
            h["dir"], h["size"] / 1048576.0, h["hidden"], h["layers"], h["vocab"],
            time.strftime("%m-%d %H:%M", time.localtime(h["mtime"])), tag))

    # 银行
    bank_n = "?"
    for dr in data_roots:
        dbp = os.path.join(dr, "triple_bank.db")
        if os.path.isfile(dbp):
            try:
                import sqlite3
                c = sqlite3.connect(dbp, timeout=30)
                c.execute("PRAGMA busy_timeout=30000")
                bank_n = "%s条 @%s" % (c.execute("select count(*) from triples").fetchone()[0], dbp)
                c.close()
            except Exception as e:
                bank_n = "err:%r" % (e,)
            break
    say("银行: %s" % bank_n)

    # ---- 起通道（node_v6 + 反向隧道；幂等，失败不阻断续训）----
    # 目的：开机后主控侧(n1.sh/mb.py)能直连，后续调试不必再人工贴命令。
    say("--- 起通道 ---")
    try:
        bp = os.path.join(WS, "node1", "bootstrap_all.py")
        ok, msg = ensure_file(bp, "bootstrap_all.py", 3000)
        say("bootstrap_all.py: %s" % msg)
        if ok:
            os.makedirs(os.path.dirname(bp), exist_ok=True)
            os.chdir(os.path.dirname(bp))
            exec(compile(open(bp, encoding="utf-8").read(), bp, "exec"), {"__name__": "__main__"})
            say("通道启动完成")
    except Exception as e:
        say("通道启动异常(不阻断): %r" % (e,))

    # ---- 反向隧道看门狗（SSH 直连的前提；幂等）----
    # 2026-10-19: 以前这层只有一条裸 while 循环，密钥文件丢了也无人发现，
    # 结果 ECS 上的 9022 是个死 socket，SSH 直连一直 255。现在由 node/tunnel.sh 负责，
    # 自检每次开机确保它在跑；密钥在 /root/.ssh/n1t2（已授权到 ECS administrators_authorized_keys）。
    say("--- 反向隧道 ---")
    try:
        base = os.path.join(WS, "node1")
        tsh = os.path.join(base, "node", "tunnel.sh")
        ok, msg = ensure_file(tsh, "tunnel.sh", 300)
        say("tunnel.sh: %s" % msg)
        if ok:
            try:
                os.chmod(tsh, 0o755)
            except Exception:
                pass
            pid = sh("pgrep -f 'node/tunnel.sh' | head -1", 10).strip()
            if pid:
                say("隧道看门狗已在跑 pid=%s" % pid)
            else:
                sh("cd %s && setsid nohup bash node/tunnel.sh >/dev/null 2>&1 </dev/null &" % base, 20)
                time.sleep(6)
                pid = sh("pgrep -f 'node/tunnel.sh' | head -1", 10).strip()
                say("隧道看门狗已拉起 pid=%s" % (pid or "(未起，请查 logs/tunnel.log)"))
        say("22 端口监听数: %s" % sh("ss -ltn 2>/dev/null | grep -c ':22 '", 10).strip())
        say("隧道日志末行: %s" % sh("tail -n 1 %s/logs/tunnel.log 2>/dev/null" % base, 10).strip()[:120])
    except Exception as e:
        say("隧道启动异常(不阻断): %r" % (e,))

    # 选续训起点
    with_meta = [h for h in models if h["meta"]]
    pick = (with_meta or models or [None])[0]
    say("--- 续训判定 ---")
    if not pick:
        say("!! 无完整模型，只报自检"); return finish()
    say("最近训练产物 => %s" % pick["dir"])
    if pick["meta"]:
        say("  sft_meta: " + json.dumps(pick["meta"], ensure_ascii=False)[:260])

    # 代码：找到 sft_full.py；缺则补到首选根
    sft = None
    for cr in code_roots:
        p = os.path.join(cr, "tools", "sft_full.py")
        if os.path.isfile(p):
            sft = p
            break
    if not sft:
        dst = os.path.join(code_roots[0] if code_roots else os.path.dirname(pick["dir"]),
                           "tools", "sft_full.py")
        ok, msg = ensure_file(dst, "sft_full.py", 3000)
        say("sft_full.py 补齐: %s (%s)" % (dst, msg))
        sft = dst if ok else None
    else:
        say("sft_full.py => %s" % sft)
    if not sft:
        say("!! sft_full.py 仍缺失，只报自检"); return finish()

    # 数据：sft_meta 名 → seed → 最新
    data, why = pick_data(pick["meta"], data_roots)
    if not data:
        # 补种子优先落到**代码根/data**（让下次自检就能发现数据根），其次模型旁
        ddir = os.path.join(code_roots[0], "data") if code_roots \
            else os.path.join(os.path.dirname(os.path.dirname(pick["dir"])), "data")
        dst = os.path.join(ddir, "seed_code_200.jsonl")
        ok, msg = ensure_file(dst, "seed_code_200.jsonl", 10000)
        say("seed 补齐: %s (%s)" % (dst, msg))
        data = [dst] if ok else []
        why = "ECS补种子"
        # 顺带补银行
        dbdst = os.path.join(ddir, "triple_bank.db")
        ok2, msg2 = ensure_file(dbdst, "triple_bank.db", 1000)
        say("triple_bank.db 补齐: %s (%s)" % (dbdst, msg2))
    say("数据 => %s (%s)" % (data, why))
    for d in data:
        say("  %-52s %d 条" % (d, rows(d)))

    # 防重
    pids = sh("pgrep -f '[s]ft_full.py'").stdout.split()
    if pids:
        say("已有 sft_full 在跑 PID=%s，不重复拉起" % ",".join(pids)); return finish()
    if not data:
        say("!! 无训练数据，无法续训"); return finish()

    # 拉起：全绝对路径
    out = os.path.join(os.path.dirname(pick["dir"]), "resume_%s" % TS)
    log = os.path.join(os.path.dirname(pick["dir"]), "..", "logs", "boot_resume_%s.log" % TS)
    log = os.path.abspath(log)
    os.makedirs(os.path.dirname(log), exist_ok=True)
    tok = pick["dir"] if os.path.isfile(os.path.join(pick["dir"], "tokenizer_config.json")) \
        else pick["dir"]

    cmd = ["python3", "-u", sft,
           "--llm", pick["dir"], "--tok", tok,
           "--data"] + data + [
           "--out", out, "--epochs", "2", "--accum", "4", "--lr", "1e-4",
           "--max-len", "512", "--threads", str(train_threads),
           "--log-every", "5", "--save-every", "50"]
    say("续训命令(参数增量/恒维): " + " ".join(cmd))
    env = dict(os.environ, OMP_NUM_THREADS=str(train_threads), MKL_NUM_THREADS=str(train_threads),
               PYTHONUNBUFFERED="1")
    with open(log, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen(cmd, cwd=os.path.dirname(os.path.dirname(sft)),
                                stdout=lf, stderr=subprocess.STDOUT,
                                start_new_session=True, env=env)
    say("已拉起 PID=%d 日志=%s" % (proc.pid, log))
    time.sleep(15)
    try:
        say("--- 15s 后 ---\n" + (open(log, encoding="utf-8", errors="replace").read()[-1600:].strip()
                                  or "(空)"))
    except Exception as e:
        say("(读日志失败 %r)" % (e,))
    say("结论: 训练进程 %s" % ("仍在运行 OK" if proc.poll() is None
                             else "已退出 FAIL(code=%s)" % proc.returncode))
    finish()


def finish():
    say("=== boot_resume 结束 ===")
    try:
        p = "/tmp/boot_resume_%s.txt" % TS
        open(p, "w", encoding="utf-8").write("\n".join(REP))
        req = urllib.request.Request(
            "%s/upload?name=boot_resume_%s.txt&key=%s" % (ECS, TS, KEY),
            data=open(p, "rb").read(), method="POST")
        r = urllib.request.urlopen(req, timeout=40)
        print("[已回传 boot_resume_%s.txt] %s" % (TS, r.read().decode("utf-8", "replace")[:80]))
    except Exception as e:
        print("[回传失败] %r" % (e,))


if __name__ == "__main__":
    main()
