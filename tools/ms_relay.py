#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ms_relay.py — 用魔搭仓库(阿里云 OSS 底座)做本机 <-> node1 的大文件中转

为什么不用信箱/隧道搬大文件(2026-10-18 实测):
  ECS 是 2C4G 窄带, 8MB 走 SSH 隧道 85s 后断链 (~94KB/s); 1.3GB 权重要 4 小时。
  魔搭仓库是 OSS 底座: 两端都直连阿里云存储, node1 下载实测 ~200MB/s。
  所以: 控制流走隧道(毫秒~1s), 大文件走仓库(百 MB/s)。

用法
  python ms_relay.py ls
  python ms_relay.py push <本地文件> [--as inbox/名字] 
  python ms_relay.py pull <仓库内路径> [--out 本地目录]
  python ms_relay.py push-dir <本地目录> --as inbox/目录名
  可选: --repo <owner/name>  默认 hhhlhuk/polaris-relay
        --model             用模型仓库而不是数据集仓库(默认 polaris-03b)

token 解析顺序: 环境变量 MODELSCOPE_ACCESS_TOKEN -> ~/.modelscope/credentials/token
"""
import argparse
import os
import sys
import time

DEFAULT_DS = "hhhlhuk/polaris-relay"
DEFAULT_MODEL = "hhhlhuk/polaris-03b"

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT_PARENT = os.path.dirname(os.path.dirname(HERE))   # D:/work/_local_ai


def _token_candidates():
    """按可信度排序的候选 token。
    UUID 格式(36位)才是真正的 AccessToken; credentials/session 里的 32 位 hex 已失效。
    """
    out = []
    t = (os.environ.get("MODELSCOPE_ACCESS_TOKEN") or "").strip()
    if t:
        out.append(("env", t))
    # 历史可用 token(浏览器登录时抓下来的 AccessToken)
    p = os.path.join(ROOT_PARENT, "deploy_aliyun", "ms_token.txt")
    if os.path.isfile(p):
        v = open(p, encoding="utf-8", errors="replace").read().strip()
        if v:
            out.append((p, v))
    d = os.path.join(os.path.expanduser("~"), ".modelscope", "credentials")
    for n in ("token", "access_token", "session"):
        q = os.path.join(d, n)
        if os.path.isfile(q):
            v = open(q, encoding="utf-8", errors="replace").read().strip()
            if v:
                out.append((q, v))
    return out


def token():
    c = _token_candidates()
    if not c:
        sys.exit("[err] 没有任何魔搭 token 候选")
    return c[0][1]


def api():
    """逐个候选试登录, 第一个成功的就用它(失败的 token 很常见, 不要一上来就死)。"""
    from modelscope.hub.api import HubApi
    cands = _token_candidates()
    if not cands:
        sys.exit("[err] 没有任何魔搭 token 候选")
    last = None
    for src, tk in cands:
        a = HubApi()
        try:
            a.login(tk)
            say("login ok <- %s (len=%d)" % (src, len(tk)))
            return a, tk
        except Exception as e:
            last = e
            say("login fail <- %s: %s" % (src, repr(e)[:90]))
    sys.exit("[err] 全部 token 登录失败: %s" % repr(last)[:200])


def human(n):
    for u in ("B", "KB", "MB", "GB"):
        if n < 1024 or u == "GB":
            return "%.1f%s" % (n, u)
        n /= 1024.0


def ensure_dataset(a, repo):
    if "/" not in repo:
        sys.exit("[err] repo 需要 owner/name 形式: %s" % repo)
    ns, name = repo.split("/", 1)
    try:
        a.create_dataset(dataset_name=name, namespace=ns, visibility=1,
                         license="Apache License 2.0",
                         description="polaris relay: 本机<->node1 大文件中转(私有)")
        say("create_dataset ok: %s" % repo)
    except Exception as e:
        say("create_dataset skip: %s" % repr(e)[:140])


def say(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def cmd_ls(a, repo, is_model):
    if is_model:
        files = a.get_model_files(repo, recursive=True)
    else:
        files = a.get_dataset_files(repo, recursive=True)
    if not files:
        say("(空仓库)")
        return
    for f in files:
        if isinstance(f, dict):
            p = f.get("Path") or f.get("path") or f.get("Name")
            s = f.get("Size") or f.get("size") or 0
            print("  %-60s %s" % (p, human(s) if s else ""))
        else:
            print("  %s" % f)


def cmd_push(a, repo, src, dest, is_model):
    if not os.path.isfile(src):
        sys.exit("[err] 不是文件: %s" % src)
    dest = dest or os.path.join("inbox", os.path.basename(src))
    size = os.path.getsize(src)
    say("push %s -> %s/%s (%s)" % (src, repo, dest, human(size)))
    t0 = time.time()
    kw = {} if is_model else {"repo_type": "dataset"}
    a.upload_file(repo_id=repo, path_or_fileobj=src, path_in_repo=dest, **kw)
    dt = time.time() - t0
    say("done %.1fs  %.2f MB/s" % (dt, size / 1048576.0 / max(dt, 1e-6)))


def cmd_push_dir(a, repo, d, destprefix, is_model):
    if not os.path.isdir(d):
        sys.exit("[err] 不是目录: %s" % d)
    total, t0 = 0, time.time()
    for root, _, fs in os.walk(d):
        for f in fs:
            p = os.path.join(root, f)
            rel = os.path.relpath(p, d).replace("\\", "/")
            tgt = "%s/%s" % (destprefix.rstrip("/"), rel) if destprefix else rel
            sz = os.path.getsize(p)
            total += sz
            say("  %-50s %s" % (tgt, human(sz)))
            kw = {} if is_model else {"repo_type": "dataset"}
            a.upload_file(repo_id=repo, path_or_fileobj=p, path_in_repo=tgt, **kw)
    dt = time.time() - t0
    say("dir done %.1fs  %s  %.2f MB/s" % (dt, human(total), total / 1048576.0 / max(dt, 1e-6)))


def cmd_pull(repo, remote, out, is_model, tk):
    from modelscope.hub.file_download import dataset_file_download, model_file_download
    out = out or os.path.join(os.getcwd(), "relay_dl")
    os.makedirs(out, exist_ok=True)
    say("pull %s/%s -> %s" % (repo, remote, out))
    t0 = time.time()
    fn = model_file_download if is_model else dataset_file_download
    p = fn(repo, file_path=remote, local_dir=out, token=tk)
    dt = time.time() - t0
    size = os.path.getsize(p)
    say("done %.1fs  %s  %.2f MB/s" % (dt, human(size), size / 1048576.0 / max(dt, 1e-6)))
    print(p)


def cmd_rm(hub, repo, remote, is_model):
    kw = {} if is_model else {"repo_type": "dataset"}
    hub.delete_files(repo_id=repo, file_paths=[remote], **kw)
    say("deleted %s/%s" % (repo, remote))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ls", "push", "pull", "push-dir", "rm"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--as", dest="as_", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--repo", default=None)
    ap.add_argument("--model", action="store_true", help="用模型仓库")
    a = ap.parse_args()

    is_model = a.model
    repo = a.repo or (DEFAULT_MODEL if is_model else DEFAULT_DS)

    if a.cmd == "pull":
        hub, tk = api()
        cmd_pull(repo, a.arg, a.out, is_model, tk)
        return

    hub, tk = api()
    if not is_model:
        ensure_dataset(hub, repo)
    if a.cmd == "ls":
        cmd_ls(hub, repo, is_model)
    elif a.cmd == "push":
        cmd_push(hub, repo, a.arg, a.as_, is_model)
    elif a.cmd == "push-dir":
        cmd_push_dir(hub, repo, a.arg, a.as_ or os.path.basename(os.path.abspath(a.arg)), is_model)
    elif a.cmd == "rm":
        cmd_rm(hub, repo, a.arg, is_model)


if __name__ == "__main__":
    main()
