#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
harvest_workbuddy_chats.py
从 WorkBuddy 本地会话存档中抽取「主人(user)轮次」，作为权重级身份嵌入的注册语料。

数据位置（默认，Windows）：
  C:/Users/admin/.workbuddy/projects/<encoded_cwd>/<session-id>.jsonl
每个 jsonl 文件每行一个 JSON message 对象，含 role/content（content 可能是 str 或 parts 数组）。

身份逻辑（见 docs/identity-memory-roadmap.md）：
  只抽 role == user/human 的轮次 —— 这才是「主人声纹」。
  assistant / system / tool 轮一律跳过（那是模型/系统声音，不构主人身份）。
  负样本(他人/通用语料)由训练侧另取，这里只出正样本。

输出：
  owner_corpus/owner_corpus_chats.jsonl   纯主人对话轮次（每行 {text,kind,src_session,ts}）
  owner_corpus/owner_corpus.jsonl         （可选）与 MEMORY.md 抽取结果合并，统一格式

用法：
  python harvest_workbuddy_chats.py
  python harvest_workbuddy_chats.py --projects-root "D:/other/.workbuddy/projects" --out owner_corpus/owner_corpus_chats.jsonl
"""
import argparse
import json
import os
import re
import sys
from datetime import datetime, timezone

# ---- 调参 ----
MIN_LEN = 6          # 主人轮最小字符数，过滤 "ok"/"好的" 之类过短噪声（可按需下调）
MAX_LEN = 4000       # 单轮上限，超长截断（避免整篇文档灌入）
USER_ROLES = {"user", "human", "owner"}
SKIP_ROLES = {"assistant", "system", "tool", "function", "agent"}

# 清理：去掉 WorkBuddy 的图片引用 @image#N、平台注入的 <system-reminder> 块、前导空白
RE_IMG = re.compile(r"@image#\d+")
RE_WS = re.compile(r"\s+")
# 平台常把 <system-reminder ...>...</system-reminder> 注入 user 轮，这不是主人声纹，必须剥掉
RE_SYSREM = re.compile(r"<system-reminder\b[\s\S]*?</system-reminder>", re.IGNORECASE)
RE_SYSREM_LEAD = re.compile(r"^\s*<system-reminder\b", re.IGNORECASE)
# WorkBuddy 把用户原话包在 <user_query>...</user_query> 里，剥掉只留主人文本
RE_USERQ = re.compile(r"</?user_query\b[^>]*>", re.IGNORECASE)
# 其余平台注入块（不是主人声纹）：后台任务通知 / 会话摘要 / 队友消息 / 上下文注入等
# 注意：user_query 不在这里 —— 它是包裹主人话的壳，只脱壳不丢内容（见 RE_USERQ）
SYS_TAGS = ["system-reminder", "task-notification", "conversation_history_summary",
            "teammate-message", "cb_summary", "additional_data",
            "memory_and_skills_reminder", "craft_mode",
            "working_memory_content", "system_prompt"]
# 先按「成对块」剥除（贪婪匹配到对应闭合标签）
RE_INJECT = re.compile(
    r"<(?:" + "|".join(re.escape(t) for t in SYS_TAGS) + r")\b[\s\S]*?</(?:"
    + "|".join(re.escape(t) for t in SYS_TAGS) + r")>", re.IGNORECASE)
# 兜底：成对剥除后仍可能残留孤立闭合标签（结构不规整时），一并清掉
RE_ORPHAN_CLOSE = re.compile(
    r"</(?:" + "|".join(re.escape(t) for t in SYS_TAGS) + r")\s*>", re.IGNORECASE)

def extract_role(obj):
    if not isinstance(obj, dict):
        return None
    r = obj.get("role") or (obj.get("message") or {}).get("role")
    if r is None and obj.get("type") in USER_ROLES:
        r = obj.get("type")
    return (r or "").lower() if isinstance(r, str) else None

def extract_text(obj):
    """从 message 对象取文本内容（支持 str / parts 数组 / message.content 嵌套）。"""
    content = obj.get("content")
    if content is None:
        content = (obj.get("message") or {}).get("content")
    if content is None:
        content = obj.get("parts")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, str):
                chunks.append(part)
            elif isinstance(part, dict):
                t = part.get("text") or part.get("content") or part.get("text_content")
                if isinstance(t, str):
                    chunks.append(t)
        return "\n".join(c for c in chunks if c)
    # 极少数存成纯字符串字段
    if isinstance(obj.get("text"), str):
        return obj["text"]
    return ""

def extract_ts(obj):
    for k in ("created_at", "timestamp", "time", "ts", "sent_at"):
        v = obj.get(k) or (obj.get("message") or {}).get(k)
        if isinstance(v, (int, float)) and v > 0:
            # 可能是毫秒
            sec = v / 1000.0 if v > 1e11 else v
            try:
                return datetime.fromtimestamp(sec, tz=timezone.utc).isoformat()
            except Exception:
                return None
    return None

def clean(text):
    t = RE_INJECT.sub("", text)   # 剥掉所有平台注入块（成对）
    t = RE_ORPHAN_CLOSE.sub("", t)  # 清掉残留孤立闭合标签
    t = RE_USERQ.sub("", t)       # 剥掉 <user_query> 包裹，只留主人文本
    t = RE_IMG.sub("", t)
    t = RE_WS.sub(" ", t).strip()
    return t

def is_owner_voice(text):
    """剔除剥除后仍为空、或整条只是系统块/工具回显的噪声轮。"""
    if not text or len(text) < MIN_LEN:
        return False
    low = text.lower()
    # 剥除后仍整条以已知系统标签开头 → 非主人声
    # 注意：不拒绝任意 <tag> —— 用户可能真粘贴 HTML/代码，属合法内容
    for tag in SYS_TAGS:
        if low.startswith(f"<{tag}"):
            return False
    return True

def main():
    ap = argparse.ArgumentParser()
    default_root = "C:/Users/admin/.workbuddy/projects"
    ap.add_argument("--projects-root", default=os.environ.get("WB_PROJECTS", default_root))
    ap.add_argument("--out", default="owner_corpus/owner_corpus_chats.jsonl")
    ap.add_argument("--merge-into", default="owner_corpus/owner_corpus.jsonl",
                    help="若指定，把结果按统一格式并入该文件（去重）")
    args = ap.parse_args()

    if not os.path.isdir(args.projects_root):
        print(f"[ERR] projects 目录不存在: {args.projects_root}", file=sys.stderr)
        print("      本会话沙箱可能禁止读取 projects/；请在拥有读权限的环境运行，", file=sys.stderr)
        print("      或设置 WB_PROJECTS / --projects-root 指向可读副本。", file=sys.stderr)
        sys.exit(2)

    out_rows = []
    seen = set()
    n_files = 0
    n_user = 0
    n_kept = 0
    sess_re = re.compile(r"([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\.jsonl$")

    for dirpath, _, files in os.walk(args.projects_root):
        for fn in files:
            if not fn.endswith(".jsonl"):
                continue
            fp = os.path.join(dirpath, fn)
            m = sess_re.search(fn)
            sess = m.group(1) if m else fn
            n_files += 1
            try:
                with open(fp, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            obj = json.loads(line)
                        except Exception:
                            continue
                        # jsonl 可能是 [msg,msg,...] 整段，也可能是逐行 msg
                        objs = obj if isinstance(obj, list) else [obj]
                        for o in objs:
                            role = extract_role(o)
                            if role not in USER_ROLES:
                                continue
                            n_user += 1
                            txt = clean(extract_text(o))
                            if not is_owner_voice(txt) or len(txt) > MAX_LEN:
                                continue
                            key = txt
                            if key in seen:
                                continue
                            seen.add(key)
                            out_rows.append({
                                "text": txt,
                                "kind": "chat_user_turn",
                                "src_session": sess,
                                "ts": extract_ts(o),
                            })
                            n_kept += 1
            except Exception as e:
                print(f"[WARN] 跳过 {fp}: {e}", file=sys.stderr)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[OK] 扫描 jsonl 文件: {n_files}")
    print(f"[OK] 命中 user 轮: {n_user} | 去重后保留: {n_kept}")
    print(f"[OK] 写出: {args.out}")

    if args.merge_into:
        merged = []
        mseen = set()
        if os.path.exists(args.merge_into):
            with open(args.merge_into, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    # 幂等：并入前先丢弃上一次跑出的 chat_user_turn 旧行，避免累积
                    if r.get("kind") == "chat_user_turn":
                        continue
                    t = r.get("text", "")
                    if t in mseen:
                        continue
                    mseen.add(t)
                    merged.append(r)
        for r in out_rows:
            if r["text"] not in mseen:
                mseen.add(r["text"])
                merged.append(r)
        with open(args.merge_into, "w", encoding="utf-8") as f:
            for r in merged:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"[OK] 合并入: {args.merge_into} (总 {len(merged)} 条)")

if __name__ == "__main__":
    main()
