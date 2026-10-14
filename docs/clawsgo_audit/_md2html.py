#!/usr/bin/env python3
# 简易 markdown -> 自包含 HTML 转换器（仅处理本 catalog 用到的语法：#/##/###、**粗体*、> 引用、- 列表、段落）
import re, html, sys

def inline(t: str) -> str:
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)\*(?!\*)", r"<em>\1</em>", t)
    return t

def convert(src: str) -> str:
    lines = src.split("\n")
    out = []
    i = 0
    n = len(lines)
    list_open = False
    quote_buf = []

    def close_list():
        nonlocal list_open
        if list_open:
            out.append("</ul>")
            list_open = False

    def flush_quote():
        nonlocal quote_buf
        if quote_buf:
            out.append("<blockquote>")
            for q in quote_buf:
                if q.strip() == "":
                    out.append("<br>")
                else:
                    out.append(f"<p>{inline(q)}</p>")
            out.append("</blockquote>")
            quote_buf = []

    while i < n:
        line = lines[i]
        if line.startswith("## "):
            close_list(); flush_quote()
            out.append(f"<h2>{inline(line[3:])}</h2>")
        elif line.startswith("### "):
            close_list(); flush_quote()
            out.append(f"<h3>{inline(line[4:])}</h3>")
        elif line.startswith("# "):
            close_list(); flush_quote()
            out.append(f"<h1>{inline(line[2:])}</h1>")
        elif line.startswith("> "):
            close_list()
            quote_buf.append(line[2:])
        elif line.startswith("- "):
            flush_quote()
            if not list_open:
                out.append("<ul>")
                list_open = True
            out.append(f"<li>{inline(line[2:])}</li>")
        elif line.strip() == "":
            close_list(); flush_quote()
        else:
            close_list(); flush_quote()
            out.append(f"<p>{inline(line)}</p>")
        i += 1
    close_list(); flush_quote()
    return "\n".join(out)

src_path = sys.argv[1]
dst_path = sys.argv[2]
with open(src_path, encoding="utf-8") as f:
    body = convert(f.read())

doc = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>未来模型问题全集 · 提交 ClawsGO</title>
<style>
  :root {{ --bg:#ffffff; --fg:#1a1a1a; --muted:#666; --accent:#b8342b; --line:#e2e2e2; --quote:#f7f3ee; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; padding:32px 40px; background:var(--bg); color:var(--fg);
         font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;
         line-height:1.75; font-size:15px; max-width:920px; margin:0 auto; }}
  h1 {{ font-size:26px; border-bottom:3px solid var(--accent); padding-bottom:10px; margin-top:8px; }}
  h2 {{ font-size:20px; color:var(--accent); margin-top:34px; border-left:4px solid var(--accent); padding-left:12px; }}
  h3 {{ font-size:16.5px; margin-top:22px; color:#222; }}
  p {{ margin:10px 0; }}
  ul {{ margin:10px 0; padding-left:22px; }}
  li {{ margin:6px 0; }}
  blockquote {{ background:var(--quote); border-left:4px solid #d9b08c; margin:14px 0; padding:10px 16px;
               border-radius:6px; color:#3a3a3a; }}
  blockquote p {{ margin:6px 0; }}
  strong {{ color:#111; }}
  em {{ color:var(--muted); }}
  hr {{ border:none; border-top:1px solid var(--line); margin:28px 0; }}
</style>
</head>
<body>
{body}
</body>
</html>
"""
with open(dst_path, "w", encoding="utf-8") as f:
    f.write(doc)
print("OK", dst_path, len(doc), "bytes")
