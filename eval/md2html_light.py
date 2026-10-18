#!/usr/bin/env python3
# 轻量 markdown->HTML（仅覆盖本项目文档用到的构造），内联 CSS，自包含。
import sys, re, html

def conv(md: str) -> str:
    out = []
    lines = md.split("\n")
    i = 0
    in_code = False
    code_buf = []
    while i < len(lines):
        line = lines[i]
        if line.strip().startswith("```"):
            if not in_code:
                in_code = True; code_buf = []; i += 1; continue
            else:
                out.append("<pre class='code'>" + html.escape("\n".join(code_buf)) + "</pre>")
                in_code = False; i += 1; continue
        if in_code:
            code_buf.append(line); i += 1; continue
        s = line.rstrip()
        if s.strip() == "":
            i += 1; continue
        if re.match(r"^={3,}\s*$", s):
            out.append("<hr>"); i += 1; continue
        m = re.match(r"^(#{1,6})\s+(.*)$", s)
        if m:
            lvl = len(m.group(1)); txt = inline(m.group(2))
            out.append(f"<h{lvl}>{txt}</h{lvl}>"); i += 1; continue
        # table: line contains | and next line is separator
        if "|" in s and i+1 < len(lines) and re.match(r"^\s*\|?[\s:|-]+\|?\s*$", lines[i+1]):
            # gather table
            rows = []
            while i < len(lines) and "|" in lines[i]:
                cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
                rows.append(cells); i += 1
            # first row header, second is separator
            thead = rows[0]; tbody = rows[2:]
            t = ["<table>","<thead><tr>" + "".join(f"<th>{inline(c)}</th>" for c in thead) + "</tr></thead>","<tbody>"]
            for r in tbody:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append("".join(t)); continue
        if re.match(r"^[-*]\s+", s):
            out.append("<ul>")
            while i < len(lines) and re.match(r"^[-*]\s+", lines[i].rstrip()):
                out.append("<li>" + inline(re.sub(r"^[-*]\s+","",lines[i].rstrip())) + "</li>"); i += 1
            out.append("</ul>"); continue
        # paragraph (merge consecutive non-special)
        para = [s]
        i += 1
        while i < len(lines) and lines[i].rstrip() != "" and not lines[i].lstrip().startswith(("#","|","-","*","```")) and not re.match(r"^={3,}$", lines[i].strip()):
            para.append(lines[i].rstrip()); i += 1
        out.append("<p>" + inline(" ".join(para)) + "</p>")
    return "\n".join(out)

def inline(t: str) -> str:
    t = html.escape(t)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"`([^`]+?)`", r"<code>\1</code>", t)
    return t

if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    md = open(src, encoding="utf-8").read()
    body = conv(md)
    css = """* { box-sizing: border-box; }
    body { font-family: -apple-system, "Segoe UI", "Microsoft YaHei", sans-serif; max-width: 980px; margin: 0 auto; padding: 28px; line-height: 1.7; color: #1d2129; background:#fff; }
    h1 { font-size: 26px; border-bottom: 3px solid #3478f6; padding-bottom: 8px; }
    h2 { font-size: 21px; margin-top: 32px; color: #1f2d3d; border-left: 5px solid #3478f6; padding-left: 10px; }
    h3 { font-size: 17px; color: #2b4a7a; margin-top: 22px; }
    h4 { font-size: 15px; color:#555; }
    table { border-collapse: collapse; width: 100%; margin: 14px 0; font-size: 13px; }
    th, td { border: 1px solid #d9e1ec; padding: 7px 9px; text-align: left; vertical-align: top; }
    th { background: #eef4ff; font-weight: 600; }
    tr:nth-child(even) td { background: #f7faff; }
    code { background: #f0f2f5; padding: 1px 5px; border-radius: 4px; font-family: "Cascadia Code", Consolas, monospace; font-size: 12.5px; }
    pre.code { background: #0f1620; color: #d6e2f0; padding: 14px; border-radius: 8px; overflow-x: auto; font-family: "Cascadia Code", Consolas, monospace; font-size: 12px; line-height: 1.45; white-space: pre; }
    blockquote { border-left: 4px solid #ffc53d; background: #fffbe6; margin: 12px 0; padding: 8px 14px; color:#614700; }
    hr { border: none; border-top: 2px dashed #c8d3e0; margin: 24px 0; }
    ul { padding-left: 22px; }
    li { margin: 3px 0; }
    p { margin: 10px 0; }
    """
    doc = f"<!doctype html><html lang='zh'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>联网自学组件架构草案</title><style>{css}</style></head><body>{body}</body></html>"
    open(dst, "w", encoding="utf-8").write(doc)
    print("wrote", dst, len(doc), "bytes")
