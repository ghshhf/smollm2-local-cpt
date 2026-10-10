# -*- coding: utf-8 -*-
"""V6 纯代码数据集成分统计 -> 自包含 HTML 报告。"""
import json, re, os, html, collections

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "role_data_v6.jsonl")
OUT = os.path.join(ROOT, "v6_report.html")

rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
N = len(rows)

# 类型 / 来源分布
by_type = collections.Counter(r.get("_type", "?") for r in rows)
by_src = collections.Counter(r.get("_src", "?") for r in rows)

# 语言分布(从 output 的代码围栏 ```lang 推断)
lang_re = re.compile(r"```([a-zA-Z0-9_+#-]+)")
langs = collections.Counter()
for r in rows:
    m = lang_re.search(r.get("output", ""))
    langs[m.group(1).lower() if m else "none"] += 1

# 输出长度分布
outs = [len(r.get("output", "")) for r in rows]
outs.sort()
def pct(p):
    return outs[min(len(outs) - 1, int(len(outs) * p))]

# 每类抽 2 条样例
samples = {}
for r in rows:
    t = r.get("_type", "?")
    samples.setdefault(t, [])
    if len(samples[t]) < 2:
        samples[t].append(r.get("instruction", "")[:110])

def tb(counter, title):
    items = counter.most_common()
    total = sum(counter.values())
    rows_html = "".join(
        f"<tr><td>{html.escape(str(k))}</td><td style='text-align:right'>{v}</td>"
        f"<td style='text-align:right'>{v/total*100:.1f}%</td></tr>"
        for k, v in items)
    return (f"<h3>{title}</h3><table class='t'><thead><tr>"
            f"<th>类别</th><th>条数</th><th>占比</th></tr></thead><tbody>{rows_html}</tbody></table>")

sample_html = ""
for t, lst in samples.items():
    cards = "".join(f"<li>{html.escape(s)}</li>" for s in lst)
    sample_html += f"<h3>类型 <code>{html.escape(t)}</code> 样例</h3><ul class='s'>{cards}</ul>"

doc = f"""<!doctype html><html lang='zh'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'>
<title>V6 纯代码数据集成分报告</title>
<style>
*{{box-sizing:border-box}}
body{{font-family:-apple-system,'Segoe UI',Roboto,'Microsoft YaHei',sans-serif;
  margin:0;background:#0f1115;color:#e6e6e6;padding:32px}}
.wrap{{max-width:920px;margin:0 auto}}
h1{{font-size:24px;margin:0 0 4px}}
.sub{{color:#8b93a1;margin:0 0 24px;font-size:13px}}
.cards{{display:flex;gap:12px;margin:0 0 24px;flex-wrap:wrap}}
.card{{flex:1;min-width:150px;background:#1a1d24;border:1px solid #2a2f3a;border-radius:10px;padding:16px}}
.card .n{{font-size:28px;font-weight:700;color:#5ad1c4}}
.card .l{{color:#8b93a1;font-size:12px;margin-top:4px}}
h3{{font-size:15px;margin:24px 0 8px;color:#cdd3dc}}
table.t{{width:100%;border-collapse:collapse;background:#1a1d24;border-radius:8px;overflow:hidden}}
table.t th,table.t td{{padding:8px 12px;text-align:left;border-bottom:1px solid #2a2f3a;font-size:13px}}
table.t th{{background:#22262f;color:#9aa3b2;font-weight:600}}
table.t tr:last-child td{{border-bottom:none}}
code{{background:#22262f;padding:1px 5px;border-radius:4px;color:#7fd1c4;font-size:12px}}
ul.s{{margin:4px 0;padding-left:18px}}
ul.s li{{font-size:13px;color:#b9c0cc;margin:3px 0}}
.hr{{height:1px;background:#2a2f3a;margin:24px 0}}
</style></head><body><div class='wrap'>
<h1>V6 纯代码数据集 · 成分报告</h1>
<p class='sub'>文件：<code>data/role_data_v6.jsonl</code> ｜ 用途：Qwen2.5-1.5B 纯代码专项 LoRA（从基座重训，不继承小说数据）</p>
<div class='cards'>
  <div class='card'><div class='n'>{N}</div><div class='l'>总条数（纯代码）</div></div>
  <div class='card'><div class='n'>{len(by_type)}</div><div class='l'>抽取类型</div></div>
  <div class='card'><div class='n'>{len(by_src)}</div><div class='l'>来源模块</div></div>
  <div class='card'><div class='n'>{len(langs)}</div><div class='l'>覆盖语言</div></div>
</div>
{tb(by_type, '按抽取类型')}
{tb(by_src, '按来源模块')}
{tb(langs, '按代码语言(围栏推断)')}
<div class='hr'></div>
<h3>输出长度分布</h3>
<table class='t'><tbody>
<tr><td>最短</td><td style='text-align:right'>{outs[0]}</td><td>字符</td></tr>
<tr><td>P25</td><td style='text-align:right'>{pct(.25)}</td><td>字符</td></tr>
<tr><td>中位数</td><td style='text-align:right'>{pct(.5)}</td><td>字符</td></tr>
<tr><td>P75</td><td style='text-align:right'>{pct(.75)}</td><td>字符</td></tr>
<tr><td>最长</td><td style='text-align:right'>{outs[-1]}</td><td>字符</td></tr>
</tbody></table>
<div class='hr'></div>
{sample_html}
<div class='hr'></div>
<p class='sub'>质量把关：① 蒸馏集仅保留含代码围栏(```python/js/...)的输出，剔除 text/知识问答污染；
② 游戏 JS 函数指令改为"理解并重现片段"的诚实措辞，且每 HTML 文件下采样≤20 个；
③ 非游戏函数要求自包含；④ 混入 20 条手写高难度范式(并发限流重试/LRU-TTL/线段树/单调栈/并查集/最小堆/Promise 并发限制器/Canvas 粒子等)。</p>
</div></body></html>"""

with open(OUT, "w", encoding="utf-8") as f:
    f.write(doc)
print("已写出", OUT)
print("类型:", dict(by_type))
print("来源:", dict(by_src))
print("语言:", dict(langs))
