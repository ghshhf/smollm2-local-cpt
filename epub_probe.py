import zipfile, re, os

epub = "D:/tgapp/telegram_mini_app/data/uploads/1007_220548_琼明神女录(完结+番外).epub"
z = zipfile.ZipFile(epub)
names = z.namelist()
print("压缩包内文件总数:", len(names))

# 文本型文件
texts = [n for n in names if n.lower().endswith(('.xhtml', '.html', '.htm', '.xml'))]
print("文本文件数:", len(texts))

def strip_tags(s):
    s = re.sub(r'<style[\s\S]*?</style>', ' ', s, flags=re.I)
    s = re.sub(r'<script[\s\S]*?</script>', ' ', s, flags=re.I)
    s = re.sub(r'<[^>]+>', '', s)
    for a, b in [('&nbsp;', ' '), ('&amp;', '&'), ('&lt;', '<'), ('&gt;', '>'), ('&quot;', '"')]:
        s = s.replace(a, b)
    s = re.sub(r'[ \t]+', ' ', s)
    s = re.sub(r'\n\s*\n+', '\n', s)
    return s

parts = []
for n in sorted(texts):
    try:
        data = z.read(n).decode('utf-8', errors='ignore')
    except Exception:
        continue
    t = strip_tags(data)
    if t.strip():
        parts.append(t)
joined = '\n'.join(parts)
joined = re.sub(r'\n{3,}', '\n\n', joined).strip()
print("提取纯文本总字符数:", len(joined))

# 章节粗切
chapters = re.split(r'(?=第\s*[一二三四五六七八九十百零0-9]+\s*章)', joined)
chapters = [c.strip() for c in chapters if len(c.strip()) > 300]
print("章节数(粗估, >300字):", len(chapters))
for i, c in enumerate(chapters[:3]):
    print(f"--- 章{i} 字数{len(c)} ---")
    print(c[:280].replace('\n', ' '))
print("...")
for i, c in enumerate(chapters[-2:]):
    print(f"--- 末章{len(chapters)-2+i} 字数{len(c)} ---")
    print(c[:200].replace('\n', ' '))

out = "D:/work/_local_ai/train_15b/shennulv_raw.txt"
open(out, 'w', encoding='utf-8').write(joined)
print("原始文本已存:", out, "字数:", len(joined))
