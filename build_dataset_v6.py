# -*- coding: utf-8 -*-
"""
V6 纯代码蒸馏数据集构建器（精细化版）
来源：
  A. tgapp 单文件游戏 HTML 的 <script> 内 JS 函数（真实复杂前端）
  B. D:/work/_scripts 量化/链上脚本（真实自包含工程）
  C. D:/code 自有项目核心（排除 node_modules/dist/vendor 噪声）
  D. distill_code_dataset.jsonl 精华（过滤掉 <120 字符的玩具）
  E. 手写高难度工程范式（覆盖 AST 难抽到的通用模式）
输出：data/role_data_v6.jsonl（纯代码，去重）
"""
import os, re, json, ast

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data")
os.makedirs(DATA, exist_ok=True)

EXCLUDE_DIRS = {"node_modules", "dist", ".next", "vendor", ".git", "__pycache__", "build", "out"}
MAX_FILE = 2_000_000          # 单个源码文件上限(字节)
MIN_LINES, MAX_LINES = 6, 100  # 函数行数窗口
MIN_OUT_CHARS = 120            # 输出代码最小字符(过滤玩具)

rows = []

# ----------------------------------------------------------------------
# 工具：从函数名/ docstring 生成自然语言描述
# ----------------------------------------------------------------------
def to_words(name):
    if not name:
        return "该功能"
    s = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', name)
    s = re.sub(r'[_\-.]', ' ', s)
    return s.strip() or name

def describe(lang, name, doc):
    if doc:
        d = doc.strip().split("\n")[0].strip()
        if len(d) > 4:
            return f"用 {lang} 实现：{d}"
    return f"用 {lang} 实现函数 `{name}`（功能：{to_words(name)}）"

# ----------------------------------------------------------------------
# A/B/C. Python 函数级抽取（AST）
# ----------------------------------------------------------------------
def extract_py(path):
    try:
        src = open(path, encoding="utf-8", errors="ignore").read()
    except Exception:
        return
    try:
        tree = ast.parse(src)
    except Exception:
        return
    lines = src.splitlines()
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not hasattr(node, "end_lineno") or node.end_lineno is None:
                continue
            a, b = node.lineno - 1, node.end_lineno
            body = "\n".join(lines[a:b])
            nlines = b - a
            if not (MIN_LINES <= nlines <= MAX_LINES):
                continue
            if len(body) < MIN_OUT_CHARS:
                continue
            doc = ast.get_docstring(node)
            # 过滤明显玩具（body 里只有 print('hi') 之类）
            if re.search(r"print\(\s*['\"](hi|hello|test|ok)['\"]\s*\)", body, re.I) and nlines <= 8:
                continue
            instr = describe("Python", node.name, doc)
            rows.append({"instruction": instr, "output": f"```python\n{body}\n```",
                         "_type": "code_fn", "_universe": "代码", "_src": os.path.basename(path)})

# ----------------------------------------------------------------------
# A. JS / HTML 函数级抽取（正则 + 括号配平）
# ----------------------------------------------------------------------
JS_FN_RE = re.compile(
    r"(?:async\s+)?function\s+([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*\{"
    r"|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?function\s*\(([^)]*)\)\s*\{"
    r"|(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*=\s*(?:async\s*)?\(([^)]*)\)\s*=>\s*\{"
    r"|^[ \t]*(?:async\s+)?([A-Za-z_$][\w$]*)\s*\(([^)]*)\)\s*\{",  # 类方法
    re.M)

def extract_js(src, fname, game=False):
    out_rows = []
    for m in JS_FN_RE.finditer(src):
        name = m.group(1) or m.group(3) or m.group(5) or m.group(7)
        if not name or name in ("if", "for", "while", "switch", "catch", "function"):
            continue
        start = m.end() - 1  # 指向 {
        # 配平括号
        depth = 0
        i = start
        n = len(src)
        while i < n:
            c = src[i]
            if c == "{": depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            elif c == "'" or c == '"' or c == "`":
                # 跳过字符串
                q = c
                i += 1
                while i < n:
                    if src[i] == "\\":
                        i += 2; continue
                    if src[i] == q:
                        i += 1; break
                    i += 1
                continue
            i += 1
        if depth != 0:
            continue
        body = src[start+1:i]
        nlines = body.count("\n") + 1
        if not (MIN_LINES <= nlines <= MAX_LINES):
            continue
        if len(body) < MIN_OUT_CHARS:
            continue
        # 游戏片段依赖全局，措辞需诚实（理解片段而非"实现完整函数"）
        instr = ("理解并重现以下 JavaScript 代码片段（来自游戏模块）的实现逻辑："
                 if game else describe("JavaScript", name, None))
        out_rows.append({"instruction": instr, "output": f"```javascript\n{body}\n```",
                         "_type": "code_fn", "_universe": "代码", "_src": fname})
    return out_rows

def extract_html_js(path):
    res = []
    try:
        src = open(path, encoding="utf-8", errors="ignore").read()
    except Exception:
        return res
    # 提取所有 <script>...</script>
    for m in re.finditer(r"<script[^>]*>(.*?)</script>", src, re.S):
        res.extend(extract_js(m.group(1), os.path.basename(path), game=True))
    return res

def iter_code_files(roots, exts):
    for root in roots:
        if not os.path.isdir(root):
            continue
        for dp, dn, fn in os.walk(root):
            dn[:] = [d for d in dn if d not in EXCLUDE_DIRS]
            for f in fn:
                if any(f.endswith(e) for e in exts):
                    p = os.path.join(dp, f)
                    if os.path.getsize(p) > MAX_FILE:
                        continue
                    yield p

# ----------------------------------------------------------------------
# D. distill_code_dataset 精华过滤
# ----------------------------------------------------------------------
def load_distill_smart(path):
    if not os.path.isfile(path):
        return
    toy = re.compile(r"print\(\s*['\"](hi|hello|test|ok|world)['\"]\s*\)", re.I)
    cnt = 0
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            out = r.get("output", "")
            instr = r.get("instruction", "")
            # 只保留真正代码(有代码围栏), 排除 text/纯知识问答污染
            if "```" not in out:
                continue
            if re.search(r"```\s*(text|markdown|md|txt)\b", out, re.I):
                continue
            if len(out) < MIN_OUT_CHARS or len(out) > 1800:
                continue
            if toy.search(out) and len(out) < 200:
                continue
            if re.search(r"hello\s+world", instr + out, re.I):
                continue
            rows.append({"instruction": instr, "output": out,
                         "_type": "code_distill", "_universe": "代码", "_src": "distill_code_dataset"})
            cnt += 1
    print(f"[D] distill 精华保留 {cnt} 条")

# ----------------------------------------------------------------------
# E. 手写高难度工程范式（覆盖 AST 难抽到的通用模式）
# ----------------------------------------------------------------------
def handwritten():
    S = []
    # 1. asyncio 并发限流 + 指数退避重试爬虫
    S.append(("用 Python(asyncio) 写一个并发爬虫：对一批 URL 用信号量限制最大并发数，"
              "每个请求带超时与指数退避重试（最多 3 次），失败时记录并返回结果列表。",
              "```python\n"
              "import asyncio, aiohttp\n\n"
              "async def fetch(session, url, sem, retries=3):\n"
              "    async with sem:\n"
              "        for attempt in range(retries):\n"
              "            try:\n"
              "                async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:\n"
              "                    resp.raise_for_status()\n"
              "                    return url, await resp.text()\n"
              "            except Exception as e:\n"
              "                if attempt == retries - 1:\n"
              "                    return url, f'ERROR: {e}'\n"
              "                await asyncio.sleep(2 ** attempt)  # 指数退避\n\n"
              "async def crawl(urls, concurrency=5):\n"
              "    sem = asyncio.Semaphore(concurrency)\n"
              "    async with aiohttp.ClientSession() as session:\n"
              "        tasks = [fetch(session, u, sem) for u in urls]\n"
              "        return await asyncio.gather(*tasks)\n\n"
              "# 用法: results = asyncio.run(crawl(url_list))\n"
              "```"))
    # 2. LRU + TTL 缓存装饰器
    S.append(("用 Python 写一个带容量上限和 TTL（过期时间）的 LRU 缓存装饰器，"
              "支持同步函数，过期或超容量时自动淘汰最久未用项。",
              "```python\n"
              "import time, functools\n\n"
              "def lru_ttl_cache(maxsize=128, ttl=60):\n"
              "    def deco(func):\n"
              "        cache = {}\n"
              "        order = []\n"
              "        @functools.wraps(func)\n"
              "        def wrapper(*args, **kw):\n"
              "            key = (args, tuple(sorted(kw.items())))\n"
              "            now = time.time()\n"
              "            if key in cache:\n"
              "                val, ts = cache[key]\n"
              "                if now - ts < ttl:\n"
              "                    order.remove(key); order.append(key)\n"
              "                    return val\n"
              "                del cache[key]\n"
              "            if len(order) >= maxsize:\n"
              "                old = order.pop(0)\n"
              "                cache.pop(old, None)\n"
              "            val = func(*args, **kw)\n"
              "            cache[key] = (val, now); order.append(key)\n"
              "            return val\n"
              "        return wrapper\n"
              "    return deco\n"
              "```"))
    # 3. 线段树：区间求和 + 单点更新
    S.append(("用 Python 实现一个线段树，支持单点更新和区间求和查询，"
              "构造复杂度 O(n)，更新/查询 O(log n)。",
              "```python\n"
              "class SegTree:\n"
              "    def __init__(self, nums):\n"
              "        self.n = len(nums)\n"
              "        self.t = [0] * (4 * self.n)\n"
              "        self._build(nums, 1, 0, self.n - 1)\n"
              "    def _build(self, a, node, l, r):\n"
              "        if l == r:\n"
              "            self.t[node] = a[l]; return\n"
              "        m = (l + r) // 2\n"
              "        self._build(a, node*2, l, m)\n"
              "        self._build(a, node*2+1, m+1, r)\n"
              "        self.t[node] = self.t[node*2] + self.t[node*2+1]\n"
              "    def update(self, i, val):\n"
              "        self._upd(1, 0, self.n-1, i, val)\n"
              "    def _upd(self, node, l, r, i, val):\n"
              "        if l == r:\n"
              "            self.t[node] = val; return\n"
              "        m = (l + r) // 2\n"
              "        if i <= m: self._upd(node*2, l, m, i, val)\n"
              "        else: self._upd(node*2+1, m+1, r, i, val)\n"
              "        self.t[node] = self.t[node*2] + self.t[node*2+1]\n"
              "    def query(self, ql, qr):\n"
              "        return self._q(1, 0, self.n-1, ql, qr)\n"
              "    def _q(self, node, l, r, ql, qr):\n"
              "        if ql <= l and r <= qr: return self.t[node]\n"
              "        if qr < l or r < ql: return 0\n"
              "        m = (l + r) // 2\n"
              "        return self._q(node*2, l, m, ql, qr) + self._q(node*2+1, m+1, r, ql, qr)\n"
              "```"))
    # 4. 带退避重试的同步 HTTP 客户端
    S.append(("用 Python 写一个健壮的 HTTP GET 客户端：带超时、最多 3 次指数退避重试、"
              "5xx/网络异常才重试，4xx 直接抛出，并返回解析后的 JSON。",
              "```python\n"
              "import time, requests\n\n"
              "def get_json(url, timeout=10, retries=3):\n"
              "    last = None\n"
              "    for attempt in range(retries):\n"
              "        try:\n"
              "            r = requests.get(url, timeout=timeout)\n"
              "            if 400 <= r.status_code < 500:\n"
              "                raise ValueError(f'client error {r.status_code}')\n"
              "            r.raise_for_status()\n"
              "            return r.json()\n"
              "        except (requests.RequestException, ValueError) as e:\n"
              "            last = e\n"
              "            if isinstance(e, ValueError):\n"
              "                raise\n"
              "            if attempt < retries - 1:\n"
              "                time.sleep(2 ** attempt)\n"
              "    raise last\n"
              "```"))
    # 5. 滑动窗口最大值
    S.append(("用 Python 实现滑动窗口最大值（单调队列），给定数组和窗口大小 k，"
              "返回每个窗口的最大值，时间复杂度 O(n)。",
              "```python\n"
              "from collections import deque\n\n"
              "def sliding_max(nums, k):\n"
              "    dq = deque()  # 存下标，队首为当前窗口最大\n"
              "    res = []\n"
              "    for i, x in enumerate(nums):\n"
              "        while dq and nums[dq[-1]] <= x:\n"
              "            dq.pop()\n"
              "        dq.append(i)\n"
              "        if dq[0] <= i - k:\n"
              "            dq.popleft()\n"
              "        if i >= k - 1:\n"
              "            res.append(nums[dq[0]])\n"
              "    return res\n"
              "```"))
    # 6. 并查集（路径压缩 + 按秩合并）
    S.append(("用 Python 实现一个并查集（Union-Find），支持合并集合与查询是否连通，"
              "带路径压缩和按秩合并，近 O(1) 均摊。",
              "```python\n"
              "class DSU:\n"
              "    def __init__(self, n):\n"
              "        self.p = list(range(n)); self.r = [0] * n\n"
              "    def find(self, x):\n"
              "        while self.p[x] != x:\n"
              "            self.p[x] = self.p[self.p[x]]  # 路径压缩\n"
              "            x = self.p[x]\n"
              "        return x\n"
              "    def union(self, a, b):\n"
              "        ra, rb = self.find(a), self.find(b)\n"
              "        if ra == rb: return False\n"
              "        if self.r[ra] < self.r[rb]: ra, rb = rb, ra\n"
              "        self.p[rb] = ra\n"
              "        if self.r[ra] == self.r[rb]: self.r[ra] += 1\n"
              "        return True\n"
              "    def connected(self, a, b):\n"
              "        return self.find(a) == self.find(b)\n"
              "```"))
    # 7. 批量流式处理大 JSON 行文件
    S.append(("用 Python 逐行流式读取一个很大的 .jsonl 文件（避免一次性载入内存），"
              "对每条记录做转换后写入输出文件，并统计处理条数。",
              "```python\n"
              "import json\n\n"
              "def transform_jsonl(in_path, out_path):\n"
              "    count = 0\n"
              "    with open(in_path, encoding='utf-8') as fin, \\\n"
              "         open(out_path, 'w', encoding='utf-8') as fout:\n"
              "        for line in fin:\n"
              "            line = line.strip()\n"
              "            if not line:\n"
              "                continue\n"
              "            rec = json.loads(line)\n"
              "            rec['processed'] = True  # 业务转换\n"
              "            fout.write(json.dumps(rec, ensure_ascii=False) + '\\n')\n"
              "            count += 1\n"
              "    return count\n"
              "```"))
    # 8. 用二分答案求「满足某条件的最大值」
    S.append(("用 Python 写「二分答案」模板：在 [lo, hi] 整数区间上，"
              "找到满足 predicate 的最大 x（假设 predicate 关于 x 单调）。",
              "```python\n"
              "def binary_search_max(lo, hi, predicate):\n"
              "    ans = lo\n"
              "    while lo <= hi:\n"
              "        mid = (lo + hi) // 2\n"
              "        if predicate(mid):\n"
              "            ans = mid\n"
              "            lo = mid + 1  # 尝试更大\n"
              "        else:\n"
              "            hi = mid - 1\n"
              "    return ans\n"
              "```"))
    # 9. JS: 防抖 + 节流
    S.append(("用 JavaScript 实现 debounce（防抖）和 throttle（节流）两个高阶函数，"
              "用于优化高频触发事件（如 resize、input）。",
              "```javascript\n"
              "function debounce(fn, wait) {\n"
              "  let t;\n"
              "  return function (...args) {\n"
              "    clearTimeout(t);\n"
              "    t = setTimeout(() => fn.apply(this, args), wait);\n"
              "  };\n"
              "}\n\n"
              "function throttle(fn, wait) {\n"
              "  let last = 0, timer;\n"
              "  return function (...args) {\n"
              "    const now = Date.now();\n"
              "    const remain = wait - (now - last);\n"
              "    if (remain <= 0) {\n"
              "      clearTimeout(timer); timer = null;\n"
              "      last = now; fn.apply(this, args);\n"
              "    } else if (!timer) {\n"
              "      timer = setTimeout(() => { last = Date.now(); timer = null; fn.apply(this, args); }, remain);\n"
              "    }\n"
              "  };\n"
              "}\n"
              "```"))
    # 10. JS: Canvas 粒子系统（带重力）
    S.append(("用 JavaScript + Canvas 实现一个简单粒子系统：粒子带位置、速度、重力，"
              "每帧更新并绘制，支持从某点发射一批粒子。",
              "```javascript\n"
              "class Particle {\n"
              "  constructor(x, y) {\n"
              "    this.x = x; this.y = y;\n"
              "    this.vx = (Math.random() - 0.5) * 4;\n"
              "    this.vy = -Math.random() * 4 - 1;\n"
              "    this.life = 1;\n"
              "  }\n"
              "  update() {\n"
              "    this.vy += 0.15;        // 重力\n"
              "    this.x += this.vx; this.y += this.vy;\n"
              "    this.life -= 0.02;\n"
              "  }\n"
              "  draw(ctx) {\n"
              "    ctx.globalAlpha = Math.max(this.life, 0);\n"
              "    ctx.fillRect(this.x, this.y, 3, 3);\n"
              "  }\n"
              "}\n"
              "```"))
    # 11. JS: Promise 并发限制器
    S.append(("用 JavaScript 写一个并发限制器：给定一组返回 Promise 的异步任务，"
              "限制同时执行的任务数不超过 n，全部完成后按原顺序返回结果。",
              "```javascript\n"
              "async function mapLimit(tasks, limit) {\n"
              "  const results = new Array(tasks.length);\n"
              "  let i = 0;\n"
              "  async function worker() {\n"
              "    while (i < tasks.length) {\n"
              "      const idx = i++;\n"
              "      results[idx] = await tasks[idx]();\n"
              "    }\n"
              "  }\n"
              "  const workers = Array.from({ length: Math.min(limit, tasks.length) }, worker);\n"
              "  await Promise.all(workers);\n"
              "  return results;\n"
              "}\n"
              "```"))
    # 12. Shell: 批量重命名 + 日志轮转
    S.append(("写一个 Bash 脚本：把当前目录下所有 .txt 文件按 'doc_0001.txt' 顺序重命名，"
              "并把大于 10MB 的日志按日期轮转压缩。要求幂等、不覆盖已存在文件。",
              "```bash\n"
              "#!/usr/bin/env bash\n"
              "shopt -s nullglob\n"
              "i=1\n"
              "for f in *.txt; do\n"
              "  ext='${f##*.}'\n"
              "  printf -v new 'doc_%04d.%s' \"$i\" \"$ext\"\n"
              "  if [[ ! -e \"$new\" ]]; then mv -- \"$f\" \"$new\"; fi\n"
              "  ((i++))\n"
              "done\n"
              "# 日志轮转\n"
              "find . -name '*.log' -size +10M -exec sh -c 'gzip \"{}.$(date +%Y%m%d)\"' \\;\n"
              "```"))
    # 13. Python: 用 dataclass + 校验装饰器建模配置
    S.append(("用 Python dataclass 定义一个带类型校验的配置对象，"
              "字段缺失或类型错误时抛出清晰异常；提供 from_dict 类方法。",
              "```python\n"
              "from dataclasses import dataclass, fields\n\n"
              "@dataclass\n"
              "class Config:\n"
              "    host: str\n"
              "    port: int\n"
              "    debug: bool = False\n"
              "    @classmethod\n"
              "    def from_dict(cls, d):\n"
              "        for f in fields(cls):\n"
              "            if f.name not in d:\n"
              "                raise KeyError(f'missing field: {f.name}')\n"
              "        return cls(**{f.name: d[f.name] for f in fields(cls)})\n"
              "```"))
    # 14. Python: 参数化 pytest 测试
    S.append(("用 Python + pytest 写一个参数化单元测试，验证一个 add 函数对多组输入（含负数、零、大数）都正确。",
              "```python\n"
              "import pytest\n\n"
              "def add(a, b):\n"
              "    return a + b\n\n"
              "@pytest.mark.parametrize('a,b,exp', [\n"
              "    (1, 2, 3), (-1, 1, 0), (0, 0, 0), (10**9, 1, 10**9 + 1),\n"
              "])\n"
              "def test_add(a, b, exp):\n"
              "    assert add(a, b) == exp\n"
              "```"))
    # 15. JS: 用 fetch 包装带超时和 JSON 的 API 调用
    S.append(("用 JavaScript 封装一个 fetch 调用：自动加超时（AbortController）、"
              "解析 JSON、对 4xx/5xx 抛出带状态码的错误。",
              "```javascript\n"
              "async function api(url, opts = {}, timeout = 10000) {\n"
              "  const ctrl = new AbortController();\n"
              "  const t = setTimeout(() => ctrl.abort(), timeout);\n"
              "  try {\n"
              "    const res = await fetch(url, { ...opts, signal: ctrl.signal });\n"
              "    if (!res.ok) throw new Error('HTTP ' + res.status);\n"
              "    return await res.json();\n"
              "  } finally {\n"
              "    clearTimeout(t);\n"
              "  }\n"
              "}\n"
              "```"))
    # 16. Python: 单调栈求下一个更大元素
    S.append(("用 Python 单调栈求数组中每个元素「右边第一个比它大的元素下标」，"
              "没有则记为 -1，时间复杂度 O(n)。",
              "```python\n"
              "def next_greater(nums):\n"
              "    n = len(nums)\n"
              "    res = [-1] * n\n"
              "    st = []  # 存下标，栈底到栈顶递减\n"
              "    for i, x in enumerate(nums):\n"
              "        while st and nums[st[-1]] < x:\n"
              "            res[st.pop()] = i\n"
              "        st.append(i)\n"
              "    return res\n"
              "```"))
    # 17. Python: 二叉堆实现优先队列
    S.append(("用 Python 实现一个最小堆优先队列：支持 push、pop（取最小值）、"
              "peek，基于数组下标父子关系，上浮下沉维护堆性质。",
              "```python\n"
              "class MinHeap:\n"
              "    def __init__(self): self.a = []\n"
              "    def push(self, x):\n"
              "        self.a.append(x); self._up(len(self.a) - 1)\n"
              "    def pop(self):\n"
              "        if not self.a: raise IndexError('empty')\n"
              "        top = self.a[0]; last = self.a.pop()\n"
              "        if self.a: self.a[0] = last; self._down(0)\n"
              "        return top\n"
              "    def _up(self, i):\n"
              "        while i > 0:\n"
              "            p = (i - 1) // 2\n"
              "            if self.a[p] <= self.a[i]: break\n"
              "            self.a[p], self.a[i] = self.a[i], self.a[p]; i = p\n"
              "    def _down(self, i):\n"
              "        n = len(self.a)\n"
              "        while True:\n"
              "            l, r = 2*i+1, 2*i+2; s = i\n"
              "            if l < n and self.a[l] < self.a[s]: s = l\n"
              "            if r < n and self.a[r] < self.a[s]: s = r\n"
              "            if s == i: break\n"
              "            self.a[s], self.a[i] = self.a[i], self.a[s]; i = s\n"
              "```"))
    # 18. JS: 深拷贝（处理循环引用 + 多种类型）
    S.append(("用 JavaScript 实现一个深拷贝 deepClone，正确处理嵌套对象、数组、"
              "Date/Regex、以及循环引用（用 WeakMap 记录已拷贝对象避免栈溢出）。",
              "```javascript\n"
              "function deepClone(val, seen = new WeakMap()) {\n"
              "  if (val === null || typeof val !== 'object') return val;\n"
              "  if (seen.has(val)) return seen.get(val);\n"
              "  if (val instanceof Date) return new Date(val);\n"
              "  if (val instanceof RegExp) return new RegExp(val);\n"
              "  const out = Array.isArray(val) ? [] : {};\n"
              "  seen.set(val, out);\n"
              "  for (const k of Reflect.ownKeys(val)) out[k] = deepClone(val[k], seen);\n"
              "  return out;\n"
              "}\n"
              "```"))
    # 19. Python: 上下文管理器管理资源
    S.append(("用 Python 写一个上下文管理器（contextmanager 装饰器），"
              "在进入时打开文件、退出时确保关闭并打印耗时，异常时也安全清理。",
              "```python\n"
              "import time\n"
              "from contextlib import contextmanager\n\n"
              "@contextmanager\n"
              "def timed_open(path, mode='r'):\n"
              "    t0 = time.time()\n"
              "    f = open(path, mode)\n"
              "    try:\n"
              "        yield f\n"
              "    finally:\n"
              "        f.close()\n"
              "        print(f'closed, cost={time.time()-t0:.3f}s')\n"
              "```"))
    # 20. Python: 正则提取日志中的字段
    S.append(("用 Python 正则从一行访问日志 `127.0.0.1 - - [10/Oct/2026:13:55:01] \"GET /api/x 200\" 3ms` "
              "中提取 IP、时间、方法、路径、状态码、耗时，返回字典。",
              "```python\n"
              "import re\n\n"
              "PAT = re.compile(\n"
              "    r'(?P<ip>\\S+) \\S+ \\S+ \\[(?P<ts>[^\\]]+)\\] \"(?P<method>\\S+) (?P<path>\\S+) (?P<code>\\d+)\" (?P<ms>\\d+)ms'\n"
              ")\n\n"
              "def parse_log(line):\n"
              "    m = PAT.search(line)\n"
              "    return m.groupdict() if m else None\n"
              "```"))
    for instr, out in S:
        rows.append({"instruction": instr, "output": out,
                     "_type": "code_paradigm", "_universe": "代码", "_src": "curated"})

# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------
def main():
    # A. 游戏 HTML（每文件下采样: 优先带注释, 上限 20, 避免单个巨文件过载）
    games_dir = "D:/tgapp/telegram_mini_app/games"
    for p in iter_code_files([games_dir], [".html"]):
        recs = extract_html_js(p)
        recs.sort(key=lambda r: (0 if ("//" in r["output"] or "/*" in r["output"]) else 1))
        rows.extend(recs[:20])
    # B. _scripts 量化/链上脚本（整文件函数级）
    scripts_dir = "D:/work/_scripts"
    for p in iter_code_files([scripts_dir], [".py", ".mjs", ".js", ".ts"]):
        if p.endswith(".py"):
            extract_py(p)
        else:
            rows.extend(extract_js(open(p, encoding="utf-8", errors="ignore").read(),
                                  os.path.basename(p), game=False))
    # C. D:/code 自有项目核心（排除 node_modules）
    code_roots = ["D:/code/mx_auto_strategy", "D:/code/audiosync-v2", "D:/code/localsend"]
    for p in iter_code_files(code_roots, [".py", ".js", ".mjs", ".ts"]):
        if p.endswith(".py"):
            extract_py(p)
        else:
            rows.extend(extract_js(open(p, encoding="utf-8", errors="ignore").read(),
                                  os.path.basename(p), game=False))
    # D. distill 精华
    load_distill_smart("D:/tgapp/telegram_mini_app/distill_code_dataset.jsonl")
    # E. 手写范式
    handwritten()

    # 去重（instruction + output）
    seen = set()
    uniq = []
    for r in rows:
        k = (r["instruction"].strip(), r["output"].strip())
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    uniq.sort(key=lambda r: (r["_type"], r["_src"], r["instruction"]))

    out_path = os.path.join(DATA, "role_data_v6.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in uniq:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 统计
    from collections import Counter
    by_type = Counter(r["_type"] for r in uniq)
    by_src = Counter(r["_src"] for r in uniq)
    print(f"[V6] 总条数(去重后): {len(uniq)}")
    print("[V6] 按类型:", dict(by_type))
    print("[V6] 入库路径:", out_path)

if __name__ == "__main__":
    main()
