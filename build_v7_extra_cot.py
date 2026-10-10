"""
V7 推理卡 · 扩展真推理字典（agent 亲手写，teacher:self）
=====================================================
与 build_v7_seed_selfteacher.py 的 20 条范式种子互补，
本文件覆盖 V6 里更多"有真实取舍点"的代码任务（CI/SQL/算法/工程化）。

匹配：instruction 包含 key 即挂对应真推理。
产物：data/v7_self_cot_extra.jsonl  (instruction, cot, _src=teacher:self)
之后 build_v7_selftaught.py 会把它 + 范式种子合并成"真推理字典"优先命中。
"""
import json, os, re
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "role_data_v6.jsonl")
OUT = os.path.join(ROOT, "data", "v7_self_cot_extra.jsonl")

# (匹配子串, 真推理链) — 每条都是针对该任务真实代码的取舍分析
EXTRA = [
    ("CI 如何做多版本矩阵测试",
     "思考步骤：\n"
     "1. 矩阵测试的核心是「笛卡尔积展开」：把 python-version / os / 依赖版本声明成多个维度，CI 引擎自动叉乘出组合，每个组合跑一遍同一套 steps。\n"
     "2. 关键陷阱是「脆弱组合拖垮整体」：一个边缘版本（如 3.13 刚出）挂了会红整个矩阵。用 `continue-on-error` 或把实验性维度标 `include: manual` 隔离。\n"
     "3. fail-fast：matrix 默认并行，但某个组合发现致命错误时应快速失败，别等所有组合跑完才暴露。\n"
     "4. 缓存：不同版本/OS 的依赖缓存 key 必须含版本维度，否则会复用错环境的缓存。\n"
     "5. 代价：组合数随维度指数增长，要克制维度数量（通常 2-3 个），否则 CI 分钟数爆掉。"),

    ("JS 如何让函数只执行一次",
     "思考步骤：\n"
     "1. 闭包持有 `done` 标志和缓存结果 `r`；返回的高阶函数每次被调用时先查 `done`。\n"
     "2. 第一次调用才真正执行 `fn(...a)`，把结果存进 `r` 并置 `done=true`；之后所有调用直接 `return r`，fn 不再执行。\n"
     "3. 用 `(...a)` rest 参数透传任意实参，`done` 初始化闭包作用域内，不会被外部篡改。\n"
     "4. 语义是「惰性记忆化 + 单次触发」：适合一次性初始化（建连接、注册、启动）。\n"
     "5. 边界：若 fn 抛异常，done 已置 true 会缓存失败——健壮版应在异常时回滚 done。线程安全不适用（JS 单线程没事）。"),

    ("SQL 分页一般怎么写",
     "思考步骤：\n"
     "1. `LIMIT n OFFSET k` 是浅分页，简单直观，但 OFFSET 越大越慢：数据库要扫描并丢弃前 k 行，深翻页是 O(k+n)。\n"
     "2. 深分页改用「键集分页（seek method）」：`WHERE id > :last_id ORDER BY id LIMIT n`，游标式推进，每次都走索引，复杂度恒定。\n"
     "3. 前提：必须有稳定有序的唯一键（自增 id 或时间戳+id），且排序键可被 WHERE 比较。\n"
     "4. 缺点：不能直接跳到第 100 页（无偏移量概念），更适合「无限滚动/下一页」场景。\n"
     "5. 取舍：浅分页给用户「跳页」体验但深翻慢；键集分页深翻快但只能顺序游标。按业务选。"),

    ("SQL 如何递归查询层级",
     "思考步骤：\n"
     "1. 用 `WITH RECURSIVE` 做递归 CTE：锚点查询取根节点（parent IS NULL），递归部分 JOIN 自身向下展开。\n"
     "2. `depth` 字段在递归里 `+1`，用来限深或展示层级；JOIN 条件 `n.id = t.parent_id` 是向下一步。\n"
     "3. 必须保证有终止：树有穷则自然停；若有环会无限递归，需加 `depth < N` 护栏。\n"
     "4. 上扩（找祖先）只需把 JOIN 方向反过来 `n.parent_id = t.id`。\n"
     "5. 性能：每递归层都要 JOIN，深树慢；数据量大可用物化路径（path 存 '1/3/7'）替代递归。"),

    ("Windows 下如何建定时任务",
     "思考步骤：\n"
     "1. `schtasks /Create` 注册任务；`/SC DAILY /ST 03:00` 定频，`/TR` 指向要跑的脚本/程序。\n"
     "2. 关键 `/RL HIGHEST`：以最高权限（提权）运行——否则非提权进程只能拿到 Users 的受限令牌，写受保护目录会「拒绝访问」且任务**静默失败**（日志里看不到报错，最坑）。\n"
     "3. 写盘/改系统目录的脚本务必加 `/RL HIGHEST`；普通用户态脚本不需要。\n"
     "4. `/RU SYSTEM` 可让任务以系统账户跑（脱离用户会话、无 UAC 限制），适合常驻服务类。\n"
     "5. 排查：任务「看似没跑」先 `schtasks /Query /V` 看 Last Result 码，0=成功，非0=失败需看具体含义。"),

    ("numpy 如何处理 NaN",
     "思考步骤：\n"
     "1. NaN 是「污染性传播」：任何 `x op NaN` 结果都是 NaN，所以均值/求和一旦混入就全毁。\n"
     "2. `np.isnan(a)` 逐元素布尔掩码；`np.nanmean/nanmax/...` 系列在计算前先跳过 NaN 元素，是正确做法。\n"
     "3. 判等陷阱：`np.nan == np.nan` 是 False（NaN 不等于任何值包括自己），所以绝不能用 `==` 检测，必须用 `isnan`。\n"
     "4. 缺失数据建模常用 NaN 占位，但下游统计/绘图要显式处理（dropna / 插值 / nan* 函数）。\n"
     "5. 注意：0 * NaN = NaN，不是 0；涉及掩码运算时顺序会影响结果。"),

    ("pandas 如何新增或修改列",
     "思考步骤：\n"
     "1. `df['new'] = 表达式` 做列间运算，pandas 会向量化（C 层逐元素），远快于 Python 循环 `for`。\n"
     "2. `apply` 是把函数逐行/列调用，**仍带 Python 解释器开销**，只在无法向量化（复杂逻辑）时用。\n"
     "3. 能用 `np.where` / 布尔索引 / 向量化运算就别 apply；性能差可达几十倍。\n"
     "4. 链式赋值（`df[mask]['x']=...`）会触发 SettingWithCopyWarning 且可能不生效，用 `.loc[mask,'x']`。\n"
     "5. 新增列不改变原索引；`assign` 返回新 DataFrame 适合管道式书写。"),

    ("ollama think 封装",
     "思考步骤：\n"
     "1. 推理模型（如 qwen3/thinker）的流式 chunk 里 thinking 和 content 分字段返回，要把「思考过程」和「最终回答」拆成独立字段，方便上层分别展示/记录。\n"
     "2. 工具调用场景下，tool_calls 可能跨多个 chunk 拼接，需累积 message 片段再解析，不能按单 chunk 硬解。\n"
     "3. 用 pydantic 模型（如 Invoice）做结构化抽取：让模型输出 JSON、用 `response_format` 或后处理校验字段类型。\n"
     "4. 流式处理要处理「thinking 阶段无 content」「content 与 tool_calls 交替」两类交错，状态机比 if-else 稳。\n"
     "5. 边界：模型可能不遵守格式；要做兜底解析与校验失败的重试/报错，别把脏 JSON 直接 getattr。"),

    ("BFS 求迷宫最短路",
     "思考步骤：\n"
     "1. BFS 在「边权相等（每步代价 1）」的网格上是求最短路的标准解：队列按层展开，第一次到达目标就是最短步数。\n"
     "2. `deque` 存 `(位置, 步数)`；visited 集合防止回头，避免死循环和重复扩展。\n"
     "3. 四方向（上下左右）扩展，越界或撞墙（grid==1）跳过；起点即目标返回 0。\n"
     "4. 为什么不用 DFS：DFS 不保证最短，且可能深陷。BFS 的「先扩展近层」天然保证首达即最优。\n"
     "5. 边界：目标不可达返回 -1/None；大网格内存用 visited 而非递归。带权图要换 Dijkstra。"),

    ("原子地写文件",
     "思考步骤：\n"
     "1. 直接覆盖写的风险：写到一半进程崩/磁盘满，文件变残缺，下游读到半截数据。\n"
     "2. 原子写 = 先写临时文件（`tempfile.mkstemp`），写成功后 `os.replace(tmp, path)` 一次性改名——rename 在同文件系统上是原子操作，读者永远看到「旧完整文件」或「新完整文件」，不会看到半截。\n"
     "3. `mkstemp(dir=同目录)` 确保 temp 与目标在同一文件系统，否则 replace 退化成 copy（非原子）。\n"
     "4. fdopen 后记得关，finally 里清理可能残留的临时文件。\n"
     "5. 跨盘/网络盘 replace 不保证原子，那种场景要加校验和+双写+标记。"),

    ("实现归并排序",
     "思考步骤：\n"
     "1. 分治：递归把数组对半拆到长度<=1（天然有序），再合并两个有序子数组。\n"
     "2. merge 双指针 i,j 从两子数组头比较，小的先放 out，指针前移；某侧耗尽把另一侧全接上。\n"
     "3. 稳定性：相等时取左侧，保持原相对顺序（归并天然稳定）。\n"
     "4. 复杂度 O(n log n) 确定，不受输入分布影响（优于快排最差 O(n²)）；代价是需要 O(n) 额外空间。\n"
     "5. 边界：空/单元素直接返回；大数组递归深度用迭代归并或提高递归上限；链表归并可 O(1) 空间。"),

    ("断言抛出了异常",
     "思考步骤：\n"
     "1. `pytest.raises(ExpectedError)` 作为上下文管理器包裹应抛错的代码，若没抛则测试失败——验证「错误路径」真的会被触发。\n"
     "2. `match='正则'` 进一步断言错误消息内容，防止「抛了别的异常也蒙混过关」。\n"
     "3. 只捕获预期的**具体**异常类型（ValueError 而非裸 Exception），否则会把真正 bug 的异常也「通过」。\n"
     "4. 用法：`with raises(ValueError, match='坏了'): boom()`，boom 内抛 ValueError 且消息含「坏了」才过。\n"
     "5. 边界：若代码可能抛多种异常，逐条分别 assert；别用 except 吞掉再手动 flag。"),

    ("笛卡尔积",
     "思考步骤：\n"
     "1. `itertools.product(a, b)` 等价于两层嵌套 for，但惰性生成、内存 O(1)、可读性高。\n"
     "2. 多序列笛卡尔积是组合爆炸的来源：k 个长度为 n 的集合，结果 n^k 项，要警惕规模。\n"
     "3. `repeat=k` 做同一集合的 k 次自积（如密码暴力/排列候选）。\n"
     "4. 比手写嵌套 for 好处：层数可变（product(*lists) 解包任意多个）、顺序确定（字典序）。\n"
     "5. 边界：空集合使结果为空；超大积改流式 yield 避免一次性 materialize。"),

    ("正则做分割但保留分隔符",
     "思考步骤：\n"
     "1. `re.split` 默认丢弃分隔符；把分隔模式放进**捕获组** `(...)`，split 会把它也作为列表元素保留。\n"
     "2. 非捕获组 `(?:...)` 则不保留——所以「想不想留分隔符」完全由组是否捕获控制。\n"
     "3. 例 `re.split(r'(\\d+)', 'a1b22c')` → `['a','1','b','22','c']`，数字成对出现可做 token 解析。\n"
     "4. 适合「切词但要记住切点在哪」的场景（如算式/日志分段）。\n"
     "5. 边界：连续分隔符会产生空串；首尾若匹配会在头尾多出空串，按需 filter。"),

    ("命令在挂断后继续跑",
     "思考步骤：\n"
     "1. 终端关掉会向子进程发 SIGHUP，前台/后台任务都被带走——这就是之前训练「静默消失」的真凶。\n"
     "2. `nohup cmd &` 忽略 HUP 信号 + 重定向输出到 nohup.out，进程在终端关后仍活。\n"
     "3. `disown` 把作业从 shell 的作业表移除，shell 退出时不再发 HUP 给它。\n"
     "4. 更彻底：`setsid cmd` 让进程脱离原会话、自建新会话，连进程组都换掉，最稳。\n"
     "5. 边界：nohup 只挡 HUP，不挡 OOM/崩溃；真正要常驻应交给系统服务/计划任务（会话0）而非靠 shell 技巧。"),

    ("拓扑排序",
     "思考步骤：\n"
     "1. Kahn 算法基于「入度」：统计每个节点入度，入度为 0 的才能排在最前（无依赖）。\n"
     "2. 队列初始放所有入度 0 节点；每次弹出一个，把它指向的邻居入度 -1，减到 0 就入队。\n"
     "3. 若最终排出的节点数 < 总节点数 = 存在环（有人依赖你、你也依赖他），返回 None 表示无法拓扑。\n"
     "4. 正确性：只有「所有前驱都排好」的节点才会入度归零，天然满足依赖顺序。\n"
     "5. 边界：入度统计要含隐式边；多个入度 0 时顺序不唯一（任意合法序均可）；DFS 版可检测后向边判环。"),

    ("环境变量缺失",
     "思考步骤：\n"
     "1. 启动即校验（fail-fast）：把必需变量列成清单，缺失就 `sys.exit` 并列出缺哪些，别等跑到一半才 KeyError 崩。\n"
     "2. 好处：错误在启动期暴露，运维一眼看出「是配置没配」，而不是半夜报个看不懂的栈。\n"
     "3. 区分「必需」与「可选」：可选给默认值，必需缺失才致命。\n"
     "4. 错误信息要可读：直接列出 `DATABASE_URL, SECRET_KEY` 而不是笼统「出错了」。\n"
     "5. 进阶：用 pydantic-settings / dataclass 集中管理配置，校验逻辑一处收敛、可测试。"),

    ("functools.partial 与 lambda",
     "思考步骤：\n"
     "1. `partial(f, a)` 生成一个「预填了部分参数的新可调用对象」，比 lambda 更轻、无额外函数帧、且 `__name__`/`__doc__` 可保留（用 update_wrapper）。\n"
     "2. `lambda` 灵活：能重排参数顺序、做运算、改返回值，partial 只能「从左往右填固定值」。\n"
     "3. 性能：partial 在 C 层绑定，调用稍快；lambda 是 Python 函数对象，略慢。\n"
     "4. 线程池/回调场景（如 `executor.submit(partial(send_email, host), to)`）partial 更干净。\n"
     "5. 边界：partial 填的是位置/关键字固定值，不能「跳过中间参数只填后面的」；那种要 lambda 或包装函数。"),

    ("numpy 向量化改写",
     "思考步骤：\n"
     "1. 慢的根因：Python 逐元素 for 循环 + 每次 `a[i]` 都要过 numpy 标量包装（Python 对象开销），两层慢叠加。\n"
     "2. 向量化：把 `out[i]=a[i]*a[i]+2*a[i]` 写成 `out = a*a + 2*a`，整个运算在 C 层对整块内存做，无 Python 循环。\n"
     "3. 加速来源：去 Python 解释器逐元素开销 + 连续内存上的 SIMD 指令。典型几十到上百倍。\n"
     "4. 内存代价：向量化常需临时数组（a*a 先建一个），大数组可能爆内存——此时用 `np.add/out=` 原地写或 numba。\n"
     "5. 何时换 numba/分块：超大数据放不下、或运算无法向量化（含条件分支/递推）时，numba JIT 兼顾速度与内存。"),

    ("阶段 1+2: 防御选股",
     "思考步骤：\n"
     "1. 这是「按市场状态（regime）切换仓位结构」的选股：regime 决定防御/进攻权重，defense_only 标志强制全防御。\n"
     "2. 防御选股挑低波动/抗跌标的，进攻选股挑高弹性；两者用不同 selector 与打分，返回 (chosen_def, chosen_off)。\n"
     "3. 关键约束：严格只用 <=t 的数据（不能用未来信息，否则回测造假），选股信号必须时间对齐。\n"
     "4. 冷却期滤除：近期刚调出的币不立刻回选，避免来回换手吃摩擦成本。\n"
     "5. 边界：regime 判错会错配；要有 fallback（defense_only 兜底），且选股结果要可解释、可回测复盘。"),

    ("跑单窗口回测",
     "思考步骤：\n"
     "1. 单窗口回测 = 给定一段价格 + 配置，跑策略算 multiple（收益倍数）/mdd（最大回撤）/sharpe（风险调整收益）。\n"
     "2. 每次跑前清缓存（如 `_ALT_RS_CACHE.clear()`），防止跨窗口状态污染——回测必须隔离。\n"
     "3. try/except 包住 run_bt：某窗口数据异常（缺失/除零）应返回失败标记而非中断整轮。\n"
     "4. 成本模型：cost_bps/tax_rate 要反映真实摩擦，否则 sharpe 虚高。\n"
     "5. 边界：暖期/样本外要区分；单窗口只是积木，walkforward 才是把多窗口串成时间序列验证。"),

    ("load_or_fetch",
     "思考步骤：\n"
     "1. 缓存模式：先查本地缓存文件是否存在，存在就直接读（省网络/算力），不存在才去 fetch 并落盘。\n"
     "2. 读缓存用 csv.reader，首行是 header，解析出字段与数据；fetch 后同样格式写回，保证读写对称。\n"
     "3. 失效策略：这里用「文件存在即命中」，简单但无 TTL——生产应加 mtime/版本号判断过期。\n"
     "4. 边界：缓存文件损坏/半截（上次写崩）要 try 兜底回退到 fetch，不能死在解析上。\n"
     "5. 并发：多进程同时 load_or_fetch 同一 key 可能重复 fetch，重场景要加文件锁/原子写。"),
]


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    # 建立 instruction -> 行 的索引（用子串匹配）
    matched = []
    used_instr = set()
    for key, cot in EXTRA:
        hit = None
        for r in rows:
            instr = r.get("instruction", "").strip()
            if key in instr and instr not in used_instr:
                hit = r
                break
        if not hit:
            print(f"[warn] 未匹配到: {key}")
            continue
        instr = hit.get("instruction", "").strip()
        code = hit.get("output", "").strip()
        used_instr.add(instr)
        matched.append({
            "instruction": instr,
            "output": cot + "\n\n实现：\n" + code,
            "_type": "reason_cot",
            "_universe": "代码",
            "_src": "v7_selftaught(teacher:self)",
        })
    with open(OUT, "w", encoding="utf-8") as f:
        for m in matched:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")
    print(f"扩展真推理 -> {OUT}  共 {len(matched)} 条 (期望 {len(EXTRA)})")


if __name__ == "__main__":
    main()
