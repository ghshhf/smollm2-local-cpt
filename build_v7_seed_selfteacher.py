"""
V7 推理卡 · 自教种子生成器（老师 = 本机最强模型，即 agent 自身）
===============================================================
这不是模板！reasoning 列表里的每一步都是针对该任务真实代码的取舍分析，
由 agent（强模型）亲手写出，模拟"老师先想一遍、小模型学这个想的过程"。

产物：data/role_data_v7_selftaught_seed.jsonl （20 条范式任务真推理链）
后续强化路径：
  - 持续扩充 reasoning 列表（更多范式/更多 code_fn 样本）→ 重跑本脚本
  - 或把 teacher 指向强端点，用 build_v7_selftaught.py 铺满 1958 条
"""
import json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "role_data_v6.jsonl")
OUT = os.path.join(ROOT, "data", "role_data_v7_selftaught_seed.jsonl")

# —— 老师（agent）亲手写的真推理链，顺序对应 role_data_v6 中 code_paradigm 的出现序 ——
REASONING = [
    # [0] Bash 重命名 + 日志轮转
    "思考步骤：\n"
    "1. 两件事拆开：①把 *.txt 按 doc_0001 顺序重命名，②把 >10MB 的 .log 按日期轮转压缩。\n"
    "2. 重命名用 `shopt -s nullglob` 防止无匹配时把 `*.txt` 当字面量；`printf -v` 做 4 位零填充；保留原扩展名（`${f##*.}`）。\n"
    "3. 必须检查目标是否已存在（`[[ ! -e $new ]]`），否则同名会互相覆盖；`mv --` 防文件名以 `-` 开头被当选项。\n"
    "4. 日志轮转用 `find -size +10M` 筛大文件，`-exec gzip` 改名带日期。复杂度：重命名 O(n)、轮转线性。\n"
    "5. 边界：空目录、重名冲突、gzip 目标已存在会失败——脚本只做最小可用，production 应加存在性判断。",
    # [1] Canvas 粒子系统
    "思考步骤：\n"
    "1. 粒子状态（位置/速度/生命）与更新（重力/位移/衰减）、绘制三者分离，便于发射器批量管理。\n"
    "2. 构造函数随机初速，`vy = -random*4 - 1` 让粒子先升后落，模拟从某点喷射。\n"
    "3. update 里 `vy += 0.15` 是重力逐帧累加，`life -= 0.02` 做淡出；位置用欧拉积分 `x += vx`。\n"
    "4. 渲染每帧 clearRect 再画；发射器 spawn 一批 push 进数组。\n"
    "5. 边界：life<=0 必须从数组移除（filter 或 swap-pop），否则数组只增不减导致内存泄漏与性能塌方。复杂度 O(粒子数)/帧。",
    # [2] JS 并发限制器 mapLimit
    "思考步骤：\n"
    "1. 核心不是 `Promise.all`（那会全部并发），而是用固定数量的 worker 协程消费一个共享索引 i。\n"
    "2. worker 循环：取当前 i 并自增（原子），`await tasks[idx]()`，把结果存到 `results[idx]` 保持原顺序。\n"
    "3. 启动 `min(limit, tasks.length)` 个 worker 并发跑，全部 task 消费完即自然结束。\n"
    "4. 结果数组预分配长度、按下标回填，避免并发 push 打乱顺序。\n"
    "5. 边界：空 tasks、limit 大于长度取 min；某任务 reject 会冒泡（如需隔离可在外层包 catch）。复杂度 O(n)，并发度≤limit。",
    # [3] debounce / throttle
    "思考步骤：\n"
    "1. 防抖=等待「安静期」后才执行（最后一次触发后 wait 内无新触发才跑）；节流=固定窗口内最多执行一次。语义不同，别混。\n"
    "2. debounce 用 `clearTimeout`+`setTimeout` 每次触发都重置计时器，只有停手才真正执行。\n"
    "3. throttle 用时间戳：`now-last>=wait` 才执行并更新 last；否则记 `remain` 延迟补一次（trailing 变体），实现 leading/trailing 两种风格。\n"
    "4. 两者都返回高阶函数闭包，持有定时器/时间戳；用 `fn.apply(this, args)` 透传 this 与参数。\n"
    "5. 边界：首次是否立即执行、参数透传、高频场景下的 this 绑定，这三点是手写最容易漏的。",
    # [4] deepClone 带循环引用
    "思考步骤：\n"
    "1. 递归 + 记忆化。基础类型/`null` 直接返回，不进入递归。\n"
    "2. 用 `WeakMap` 记录「原对象→其副本」，检测并打破循环引用；用 WeakMap 而非 Map 是因为不阻止 GC，正好适合这个临时映射场景。\n"
    "3. `Date`/`RegExp` 必须新建同值实例（`new Date(val)` / `new RegExp(val)`），否则会共享引用。\n"
    "4. Array/Object 新建容器后，先把自己放入 seen 再递归填字段——顺序很关键，先放后递归才能在互引时查到自身、不死循环。\n"
    "5. 边界：函数一般不深拷贝、Symbol key、Map/Set 未处理（可扩展）。复杂度 O(节点数)。",
    # [5] fetch 超时 + 错误处理
    "思考步骤：\n"
    "1. 用 `AbortController` 在 timeout 毫秒后 `abort()`，中断挂起的 fetch，避免请求永远卡住拖垮调用方。\n"
    "2. try 内 `await fetch`，`!res.ok` 直接抛 HTTP 错误（4xx/5xx 都算失败，交由调用方区分）。\n"
    "3. 解析 JSON 可能抛 `SyntaxError`（返回非 JSON），也要捕获并给出可读错误。\n"
    "4. finally 清掉定时器，避免 timeout 后仍触发 abort 影响别的请求。\n"
    "5. 边界：`abort` 自身抛 `AbortError`，要与 HTTP 错误区分；timeout 取值要合理（默认 10s）。",
    # [6] pytest 参数化
    "思考步骤：\n"
    "1. 把多组 (输入, 期望) 抽成参数表，用 `@pytest.mark.parametrize` 驱动同一测试函数多次运行，每组独立失败定位。\n"
    "2. 覆盖负数、零、大数等边界——负负得零、零加零、大数不溢出（Python int 无限精度其实不会溢，但验证语义正确）。\n"
    "3. `assert add(a,b)==exp` 直接比对，失败信息清晰。\n"
    "4. 优势：新增用例只加一行表项，不复制函数体，DRY 且覆盖度可见。\n"
    "5. 边界：参数表类型要一致；浮点场景需用 `approx` 而非 `==`。",
    # [7] dataclass 配置校验
    "思考步骤：\n"
    "1. `@dataclass` 省掉手写 `__init__` 样板；`from_dict` 做字段存在性 + 类型双重校验。\n"
    "2. 遍历 `fields(cls)`：缺字段抛 `KeyError` 带字段名，类型不符抛 `TypeError` 带期望/实际，错误信息要可读便于排查。\n"
    "3. 类型检查用 `isinstance`，对可选字段（有 default）不强制要求传入。\n"
    "4. 返回 `cls(**cleaned)`，把校验后的值交给 dataclass 构造。\n"
    "5. 边界：嵌套类型、Union、List[...] 需扩展校验逻辑；本例只做一层基础校验。",
    # [8] 二分答案
    "思考步骤：\n"
    "1. 「二分答案」前提是答案具单调性：若 x 满足 predicate，则更大（或更小）的也满足，可二分收缩。\n"
    "2. 本题找「最大 x 使 predicate 成立」：成立时记录 `ans=mid` 并向右探 `lo=mid+1`；不成立向左 `hi=mid-1`。\n"
    "3. 循环条件 `lo<=hi`，`mid=(lo+hi)//2` 用整除防浮点。\n"
    "4. 初始化 `ans=lo`：当 predicate 全不成立时返回合理默认，避免返回 None 引发下游错误。\n"
    "5. 复杂度 O(log(hi-lo) · predicate 代价)。边界：predicate 必须是纯函数、无副作用。",
    # [9] contextmanager 计时打开
    "思考步骤：\n"
    "1. `@contextmanager` 把「进入 yield 资源 / 退出清理」写成一个生成器，比手写 `__enter__/__exit__` 简洁。\n"
    "2. 进入时 `open` 文件并记 `t0`；`yield f` 把文件交给 `with` 块使用。\n"
    "3. `finally` 保证无论块内是否抛异常，都关闭文件并打印耗时——finally 语义就是「必然执行」。\n"
    "4. 异常会透传给调用方（不吞），但 finally 的清理仍执行，符合资源管理预期。\n"
    "5. 边界：yield 只能有一次；mode 默认 'r'；若 open 本身就失败，finally 不会跑（还没拿到 f）。",
    # [10] HTTP 重试 + 指数退避
    "思考步骤：\n"
    "1. 重试只对「有价值」的错误：网络异常和 5xx（服务端临时故障）重试；4xx 是客户端错，重发无意义，直接抛。\n"
    "2. 指数退避 `sleep(2**attempt)` 避免雪崩式重试把服务打更死。\n"
    "3. 循环 `retries` 次，成功即返回；最后一次仍失败要抛出（原错或汇总）。\n"
    "4. 超时(`timeout`)算网络错，可重试；退避应设上限防止等太久。\n"
    "5. 边界：4xx 中 429(限流)其实该退避，这里简化为不重试，真实场景应特判。",
    # [11] LRU + TTL 缓存装饰器
    "思考步骤：\n"
    "1. dict 存「值+过期时间」，order 列表维护 LRU 顺序（访问/写入移到末尾，淘汰头部）。\n"
    "2. TTL：访问时 `now>expire` 则当缺失、重新计算并刷新；容量超 `maxsize` 淘汰 `order[0]`。\n"
    "3. key 用参数元组，`kw` 排序后转 tuple 保证可哈希且等价调用命中同一 key。\n"
    "4. `@functools.wraps` 保留原函数元信息。\n"
    "5. 边界：本例单线程够用，非线程安全；ttl=0 表示永不过期；有副作用的函数缓存要谨慎。复杂度 O(1) 均摊。",
    # [12] 单调栈 下一个更大
    "思考步骤：\n"
    "1. 单调栈维护「待解决」的下标，栈内值递减。遍历数组，遇更大 x 就不断弹出栈顶——那些元素的「右边第一个更大」正是当前 i。\n"
    "2. 弹出后把 i 压入；栈空说明左边已全部解决。最终仍留栈里的元素「右边没有更大」→ res 保持初始 -1。\n"
    "3. 比较用严格 `<`，相等时不弹（保留更近的候选）。\n"
    "4. 每个元素进出栈各一次，总 O(n)，比暴力 O(n²) 优。\n"
    "5. 边界：空数组、单调增/减数组、重复元素。",
    # [13] 并查集 路径压缩 + 按秩
    "思考步骤：\n"
    "1. parent 数组 + rank 数组。find 做路径压缩：边找根边把节点直接挂到根，把树压平，加速后续查询。\n"
    "2. union 按秩合并：矮树挂到高树根下，保持树深度小，保证近 O(1) 均摊。\n"
    "3. 比较两元素根相等即连通。\n"
    "4. rank 可用深度或大小；路径压缩后 rank 与真实深度不完全一致，但不影响正确性。\n"
    "5. 应用：连通分量、Kruskal 最小生成树、岛屿类问题。初始化各自独立。",
    # [14] 最小堆
    "思考步骤：\n"
    "1. 数组实现完全二叉树：父 i 的子为 2i+1、2i+2；父为 (i-1)//2。\n"
    "2. push：尾插后 `_up` 上浮，与父比、小则交换，直到根或有序。\n"
    "3. pop：取 a[0]，尾元素放堆顶后 `_down` 下沉，与较小子比、交换直到有序；空堆抛错。\n"
    "4. peek 直接读 a[0]，O(1)。\n"
    "5. 边界：单元素、重复值、_up/_down 的终止条件。复杂度 O(log n) 操作。",
    # [15] 线段树
    "思考步骤：\n"
    "1. 线段树用 4n 数组，节点存区间和。_build 递归二分，叶子=单元素，内部节点=左右子树和。\n"
    "2. 单点更新：改叶子值后沿路径向上回溯更新所有祖先，O(log n)。\n"
    "3. 区间查询：递归拆分，节点区间完全被查询区间覆盖则直接返回，否则下钻左右。\n"
    "4. 下标约定从 1 还是 0 要统一（本例根节点 1，区间 [l,r]）。\n"
    "5. 边界：n=0；建树 O(n)、查改 O(log n)。适合动态前缀和/区间统计，优于树状数组处在区间查询更直观。",
    # [16] 滑动窗口最大值（单调队列）
    "思考步骤：\n"
    "1. 单调队列存下标，队首始终是当前窗口最大。遍历数组：先剔队尾 `<=x` 的（它们不可能再成最大），压入 i。\n"
    "2. 剔队首超出窗口 `[i-k, i-1]` 的（`dq[0] <= i-k`）。\n"
    "3. `i>=k-1` 窗口形成后才记录 `nums[dq[0]]`。\n"
    "4. 每个元素进出队列各一次，总 O(n)，比堆的 O(n log k) 更优。\n"
    "5. 边界：k=1、k=n、相等元素处理（用 `<=` 剔尾保证队首是最左最大）。",
    # [17] 正则解析日志
    "思考步骤：\n"
    "1. 用命名组 `(?P<name>...)` 把一行日志拆成结构化字段：ip、ts、method、path、code、ms。\n"
    "2. 模式按格式逐段写：反引号 S+ 匹配非空格、方括号转义匹配时间戳、引号内是 method/path/code、末尾数字加 ms。\n"
    "3. `search` 命中返回 `groupdict()`，未命中返回 None——对单行坏数据容错而非整体崩溃。\n"
    "4. 正则预编译复用（`re.compile`），逐行处理大文件也 OK。\n"
    "5. 边界：字段含空格需调整量词；时区/格式变化要改 pattern。",
    # [18] 流式 jsonl 转换
    "思考步骤：\n"
    "1. 逐行读避免把大文件一次性载入内存——这是处理 GB 级 jsonl 的关键。\n"
    "2. `with` 同时开输入/输出，`for line in fin` 惰性迭代；`strip()` 跳空行。\n"
    "3. 每行 `json.loads`→转换→`json.dumps` 写出；`count` 统计处理条数。\n"
    "4. 单条坏 JSON 用 try 包住，避免一条错整批崩。\n"
    "5. 边界：编码指定 utf-8；写入可加缓冲；转换函数应是纯函数便于测试。复杂度 O(行数)。",
    # [19] asyncio 并发爬虫
    "思考步骤：\n"
    "1. asyncio 协程比线程池省资源。Semaphore 限制最大并发，防止打爆目标服务器或被封。\n"
    "2. 每个 fetch 在 `sem` 内做带超时（`aiohttp.ClientTimeout`）的请求，避免单请求卡死整池。\n"
    "3. 失败指数退避重试最多 3 次（`await asyncio.sleep(2**attempt)`），限流/5xx 才有意义重试。\n"
    "4. 失败记录到结果列表而非中断整体；`gather` 收集全部结果/异常。\n"
    "5. 边界：finally 关 session；超时抛 `asyncio.TimeoutError`；合理设置并发数与退避上限。",
]

def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    para = [r for r in rows if r.get("_type") == "code_paradigm"]
    assert len(para) == len(REASONING), f"范式数 {len(para)} != 推理条数 {len(REASONING)}"
    out = []
    for r, cot in zip(para, REASONING):
        instr = r.get("instruction", "").strip()
        code = r.get("output", "").strip()
        out.append({
            "instruction": instr,
            "output": cot + "\n\n实现：\n" + code,
            "_type": "reason_cot",
            "_universe": "代码",
            "_src": "v7_selftaught(teacher:self)",
        })
    with open(OUT, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"自教种子生成 {len(out)} 条 -> {OUT}")

if __name__ == "__main__":
    main()
