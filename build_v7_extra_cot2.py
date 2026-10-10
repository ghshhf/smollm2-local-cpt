"""
V7 推理卡 · 第二扩展库（agent 亲手写，teacher:self）— 自包含真推理
========================================================================
与 seed(20) + extra(22) 互补，本文件再补 58 条**自包含**真推理链，
每条都带 instruction + 真推理(cot) + 工作代码，不依赖 role_data_v6 匹配。

匹配策略：直接写盘为 data/v7_self_cot_extra2.jsonl，
build_v7_selftaught.py 会把它并入「真推理字典」一起注入最终数据集。

目标：真推理链总量 20 + 22 + 58 = 100 条。
"""
import json, os
ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "data", "v7_self_cot_extra2.jsonl")

# (instruction, 真推理链, 代码) — 每条都是针对该任务真实代码的取舍分析
BANK = [

("实现快速幂（二进制幂）",
 "思考步骤：\n"
 "1. 朴素连乘是 O(n)，但指数是 2 的幂可用平方减半：a^e = (a^(e/2))^2，指数每次折半，O(log e)。\n"
 "2. 二进制展开更通用：把指数 e 写成二进制，e=Σ b_i·2^i，则 a^e = Π (a^(2^i))^{b_i}，只在当前位是 1 时乘入结果。\n"
 "3. 循环里 `base = base*base % mod` 不断平方，`e>>=1` 逐位右移；`e & 1` 取最低位决定是否乘入。\n"
 "4. 必须边乘边取模（(res*base)%mod），否则中间值指数爆炸溢出。\n"
 "5. 边界：e=0 返回 1；负指数需先求逆元（模意义下），本例只处理非负。复杂度 O(log e)。",
 "def pow_mod(base, e, mod):\n"
 "    res = 1 % mod\n"
 "    base %= mod\n"
 "    while e > 0:\n"
 "        if e & 1:\n"
 "            res = (res * base) % mod\n"
 "        base = (base * base) % mod\n"
 "        e >>= 1\n"
 "    return res"),

("二叉树层序遍历（BFS 队列）",
 "思考步骤：\n"
 "1. 层序 = 广度优先，用队列 FIFO：先把根入队，循环取队首、访问、把左右孩子入队。\n"
 "2. 关键难点是「分层」：用 `len(q)` 快照本层节点数，一次性处理完这一层再进入下一层，才能把每层分组。\n"
 "3. 空孩子不进队（或进 None 占位）——这里只进有效节点，结果不含 None。\n"
 "4. 为什么不用 DFS：DFS 会先钻到底，天然先序/后序，无法保证同层相邻。\n"
 "5. 边界：空树返回空列表；单节点；退化成链表时队列最多 1 个元素。复杂度 O(n)。",
 "from collections import deque\n"
 "def level_order(root):\n"
 "    if not root:\n"
 "        return []\n"
 "    q = deque([root]); res = []\n"
 "    while q:\n"
 "        level = []\n"
 "        for _ in range(len(q)):\n"
 "            n = q.popleft()\n"
 "            level.append(n.val)\n"
 "            if n.left: q.append(n.left)\n"
 "            if n.right: q.append(n.right)\n"
 "        res.append(level)\n"
 "    return res"),

("两数之和（哈希表一遍过）",
 "思考步骤：\n"
 "1. 暴力两重循环 O(n^2)；用哈希表把「值→下标」缓存，边走边查互补值，降到 O(n)。\n"
 "2. 遍历到 x 时，先查 `target-x` 是否已在表中：在则直接返回这对下标（互补值一定在 x 之前出现过）。\n"
 "3. 查不到才把 x 存入表。顺序很重要：先查后存，避免自己和自己匹配（如 target=6, x=3 会误配）。\n"
 "4. 只要求返回任意一对，找到即返回，不需遍历完。\n"
 "5. 边界：保证有解则无需处理无解；若可能无解要返回特定标记。空间换时间，O(n) 额外空间。",
 "def two_sum(nums, target):\n"
 "    seen = {}\n"
 "    for i, x in enumerate(nums):\n"
 "        if target - x in seen:\n"
 "            return [seen[target - x], i]\n"
 "        seen[x] = i\n"
 "    return []"),

("反转单链表（迭代三指针）",
 "思考步骤：\n"
 "1. 核心：遍历时把当前节点的 next 指向前一个，但需要先保存下一个节点，否则改完指针就丢失后续链。\n"
 "2. 三指针 prev/cur/next：每步 `nxt = cur.next` 暂存，`cur.next = prev` 反转，`prev = cur; cur = nxt` 推进。\n"
 "3. 循环结束 cur 为 None，prev 正好是新头。\n"
 "4. 为什么不用递归：递归简洁但 O(n) 栈深，长链表会栈溢出；迭代 O(1) 空间更稳。\n"
 "5. 边界：空链表/单节点直接返回；处理时 prev 初值 None（新尾指向 None）。复杂度 O(n)。",
 "def reverse(head):\n"
 "    prev = None; cur = head\n"
 "    while cur:\n"
 "        nxt = cur.next\n"
 "        cur.next = prev\n"
 "        prev = cur\n"
 "        cur = nxt\n"
 "    return prev"),

("最长无重复字符子串（滑动窗口）",
 "思考步骤：\n"
 "1. 暴力枚举所有子串 O(n^2)；滑动窗口用「右扩+左缩」在 O(n) 内维护「窗口内无重复」。\n"
 "2. 右指针 r 一直右扩，把字符纳入；若 `s[r]` 已在窗口（last 位置 >= l），左指针 l 跳到重复字符的下一个。\n"
 "3. 用字典 last 记录「字符→最近下标」，O(1) 判断重复并定位 l 的新位置。\n"
 "4. 每步更新最大长度 `r-l+1`（注意 l 是「已不含重复」的左边界）。\n"
 "5. 边界：空串返回 0；全相同字符时窗口宽度恒 1；ASCII 与 Unicode 都靠字典处理。复杂度 O(n)。",
 "def length_of_longest(s):\n"
 "    last = {}; l = 0; best = 0\n"
 "    for r, ch in enumerate(s):\n"
 "        if ch in last and last[ch] >= l:\n"
 "            l = last[ch] + 1\n"
 "        last[ch] = r\n"
 "        best = max(best, r - l + 1)\n"
 "    return best"),

("实现堆排序",
 "思考步骤：\n"
 "1. 分两阶段：先自底向上 `heapify` 把数组建成大顶堆（父 >= 子），再反复把堆顶（最大）换到末尾并收缩堆。\n"
 "2. 建堆从最后一个非叶节点 `n//2-1` 倒着下沉，保证每个子树都满足堆性质。\n"
 "3. 排序循环：swap(a[0], a[i]) 把最大放末尾，对缩小后的堆 `i` 调用 sift_down(0)。\n"
 "4. 大顶堆排出来是升序（最大不断落到末尾）；要降序用小顶堆。\n"
 "5. 边界：sift_down 终止条件是「无子或已比子大」；空/单元素直接返回。复杂度 O(n log n)，原地无额外空间。",
 "def heap_sort(a):\n"
 "    n = len(a)\n"
 "    def down(i, size):\n"
 "        while True:\n"
 "            l, r = 2*i+1, 2*i+2; big = i\n"
 "            if l < size and a[l] > a[big]: big = l\n"
 "            if r < size and a[r] > a[big]: big = r\n"
 "            if big == i: break\n"
 "            a[i], a[big] = a[big], a[i]; i = big\n"
 "    for i in range(n//2-1, -1, -1): down(i, n)\n"
 "    for i in range(n-1, 0, -1):\n"
 "        a[0], a[i] = a[i], a[0]; down(0, i)\n"
 "    return a"),

("岛屿数量（DFS 网格）",
 "思考步骤：\n"
 "1. 网格里相邻的 '1' 连成一座岛。遍历每个格子，遇 '1' 就 dfs 把整座岛（四连通）全标成 '0'，岛数 +1。\n"
 "2. dfs 把当前格置 '0'（已访问标记，防重复计数），再向上下左右递归；越界或遇 '0' 停止。\n"
 "3. 为什么原地改网格当 visited：省一个 m*n 的 visited 矩阵，空间 O(1) 额外。\n"
 "4. BFS（队列）同样可行，DFS 递归写法最短，但大网格有栈溢出风险——极端情况换显式栈。\n"
 "5. 边界：空矩阵；单格；斜向不算连通（题目通常只上下左右）。复杂度 O(m*n)。",
 "def num_islands(grid):\n"
 "    if not grid: return 0\n"
 "    m, n = len(grid), len(grid[0]); cnt = 0\n"
 "    def dfs(i, j):\n"
 "        if i < 0 or j < 0 or i >= m or j >= n or grid[i][j] != '1':\n"
 "            return\n"
 "        grid[i][j] = '0'\n"
 "        for di, dj in ((1,0),(-1,0),(0,1),(0,-1)):\n"
 "            dfs(i+di, j+dj)\n"
 "    for i in range(m):\n"
 "        for j in range(n):\n"
 "            if grid[i][j] == '1':\n"
 "                dfs(i, j); cnt += 1\n"
 "    return cnt"),

("编辑距离（动态规划）",
 "思考步骤：\n"
 "1. 把 word1 变 word2 的最少单字符操作（增/删/改）。定义 dp[i][j]=word1[:i] 到 word2[:j] 的最小步数。\n"
 "2. 若末字符相同，dp[i][j]=dp[i-1][j-1]（不用操作）；否则取「改/删/增」三者最小值 +1。\n"
 "3. 改：dp[i-1][j-1]+1；删：dp[i-1][j]+1（删 word1 末字符）；增：dp[i][j-1]+1（补 word1 末字符匹配 word2 末字符）。\n"
 "4. 初始化边界：空串到另一串需 len 次增/删。\n"
 "5. 边界：可滚动数组把 O(mn) 空间降到 O(n)；空串特判。复杂度 O(mn)。",
 "def min_distance(w1, w2):\n"
 "    m, n = len(w1), len(w2)\n"
 "    dp = list(range(n+1))\n"
 "    for i in range(1, m+1):\n"
 "        prev = dp[0]; dp[0] = i\n"
 "        for j in range(1, n+1):\n"
 "            tmp = dp[j]\n"
 "            if w1[i-1] == w2[j-1]:\n"
 "                dp[j] = prev\n"
 "            else:\n"
 "                dp[j] = 1 + min(prev, dp[j], dp[j-1])\n"
 "            prev = tmp\n"
 "    return dp[n]"),

("0/1 背包（动态规划）",
 "思考步骤：\n"
 "1. 每个物品「选或不选」，目标价值最大且不超过容量。dp[j]=容量 j 下的最大价值，从 0 推到 W。\n"
 "2. 关键：容量必须**逆序**遍历（j 从 W 降到 w），否则同一物品会被重复选多次（变成完全背包）。\n"
 "3. 转移：dp[j] = max(dp[j], dp[j-w]+v)，即「不选」与「选当前物品」取优。\n"
 "4. 初始化 dp 全 0（允许不选任何物品价值为 0）；若要求必须装满，初值要设 -inf 仅 dp[0]=0。\n"
 "5. 边界：物品重量 > 当前容量则跳过；空间可压成一维 O(W)。复杂度 O(nW)。",
 "def knapsack(W, weights, values):\n"
 "    dp = [0] * (W + 1)\n"
 "    for w, v in zip(weights, values):\n"
 "        for j in range(W, w - 1, -1):\n"
 "            dp[j] = max(dp[j], dp[j - w] + v)\n"
 "    return dp[W]"),

("实现 Trie 前缀树",
 "思考步骤：\n"
 "1. Trie 用树节点存字符，根到某节点的路径构成一个前缀；每个节点持 children 字典和 is_end 标记。\n"
 "2. 插入：逐字符沿 children 下行，缺则建节点，末尾置 is_end=True。\n"
 "3. 查找词：同样下行，能走完且末节点 is_end 才是「完整词」（否则只是前缀，如 'app' 是 'apple' 的前缀）。\n"
 "4. 前缀查询 startsWith：能走完路径即可，不要求 is_end。\n"
 "5. 边界：空串；重复插入幂等；删除较复杂（需判断是否仍有其他词共用后缀），本例不实现删除。复杂度 O(词长)。",
 "class Trie:\n"
 "    def __init__(self):\n"
 "        self.children = {}\n"
 "        self.is_end = False\n"
 "    def insert(self, w):\n"
 "        node = self\n"
 "        for c in w:\n"
 "            node = node.children.setdefault(c, Trie())\n"
 "        node.is_end = True\n"
 "    def search(self, w):\n"
 "        node = self\n"
 "        for c in w:\n"
 "            if c not in node.children: return False\n"
 "            node = node.children[c]\n"
 "        return node.is_end\n"
 "    def starts_with(self, p):\n"
 "        node = self\n"
 "        for c in p:\n"
 "            if c not in node.children: return False\n"
 "            node = node.children[c]\n"
 "        return True"),

("有序数组去重（双指针）",
 "思考步骤：\n"
 "1. 已排序数组里重复元素相邻，用慢指针 slow 指「下一个可写位置」，快指针 fast 扫描。\n"
 "2. fast 每遇到与 `nums[slow]` 不同的值，slow 先 +1 再写入，保证 slow 左侧全是唯一值。\n"
 "3. 原地修改、O(1) 额外空间；返回 slow+1 即唯一元素个数。\n"
 "4. 为什么不用集合：集合 O(n) 空间且打乱顺序，这里要求原地保序。\n"
 "5. 边界：空数组返回 0；全相同元素时 slow 不动，结果长度 1；重复次数 >2 的变体用 slow-1 比较。复杂度 O(n)。",
 "def remove_duplicates(nums):\n"
 "    if not nums: return 0\n"
 "    slow = 0\n"
 "    for fast in range(1, len(nums)):\n"
 "        if nums[fast] != nums[slow]:\n"
 "            slow += 1\n"
 "            nums[slow] = nums[fast]\n"
 "    return slow + 1"),

("旋转有序数组搜值（二分）",
 "思考步骤：\n"
 "1. 数组被旋转但仍局部有序，普通二分失效，但每次 mid 必有一侧完整有序——利用这点。\n"
 "2. 判 mid 落在左半还是右半：若 `nums[lo]<=nums[mid]` 左半有序，否则右半有序。\n"
 "3. 在有序那一侧判断 target 是否落在其区间：是则在该侧二分收缩，否则去另一侧。\n"
 "4. 用 `lo<hi` 循环、整除取 mid；找到即返回下标。\n"
 "5. 边界：无重复元素时好判；若有重复（如 [1,1,1,3,1]）需 `lo++` 退化处理。复杂度 O(log n)。",
 "def search_rotated(nums, target):\n"
 "    lo, hi = 0, len(nums) - 1\n"
 "    while lo <= hi:\n"
 "        mid = (lo + hi) // 2\n"
 "        if nums[mid] == target:\n"
 "            return mid\n"
 "        if nums[lo] <= nums[mid]:\n"
 "            if nums[lo] <= target < nums[mid]:\n"
 "                hi = mid - 1\n"
 "            else:\n"
 "                lo = mid + 1\n"
 "        else:\n"
 "            if nums[mid] < target <= nums[hi]:\n"
 "                lo = mid + 1\n"
 "            else:\n"
 "                hi = mid - 1\n"
 "    return -1"),

("合并区间",
 "思考步骤：\n"
 "1. 先按区间起点排序，使可能重叠的区间相邻，才能线性扫描合并。\n"
 "2. 维护当前合并结果 cur，遍历每个区间：若它的起点 <= cur 的终点则重叠，把 cur 终点扩到 max；否则 cur 与下一个无交，把 cur 收进答案、开始新 cur。\n"
 "3. 用 `max(cur[1], iv[1])` 处理「被包含」情况（如 [1,5] 与 [2,3]）。\n"
 "4. 为什么先排序：不排序时重叠区间可能分散，单次扫描会漏。\n"
 "5. 边界：空输入返回空；单区间；完全包含/部分重叠都正确。复杂度 O(n log n) 由排序主导。",
 "def merge(intervals):\n"
 "    if not intervals: return []\n"
 "    intervals.sort(key=lambda x: x[0])\n"
 "    res = [list(intervals[0])]\n"
 "    for s, e in intervals[1:]:\n"
 "        if s <= res[-1][1]:\n"
 "            res[-1][1] = max(res[-1][1], e)\n"
 "        else:\n"
 "            res.append([s, e])\n"
 "    return res"),

("最小栈（辅助栈）",
 "思考步骤：\n"
 "1. 普通栈 push/pop O(1)，但取最小需 O(n)；用「辅助栈」同步记录「到当前位置为止的最小值」。\n"
 "2. 每 push x，辅助栈压入 `min(aux.top, x)`——即压栈后栈内最小值；pop 时主栈与辅助栈同步弹出。\n"
 "3. 取最小 = 直接读 aux.top，O(1)。\n"
 "4. 为什么不同步存最小值的下标：值可能重复、删除时难判断；存「运行最小值」更稳。\n"
 "5. 边界：空栈 pop/getMin 应抛错或返回哨兵；辅助栈与主栈严格同步长度。复杂度 O(1) 各操作。",
 "class MinStack:\n"
 "    def __init__(self):\n"
 "        self.st = []; self.au = []\n"
 "    def push(self, x):\n"
 "        self.st.append(x)\n"
 "        m = x if not self.au else min(self.au[-1], x)\n"
 "        self.au.append(m)\n"
 "    def pop(self):\n"
 "        self.st.pop(); self.au.pop()\n"
 "    def top(self):\n"
 "        return self.st[-1]\n"
 "    def get_min(self):\n"
 "        return self.au[-1]"),

("二叉树最近公共祖先（LCA）",
 "思考步骤：\n"
 "1. 后序递归：对当前节点，若它等于 p 或 q，或左右子树分别 contains 了 p/q，则当前节点是 LCA。\n"
 "2. 递归返回「在子树中是否找到 p 或 q」：左子树找到 + 右子树找到 => 当前是 LCA；仅一侧找到则把那侧结果向上传。\n"
 "3. 利用「p、q 必在树中」前提，第一次出现左右都非空的就是答案。\n"
 "4. 为什么后序：要先知道子树结果才能判定当前节点角色。\n"
 "5. 边界：p 是 q 的祖先时返回 p；节点不存在需先校验；本例假定两节点均在树中。复杂度 O(n)。",
 "def lowest_common_ancestor(root, p, q):\n"
 "    def dfs(node):\n"
 "        if not node or node in (p, q):\n"
 "            return node\n"
 "        left = dfs(node.left)\n"
 "        right = dfs(node.right)\n"
 "        if left and right:\n"
 "            return node\n"
 "        return left or right\n"
 "    return dfs(root)"),

("环形链表检测（快慢指针）",
 "思考步骤：\n"
 "1. 用快(2步)/慢(1步)双指针：若有环，快指针终会套圈追上慢指针；无环则快指针先到 None。\n"
 "2. 相遇后，把其中一个指针拉回表头，两指针都改 1 步前进，再次相遇点即环入口。\n"
 "3. 数学性质：头到入口距离 = 相遇点沿环到入口距离的某倍数偏移，故同步 1 步必在入口重逢。\n"
 "4. 为什么不用集合记录 visited：集合 O(n) 空间，双指针 O(1) 空间更优。\n"
 "5. 边界：空表/单节点无环返回 None；环长度为 1（自环）也能正确识别。复杂度 O(n) 时间 O(1) 空间。",
 "def detect_cycle(head):\n"
 "    slow = fast = head\n"
 "    while fast and fast.next:\n"
 "        slow = slow.next; fast = fast.next.next\n"
 "        if slow is fast:\n"
 "            slow = head\n"
 "            while slow is not fast:\n"
 "                slow = slow.next; fast = fast.next\n"
 "            return slow\n"
 "    return None"),

("KMP 字符串匹配",
 "思考步骤：\n"
 "1. 朴素匹配最坏 O(mn)；KMP 用「部分匹配表（next/前缀函数）」在主串指针不回退，降到 O(m+n)。\n"
 "2. 前缀函数 lps[i] = 模式前 i+1 字符中「最长的真前缀=真后缀」长度，刻画「失配后能复用多少已匹配」。\n"
 "3. 文本指针 i 与模式指针 j：匹配则同进；失配时 j 不回 0 而是跳到 lps[j-1]，复用前面一致的部分。\n"
 "4. 求 lps 本身用「模式与自己匹配」的同理逻辑构造。\n"
 "5. 边界：模式空串；重复字符模式（如 'aaaa'）lps 全 1；返回所有命中下标。复杂度 O(m+n)。",
 "def kmp(text, pat):\n"
 "    if not pat: return [0]\n"
 "    m = len(pat); lps = [0]*m; j = 0\n"
 "    for i in range(1, m):\n"
 "        while j and pat[i] != pat[j]: j = lps[j-1]\n"
 "        if pat[i] == pat[j]: j += 1\n"
 "        lps[i] = j\n"
 "    res = []; j = 0\n"
 "    for i, c in enumerate(text):\n"
 "        while j and c != pat[j]: j = lps[j-1]\n"
 "        if c == pat[j]: j += 1\n"
 "        if j == m: res.append(i-m+1); j = lps[j-1]\n"
 "    return res"),

("子集枚举（回溯）",
 "思考步骤：\n"
 "1. 求所有子集 = 每个元素「选或不选」的 2^n 种组合，回溯逐位决策最直观。\n"
 "2. dfs 维护当前路径 path 和起点 start：每层把 nums[start] 选入 path、递归到 start+1、再回溯弹出。\n"
 "3. 每层开头（无论选不选）先把当前 path 加入答案——因为 path 本身就代表一个已定子集。\n"
 "4. 为什么从 start 推进而非全排列：避免产生同一子集的重复排列（如 [1,2] 与 [2,1] 视为同集）。\n"
 "5. 边界：空数组有 1 个子集（空集）；可加去重（排序后跳过相同起点的相同值）。复杂度 O(2^n)。",
 "def subsets(nums):\n"
 "    res = []\n"
 "    def dfs(start, path):\n"
 "        res.append(list(path))\n"
 "        for i in range(start, len(nums)):\n"
 "            dfs(i + 1, path + [nums[i]])\n"
 "    dfs(0, [])\n"
 "    return res"),

("约瑟夫问题（数学递归）",
 "思考步骤：\n"
 "1. n 人报数每 m 个淘汰，求最后幸存者位置。暴力模拟 O(nm)；用递推 O(n)。\n"
 "2. 关键观察：第一轮淘汰位置 (m-1)%n 后，剩余 n-1 人重新编号——问题递归成规模 n-1 的子问题。\n"
 "3. 递推式 f(n) = (f(n-1) + m) % n，f(1)=0（只剩 1 人时位置 0）。从 1 推到 n 即可。\n"
 "4. 含义：子问题的解在新编号下偏移 m 位，取模回到 0-based 物理位置。\n"
 "5. 边界：m=1 每轮淘汰第 1 个，幸存者是最后一个；n=0 无意义。复杂度 O(n) 迭代、O(1) 空间。",
 "def josephus(n, m):\n"
 "    if n < 1 or m < 1: return -1\n"
 "    ans = 0\n"
 "    for i in range(2, n + 1):\n"
 "        ans = (ans + m) % i\n"
 "    return ans"),

("最大子数组和（Kadane）",
 "思考步骤：\n"
 "1. 求连续子数组最大和。暴力 O(n^2)；Kadane 一次扫描 O(n)：维护「以当前元素结尾的最大子数组和」。\n"
 "2. 到位置 i 时，要么把 a[i] 接在前面最优段后（cur+a[i]），要么从 a[i] 重新开始（a[i]），取较大者更新 cur。\n"
 "3. 全局最大 best 在每步与 cur 比较更新。\n"
 "4. 为什么不能全局只取 max：负数会拖累，必须允许「在更负之前重置起点」。\n"
 "5. 边界：全负数时返回最大那个负数（重置不跨过 0）；空数组特判。复杂度 O(n) 一维扫描。",
 "def max_subarray(nums):\n"
 "    best = cur = nums[0]\n"
 "    for x in nums[1:]:\n"
 "        cur = max(x, cur + x)\n"
 "        best = max(best, cur)\n"
 "    return best"),

("事件循环：微任务与宏任务",
 "思考步骤：\n"
 "1. JS 单线程靠事件循环：同步代码先跑完，再依次清空「微任务队列」，最后取一个「宏任务」并重复。\n"
 "2. 微任务（Promise.then、queueMicrotask、MutationObserver）优先级高于宏任务（setTimeout、I/O、UI 事件）。\n"
 "3. 每个宏任务执行后，会先把当时所有微任务清空，才进入下一个宏任务——这决定了 await 之后的代码晚于同步但早于定时器。\n"
 "4. 关键陷阱：`setTimeout(fn,0)` 不保证 0ms，它只是排到宏任务队尾；Promise 回调一定先于它。\n"
 "5. 边界：微任务里无限 enqueue 会饿死宏任务（事件循环卡死在微任务阶段）；await 本质是把后续代码压成微任务。"),

("用 threading.Lock 防竞态",
 "思考步骤：\n"
 "1. 多线程共享可变状态（如计数器）时，读-改-写非原子，线程切换会交错导致丢更新。\n"
 "2. Lock 把临界区串行化：进入 `with lock:` 才允许改共享变量，退出自动释放（即使抛异常也释放，避免死锁）。\n"
 "3. 绝不能漏加锁的「看似只读」路径——读也得加锁，否则可能读到写了一半的中间值。\n"
 "4. 粒度权衡：锁范围太大会串行化过多、丧失并发；太小可能漏护住某些路径，要刚好罩住临界段。\n"
 "5. 边界：重入需用 RLock；asyncio 里用 asyncio.Lock 而非 threading.Lock（两者不互通）。",
 "import threading\n"
 "counter = 0; lock = threading.Lock()\n"
 "def inc():\n"
 "    global counter\n"
 "    for _ in range(100000):\n"
 "        with lock:\n"
 "            counter += 1"),

("queue.Queue 实现生产者消费者",
 "思考步骤：\n"
 "1. 生产者往队列放任务、消费者取任务，Queue 自带锁和条件变量，天然线程安全，省去手搓同步。\n"
 "2. `task_done()` + `join()` 让主线程等所有任务被消费完，比忙等 sleep 干净。\n"
 "3. 哨兵值（如 None）或 `put(None)` 通知消费者「没有更多任务」以优雅退出。\n"
 "4. 为什么不用 list + Lock 手搓：Queue 已处理「空时阻塞、满时阻塞」的边界，手搓易出竞态。\n"
 "5. 边界：maxsize 限制防内存爆；多消费者时哨兵要发 N 个（每个消费者一个）或用 Event 关店。",
 "import threading, queue\n"
 "q = queue.Queue()\n"
 "def consumer():\n"
 "    while True:\n"
 "        item = q.get()\n"
 "        if item is None:\n"
 "            q.task_done(); break\n"
 "        process(item); q.task_done()"),

("多进程绕过 GIL（CPU 密集）",
 "思考步骤：\n"
 "1. CPython 的 GIL 让多线程在 CPU 密集任务里无法真正并行；multiprocessing 起独立进程各有自己的 GIL，能吃满多核。\n"
 "2. 适合纯计算（数值、压缩、图像处理）；若任务是 I/O 密集（网络/磁盘），线程/协程更轻、无需进程开销。\n"
 "3. ProcessPoolExecutor 比手写 Process 更省心，map 自动把任务分片派发给进程池并返回结果。\n"
 "4. 代价：进程间不共享内存，传大对象要 pickle 走队列，开销大——数据应原地算、只回传小结果。\n"
 "5. 边界：进程数一般 = CPU 核数；Windows 上必须把入口包在 `if __name__=='__main__'` 防递归 spawn。",
 "from concurrent.futures import ProcessPoolExecutor\n"
 "def cpu_heavy(x):\n"
 "    return sum(i*i for i in range(x))\n"
 "def run(data):\n"
 "    with ProcessPoolExecutor() as ex:\n"
 "        return list(ex.map(cpu_heavy, data))"),

("用信号实现优雅退出",
 "思考步骤：\n"
 "1. 长期运行的服务收到 SIGTERM/SIGINT 不应立刻杀进程，否则正在处理的请求会中断、数据可能损坏。\n"
 "2. 注册信号处理函数只做「置标志位 stopped=True」，主循环每轮检查标志，自然走到安全点再退出。\n"
 "3. 信号处理函数要尽量轻（只设变量），别在里面做 I/O 或复杂逻辑——信号处理上下文受限。\n"
 "4. 退出前清理：关连接、刷盘、通知下游「我要下线」，比 `kill -9` 优雅得多。\n"
 "5. 边界：SIGKILL(-9) 不可捕获，优雅退出只对可捕获信号有效；多进程要逐一转发信号。",
 "import signal\n"
 "stopped = False\n"
 "def handle(signum, frame):\n"
 "    global stopped; stopped = True\n"
 "signal.signal(signal.SIGTERM, handle)\n"
 "signal.signal(signal.SIGINT, handle)\n"
 "while not stopped:\n"
 "    do_work()\n"
 "cleanup()"),

("asyncio.gather 容错并发",
 "思考步骤：\n"
 "1. gather 并发跑多个协程并等全部完成，比串行 await 快得多；默认任一抛异常会立即取消其余并冒泡。\n"
 "2. `return_exceptions=True` 让异常被当作正常结果返回而非中断整批——适合「允许部分失败」的场景。\n"
 "3. 结果顺序与传入顺序一致（不是完成顺序），方便与输入对齐。\n"
 "4. 真正要隔离单个失败，应在每个协程内部 try 包好，而非靠 gather 兜底——gather 只是最后一道网。\n"
 "5. 边界：gather 接收的是协程/任务对象；传普通函数会报错；大量任务建议分批避免过度并发。",
 "import asyncio\n"
 "async def run_all(tasks):\n"
 "    results = await asyncio.gather(*tasks, return_exceptions=True)\n"
 "    ok = [r for r in results if not isinstance(r, Exception)]\n"
 "    errs = [r for r in results if isinstance(r, Exception)]\n"
 "    return ok, errs"),

("令牌桶限流",
 "思考步骤：\n"
 "1. 令牌桶以固定速率往桶里加令牌，请求来时取走一个令牌才放行；桶空则限流（拒绝/排队）。\n"
 "2. 与「固定窗口计数器」比，令牌桶允许短时突发（桶攒满时一次性放过去），更平滑、对合法突发友好。\n"
 "3. 实现要点：不真跑定时器，而是取令牌时按 `now-last` 计算应补多少令牌（`min(cap, acc+rate*dt)`），惰性补桶。\n"
 "4. 桶容量 = 最大突发量；速率 = 长期平均允许 QPS。两者独立调参。\n"
 "5. 边界：多进程需共享桶（Redis/原子），单进程用锁即可；时间用单调时钟防系统时间回拨。",
 "import time\n"
 "class TokenBucket:\n"
 "    def __init__(self, rate, cap):\n"
 "        self.rate, self.cap, self.tok, self.ts = rate, cap, cap, time.monotonic()\n"
 "    def allow(self):\n"
 "        now = time.monotonic(); dt = now - self.ts\n"
 "        self.tok = min(self.cap, self.tok + dt * self.rate); self.ts = now\n"
 "        if self.tok >= 1:\n"
 "            self.tok -= 1; return True\n"
 "        return False"),

("JWT 签发与校验（HMAC）",
 "思考步骤：\n"
 "1. JWT = header.payload.signature，签名用服务端密钥对前两段做 HMAC，客户端无法篡改（改了签名对不上）。\n"
 "2. 签发：把声明（sub/exp/role）base64url 编码进 payload，用密钥算签名拼成三段串。\n"
 "3. 校验：重新算签名比对，并查 exp 是否过期——两者都过才信任该 token。\n"
 "4. 为什么不能把敏感信息放 payload：payload 只编码不加密，base64 一解码就可见，务必只放非机密声明。\n"
 "5. 边界：密钥必须服务端保密（泄露=可伪造任意 token）；过期时间 exp 用 UTC；算法固定写死 HMAC，防 alg:none 降级攻击。",
 "import hmac, hashlib, base64, json, time\n"
 "def b64u(d): return base64.urlsafe_b64encode(d).rstrip(b'=')\n"
 "def sign(payload, secret):\n"
 "    seg = b64u(json.dumps(payload).encode())\n"
 "    sig = hmac.new(secret, seg, hashlib.sha256).digest()\n"
 "    return seg + b'.' + b64u(sig)\n"
 "def verify(token, secret):\n"
 "    seg, sig = token.rsplit(b'.', 1)\n"
 "    exp = json.loads(base64.urlsafe_b64decode(seg + b'=='))['exp']\n"
 "    ok = hmac.compare_digest(hmac.new(secret, seg, hashlib.sha256).digest(), base64.urlsafe_b64decode(sig + b'=='))\n"
 "    return ok and exp > time.time()"),

("WebSocket 心跳保活",
 "思考步骤：\n"
 "1. TCP/WS 长连接可能因 NAT/代理超时静默断开，双方却以为还连着——心跳 ping/pong 用来探测真实存活。\n"
 "2. 客户端/服务端周期性（如 25s）发 ping，对端回 pong；超过 N 个周期没收到 pong 就判定掉线、主动重连。\n"
 "3. 间隔要小于中间设备的空闲超时（常见 30-60s），留余量防被掐。\n"
 "4. 心跳包要极小（几字节），高频心跳会占带宽，低频又可能漏检——按网络环境折中。\n"
 "5. 边界：应用层消息也算活跃可顺延心跳；重连要带退避，避免服务端被瞬间 reconnect 洪流打爆。"),

("CORS 预检与处理",
 "思考步骤：\n"
 "1. 浏览器对「非简单请求」（带自定义头/非 GET/带凭据）先发 OPTIONS 预检，确认服务器许可才发真实请求。\n"
 "2. 服务端对 OPTIONS 要回 204 + Access-Control-Allow-* 头（允许的源/方法/头），否则真实请求被浏览器拦截。\n"
 "3. 带凭据（cookie）时 `Access-Control-Allow-Origin` 不能写 `*`，必须明确列出请求方源，且 `Allow-Credentials: true`。\n"
 "4. 常见坑：预检请求不带 body，后端路由若要求 body 会 400；要单独放行 OPTIONS 或中间件统一处理。\n"
 "5. 边界：缓存预检结果用 `Max-Age` 减少重复 OPTIONS；生产应白名单校验 Origin 而非反射回传任意源。"),

("幂等接口设计（去重键）",
 "思考步骤：\n"
 "1. 网络重试/客户端双击会导致同一操作多次到达，非幂等接口（如扣款）会重复执行——需让「重复请求 = 一次效果」。\n"
 "2. 做法：客户端带唯一 idempotency-key，服务端用表/缓存记录「已处理过的 key → 上次结果」，重复 key 直接返回原结果。\n"
 "3. key 的粒度要正确：应是「一次业务动作」而非「一次 HTTP 请求」，否则跨业务的相同 payload 会误判重复。\n"
 "4. 存储要带 TTL 过期，否则表无限增长；高并发下对 key 的「查+写」要原子（唯一索引/分布式锁）防竞态双写。\n"
 "5. 边界：幂等只保证「同 key 同效果」，不同 key 的等价语义仍需业务层校验；失败请求不应占 key（允许重试）。"),

("数据库索引为什么快（B+树）",
 "思考步骤：\n"
 "1. 无索引的查询是全表扫描 O(n)；B+树索引把查找降到 O(log n)，因为树高很低（一个页存很多键，百万行也才 3-4 层）。\n"
 "2. B+树所有数据在叶子节点且用链表串起，范围查询（BETWEEN、ORDER BY）只需顺着叶子链表走，不必回树。\n"
 "3. 非聚簇索引叶子存的是主键值，拿到主键还要「回表」查聚簇索引取整行——覆盖索引（索引含所需列）可避免回表。\n"
 "4. 代价：索引占用空间、拖慢写入（每次 INSERT/UPDATE 要维护树），所以不能乱建。\n"
 "5. 边界：低基数字段（如性别）建索引收益低，优化器可能直接全表；复合索引遵循最左前缀才生效。"),

("事务隔离级别与幻觉",
 "思考步骤：\n"
 "1. 隔离级别解决并发事务三类问题：脏读（读别人未提交）、不可重复读（同查询两次结果变）、幻读（范围查询多了新行）。\n"
 "2. 读未提交最低（啥都不挡），串行化最高（全加锁，最慢）；读已提交挡脏读，可重复读挡前两者，但标准下幻读仍可能发生。\n"
 "3. 多数库（如 MySQL InnoDB）在「可重复读」用 MVCC + 间隙锁实际也挡住了幻读，比 SQL 标准更严。\n"
 "4. 级别越高越安全但并发越低——要在一致性与吞吐间权衡，不是无脑拉满。\n"
 "5. 边界：长事务会拖住 MVCC 旧版本清理，撑大回滚段；只读事务也建议显式标记降开销。"),

("N+1 查询问题与预加载",
 "思考步骤：\n"
 "1. N+1：先查 N 条主记录，再为每条循环发一条查关联（如作者），共发出 1+N 条 SQL，网络往返爆炸。\n"
 "2. 根因是「对象关系映射的懒加载」——访问关系属性时才触发查询，循环里访问就触发 N 次。\n"
 "3. 解决：一次性 JOIN 或 `IN (ids)` 批量查关联，再在内存里按 id 组装；ORM 叫 eager load / prefetch / selectinload。\n"
 "4. 取舍：JOIN 会重复主表字段（数据冗余），IN 查询分两批但要手动拼装；数据量极大时游标流式更稳。\n"
 "5. 边界：关联很多层时全预加载会拉爆内存，应按需 select 列、分页；用 `EXPLAIN` 验证 SQL 条数。"),

("乐观锁与版本号",
 "思考步骤：\n"
 "1. 并发改同一行：悲观锁（SELECT ... FOR UPDATE）先锁住行，安全但吞吐低、易死锁。\n"
 "2. 乐观锁假设冲突少：更新时带条件 `WHERE id=? AND version=?`，并把 version+1；若影响行数=0 说明期间被别人改了，重试或报错。\n"
 "3. 本质是把「冲突检测」交给数据库原子更新，避免长事务持锁。\n"
 "4. 适用：读多写少、冲突概率低；若冲突频繁，反复重试反而更慢，应换悲观或分片。\n"
 "5. 边界：version 也可用时间戳/状态字段；CAS 失败要有明确重试上限，别无限循环。"),

("布隆过滤器",
 "思考步骤：\n"
 "1. 判断「某元素是否可能在集合中」——存在误判（可能说在，实际不在），但不会漏判（在的必说在）。适合缓存穿透防护。\n"
 "2. 原理：m 位位数组 + k 个独立哈希，插入时把对应 k 个位设 1；查询时 k 位全 1 才算「可能存在」。\n"
 "3. 某个位为 0 就绝对不存在（因为插入必置 1）；全 1 可能是别的元素碰巧把位都置 1 了（假阳性）。\n"
 "4. 调参：位数 m 越大、哈希数 k 越合适，误判率越低，但空间和计算成本上升；有公式按预期误判率反推 m、k。\n"
 "5. 边界：不支持删除（清位会影响别的元素）；需删除用计数布隆（Counting Bloom）；位数组可放 Redis 共享。"),

("用生成器惰性迭代省内存",
 "思考步骤：\n"
 "1. 返回列表会把全部结果一次性 materialize 到内存；生成器 `yield` 是惰性的，每次要一个产一个，内存恒定 O(1)。\n"
 "2. 处理大文件/大序列时这是关键：比如逐行读 10GB 日志，绝不该 `readlines()` 全读。\n"
 "3. 生成器函数遇到 yield 暂停并保留局部状态，下次 next 从暂停处继续——天然支持流式管道。\n"
 "4. 代价：生成器只能遍历一次（耗完即空），要重复用得重新造；且不能下标随机访问。\n"
 "5. 边界：生成器里含 `return value` 会作为 StopIteration.value；异常会穿透到消费方，需在使用处捕获。",
 "def read_batches(path, n):\n"
 "    buf = []\n"
 "    with open(path, encoding='utf-8') as f:\n"
 "        for line in f:\n"
 "            buf.append(line)\n"
 "            if len(buf) == n:\n"
 "                yield buf; buf = []\n"
 "    if buf: yield buf"),

("用 __slots__ 省内存",
 "思考步骤：\n"
 "1. 普通类每个实例有 `__dict__`（哈希表）存属性，几十字节开销；海量小对象时这开销比数据本身还大。\n"
 "2. `__slots__ = ('x','y')` 让 Python 不为实例建 `__dict__`，改用固定大小的数组按槽存属性，省内存且访问略快。\n"
 "3. 限制：声明后不能动态加未列出的属性（没有 `__dict__`），继承时子类也要各自声明 slots 才能继续省。\n"
 "4. 适用：成千上万的同构小对象（如解析后的数据点、树节点）；少量对象没必要折腾。\n"
 "5. 边界：含 `__weakref__`/`__dict__` 需求时要显式加进 slots；多继承下 slots 行为较复杂，需实测。",
 "class Point:\n"
 "    __slots__ = ('x', 'y')\n"
 "    def __init__(self, x, y):\n"
 "        self.x = x; self.y = y"),

("用 property 控制属性访问",
 "思考步骤：\n"
 "1. 直接暴露属性字段易导致「外部随意改、破坏不变式」；property 把方法伪装成属性，读写时插进校验/计算逻辑。\n"
 "2. `@property` 定义 getter，`@x.setter` 定义 setter，setter 里可校验合法性（如温度不能为负）再赋值。\n"
 "3. 对外 API 保持 `obj.temp = 25` 的简洁，内部却有保护——比手写 get_temp/set_temp 更 Pythonic。\n"
 "4. 还可做「惰性计算属性」：首次访问才算并缓存，之后直接返回。\n"
 "5. 边界：property 每次访问都走方法调用（轻微开销）；需要删除行为加 `@x.deleter`；与 `__slots__` 配合需注意。",
 "class Celsius:\n"
 "    def __init__(self, t=0):\n"
 "        self._t = t\n"
 "    @property\n"
 "    def temp(self):\n"
 "        return self._t\n"
 "    @temp.setter\n"
 "    def temp(self, v):\n"
 "        if v < -273.15: raise ValueError('低于绝对零度')\n"
 "        self._t = v"),

("带参数的装饰器",
 "思考步骤：\n"
 "1. 普通装饰器是「函数套函数」；当装饰器自身还要接收参数（如重试次数），就变成三层嵌套：外层收参数、中层收函数、内层是包装。\n"
 "2. 调用顺序：decorator(arg) 先执行返回「真正的装饰器」，再 `()(func)` 包住函数——语法糖 `@decorator(arg)` 自动完成这两步。\n"
 "3. 用 `functools.wraps` 把原函数名/文档拷到 wrapper，否则被装饰后 `__name__` 变成 'wrapper'，日志/调试会乱。\n"
 "4. 参数要在最外层闭包捕获，wrapper 通过自由变量访问，避免每次调用重新解析。\n"
 "5. 边界：类方法装饰器要处理 self（用 `*args` 透传）；装饰器有状态要注意多线程安全。",
 "import functools\n"
 "def retry(times=3):\n"
 "    def deco(fn):\n"
 "        @functools.wraps(fn)\n"
 "        def wrapper(*a, **k):\n"
 "            for i in range(times):\n"
 "                try: return fn(*a, **k)\n"
 "                except Exception as e:\n"
 "                    if i == times-1: raise\n"
 "        return wrapper\n"
 "    return deco"),

("用 Protocol 定义结构化子类",
 "思考步骤：\n"
 "1. 传统抽象基类 ABC 要求显式继承，耦合强；Protocol（静态鸭子类型）只声明「需要哪些方法/属性」，任何长这样的对象都算合规。\n"
 "2. 好处：不侵入原类、跨库对象也能复用接口；类型检查器（mypy/pyright）在调用处做结构匹配，不要求继承关系。\n"
 "3. 与 typing 的 `runtime_checkable` 配合可在运行时 isinstance 检查（仅检查属性/方法存在，不查签名）。\n"
 "4. 取舍：Protocol 是「结构契约」而非强制实现，运行时默认不校验行为正确性，靠测试兜底。\n"
 "5. 边界：方法签名不被 Protocol 强约束（非 runtime_checkable 时连存在性都不查）；复杂多态仍可用 ABC。",
 "from typing import Protocol\n"
 "class Renderable(Protocol):\n"
 "    def render(self) -> str: ...\n"
 "def show(item: Renderable) -> str:\n"
 "    return item.render()"),

("EAFP 与 LBYL 错误处理风格",
 "思考步骤：\n"
 "1. EAFP（请求原谅比请求许可容易）：先做事、出异常再 catch——`try: d[k] except KeyError:`。LBYL（先看再跳）：先 `if k in d:` 再取值。\n"
 "2. Python 社区偏好 EAFP：异常路径正常时几乎零成本，且避免了「检查与执行之间状态被别的线程改掉」的竞态（TOCTOU）。\n"
 "3. 但 LBYL 在「前置条件复杂、异常类型多、或失败很常见且代价高」时更清晰、可读。\n"
 "4. 关键：except 要抓**具体**异常，别裸 `except:` 把真 bug 也吞掉。\n"
 "5. 边界：频繁触发的异常在紧循环里 EAFP 会有性能开销（异常构造贵），这种热点改用 LBYL 或预判。"),

("pandas groupby 聚合",
 "思考步骤：\n"
 "1. groupby 把数据按某键分桶，再对每个桶应用聚合（sum/mean/count）或变换——这是数据分析的核心操作。\n"
 "2. 性能：groupby 在 C 层分组，远快于 Python 手搓 for；聚合尽量用内置函数而非 apply 逐组调用。\n"
 "3. 多列聚合用字典 `{ 'price':'mean', 'qty':'sum' }`，一次算出多指标；agg 还支持多函数。\n"
 "4. 分组后索引变成分组键，用 `reset_index()` 还原成普通列方便后续 join/导出。\n"
 "5. 边界：含 NaN 的组默认被排除（dropna）；大分组内存吃紧时可分块或折到 SQL 端做。",
 "import pandas as pd\n"
 "df = pd.DataFrame({'cat':['a','a','b'], 'v':[1,2,3], 'n':[10,20,30]})\n"
 "out = df.groupby('cat').agg(v_mean=('v','mean'), n_sum=('n','sum')).reset_index()"),

("pivot_table 透视表",
 "思考步骤：\n"
 "1. 透视表把「长表」按行键、列键两个维度展开成矩阵，单元格填聚合值——类似 Excel 透视，适合交叉分析。\n"
 "2. index=行分组、columns=列分组、values=被聚合的字段、aggfunc=聚合方式（默认 mean）。\n"
 "3. 比多层 groupby + unstack 更直观；缺失的组合会变成 NaN（可用 fill_value 填 0）。\n"
 "4. 多值/多聚合时行列都可能出现多层索引（MultiIndex），取值要按层级。\n"
 "5. 边界：组合爆炸（行×列巨大）会撑爆内存，应先筛选维度；大数据用 crosstab 或数据库透视。",
 "import pandas as pd\n"
 "df = pd.DataFrame({'date':['d1','d1','d2'],'city':['x','y','x'],'sales':[100,200,150]})\n"
 "pv = df.pivot_table(index='date', columns='city', values='sales', aggfunc='sum', fill_value=0)"),

("时间序列 resample 重采样",
 "思考步骤：\n"
 "1. 时序数据常需从细粒度（秒）聚到粗粒度（分钟/小时），resample 按时间频率分组再聚合，比手搓 groupby 更贴合时间语义。\n"
 "2. 索引必须是 DatetimeIndex，规则用 '1min'/'1H'/'D' 等频率字符串指定桶宽。\n"
 "3. 聚合同 groupby（ohlc/sum/mean），还支持 `closed='left'`/`label` 控制桶边界归属。\n"
 "4. 升采样（细化）需要 fillna/ffill/插值填补空洞；降采样（粗化）是聚合。\n"
 "5. 边界：时区要统一（naive vs aware 会报错）；不规则时间戳 resample 仍按规则补出空桶。",
 "import pandas as pd\n"
 "idx = pd.date_range('2026-01-01', periods=100, freq='S')\n"
 "s = pd.Series(range(100), index=idx)\n"
 "per_min = s.resample('1min').sum()"),

("用 np.select 替代嵌套 if",
 "思考步骤：\n"
 "1. 多分支按条件赋值（如分数档位）常写一长串 if/elif，既慢又难读；np.select 用「条件列表+对应取值列表」向量化一次性解决。\n"
 "2. 每个元素独立判断所有条件，命中第一个为 True 的分支取值；default 收尾处理全不满足的情况。\n"
 "3. 全程在 C 层对整列运算，比 Python 逐元素 if 快几个数量级，且代码声明式更清晰。\n"
 "4. 条件必须是布尔数组（可用向量化比较生成），不能是标量判断。\n"
 "5. 边界：条件有重叠时「先出现先命中」，顺序即优先级；全覆盖不到的务必给 default，否则填 NaN。",
 "import numpy as np, pandas as pd\n"
 "score = pd.Series([55, 70, 85, 95])\n"
 "cond = [score < 60, score < 80, score < 90]\n"
 "choice = ['不及格', '及格', '良好']\n"
 "level = np.select(cond, choice, default='优秀')"),

("大文件分块读写",
 "思考步骤：\n"
 "1. 几 GB 的文件不能一次性 read() 进内存，否则 OOM；按固定大小/行数分块处理，内存恒定。\n"
 "2. 文本用 `for line in f` 天然惰性逐行；二进制/复杂格式用 `iter(lambda: f.read(CHUNK), b'')` 按字节块迭代。\n"
 "3. 每块处理完立即写出或累加统计，不囤积——处理完即丢弃，峰值内存只等于一块大小。\n"
 "4. 写大文件同理：边算边 append，别先攒成巨大 list 再一次性 dump。\n"
 "5. 边界：块边界可能切断一条记录（如半行），需缓冲跨块拼接逻辑；CSV 可用 pandas chunksize 代劳。",
 "CHUNK = 1 << 20  # 1MB\n"
 "def count_lines(path):\n"
 "    n = 0\n"
 "    with open(path, 'rb') as f:\n"
 "        while True:\n"
 "            buf = f.read(CHUNK)\n"
 "            if not buf: break\n"
 "            n += buf.count(b'\\n')\n"
 "    return n"),

("用 unittest.mock 隔离外部依赖",
 "思考步骤：\n"
 "1. 单测不该真连数据库/发 HTTP，否则慢、不稳定、还污染线上——mock 把依赖替换成可控假对象。\n"
 "2. `@patch('module.ClassName')` 在测试期间把目标替换成 MagicMock，调用返回预设值，并可断言「被调用几次、传了什么参」。\n"
 "3. 关键：patch 的路径是「被测试代码里 import 的那个名字」，不是定义处——常踩坑（patch 错模块等于没 patch）。\n"
 "4. 用 `return_value` 设返回，`side_effect` 可设抛异常或按输入返回不同值，模拟各种分支。\n"
 "5. 边界：mock 过度会让测试脱离真实行为（假绿），关键路径仍要有少量集成测试兜底。",
 "from unittest import mock\n"
 "def test_order(save, send):\n"
 "    with mock.patch('orders.db.insert') as m:\n"
 "        save({'id':1})\n"
 "        m.assert_called_once()"),

("参数化测试进阶",
 "思考步骤：\n"
 "1. 同一逻辑要验证多组输入输出，手写 N 个测试函数冗余且易漏；参数化把「用例表」抽出来驱动同一函数多次跑。\n"
 "2. pytest 用 `@pytest.mark.parametrize('a,b,exp', [(1,2,3),(0,0,0),(-1,1,0)])`，每组独立成一条测试，失败定位精准。\n"
 "3. 进阶：用例可从外部文件（CSV/JSON）加载，用列表推导生成参数，实现「数据驱动测试」。\n"
 "4. ids 参数给每组起名，报告里能看出哪条挂了，而不是冰冷的 [0]/[1]。\n"
 "5. 边界：浮点结果用 pytest.approx 比相等；异常用例用 pytest.raises 配合 match 校验错误信息。",
 "import pytest\n"
 "@pytest.mark.parametrize('a,b,exp', [(1,2,3),(0,0,0),(-1,1,0)], ids=['正','零','负'])\n"
 "def test_add(a, b, exp):\n"
 "    assert add(a, b) == exp"),

("结构化日志",
 "思考步骤：\n"
 "1. 打印 `print('user', uid, 'login')` 的纯文本日志难检索、难解析；结构化日志输出 JSON（含时间/级别/字段），机器友好。\n"
 "2. 用 logging + JSON formatter（或 structlog），每条带固定字段（ts/level/event）加随业务附带的上下文字段（uid/req_id）。\n"
 "3. 关键价值：能按字段过滤聚合（如「统计 ERROR 且 module=pay 的条数」）、串联一次请求的全链路（trace_id）。\n"
 "4. 取舍：开发期人类读 JSON 不如文本直观，可本地用纯文本、生产用 JSON，靠配置切换。\n"
 "5. 边界：别把敏感信息（密码/令牌）打进日志；高频日志要评估磁盘/采集成本。"),

("健康检查端点",
 "思考步骤：\n"
 "1. 服务暴露 /healthz（存活）与 /readyz（就绪）让编排系统（k8s）判断该不该路由流量、该不该重启。\n"
 "2. 存活探针只查进程是否假死（轻量）；就绪探针要查依赖（数据库/缓存/下游）是否可用——依赖挂了就摘流量而非被杀。\n"
 "3. 探针里**不要**做重活（全表扫描），否则探针本身拖垮服务，形成「探针拖慢→超时→被重启」死循环。\n"
 "4. 返回应快速且幂等，状态码 200=健康、5xx=异常；body 可带各依赖状态便于排查。\n"
 "5. 边界：就绪与存活分开，避免「依赖抖动就无限重启」；探针超时/阈值要留缓冲防抖动误杀。"),

("参数化查询防 SQL 注入",
 "思考步骤：\n"
 "1. 字符串拼接 SQL（`f'SELECT * WHERE name={name}'`）是注入温床：攻击者传 `name=' OR '1'='1` 改变语义。\n"
 "2. 参数化查询把 SQL 模板与数据分开传：占位符（? 或 %s 或命名 :name）由驱动做转义/绑定，数据永远不被当 SQL 执行。\n"
 "3. 本质是「代码与数据分离」——数据库收到的是「带参数的语句」，用户输入只是值，即便含引号也破坏不了结构。\n"
 "4. 表名/列名不能参数化（那是标识符），必须白名单校验或映射，不能让用户任意指定。\n"
 "5. 边界：ORM 默认参数化但用了 raw/text 拼接仍危险；LIKE 的 % 要显式绑定而非拼进字符串。",
 "import sqlite3\n"
 "def find_user(cur, name):\n"
 "    return cur.execute('SELECT id FROM users WHERE name = ?', (name,)).fetchall()"),

("密码哈希存储（bcrypt 思路）",
 "思考步骤：\n"
 "1. 绝不能明文存密码，也不能用普通哈希（MD5/SHA）——快且易被彩虹表反查；要用「慢哈希」+「每用户盐」。\n"
 "2. 盐是随机值，和用户密码拼接后再哈希，保证相同密码在不同用户存出不同结果，挡掉预计算表。\n"
 "3. bcrypt/scrypt/argon2 故意设计得慢且可调速（cost 因子），让暴力穷举在算力上不划算。\n"
 "4. 校验时取存储的盐+哈希，对新输入走同样过程比对——不解密、只重算比对，符合「永不恢复明文」原则。\n"
 "5. 边界：盐要足够随机（csprng）；cost 随硬件升级定期调高；绝不要把盐当秘密、也别存明文日志。"),

("速率限制防爆破",
 "思考步骤：\n"
 "1. 登录/短信验证码接口若不限速，攻击者可以无限重试密码或轰炸用户——速率限制是底线防护。\n"
 "2. 维度要选对：按「账号」「IP」「设备」分别或组合限流，单看 IP 易被代理池绕过，单看账号易被同 IP 多点试探。\n"
 "3. 失败次数阈值 + 锁定窗口：如 5 次失败锁 15 分钟；但纯锁定会被利用做「拒绝服务」（锁掉所有合法账号）。\n"
 "4. 更稳：指数退避 + 渐进延长等待，而非硬锁；配合验证码在高频时弹出。\n"
 "5. 边界：限流自身状态要持久化（重启不丢），分布式用共享存储；误伤正常用户要留申诉/人工通道。"),

("训练/验证/测试集划分",
 "思考步骤：\n"
 "1. 模型要在「没见过的数据」上评估才可信，所以拆成训练（学参数）、验证（调超参/早停）、测试（最终報告）三份。\n"
 "2. 测试集必须全程只碰一次，用来报最终指标；拿测试集反复调参等于把测试信息泄进模型，指标会虚高。\n"
 "3. 划分要保证分布一致（随机分层），时间序列数据则不能随机洗牌——必须按时间切，否则未来信息泄漏进训练。\n"
 "4. 数据量少时用交叉验证（K 折）更充分利用；但 CV 成本高，大数据单拆即可。\n"
 "5. 边界：类别不平衡时按 strata 分层抽样，保证各集比例一致；划分前先打乱防原始顺序偏差。"),

("过拟合识别与正则",
 "思考步骤：\n"
 "1. 过拟合 = 训练集误差很低、验证集误差高，模型背下了噪声而非学到规律——泛化差。\n"
 "2. 识别看学习曲线：train 持续降、val 先降后升，拐点即开始过拟合的时刻，对应早停（early stopping）。\n"
 "3. 正则化约束模型复杂度：L2（权重衰减）压小权重、L1 促稀疏；dropout 训练时随机屏蔽神经元强制冗余表达。\n"
 "4. 数据侧：更多数据、数据增强、降特征维度都能缓解；模型侧：减参数量、加约束。\n"
 "5. 边界：正则太强会欠拟合（train 和 val 都差）；要调 λ 平衡偏差-方差，靠验证集选。"),

("类别不平衡处理",
 "思考步骤：\n"
 "1. 正负样本悬殊（如欺诈 0.1%）时，模型倾向全预测多数类也能拿高准确率，但对稀有类毫无召回——准确率成陷阱指标。\n"
 "2. 重采样：少数类过采样（SMOTE 合成近邻点）或多数类欠采样，平衡训练分布；也可两者结合。\n"
 "3. 损失层面：class_weight 给少数类更高权重，或改用 Focal Loss 让模型专注难分样本。\n"
 "4. 评估必须用召回率/PR-AUC/F1 而非准确率；业务上常看「捕获 top-k 高风险」的覆盖。\n"
 "5. 边界：过采样可能放大噪声、欠采样丢信息；SMOTE 不在特征空间乱插值（防生成无意义样本）。"),

("Dijkstra 最短路径（优先队列）",
 "思考步骤：\n"
 "1. 边权非负时求单源最短路，用优先队列（最小堆）每次取「当前距离最小」的节点松弛，保证出堆即最终最短路（贪心正确性来自非负权）。\n"
 "2. dist 数组初始化源=0、其余=inf；heap 存 (距离, 节点)，按距离升序弹出。\n"
 "3. 松弛：对邻接边 `nd = d + w`，若 `nd < dist[v]` 则更新 dist[v] 并把 (nd, v) 压入堆（允许同一节点多次入堆，但只有最小那次有效）。\n"
 "4. 用「出堆时若 d>dist[node] 直接跳过」过滤掉过期的堆条目，避免无效展开。\n"
 "5. 边界：负权边会使贪心失效（需用 Bellman-Ford/SPFA）；稀疏图用邻接表 + 堆 O((V+E)logV)；稠密图用朴素 O(V^2) 更省。",
 "import heapq\n"
 "def dijkstra(graph, src):\n"
 "    dist = {src: 0}; pq = [(0, src)]\n"
 "    while pq:\n"
 "        d, u = heapq.heappop(pq)\n"
 "        if d > dist.get(u, float('inf')): continue\n"
 "        for v, w in graph[u]:\n"
 "            nd = d + w\n"
 "            if nd < dist.get(v, float('inf')):\n"
 "                dist[v] = nd\n"
 "                heapq.heappush(pq, (nd, v))\n"
 "    return dist"),
]

def main():
    out = []
    seen = set()
    for entry in BANK:
        if len(entry) == 3:
            instr, cot, code = entry
            output = cot + "\n\n实现：\n" + code
        else:
            instr, cot = entry
            output = cot
        assert instr not in seen, f"重复 instruction: {instr}"
        seen.add(instr)
        out.append({
            "instruction": instr,
            "output": output,
            "_type": "reason_cot",
            "_universe": "代码",
            "_src": "v7_selftaught(teacher:self)",
        })
    with open(OUT, "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"第二扩展库 -> {OUT}  共 {len(out)} 条")

if __name__ == "__main__":
    main()
