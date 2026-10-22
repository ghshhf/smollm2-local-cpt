# -*- coding: utf-8 -*-
"""代码种子 第 3 批（101-150）：字符串 -> 数据结构 -> 算法 -> 数学。

每条 = (instruction, code, note)
硬约束同 p1/p2：独立可执行、只用标准库、有 stdout、结果确定、不与 holdout 同题。
"""

S = []


def add(instruction, code, note):
    S.append((instruction, code.strip('\n'), note))


# ---------- 字符串处理（101-112） ----------
add('写一段 Python 代码，统计一句话里每个单词出现的次数。', '''
from collections import Counter

def word_freq(text):
    return Counter(text.split())

print(word_freq("the cat and the dog and the bird"))
''', 'split 按空白切词，Counter 直接统计频次。')

add('写一段 Python 代码，对列表去重并保持原来的先后顺序。', '''
def dedup(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

print(dedup([3, 1, 3, 2, 1, 5]))
''', '用集合记录见过的元素，列表保持插入顺序。')

add('写一段 Python 代码，把一句话里每个单词的首字母改成大写。', '''
def title_case(s):
    return " ".join(w[:1].upper() + w[1:] for w in s.split())

print(title_case("hello polaris world"))
''', '逐词取首字符转大写，其余保持不变。')

add('写一段 Python 代码，把驼峰命名转换成下划线命名。', '''
def camel_to_snake(s):
    out = []
    for ch in s:
        if ch.isupper():
            out.append("_")
            out.append(ch.lower())
        else:
            out.append(ch)
    return "".join(out).lstrip("_")

print(camel_to_snake("camelCaseName"))
''', '遇到大写字母先补下划线再转小写。')

add('写一段 Python 代码，判断一句话忽略大小写和标点后是不是回文。', '''
def is_palindrome(s):
    t = [c.lower() for c in s if c.isalnum()]
    return t == t[::-1]

print(is_palindrome("A man, a plan, a canal: Panama"))
''', '只保留字母数字再比较正序与倒序。')

add('写一段 Python 代码，求一组单词的最长公共前缀。', '''
def common_prefix(words):
    if not words:
        return ""
    base = words[0]
    for i, ch in enumerate(base):
        for w in words[1:]:
            if i >= len(w) or w[i] != ch:
                return base[:i]
    return base

print(common_prefix(["flower", "flow", "flight"]))
''', '以第一个词为基准逐列比较，第一处不同就截断。')

add('写一段 Python 代码，计算两个单词之间的编辑距离。', '''
def edit_distance(a, b):
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[-1]

print(edit_distance("kitten", "sitting"))
''', '动态规划：增删改三种操作取最小，滚动数组省空间。')

add('写一段 Python 代码，检查一串括号是否配对完整。', '''
def brackets_ok(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack

print(brackets_ok("{[()()]}"), brackets_ok("([)]"))
''', '左括号入栈，右括号弹栈比对，最后栈空才算配对。')

add('写一段 Python 代码，把连续重复字符压缩成字符加次数。', '''
def compress(s):
    if not s:
        return ""
    out = []
    count = 1
    for i in range(1, len(s)):
        if s[i] == s[i - 1]:
            count += 1
        else:
            out.append(s[i - 1] + str(count))
            count = 1
    out.append(s[-1] + str(count))
    return "".join(out)

print(compress("aabbbcc"))
''', '边扫描边累计重复次数，遇到变化就落一段。')

add('写一段 Python 代码，把字符串里的大小写互换。', '''
def swap_case(s):
    return "".join(c.lower() if c.isupper() else c.upper() for c in s)

print(swap_case("Hello Polaris"))
''', '逐字符判断当前字母大小写再取反。')

add('写一段 Python 代码，返回字符串里每个字符出现的所有位置。', '''
def char_positions(s):
    pos = {}
    for i, ch in enumerate(s):
        pos.setdefault(ch, []).append(i)
    return pos

print(char_positions("banana"))
''', '字典的值用列表存下标，setdefault 省一次判断。')

add('写一段 Python 代码，把模板里的占位符替换成给定值。', '''
def render(template, values):
    for k, v in values.items():
        template = template.replace("{" + k + "}", str(v))
    return template

print(render("你好 {name}，今日温度 {temp} 度", {"name": "主人", "temp": 26}))
''', '占位符用花括号，循环替换即可。')

# ---------- 数据结构（113-126） ----------
add('写一段 Python 代码，用列表实现一个栈并演示进出栈。', '''
class Stack:
    def __init__(self):
        self.items = []

    def push(self, x):
        self.items.append(x)

    def pop(self):
        return self.items.pop()

s = Stack()
for x in [1, 2, 3]:
    s.push(x)
print(s.pop(), s.pop(), s.items)
''', '栈后进先出，用列表尾部当栈顶。')

add('写一段 Python 代码，用双端队列实现一个先进先出的队列。', '''
from collections import deque

q = deque()
for x in ["a", "b", "c"]:
    q.append(x)
print(q.popleft(), q.popleft(), list(q))
''', 'deque 两头操作都是 O(1)，比列表 pop(0) 快。')

add('写一段 Python 代码，把一条单链表反转后按顺序打印。', '''
class Node:
    def __init__(self, val, nxt=None):
        self.val = val
        self.nxt = nxt

def reverse(head):
    prev = None
    while head:
        head.nxt, prev, head = prev, head, head.nxt
    return prev

n = Node(1, Node(2, Node(3)))
r = reverse(n)
vals = []
while r:
    vals.append(r.val)
    r = r.nxt
print(vals)
''', '三指针就地反转，不新建节点。')

add('写一段 Python 代码，对一棵二叉树做中序遍历。', '''
class TreeNode:
    def __init__(self, val, left=None, right=None):
        self.val, self.left, self.right = val, left, right

def inorder(node, out):
    if node:
        inorder(node.left, out)
        out.append(node.val)
        inorder(node.right, out)

root = TreeNode(2, TreeNode(1), TreeNode(3))
out = []
inorder(root, out)
print(out)
''', '左-根-右递归，二叉搜索树会得到有序序列。')

add('写一段 Python 代码，从一个列表里取出最大的三个数。', '''
import heapq

def top_k(nums, k):
    return heapq.nlargest(k, nums)

print(top_k([5, 1, 9, 3, 7, 2], 3))
''', 'heapq.nlargest 内部维护小顶堆，比全排序省。')

add('写一段 Python 代码，实现一个固定容量的 LRU 缓存。', '''
from collections import OrderedDict

class LRU:
    def __init__(self, cap):
        self.cap = cap
        self.d = OrderedDict()

    def get(self, k):
        if k not in self.d:
            return -1
        self.d.move_to_end(k)
        return self.d[k]

    def put(self, k, v):
        if k in self.d:
            self.d.move_to_end(k)
        self.d[k] = v
        if len(self.d) > self.cap:
            self.d.popitem(last=False)

lru = LRU(2)
lru.put("a", 1)
lru.put("b", 2)
lru.get("a")
lru.put("c", 3)
print(list(lru.d.keys()), lru.get("b"))
''', 'OrderedDict 记录访问顺序，超容就淘汰最久未用的。')

add('写一段 Python 代码，用并查集统计有多少个连通分量。', '''
class DSU:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra

dsu = DSU(5)
dsu.union(0, 1)
dsu.union(3, 4)
print(len({dsu.find(i) for i in range(5)}))
''', '路径压缩让查找接近常数，根节点个数就是连通分量数。')

add('写一段 Python 代码，把两个字典按相同键的值相加合并。', '''
def merge_sum(a, b):
    out = dict(a)
    for k, v in b.items():
        out[k] = out.get(k, 0) + v
    return out

print(merge_sum({"x": 1, "y": 2}, {"y": 3, "z": 4}))
''', '同名键累加，新键直接并入。')

add('写一段 Python 代码，把多个有序列表合并成一个有序列表。', '''
import heapq

def merge_k(lists):
    h = []
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(h, (lst[0], i, 0))
    out = []
    while h:
        v, i, j = heapq.heappop(h)
        out.append(v)
        if j + 1 < len(lists[i]):
            heapq.heappush(h, (lists[i][j + 1], i, j + 1))
    return out

print(merge_k([[1, 4, 7], [2, 5], [3, 6, 9]]))
''', '堆里存每个列表当前队首，弹出后补下一个。')

add('写一段 Python 代码，用双向链表依次追加节点并从尾部反向打印。', '''
class DNode:
    def __init__(self, val):
        self.val = val
        self.prev = None
        self.next = None

class DList:
    def __init__(self):
        self.head = None
        self.tail = None

    def append(self, val):
        node = DNode(val)
        if self.tail is None:
            self.head = self.tail = node
        else:
            node.prev = self.tail
            self.tail.next = node
            self.tail = node

dl = DList()
for v in [1, 2, 3]:
    dl.append(v)
out = []
cur = dl.tail
while cur:
    out.append(cur.val)
    cur = cur.prev
print(out)
''', '维护头尾指针，双向链表可以反向遍历。')

add('写一段 Python 代码，用字典树判断若干前缀是否存在。', '''
class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, word):
        node = self.root
        for ch in word:
            node = node.setdefault(ch, {})
        node["#"] = True

    def starts(self, prefix):
        node = self.root
        for ch in prefix:
            if ch not in node:
                return False
            node = node[ch]
        return True

t = Trie()
for w in ["cat", "car", "dog"]:
    t.insert(w)
print(t.starts("ca"), t.starts("do"), t.starts("bat"))
''', '每层字典代表一个字符，前缀存在就说明有词以此开头。')

add('写一段 Python 代码，实现一个固定长度的环形缓冲区。', '''
class Ring:
    def __init__(self, n):
        self.n = n
        self.buf = [None] * n
        self.i = 0

    def write(self, x):
        self.buf[self.i] = x
        self.i = (self.i + 1) % self.n

r = Ring(3)
for x in [1, 2, 3, 4, 5]:
    r.write(x)
print(r.buf)
''', '写指针取模回绕，旧数据被自然覆盖。')

add('写一段 Python 代码，演示两个集合的交集、并集和差集。', '''
a = {1, 2, 3, 4}
b = {3, 4, 5}
print(sorted(a & b), sorted(a | b), sorted(a - b))
''', '集合运算符直接给出交并差，排序只为输出稳定。')

add('写一段 Python 代码，统计一句话里出现次数最多的两个词。', '''
from collections import Counter

def top_words(text, k):
    return Counter(text.split()).most_common(k)

print(top_words("a b a c b a", 2))
''', 'most_common 按频次排序取前 k。')

# ---------- 算法（127-138） ----------
add('写一段 Python 代码，用快速排序把列表排成升序。', '''
def quick_sort(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    mid = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + mid + quick_sort(right)

print(quick_sort([5, 2, 9, 1, 7]))
''', '按基准值三路划分再递归，等于基准的放中间。')

add('写一段 Python 代码，用归并排序把列表排成升序。', '''
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    a, b = merge_sort(arr[:mid]), merge_sort(arr[mid:])
    out = []
    i = j = 0
    while i < len(a) and j < len(b):
        if a[i] <= b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    return out + a[i:] + b[j:]

print(merge_sort([8, 3, 5, 1, 9, 2]))
''', '分治到单元素再两两归并，稳定排序。')

add('写一段 Python 代码，在有序列表里找第一个不小于目标值的位置。', '''
def lower_bound(arr, target):
    lo, hi = 0, len(arr)
    while lo < hi:
        mid = (lo + hi) // 2
        if arr[mid] < target:
            lo = mid + 1
        else:
            hi = mid
    return lo

print(lower_bound([1, 3, 3, 5, 7], 3))
''', '左闭右开二分，返回插入位置而不是是否命中。')

add('写一段 Python 代码，用广度优先搜索求网格中两点的最短步数。', '''
from collections import deque

def bfs(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    q = deque([(start, 0)])
    seen = {start}
    while q:
        (r, c), d = q.popleft()
        if (r, c) == goal:
            return d
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] == 0 and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append(((nr, nc), d + 1))
    return -1

print(bfs([[0, 0, 0], [0, 1, 0], [0, 0, 0]], (0, 0), (2, 2)))
''', 'BFS 逐层扩展，第一次到达终点就是最短路。')

add('写一段 Python 代码，统计 0/1 网格里有多少个连通块。', '''
def count_islands(grid):
    rows, cols = len(grid), len(grid[0])
    seen = set()
    count = 0

    def dfs(r, c):
        stack = [(r, c)]
        while stack:
            x, y = stack.pop()
            if (x, y) in seen or not (0 <= x < rows and 0 <= y < cols) or grid[x][y] != 1:
                continue
            seen.add((x, y))
            stack.extend([(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)])

    for i in range(rows):
        for j in range(cols):
            if grid[i][j] == 1 and (i, j) not in seen:
                count += 1
                dfs(i, j)
    return count

print(count_islands([[1, 1, 0], [0, 1, 0], [0, 0, 1]]))
''', '每个未访问的 1 触发一次 DFS，次数就是连通块数。')

add('写一段 Python 代码，用 Dijkstra 算法求单源最短路径。', '''
import heapq

def dijkstra(graph, src):
    dist = {n: float("inf") for n in graph}
    dist[src] = 0
    pq = [(0, src)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]:
            continue
        for v, w in graph[u]:
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                heapq.heappush(pq, (nd, v))
    return dist

g = {"A": [("B", 1), ("C", 4)], "B": [("C", 2), ("D", 5)], "C": [("D", 1)], "D": []}
print(dijkstra(g, "A"))
''', '贪心取当前最近点松弛邻居，非负权才成立。')

add('写一段 Python 代码，用动态规划解 0-1 背包问题。', '''
def knapsack(weights, values, cap):
    dp = [0] * (cap + 1)
    for w, v in zip(weights, values):
        for c in range(cap, w - 1, -1):
            dp[c] = max(dp[c], dp[c - w] + v)
    return dp[cap]

print(knapsack([2, 3, 4], [3, 4, 5], 5))
''', '容量从大到小更新，保证每件物品只用一次。')

add('写一段 Python 代码，求两个字符串的最长公共子序列长度。', '''
def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            if a[i - 1] == b[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]

print(lcs("ABCBDAB", "BDCABA"))
''', '二维表格递推，字符相等就取左上角加一。')

add('写一段 Python 代码，求一个序列的最长递增子序列长度。', '''
import bisect

def lis(nums):
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)

print(lis([10, 9, 2, 5, 3, 7, 101, 18]))
''', '维护每个长度对应的最小结尾，二分替换，O(n log n)。')

add('写一段 Python 代码，用贪心算法把金额换成最少的纸币张数。', '''
def change(amount, coins):
    out = {}
    for c in sorted(coins, reverse=True):
        n, amount = divmod(amount, c)
        if n:
            out[c] = n
    return out, amount

print(change(63, [1, 5, 10, 20, 50]))
''', '从大面额开始尽量换，余数继续往下走。')

add('写一段 Python 代码，用回溯输出三个数字的全部排列。', '''
def permutations(nums):
    out = []

    def backtrack(path, rest):
        if not rest:
            out.append(path)
            return
        for i, x in enumerate(rest):
            backtrack(path + [x], rest[:i] + rest[i + 1:])

    backtrack([], nums)
    return out

print(permutations([1, 2, 3]))
''', '每次从剩余元素里选一个，选完即回溯。')

add('写一段 Python 代码，对有向无环图做拓扑排序。', '''
from collections import deque

def topo(n, edges):
    indeg = [0] * n
    adj = [[] for _ in range(n)]
    for a, b in edges:
        adj[a].append(b)
        indeg[b] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else None

print(topo(4, [(0, 1), (0, 2), (1, 3), (2, 3)]))
''', '不断取出入度为零的点，输出长度等于点数才无环。')

# ---------- 数学与数值（139-150） ----------
add('写一段 Python 代码，用筛法列出 30 以内的所有质数。', '''
def sieve(n):
    is_p = [True] * (n + 1)
    is_p[0] = is_p[1] = False
    for i in range(2, int(n ** 0.5) + 1):
        if is_p[i]:
            for j in range(i * i, n + 1, i):
                is_p[j] = False
    return [i for i, v in enumerate(is_p) if v]

print(sieve(30))
''', '从 2 开始把倍数划掉，剩下的就是质数。')

add('写一段 Python 代码，同时求两个数的最大公约数和最小公倍数。', '''
import math

def gcd_lcm(a, b):
    g = math.gcd(a, b)
    return g, a * b // g

print(gcd_lcm(12, 18))
''', '最小公倍数等于两数之积除以最大公约数。')

add('写一段 Python 代码，用快速幂求 2 的 100 次方对 1000 取模。', '''
def power_mod(base, exp, mod):
    result = 1
    base %= mod
    while exp:
        if exp & 1:
            result = result * base % mod
        base = base * base % mod
        exp >>= 1
    return result

print(power_mod(2, 100, 1000))
''', '指数二进制拆解，每步取模防溢出。')

add('写一段 Python 代码，计算 20 的阶乘并打印。', '''
def factorial(n):
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out

print(factorial(20))
''', 'Python 整数无上限，大阶乘也能精确算出。')

add('写一段 Python 代码，计算两个 2x2 矩阵的乘积。', '''
def matmul(a, b):
    return [[sum(a[i][k] * b[k][j] for k in range(len(b))) for j in range(len(b[0]))]
            for i in range(len(a))]

print(matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]]))
''', '三重循环按定义相乘，外层固定结果行。')

add('写一段 Python 代码，用梯形法计算 x 平方在 0 到 3 上的定积分。', '''
def trapezoid(f, a, b, n):
    h = (b - a) / n
    s = 0.5 * (f(a) + f(b))
    for i in range(1, n):
        s += f(a + i * h)
    return s * h

print(round(trapezoid(lambda x: x * x, 0, 3, 1000), 4))
''', '把区间切细，用梯形面积近似曲线下面积。')

add('写一段 Python 代码，用辛普森法计算 1/x 在 1 到 2 上的定积分。', '''
def simpson(f, a, b, n):
    if n % 2:
        n += 1
    h = (b - a) / n
    s = f(a) + f(b)
    for i in range(1, n):
        s += (4 if i % 2 else 2) * f(a + i * h)
    return s * h / 3

print(round(simpson(lambda x: 1 / x, 1, 2, 100), 6))
''', '偶数段抛物线拟合，精度高于梯形法。')

add('写一段 Python 代码，用最小二乘拟合直线并给出斜率和截距。', '''
def linreg(xs, ys):
    n = len(xs)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = sum((x - mx) ** 2 for x in xs)
    k = num / den
    return round(k, 4), round(my - k * mx, 4)

print(linreg([1, 2, 3, 4], [2, 4, 6, 8]))
''', '斜率等于协方差除以方差，截距过均值点。')

add('写一段 Python 代码，计算一组数的均值、方差和标准差。', '''
def stats(nums):
    n = len(nums)
    mean = sum(nums) / n
    var = sum((x - mean) ** 2 for x in nums) / n
    return round(mean, 3), round(var, 3), round(var ** 0.5, 3)

print(stats([2, 4, 4, 4, 5, 5, 7, 9]))
''', '方差用总体口径，标准差是方差开方。')

add('写一段 Python 代码，用线性插值求一组数的中位数和四分位数。', '''
def quantile(nums, q):
    s = sorted(nums)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return round(s[lo] + (s[hi] - s[lo]) * frac, 3)

nums = [1, 2, 3, 4, 5, 6, 7, 8]
print(quantile(nums, 0.5), quantile(nums, 0.25), quantile(nums, 0.75))
''', '分位数位置线性插值，中位数就是 0.5 分位。')

add('写一段 Python 代码，计算从 10 个里取 3 个的组合数。', '''
def comb(n, k):
    if k > n - k:
        k = n - k
    out = 1
    for i in range(k):
        out = out * (n - i) // (i + 1)
    return out

print(comb(10, 3), comb(20, 10))
''', '连乘连除用整除，避免浮点误差。')

add('写一段 Python 代码，按年复利计算一万元十年后的本息。', '''
def compound(principal, rate, years):
    for _ in range(years):
        principal *= 1 + rate
    return round(principal, 2)

print(compound(10000, 0.05, 10))
''', '每年本金乘以 1 加利率，逐年滚存。')
