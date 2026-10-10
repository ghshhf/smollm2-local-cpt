"""把编程卡 card_code.jsonl(1463 条) 拆成 4 张更细的专卡。

拆分的依据是**来源性质**而不是主观难度（难度不可复核，来源可复核）：

  1. card_code_lang    语言与库问答      _type=code_distill(521) + 泛用 code(148) 里不属量化/工程的
  2. card_code_quant   量化与交易          backtest/auto_trader/rebalance/选股/净值/因子…
  3. card_code_data    数据获取与处理      data_sources/market_data/fetch/parser/panel 拉取重建
  4. card_code_engine  工程能力            并发/数据库/网络/安全/测试/构建/脚本工具

每条只进一张卡（互斥），顺序即优先级：量化 -> 数据 -> 工程 -> 语言(兜底)。
去重沿用 build_cards_by_domain 的口径：instruction 相同只留一条。

用法：python build_code_subcards.py
输出：data/card_code_lang.jsonl / _quant / _data / _engine
"""
import json
import os
import re
import collections

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "card_code.jsonl")

# ---- 分类规则：按顺序匹配，第一个命中即归属 ----------------------------
# 1) 量化/交易：回测、策略、选股、净值、权重、行情、因子 —— 金融领域知识
QUANT = re.compile(
    r'回测|净值|nav|rebalance|再平衡|选股|策略|因子|权重|涨幅|收益率|持仓|仓位|'
    r'择时|轮动|轮换|买入持有|buyhold|hold|信号|指标|夏普|sharpe|回撤|drawdown|'
    r'最大回撤|年化|波动|止损|止盈|指数增强|对冲|套利|网格|定投|'
    r'bt_|backtest|_bt|stock|crypto|coin|btc|eth|us_|ashare|etf|'
    r'市场|行情|指数|标的|组合|blend|exposure|leverage|slippage|成本模型',
    re.I)

# 2) 数据获取与处理：抓取、清洗、存储、面板重建
DATA = re.compile(
    r'数据源|抓取|拉取|下载|获取行情|采集|爬|爬虫|api|接口|请求|重试|缓存|'
    r'data_?source|market_?data|fetch|parser|解析|清洗|清洗数据|预处理|'
    r'panel|面板|数据契约|contract|store|存储|落盘|写入|csv|jsonl|parquet|'
    r'数据库|sql|schema|表结构|迁移|索引',
    re.I)

# 3) 工程能力：并发、测试、安全、构建、脚本工具链
ENGINE = re.compile(
    r'并发|线程|进程|锁|死锁|原子|队列|线程池|协程|async|await|GIL|'
    r'测试|test_|pytest|unit|单元测试|覆盖率|mock|stub|'
    r'安全|加密|注入|越权|xss|csrf|鉴权|密钥|脱敏|'
    r'构建|打包|部署|依赖|虚拟环境|pip|编译|docker|ci/cd|日志|监控|异常|'
    r'内存|性能|优化|基准|profil|瓶颈|批处理|流式|零拷贝|'
    r'脚本|工具|命令行|cli|配置|config|文档|注释|代码审查|重构|命名|'
    r'selfcheck|selftest|selfcheck|verify|校验|自检|run_|_v2|版本',
    re.I)

# 4) 语言与库问答：题面点名某个**具体语言/库/工具**，或纯前端/JS 概念（特征最硬）
# 🔴 两个坑（都已实测复现，别再加回来）：
#   ① 不能放「如何/怎么/为什么」等泛问句式 —— 几乎所有实现题都是"用 Python 实现：…"开头，
#      放进来会让 1463 条全吸进lang。
#   ② 不能放裸 `python` —— "用 Python 实现"是本卡的**实现语言**而非考点，
#      留着同样会全吸。这里只留"语言/库作为考点"的强特征。
LANG = re.compile(
    r'javascript|typescript|java\b|rust\b|golang|pandas|numpy|react|vue|'
    r'jquery|node|正则|regex|cookie|localstorage|sessionstorage|promise|'
    r'闭包|装饰器|生成器|迭代器|原型链|类型守卫|泛型|'
    r'docker|git|traceback|logging|html|css\b',
    re.I)


# 判定顺序（三段，每段依据都不同，从硬到软）：
#   ① _type == code_distill  —— 这是**确定性的题型标记**，整条都是语言/库问答，
#      答案里即使出现"数据""回测"字样也不该改判（它是"pandas 怎么落盘"而不是数据工程本身）。
#   ② 领域双证据（题面+答案前段）—— quant / data / engine。
#   ③ 题面点名语言/库 -> lang；都不中则 lang 兜底。
# 为什么必须双证据：很多实现题题面只有函数名(如 `cmd_check`)，只看题面判不出领域。
def classify(row):
    instr = row.get("instruction", "")
    # ① 确定性题型标记优先
    if row.get("_type") == "code_distill":
        return "lang"
    # ② 领域双证据
    text = instr + "\n" + row.get("output", "")[:900]
    if QUANT.search(text):
        return "quant"
    if DATA.search(text):
        return "data"
    if ENGINE.search(text):
        return "engine"
    # ③ 题面点名语言/库/前端概念
    if LANG.search(instr):
        return "lang"
    return "lang"


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    buckets = collections.defaultdict(list)
    seen = set()
    dup = 0
    for r in rows:
        key = r["instruction"].strip()
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        buckets[classify(r)].append(r)

    names = {"lang": "card_code_lang", "quant": "card_code_quant",
             "data": "card_code_data", "engine": "card_code_engine"}
    print(f"源数据 {len(rows)} 条，去重 -{dup} 条后 {len(seen)} 条")
    total = 0
    for k, out in names.items():
        b = buckets.get(k, [])
        total += len(b)
        path = os.path.join(ROOT, "data", out + ".jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in b:
                r["_subcard"] = k
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {out:20s} {len(b):5d} 条 -> {path}")
    print(f"  合计 {total} 条（应等于去重后 {len(seen)} 条）")

    # 每张卡的来源 top5，便于人工核对分类是否合理
    for k, out in names.items():
        b = buckets.get(k, [])
        c = collections.Counter(r.get("_src", "?") for r in b)
        top = ", ".join(f"{a}({n})" for a, n in c.most_common(5))
        print(f"\n  [{out}] 主要来源: {top}")


if __name__ == "__main__":
    main()