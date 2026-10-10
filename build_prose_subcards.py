"""把文风卡 card_prose.jsonl(470 条) 按**题材**拆成 4 张卡。

为什么按题材而不是按作品：12 部作品切 12 张卡属于过度切割（每张 20 条太薄、
每张都要单独加载 3GB 基座）。按题材聚类后每张 100~170 条，够撑起一个题材的表达习惯。

用户口径：「与其说是文风，不如说是文章的内容」—— 数据也证实了：12 部的题材差异
远大于叙述腔调差异，所以按题材切才是切在真正起作用的地方。

分组（题材聚类，来自实际开头抽样核对，不是按名字猜）：
  suspense  恐怖 / 怪谈 / 悬疑 —— 灯塔守望人、隔离第七天、记忆当铺、邻居的猫、夜班守则、十三楼案件
  classic   古典仙侠 / 武侠    —— 寒山剑出、琼明神女录
  scifi     近未来 / 赛博 / 废土 —— 最后充电站、长安归路
  urban     都市日常 / 治愈      —— 咖啡两座
  adult     成人向同人（题材隔离，单独成卡避免污染通用题材）—— 催眠眼镜同人
  12 条风格指令（prose_style）并入 urban，作为通用叙事能力的底子

用法：python build_prose_subcards.py
输出：data/card_prose_{suspense,classic,scifi,urban,adult}.jsonl
"""
import json
import os
import re
import collections

ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "data", "card_prose.jsonl")

# 作品 -> 题材。注意键要与 card_prose.jsonl 里的 _universe 完全一致。
GROUP = {
    # 恐怖 / 怪谈 / 悬疑
    "dengta_shouyeren": "suspense",   # 灯塔守望人（深海克苏鲁）
    "geli_diqitian": "suspense",      # 隔离第七天（规则怪谈）
    "jiyi_dangpu": "suspense",        # 记忆当铺（都市怪谈）
    "linju_de_mao": "suspense",       # 邻居的猫（都市怪谈）
    "yeban_shouze": "suspense",       # 夜班守则（规则怪谈）
    "shisanlou_anjian": "suspense",   # 十三楼案件（悬疑）
    # 古典仙侠 / 武侠
    "jianchu_hanshan": "classic",     # 寒山剑出
    "琼明神女录": "classic",
    "qiongming_shenlvl": "classic",
    # 近未来 / 赛博 / 废土
    "zuihou_chongdianzhan": "scifi",  # 最后充电站（电网崩溃/逆变器）
    # 唐代背景悬疑（县衙/户卷/渠水渗点）—— 按实际开头看是悬疑账册推理，不是科幻
    "changan_guilu": "suspense",
    # 都市日常 / 治愈 / 近未来 —— scifi(22) 与 urban(20) 各自太薄，合并成一张通用叙事卡。
    # 合并依据：两者的叙事节奏都是"克制、白描、日常质感"，是同一类能力而不是两种题材。
    "kafei_liangzuo": "urban",        # 咖啡两座
    "zuihou_chongdianzhan": "urban",  # 最后充电站（原 scifi：仅 22 条，并入）
    # 成人向同人：单独隔离
    "催眠眼镜同人": "adult",
    "催眠眼镜": "adult",
}

# _universe 里没登记的作品走题材特征兜底
FALLBACK = [
    (re.compile(r'仙侠|修仙|门派|师尊|江湖|武林|剑法|丹炉|仙'), "classic"),
    (re.compile(r'恐怖|诡异|复苏|规则|夜班|深海|灯塔|克苏鲁|怪谈|失踪|尸体'), "suspense"),
    (re.compile(r'赛博|义体|机械|电网|充电站|废土|未来|终端|服务器|数据'), "scifi"),
    (re.compile(r'公寓|咖啡|邻居|小区|办公室|日常|治愈'), "urban"),
]


def classify(row):
    uni = row.get("_universe", "")
    if uni in GROUP:
        return GROUP[uni]
    # 风格指令（prose_style）作通用叙事能力的底子
    if row.get("_type") == "prose_style":
        return "urban"
    body = (row.get("output", "") or "")[:600]
    for pat, g in FALLBACK:
        if pat.search(body):
            return g
    return "urban"


NAMES = {"suspense": "card_prose_suspense", "classic": "card_prose_classic",
         "urban": "card_prose_urban", "adult": "card_prose_adult"}


def main():
    rows = [json.loads(l) for l in open(SRC, encoding="utf-8")]
    buckets = collections.defaultdict(list)
    seen, dup = set(), 0
    for r in rows:
        k = r["instruction"].strip()
        if k in seen:
            dup += 1
            continue
        seen.add(k)
        buckets[classify(r)].append(r)

    print(f"源数据 {len(rows)} 条，去重 -{dup} 条后 {len(seen)} 条")
    total = 0
    for k, out in NAMES.items():
        b = buckets.get(k, [])
        total += len(b)
        path = os.path.join(ROOT, "data", out + ".jsonl")
        with open(path, "w", encoding="utf-8") as f:
            for r in b:
                r["_genre"] = k
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {out:22s} {len(b):5d} 条")
    print(f"  合计 {total} 条（应等于 {len(seen)}）")

    # 核对有没有作品没被登记到 GROUP（防漏）
    allu = {r.get("_universe", "") for r in rows}
    miss = [u for u in allu if u not in GROUP and u != ""]
    if miss:
        print(f"\n  ⚠️ 未登记到 GROUP 的 _universe（走了兜底规则）: {miss}")


if __name__ == "__main__":
    main()