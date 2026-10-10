#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
对 R2 小说语料 (zh_text_r2.jsonl, 37721 篇, 无 src 字段) 按内容关键词做题材分类,
输出 zh_text_r2_cat.jsonl: 文本保持原样(训练脚本只读 text), 仅追加 cat(题材) 字段。
分类是元数据, 切到该文件对正在运行的 R2 无缝(文本不变, 仅下次 chunk 重启时换语料路径)。
用法: python classify_r2_novels.py
"""
import os, re, json, hashlib, collections

ROOT = "D:/work/_local_ai/train_15b"
PC = os.path.join(ROOT, "pretrain_corpus")
SRC = os.path.join(PC, "zh_text_r2.jsonl")
OUT = os.path.join(PC, "zh_text_r2_cat.jsonl")

# 题材 -> 关键词(在正文前 800 字 + 末 200 字里计分). 现代网文用语优先, 顺序靠前的更具体.
GENRES = {
    "历史穿越": ["穿越", "古代", "皇帝", "王朝", "将军", "三国", "大唐", "大秦", "妃", "权臣", "诸侯", "御前", "皇宫", "太子", "公主", "重生古代", "冷宫", "丞相", "将军府"],
    "仙侠武侠": ["修仙", "修真", "剑修", "宗门", "武林", "内功", "灵根", "渡劫", "法宝", "江湖", "侠客", "真气", "金丹", "元婴", "飞升", "仙门", "道侣"],
    "都市言情": ["总裁", "豪门", "老公", "老婆", "恋爱", "校园", "秘书", "网红", "闪婚", "甜宠", "霸道", "花都", "赘婿", "千金", "军", "少将", "书记", "局长", "娱乐圈", "明星", "女神", "妈咪", "宝宝", "婚后", "集团", "公司", "都市", "医院", "总裁夫人", "影帝", "顶流"],
    "科幻末世": ["星球", "宇宙", "飞船", "外星", "末世", "丧尸", "病毒", "机甲", "星际", "文明", "曲率", "虫洞", "量子", "克隆", "未来", "基地", "废土", "蓝星", "变异"],
    "游戏网游": ["游戏", "玩家", "副本", "NPC", "公会", "装备", "等级", "网游", "全息", "技能书", "BOSS", "竞技场", "开服", "电竞", "主播", "段位"],
    "悬疑灵异": ["诡异", "鬼", "灵异", "侦探", "案件", "悬疑", "恐怖", "阴间", "诡秘", "凶宅", "探案", "罪案", "失踪", "诡计", "法医"],
    "玄幻奇幻": ["玄幻", "魔法", "精灵", "巨龙", "斗气", "武魂", "异界", "召唤", "位面", "魔兽", "兽人", "矮人", "修炼", "大陆", "王座", "魔", "妖", "血脉", "进化", "怪物", "世界"],
    "系统异能": ["系统", "面板", "属性", "技能", "任务", "抽奖", "兑换", "空间", "异能", "觉醒", "天赋", "金手指", "复活", "升级", "数据", "加点", "副本", "宿主", "积分"],
}
DEFAULT = "其他"

def classify(text):
    head = text[:800]
    tail = text[-200:]
    window = head + tail
    best, best_score = DEFAULT, 0
    for g, kws in GENRES.items():
        s = sum(window.count(k) for k in kws)
        if s > best_score:
            best, best_score = g, s
    return best

def main():
    n = 0
    chars = 0
    cnt = collections.Counter()
    with open(SRC, encoding="utf-8", errors="ignore") as f, \
         open(OUT, "w", encoding="utf-8") as fo:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if isinstance(d, dict):
                    t = d.get("text") or ""
                else:
                    t = str(d)
            except Exception:
                t = line
            t = t.strip()
            if len(t) < 8:
                continue
            cat = classify(t)
            fo.write(json.dumps({"text": t, "src": "novel", "cat": cat}, ensure_ascii=False) + "\n")
            n += 1; chars += len(t); cnt[cat] += 1
    print(f"已分类 {n:,} 篇 / {chars:,} 字符 -> {OUT}")
    print("题材分布:")
    for g, c in cnt.most_common():
        print(f"  {g:6s} : {c:>6,}  ({c/max(1,n)*100:.1f}%)")

if __name__ == "__main__":
    main()
