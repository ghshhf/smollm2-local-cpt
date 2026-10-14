# V1-02 交叉裁决：Gate7 删除屏障的并发/复活竞态如何真正堵死

> 2026-10-14 · 三模型独立交叉（gemini-3.8-flash / nemotron-3-ultra / openrouter-free）+ 原始答案存 `fleet_answers/V1-02/`
> 这份裁决**直接推翻了我们 gate7_deletion.py 的现有实现**——现实现被判定为"伪删除"。

## 一、三方一致与分歧

| 维度 | gemini | nemotron | openrouter | 裁决 |
|---|---|---|---|---|
| 总体立场 | 触发器 + 载荷物理抹除 +栅栏token，**缺一不可** | 触发器（最后防线）+ Append-only 日志（审计链），两者**正交** | Append-only 日志为**真理源** + 触发器兜底，分工明确 | **三方一致：两者都要，缺一不可** |
| 能否只靠应用层 | 否（G7-05 已证伪） | 否 | 否（漏联表就读到已删数据） | **确认：应用层单独不行** |
| 能否只靠触发器 | 无审计链 | 无审计链 | 拦不住 DELETE+REINSERT / REPLACE INTO / VACUUM 重写 / DBA 改页 | **不能只靠触发器** |
| 能否只靠日志 | 拦不住绕过 | 日志本身可被删/改 | 应用层忘联表就失效 | **不能只靠日志** |
| 恢复后复活 | WAL checkpoint + VACUUM + 备份密钥丢弃 | — | — | gemini 最完整 |

## 二、🔴 对我们现有实现的裁决（重要）

我们 `gate7_deletion.py` 当前做法：`memories.tombstone=1` + 清索引/摘要/缓存 + 备份打标记。

**gemini 判定这是"伪删除"**，理由：
1. 数据残留违背宪法——记录还在、明文 payload 与 embedding 仍在 B-Tree 叶子/WAL，内存 dump 或越权 `SELECT content` 仍可读；
2. 不可逆性失效——锁住 flag 也挡不住绕过视图直读。

**nemotron/openrouter 补充**：触发器拦不住 `DELETE + REINSERT 同 PK`、拦不住 `REPLACE INTO`、拦不住 VACUUM 后物理重写、拦不住 DBA 手工改页。

⇒ **我们的 G7-05 那个"诊断项"被三方一致认定为真缺陷，且我们的缓解（应用层规避）不足。**

## 三、采纳的修正方案（三方交集）

### 3.1 防复活：三道闸
1. **DB 触发器**（SQLite/PG 通用语法差异）：
   ```sql
   -- SQLite
   CREATE TRIGGER trg_prevent_resurrection
   BEFORE UPDATE OF tombstone ON memories
   FOR EACH ROW WHEN OLD.tombstone=1 AND NEW.tombstone=0
   BEGIN SELECT RAISE(ABORT,'RESURRECTION_FORBIDDEN'); END;
   ```
   （PG 用 `plpgsql` 函数 + `RAISE EXCEPTION`。north 特别指出：触发器要**只拦复活不拦正常写**，`BEFORE UPDATE OF tombstone` 精准指向该列。）
2. **Append-only 删除日志** `deletion_log(memory_id, deleted_at, operator, reason, prev_hash)`——只 INSERT，永不 UPDATE/DELETE；作为"是否已删"的**唯一真理源**（所有查询联表此表，不只信`tombstone`）。
3. **业务写必须带版本号乐观锁**（gemini 的栅栏token）：业务 UPDATE 强制 `WHERE id=? AND version=:read_v AND tombstone=0`，`rowcount==0` 即丢弃脏写、绝不重试。

### 3.2 物理擦除（gemini 独有但必要）
删除事务内**同时把 `content`/`embedding` 置NULL**（不留明文），而非只改 flag。

### 3.3 诚实三态呈现（gemini 最完整，采纳为契约）
| 阶段 | 标识 | 承诺 |
|---|---|---|
| S1 | COMMITTED_ERASED 已注销 | 载荷已置空/粉碎；检索0 命中；旧任务不可复活 |
| S2 | PENDING_MEDIA_PURGE 待覆写 | WAL/旧快照可能仍有冷数据碎片；ID 永久进冷介质黑名单 |
| S3 | CRYPTO_PURGED 物理消亡 | 已 checkpoint+VACUUM；备份轮转完成或解密密钥已丢弃 |

**禁止**对用户宣称"数据已在全宇宙彻底消失"。S1 时就诚实告知 S2 待覆写。

## 四、可证伪判据（任一命中=方案判负）

1. **复活穿透**：在有触发器+带`WHERE tombstone=0` 写入下，任何已进 `deletion_log` 的 ID 再次出现在 `SELECT * FROM memories WHERE tombstone=0`。
2. **死尸载荷**：删除后、不执行 VACUUM，`SELECT content,embedding` 仍查出非空。
3. **压测撕裂**：并发压测中出现 `deletion_log` 有该 ID 但 `memories.tombstone=0` 的不一致，或 DB 锁死超5 秒。

## 五、node1 压测用例
gemini 给了 `test_gate7_concurrency.py`：四路竞争（并发更新/慢任务脏覆写/并发删除/删除后强复活）。⇒ 已排入待跑。

## 六、待办
- 🔴 按裁决**重写 gate7_deletion.py**：加触发器防复活、Append-only 日志为真理源、版本乐观锁、删除时载荷置空、S1/S2/S3 三态。
- ⏳ 跑 node1 四路并发压测（G7-05 升级为真测试）。

## 七、原始答案
- `fleet_answers/V1-02/ans_kris.gemini.md`（349 行，最完整：三态契约+栅栏token+压测脚本）
- `fleet_answers/V1-02/ans_kris.nemotron.md`（双栈SQL）
- `fleet_answers/V1-02/ans_zen.openrouter.md`（真理源设计）