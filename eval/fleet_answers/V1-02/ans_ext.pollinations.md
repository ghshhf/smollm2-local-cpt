# ext.pollinations (vendor=openai-oss, served=gpt-oss-20b, 29.1s)

## 1   机制 / 理论推荐

| 目标 | 方案 | 理论依据 | 备注 |
|---|---|---|---|
| **不可复活删除** | **① 禁止 1→0 的 UPDATE（触发器）** | 通过 *BEFORE UPDATE* 触发器在 `memories` 表中校验 `tombstone` 的状态，任何 1→0 的尝试都会 `RAISE` 错误。<br>在 PostgreSQL 里较为直接；SQLite 也支持触发器，但在 WAL 模式下必须保证**事务一致性**。 | 触发器几乎不占用磁盘空间，开销低。本方案能阻止所有程序级恢复，但若你愿意 “日志化” 删除，推荐**②** |
| **日志化（Append‑Only）** | **② 创建 `memories_del_log`（或 “tombstones”）表，只插入删除事件；永不 UPDATE** | **Append‑Only** 的核心优点:<br>• 任何恢复操作必须 **INSERT** 同样的“alive” 行，保证删除不可回滚<br>• 仅写入，减少冲突排队<br>• 方便审计 / 回滚（追踪时间戳）<br>• 兼容 PostgreSQL（MVCC）和 SQLite（WAL）<br>单表插入不受 MVCC 的写写冲突限制。 | 需要在应用层统一使用，避免直接 UPDATE `memories`。最好在删除时 **先** 写入 `del_log`，再在程序里把这条记忆标记为已过期（软删）。 |
| **两者组合** | **③ 触发器 + Append‑Only** | 触发器防止“数据逆向写”错误，Append‑Only 让数据库结构更天然不允许恢复。<br>此组合在极端恶意场景下最安全：即使程序以管理员身份对表进行 UPDATE，也被触发器拦截；而且数据恢复需要 **额外** INSERT 记录，常规流程不会触发。 | 牺牲少量写性能（额外表写 + 触发器计算），但风险极低。 |

**结论**  
> **推荐两者组合（③）**。如果对写吞吐极度敏感，只满足触发器（①）也可；但在安全性与一致性需求高的研发/生产场景，使用 Append‑Only 作为主模型，触发器作为防护墙能让你在两端都拥有完全不可恢复的删除。

---

## 2   可执行方案 / 伪代码

### 2‑1 PostgreSQL

```sql
-- 1. main table
CREATE TABLE memories (
    mem_id   BIGSERIAL PRIMARY KEY,
    user_id  BIGINT NOT NULL,
    content  TEXT NOT NULL,
    tombstone BOOLEAN NOT NULL DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- 2. delete‑log (append‑only)
CREATE TABLE mem_del_log (
    log_id     BIGSERIAL PRIMARY KEY,
    mem_id     BIGINT NOT NULL REFERENCES memories(mem_id) ON DELETE CASCADE,
    deleted_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT now()
);

-- 3. forbid stash 1->0
CREATE OR REPLACE FUNCTION forbid_undeletion()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'UPDATE' AND OLD.tombstone = TRUE AND NEW.tombstone = FALSE THEN
        RAISE EXCEPTION 'Undelete is forbidden';
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER no_undeletion
BEFORE UPDATE OF tombstone ON memories
