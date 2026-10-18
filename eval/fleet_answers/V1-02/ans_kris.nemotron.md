# kris.nemotron (vendor=nvidia, served=nvidia/nemotron-3-ultra-550b-a55b, 48.2s)

### ① 机制/最佳理论

**核心结论：必须「DB 层触发器禁止 tombstone 1→0」+「Append-only 删除日志」双重保障，缺一不可。**  
- **触发器**是“最后一道防线”，直接在存储引擎层拦截 `UPDATE ... SET tombstone=0 WHERE tombstone=1`，成本极低（单行检查），覆盖所有客户端（含直连 DB 的运维脚本）。  
- **Append-only 删除日志**（`memory_tombstone_log(memory_id, deleted_at, operator, reason)`）提供**完整审计链**与**时间旅行证据**，便于事后追责、合规审计、甚至“误删恢复走人工工单而非 SQL”。  
- 两者**正交**：触发器防“误/恶意复活”，日志留“证据链”；触发器失效（如 DDL 绕过）仍有日志兜底。

> **推翻常见误区**  
> - “只靠应用层规避”已被 G7-05 证伪。  
> - “只用触发器”无审计链，无法满足“删除权 0/1 硬需求”中的“可证明”子条款。  
> - “只用日志”无法阻止 `UPDATE` 绕过，日志本身可被删/改（除非再加触发器保护日志表）。

---

### ② 可执行方案（SQLite / PostgreSQL 双栈通用）

#### A. Schema 最小增量
```sql
-- 1. 删除日志表（append-only，无 UPDATE/DELETE 权限）
CREATE TABLE memory_tombstone_log (
    memory_id     BIGINT PRIMARY KEY REFERENCES memories(id) ON DELETE RESTRICT,
    deleted_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    operator      TEXT NOT NULL,          -- 'user' | 'system:ttl' | 'admin:ticket#123'
    reason        TEXT NOT NULL,
    -- 可选：快照字段便于审计，不加索引
    snapshot_json JSONB
);

-- 2. memories 表仅加触发器，不改既有字段
--    假设 memories 已有 tombstone BOOLEAN NOT NULL DEFAULT 0
```

#### B. 触发器（禁止复活）
```sql
-- PostgreSQL
CREATE OR REPLACE FUNCTION forbid_tombstone_resurrect()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF OLD.tombstone = true AND NEW.tombstone = false THEN
        RAISE EXCEPTION 'RESURRECT_FORBIDDEN: memory_id=% cannot flip tombstone 1->0', OLD.id
            USING ERRCODE = 'P0001';  -- 自定义错误码，应用层可精准捕获
    END IF;
    RETURN NEW;
END $$;

CREATE TRIGGER trg_forbid_resurrect
BEFORE UPDATE OF tombstone ON memories
FOR EACH ROW EXECUTE FUNCTION forbid_tombstone_resurrect();

-- SQLite（需编译启用 TRIGGER，默认开启）
CREATE TRIGGER trg_forbid_resurrect
BEFORE UPDATE OF tombstone ON memories
FOR EACH ROW
WHEN OLD.tombstone = 1 AND NEW.tombstone = 0
BEGIN
    SELECT RAISE(ABORT, 'RESURRECT_FORBIDDEN: memory_id=' || OLD.id || ' cannot flip tombstone 1->0');
END;
```

#### C. 删除统一入口（应用层唯一合法路径）
```python
# Python 伪代码，SQLite/PostgreSQL 通用
def hard_delete_memory(conn, memory_id: int, operator: str, reason: str):
    """
    单事务：写日志 + 翻 tombstone=1
    失败即回滚，绝无中间态。
    """
    with conn.transaction(isolation='SERIALIZABLE'):  # 见下节隔离级别
        # 1. 幂等性：已删除直接返回成功（避免并发重复报错）
        cur = conn.execute(
            "SELECT tombstone FROM memories WHERE id=%s FOR SHARE", (memory_id,))
        row = cur.fetchone()
        if not row:
            raise KeyError(f"memory_id={memory_id} not exist")
        if row[0]:
            return  # 已删除，幂等成功

        # 2. 写日志（append-only）
        conn.execute(
            """INSERT INTO memory_tombstone_log(memory_id, operator, reason, snapshot_json)
               SELECT id, %s, %s, to_jsonb(m) FROM memories m WHERE id=%s""",
            (operator, reason, memory_id))

        # 3. 翻 tombstone（触发器守住 1→0）
        conn.execute(
            "UPDATE memories SET tombstone=1, updated_at=now() WHERE id=%s",
            (memory_id,))
```

#### D. 权限收口（防御深度）
| 对象 | 授权角色 | 权限 |
|------|----------|------|
| `memories` | `app_rw` | `SELECT, INSERT, UPDATE(tombstone=1 only via trigger)` |
| `memory_tombstone_log` | `app_rw` | `INSERT ONLY`（`REVOKE UPDATE