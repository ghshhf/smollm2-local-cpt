# zen.longcat (vendor=longcat, served=longcat-2.5-preview-free, 77.8s)

① **机制/最佳理论：触发器 vs Append-only 日志 vs 混合方案**
- **最佳理论**：在数据库安全模型中，**Append-only（仅追加）是防篡改的黄金标准**。触发器本质上是“实时拦截”，它依赖数据库引擎的执行权限，存在被 DBA 或高权限恶意进程 `DISABLE TRIGGER` 绕过的风险。而 Append-only 将“删除”视为一种不可变事件（Event Sourcing），物理删除意图作为一条新记录写入日志，原有数据行永不更新，从而在数学上杜绝了 `tombstone 1→0` 的状态跃迁。
- **推荐方案**：**两者结合（纵深防御）**。
  1.  **Append-only 删除日志**：作为最终审计防线，任何删除操作必须先写入 `memories_tombstone_log` 表。
  2.  **BEFORE UPDATE 触发器**：作为实时防线，在业务层尝试将 `memories.tombstone` 从 1 改为 0 时直接抛出异常，阻断绝大多数常规误操作和低权限攻击。

② **可执行方案或伪代码**

**A. 混合方案建表与约束（PostgreSQL）**
```sql
-- 1. 主表：仅允许 tombstone 从 0->1
CREATE TABLE memories (
    id UUID PRIMARY KEY,
    content TEXT,
    tombstone BOOLEAN DEFAULT 0 CHECK (tombstone IN (0, 1)),
    version INTEGER DEFAULT 1 -- 用于乐观锁
);

-- 2. 删除日志表（Append-only）
CREATE TABLE memories_tombstone_log (
    log_id BIGSERIAL PRIMARY KEY,
    memory_id UUID NOT NULL REFERENCES memories(id),
    deleted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    deleted_by TEXT NOT NULL
);

-- 3. 触发器：禁止 tombstone 回滚
CREATE OR REPLACE FUNCTION prevent_tombstone_revival()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.tombstone = 1 AND NEW.tombstone = 0 THEN
        RAISE EXCEPTION 'Fatal: Attempt to revive deleted memory ID %', OLD.id;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_no_revival
BEFORE UPDATE ON memories
FOR EACH ROW EXECUTE FUNCTION prevent_tombstone_revival();

-- 4. 删除操作必须记录的触发器
CREATE OR REPLACE FUNCTION log_deletion()
RETURNS TRIGGER AS $$
BEGIN
    IF NEW.tombstone = 1 AND OLD.tombstone = 0 THEN
        INSERT INTO memories_tombstone_log (memory_id, deleted_by)
        VALUES (NEW.id, CURRENT_USER);
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_log_deletion
BEFORE UPDATE ON memories
FOR EACH ROW EXECUTE FUNCTION log_deletion();
```

**B. 并发隔离与锁策略**
- **策略**：采用 **悲观锁（Pessimistic Locking）** 配合 **Read Committed** 隔离级别。
- **理由**：删除操作涉及状态流转（0->1）和日志写入，属于写操作，必须使用 `SELECT FOR UPDATE` 锁定目标行，防止在标记删除前被其他事务读取并修改内容。乐观锁在写冲突高的删除场景下会导致大量重试，而 MVCC 在 `Read Committed` 下无法阻止“写后读”导致的状态不一致。

```sql
-- 删除流程伪代码
BEGIN;
-- 1. 加排他锁，阻止并发读写
SELECT * FROM memories WHERE id = 'target_uuid' FOR UPDATE;
-- 2. 执行删除标记
UPDATE memories SET tombstone = 1, version = version + 1 WHERE id = 'target_uuid';
COMMIT;
```

③ **可证伪判据与诚实性呈现**

**A. 诚实性呈现**
- **原则**：状态的呈现必须基于**已确认的事实**