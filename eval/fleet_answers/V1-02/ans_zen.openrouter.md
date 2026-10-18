# zen.openrouter (vendor=openrouter, served=openrouter/free, 58.7s)

### ① 机制/最佳理论：**「双轨制：Append-Only 删除日志为真理源 + 触发器做兜底拦截」**

**核心判断**：**不选二者其一，必须两者都要，且分工明确**。
*   **Append-Only `deletion_log`（主防线/审计源）**：记录「谁、何时、删了哪个 memory_id、删除前快照 hash」。**永不 UPDATE/DELETE**，只 INSERT。所有「是否已删」判定 **只查此表**（或其物化视图 `deleted_memory_ids`）。
*   **`memories` 表触发器（副防线/误操作拦截）**：`BEFORE UPDATE ON memories WHEN OLD.tombstone=1 AND NEW.tombstone=0 THEN RAISE(ABORT, 'RESURRECTION_FORBIDDEN')`。不负责业务判定，只负责**在事务层把「手滑/注入/绕过应用层」的复活语句直接报错回滚**。
*   **为什么不单用触发器**：触发器只能拦 `UPDATE`，拦不住 `DELETE + REINSERT 同 PK`、拦不住 `REPLACE INTO`、拦不住 `VACUUM` 后物理重写、拦不住 DBA 手工改页。Append-Only 日志才是**可审计、可重放、可形式化验证**的「单一事实源」。
*   **为什么不单用 Append-Only**：应用层若忘了联表查 `deletion_log`，就会读到「已删数据」。触发器让 `memories.tombstone=1` 变成**物理不可逆的单向阀**，配合 `CHECK (tombstone IN (0,1))`，把「复活」变成**硬报错**，倒逼所有代码路径走「新建新记录 + 关联旧 ID」而非「翻旧账」。

---

### ② 可执行方案 / 伪代码（SQLite / PostgreSQL 双写）

#### A. Schema 定义（SQLite 语法，PG 只改 `AUTOINCREMENT`→`GENERATED ALWAYS AS IDENTITY`、`RAISE`→`RAISE EXCEPTION`）
```sql
-- 1. 主表：tombstone 只能 0→1，不可 1→0
CREATE TABLE memories (
    memory_id   INTEGER PRIMARY KEY,
    owner_id    INTEGER NOT NULL,
    content     BLOB    NOT NULL,          -- 加密载荷
    content_hash BLAKE3 NOT NULL,          -- 内容去重/完整性
    tombstone   INTEGER NOT NULL DEFAULT 0 CHECK (tombstone IN (0,1)),
    created_at  INTEGER NOT NULL,          -- unix ms
    updated_at  INTEGER NOT NULL,
    -- 业务唯一键防重（可选）
    UNIQUE (owner_id, content_hash) WHERE tombstone=0
);

-- 2. 删除日志：Append-Only，主键自增，禁止任何 UPDATE/DELETE
CREATE TABLE deletion_log (
    log_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    memory_id    INTEGER NOT NULL REFERENCES memories(memory_id) ON DELETE RESTRICT,
    owner_id     INTEGER NOT NULL,
    deleted_at   INTEGER NOT NULL,         -- unix ms
    deleted_by   TEXT    NOT NULL,         -- 'user' | 'ttl' | 'admin' | 'gc'
    prev_hash    BLAKE3  NOT NULL,         -- 删除前 content_hash，防篡改
    -- 便于「已删除集合」快速查询
    UNIQUE (memory_id)                    -- 同一 memory_id 只能删一次
);

-- 3. 物化视图：供「是否已删」极速判定（SQLite 用触发器维护，PG 用 MATERIALIZED VIEW + REFRESH CONCURRENTLY）
CREATE TABLE deleted_memory_ids (
    memory_id INTEGER PRIMARY KEY REFERENCES memories(memory_id) ON DELETE RESTRICT
);

-- 4. 触发器：双向维护
-- 4.1 禁止复活（主表层面）
CREATE TRIGGER trg_forbid_resurrection
BEFORE UPDATE ON memories
WHEN OLD.tombstone = 1 AND NEW.tombstone = 0
BEGIN
    SELECT RAISE(ABORT, 'RESURRECTION_FORBIDDEN: memory_id=' || OLD.memory_id);
END;

-- 4.2 删除时：写日志 + 标记主表 + 刷物化视图（单事务原子）
CREATE TRIGGER trg_cascade_delete
AFTER UPDATE OF tombstone ON memories
WHEN NEW.tombstone = 1 AND OLD.tombstone = 0
BEGIN
    INSERT INTO deletion_log (memory_id, owner_id, deleted_at, deleted_by, prev_hash)
    VALUES (NEW.memory_id, NEW.owner_id, strftime('%s','now')*1000, 'user', OLD.content_hash);
    INSERT OR IGNORE INTO deleted_memory_ids (memory_id) VALUES (NEW.memory_id);
END;

-- 4.3 防御性：若有人绕过 UPDATE 直接 DELETE 行（极少见），也要留痕
CREATE TRIGGER trg_on_delete_row
BEFORE DELETE ON memories
BEGIN
    INSERT INTO deletion_log (memory_id, owner_id, deleted_at, deleted_by, prev_hash)
    VALUES (OLD.memory_id, OLD.owner_id, strftime('%s','now')