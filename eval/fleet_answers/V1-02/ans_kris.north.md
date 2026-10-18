# kris.north (vendor=cohere, served=cohere/north-mini-code:free, 42.5s)

**①机制/最佳理论**

| # | 建议 | 理论基础 |
|---|------|----------|
| 1️⃣ | **不可复活删除标记** – 删除行 **只** 允许 `tombstone` 由 `0 → 1`，**禁止** `1 → 0`。同时维护一个 **只增不改的删除日志**（`delete_log`）记录每个被标记的行 ID、时间戳和操作者。 | *不可变性* + *审计跟踪*。触发器保证逻辑不可逆；追加日志提供了独立于事务的证明链。 |
| 2️⃣ | **隔离与锁** – 将删除操作视为 **悲观写**（获取行级锁），而读取操作使用 **乐观 MVCC** 快照。隔离级别选择 `READ COMMITTED`（PostgreSQL）或 SQLite 的 `SERIALIZABLE` 内置快照。 | *MVCC* 使并发读不会被删除锁阻塞；*悲观写* 保证同一行不会同时被多个删除者 resurrection。 |
| 3️⃣ | **诚实的状态表示** – UI/API 展示状态 `status = "pending_deletion"`，同时显示 **到期时间**（`expires_at`）和 **不可恢复标志**（如“已标记为删除，将在 X 天后永久清除”）。不显示“已删除”或“不存在”。 | 防止用户将“已删除”与“可恢复”混淆。用户看到的是一个 **临时保留区**，而非已完成的操作。 |

---

**②可执行方案（伪代码/SQL）**

> **假设** – 表 `memories(id PK, data JSON, tombstone INT, expires_at TIMESTAMPTZ)`。

### 1️⃣ 不可复活删除 – PostgreSQL

```sql
-- 1. 删除日志表（只写）
CREATE TABLE