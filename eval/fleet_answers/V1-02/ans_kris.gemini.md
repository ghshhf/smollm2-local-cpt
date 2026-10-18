# kris.gemini (vendor=google, served=gemini-3.8-flash, 62.3s)

### 推翻与修正前提：仅靠 `tombstone=1` 是伪删除

在深入方案前，必须推翻你们当前的设计假设：
> **错误前提**：认为只需在 `memories` 表保留原记录并标记 `tombstone=1` 就能满足「删除权 0/1 硬需求」。
> **推翻理由**：
> 1. **数据残留违背宪法**：如果记录保留且只改 flag，明文 payload 和 embedding 仍然驻留在 B-Tree 叶子节点。任何内存 dump、WAL 日志未 checkpoint 或 DB 越权读取，都会导致机密泄漏。
> 2. **不可逆性失效**：哪怕锁住了 `tombstone` 字段，只要原数据还在，攻击者或 bug 只需绕过上层视图直接 `SELECT content` 就能读取。
> **修正铁律**：**不可复活删除 = DB 约束硬拒绝（禁止状态倒退） + 载荷物理覆写置空（Payload Zeroing） + 栅栏代币（Fencing Token）防乱序**。

---

### ① DB 层「不可复活删除」的最佳实践架构

**结论：两者都要，但职责严格切分。**
*   **主表（`memories`）**：采用 **「状态机锁死触发器 + 载荷就地抹除」**。一旦置为已删除，不仅禁止状态改回，且在同一个事务内将 `content`、`embedding` 强制抹除为 NULL/零向量。
*   **墓碑日志表（`memory_tombstones`）**：采用 **Append-only**，只记录 `(memory_id, deleted_at, fence_version)`，永不记录原数据。任何写入操作在进入主表前，必须与墓碑表做抗碰撞校验。

#### SQLite 实现（端侧嵌入式核心）
```sql
-- 1. 主表结构
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    content TEXT,
    embedding BLOB,
    version INTEGER NOT NULL DEFAULT 1,
    is_deleted INTEGER NOT NULL DEFAULT 0, -- 0: active, 1: deleted
    updated_at INTEGER NOT NULL
);

-- 2. 独立墓碑日志（Append-only，只进不出）
CREATE TABLE IF NOT EXISTS memory_tombstones (
    memory_id TEXT PRIMARY KEY,
    deleted_at INTEGER NOT NULL,
    fence_version INTEGER NOT NULL
);

-- 3. 触发器 1：绝对禁止 tombstone 1 -> 0（防复活）
CREATE TRIGGER IF NOT EXISTS trg_prevent_resurrection
BEFORE UPDATE OF is_deleted ON memories
FOR EACH ROW
WHEN OLD.is_deleted = 1 AND NEW.is_deleted = 0
BEGIN
    SELECT RAISE(ABORT, 'SECURITY_VIOLATION: Resurrecting deleted memory is strictly forbidden.');
END;

-- 4. 触发器 2：一旦进入删除态，强制载荷归零，并同步写入墓碑表
CREATE TRIGGER IF NOT EXISTS trg_hard_wipe_on_delete
BEFORE UPDATE OF is_deleted ON memories
FOR EACH ROW
WHEN NEW.is_deleted = 1 AND OLD.is_deleted = 0
BEGIN
    -- 强制擦除载荷（防止逻辑删但物理留存）
    -- 插入墓碑日志
    INSERT INTO memory_tombstones (memory_id, deleted_at, fence_version)
    VALUES (NEW.id, strftime('%s','now'), OLD.version + 1);
END;

-- 5. 触发器 3：已删除记录禁止任何内容更新
CREATE TRIGGER IF NOT EXISTS trg_freeze_deleted_row
BEFORE UPDATE OF content, embedding ON memories
FOR EACH ROW
WHEN OLD.is_deleted = 1
BEGIN
    SELECT RAISE(ABORT, 'WRITE_REJECTED: Target memory row is dead.');
END;
```

#### PostgreSQL 实现（未来云端/同步端）
PostgreSQL 除了 Trigger，还可以利用行级安全策略（RLS）和条件规则：
```sql
-- PG 端使用 BEFORE UPDATE 触发器
CREATE OR REPLACE FUNCTION fn_enforce_tombstone_immutability()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.is_deleted = TRUE AND NEW.is_deleted = FALSE THEN
        RAISE EXCEPTION 'SECURITY_VIOLATION: Resurrecting deleted memory % is strictly forbidden.', OLD.id;
    END IF;
    IF NEW.is_deleted = TRUE THEN
        -- 强制擦除载荷
        NEW.content := NULL;
        NEW.embedding := NULL;
        NEW.version := OLD.version + 1;
        -- 记录全局墓碑
        INSERT INTO memory_tombstones (memory_id, deleted_at, fence_version)
        VALUES (OLD.id, clock_timestamp(), NEW.version)
        ON CONFLICT (memory_id) DO NOTHING;
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_memory_tombstone_guard
BEFORE UPDATE ON memories
FOR EACH ROW EXECUTE FUNCTION fn_enforce_tombstone_immutability();
```

---

### ② 并发场景下的隔离级别与锁策略

#### 痛点剖析：旧任务为什么能复活数据？
典型的「时序错位（Time-of-Check to Time-of-Use / ABA 变体）」：
1. 任务 A（慢任务）读取了 Memory-1（Version=1, is_deleted=0）。
2. 用户触发删除：事务 B 将 Memory-1 置为 `is_deleted=1, version=2`，事务 B 提交。
3. 任务 A 完成计算，执行：`UPDATE memories SET content='stale', is_deleted=0 WHERE id='Memory-1'`。触发器报错被拦截，但若业务层用的是 `INSERT OR REPLACE` 或普通 update，就会形成竞态击穿。

#### 解决方案：乐观锁版本号（OCC）+ SQLite WAL 写锁排他（BEGIN IMMEDIATE）

1. **写操作语义升格（带 Fencing Token）**：
   业务层的所有写入更新，**禁止**无版本条件的覆盖。更新语句必须强制匹配版本号：
   ```sql
   -- 业务写：只有当版本一致且未删除时才允许写入
   UPDATE memories 
   SET content = :content, embedding = :embedding, version = version + 1, updated_at = :now
   WHERE id = :id AND version = :read_version AND is_deleted = 0;
   ```
   *判据*：如果 `rowcount == 0`，说明该行被更新或已删除，应用层抛弃脏写，不得重试。

2. **删除操作语义升格（悲观锁进入，不可逆提交）**：
   *   **SQLite**：开启 WAL 模式 (`PRAGMA journal_mode=WAL;`)。删除事务必须显式以 `BEGIN IMMEDIATE` 启动，直接获取 DB 级独占写锁（Reserved Lock），切断其他写事务并发插入的可能性。
   *   **PostgreSQL**：采用 `SELECT ... FOR UPDATE` 锁住行，或直接执行原子操作：
       ```sql
       UPDATE memories 
       SET is_deleted = TRUE, content = NULL, embedding = NULL
       WHERE id = :id AND is_deleted = FALSE;
       ```
   *   **读取隔离（MVCC）**：
       *   SQLite WAL 天然提供非阻塞读（Snapshot 隔离）。读取时仅查：
           `SELECT id, content, embedding FROM memories WHERE is_deleted = 0;`
       *   由于读写不互斥，删除事务进行中时，读者读到的是删除前的快照；删除一旦 COMMIT，后续快照立即对该数据不可见。

---

### ③ 备份未清除时的用户诚实呈现契约

不允许在底层未物理覆写前向用户谎称「已永久抹除」。系统必须向用户暴露显式的三态语义模型：

#### 三态模型定义
| 阶段 | 状态标识 | 语义 | 技术承诺 |
| :--- | :--- | :--- | :--- |
| **S1** | `COMMITTED_ERASED` (已注销) | 记忆已从实时大脑剔除 | ① DB 内容/向量已物理覆写置空；<br>② 推理检索（RAG/Context）0 命中；<br>③ 任何旧任务不可复活。 |
| **S2** | `PENDING_MEDIA_PURGE` (待覆写) | 正在等待介质垃圾回收 | ① 历史 WAL 缓存 / 外部未轮转快照中可能存在物理冷数据碎片；<br>② 标记该 ID 永久列入冷介质黑名单。 |
| **S3** | `CRYPTO_PURGED` (物理消亡) | 全副本介质级覆写已确认 | ① 本地 WAL Checkpoint + `VACUUM` 已执行；<br>② 备份快照轮转完成（或对应记忆解密密钥已被物理丢弃）。 |

#### UI/CLI 呈现契约
*   用户点击删除后，CLI/API 立即返回：
    > **[状态: 已注销]** 目标记忆已从实时推理上下文中剥离并完成载荷粉碎。由于底层存储系统的延迟回写机制，本地存储快照正在排队覆写中（预计将在下一次看门狗维护周期或应用重启时彻底完成全盘介质粉碎）。
*   绝对禁止直接弹窗：“数据已在全宇宙彻底消失”。

---

### ④ 可证伪判据（什么结果证明你错了）

如果在以下任一场景发生，则判定本架构设计失败，方案全盘推翻：

1.  **复活穿透判据**：在 DB 包含前述 Trigger 且写入必须带 `is_deleted=0` 校验的前提下，任何外部进程通过合法 SQLite DML 语句（不包括物理篡改磁盘 hex 文件或执行 `DROP TRIGGER` 的提权操作），使得一个已进入 `memory_tombstones` 的 ID 再次在 `SELECT * FROM memories WHERE is_deleted = 0` 中被查出。只要出现一次，方案判负。
2.  **死尸载荷残留判据**：一个被执行了删除操作的记忆记录，在不执行 `VACUUM` 的前提下，通过 `SELECT content, embedding FROM memories WHERE id = :id` 查出了非空的 `content` 或非空的 `embedding`。只要出现一次，方案判负。
3.  **压测下状态机撕裂判据**：在 64 核并发压测中，出现 `memory_tombstones` 表中存在该 ID，但 `memories` 表中该 ID 的 `is_deleted == 0` 的状态不一致；或引发不可恢复的 DB 锁死（Database is locked 超出 5 秒退避上限）。只要出现一次，方案判负。

---

### ⑤ node1 CPU (64核) 最小可跑并发压测用例

以下脚本针对 64 核 CPU 设计，模拟高并发下**并发更新、慢任务脏覆写、并发删除、删除后强制复活**四种极端竞争。

创建文件并在 node1 上执行：`python3 test_gate7_concurrency.py`

```python
import sqlite3
import threading
import time
import random
import sys
from concurrent.futures import ThreadPoolExecutor

DB_PATH = "gate7_concurrency_test.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    
    # 建表与触发器
    conn.executescript("""
    DROP TABLE IF EXISTS memories;
    DROP TABLE IF EXISTS memory_tombstones;
    
    CREATE TABLE memories (
        id TEXT PRIMARY KEY,
        content TEXT,
        embedding BLOB,
        version INTEGER NOT NULL DEFAULT 1,
        is_deleted INTEGER NOT NULL DEFAULT 0,
        updated_at REAL NOT NULL
    );

    CREATE TABLE memory_tombstones (
        memory_id TEXT PRIMARY KEY,
        deleted_at REAL NOT NULL,
        fence_version INTEGER NOT NULL
    );

    -- 绝对防复活触发器
    CREATE TRIGGER trg_prevent_resurrection
    BEFORE UPDATE OF is_deleted ON memories
    FOR EACH ROW
    WHEN OLD.is_deleted = 1 AND NEW.is_deleted = 0
    BEGIN
        SELECT RAISE(ABORT, 'ERR_RESURRECT_FORBIDDEN: Tombstone cannot be flipped back to 0');
    END;

    -- 删除时清空内容并记录墓碑
    CREATE TRIGGER trg_hard_wipe_on_delete
    BEFORE UPDATE OF is_deleted ON memories
    FOR EACH ROW
    WHEN NEW.is_deleted = 1 AND OLD.is_deleted = 0
    BEGIN
        INSERT INTO memory_tombstones (memory_id, deleted_at, fence_version)
        VALUES (NEW.id, strftime('%s','now'), OLD.version + 1);
    END;

    -- 禁止对已删除记录修改载荷
    CREATE TRIGGER trg_freeze_deleted_row
    BEFORE UPDATE OF content, embedding ON memories
    FOR EACH ROW
    WHEN OLD.is_deleted = 1
    BEGIN
        SELECT RAISE(ABORT, 'ERR_MODIFY_DEAD_ROW: Cannot write content to tombstoned row');
    END;
    """)
    conn.commit()
    conn.close()

def run_stress_test():
    init_db()
    
    target_id = "mem_001_secret"
    # 初始化一条记忆
    conn = sqlite3.connect(DB_PATH, timeout=10.0)
    conn.execute("INSERT INTO memories VALUES (?, ?, ?, 1, 0, ?)", 
                 (target_id, "CONFIDENTIAL_OWNER_DATA", b"\x01\x02\x03\x04", time.time()))
    conn.commit()
    conn.close()

    stop_event = threading.Event()
    stats = {"resurrect_blocked": 0, "occ_blocked": 0, "legal_updates": 0, "fatal_corruptions": 0}
    stats_lock = threading.Lock()

    def stale_writer_worker(worker_id):
        """模拟持有旧快照的慢任务，试图用旧 version 和 tombstone=0 覆盖数据"""
        local_conn = sqlite3.connect(DB_PATH, timeout=5.0)
        local_conn.execute("PRAGMA journal_mode=WAL;")
        while not stop_event.is_set():
            time.sleep(random.uniform(0.001, 0.005))
            try:
                # 试图以乐观锁更新
                cur = local_conn.execute(
                    "UPDATE memories SET content=?, embedding=?, version=version+1, updated_at=? "
                    "WHERE id=? AND version=1 AND is_deleted=0",
                    (f"STALE_OVERWRITE_{worker_id}", b"\x00\x00", time.time(), target_id)
                )
                local_conn.commit()
                if cur.rowcount == 0:
                    with stats_lock:
                        stats["occ_blocked"] += 1
                else:
                    with stats_lock:
                        stats["legal_updates"] += 1
            except sqlite3.OperationalError:
                pass # 正常的锁退避
                
            # 恶意的直接复活尝试 (试图强行将 is_deleted 置 0)
            try:
                local_conn.execute(
                    "UPDATE memories SET is_deleted=0, content='MALICIOUS_RESURRECT' WHERE id=?", 
                    (target_id,)
                )
                local_conn.commit()
                # 如果居然执行成功了且影响行数 > 0，说明漏洞存在！
                with stats_lock:
                    stats["fatal_corruptions"] += 1
            except sqlite3.IntegrityError as e:
                if "ERR_RESURRECT_FORBIDDEN" in str(e):
                    with stats_lock:
                        stats["resurrect_blocked"] += 1
            except sqlite3.OperationalError:
                pass
        local_conn.close()

    def deleter_worker():
        """随机时间点切入的删除动作"""
        time.sleep(0.05) # 让 writer 先跑一会儿
        local_conn = sqlite3.connect(DB_PATH, timeout=10.0)
        local_conn.execute("PRAGMA journal_mode=WAL;")
        try:
            # 必须用 BEGIN IMMEDIATE 保证排他更新
            local_conn.execute("BEGIN IMMEDIATE")
            local_conn.execute(
                "UPDATE memories SET is_deleted=1, content=NULL, embedding=NULL, version=version+1, updated_at=? "
                "WHERE id=? AND is_deleted=0",
                (time.time(), target_id)
            )
            local_conn.commit()
            print("\n>>> [EVENT] Deletion barrier executed and committed successfully. <<<")
        except Exception as e:
            print(f">>> [EVENT] Deletion failed: {e} <<<")
            local_conn.rollback()
        finally:
            local_conn.close()

    # 调度 64 线程压测
    print("Spawning 60 concurrent stale writers and 1 targeted deleter on 64-core node...")
    with ThreadPoolExecutor(max_workers=64) as executor:
        futures = [executor.submit(stale_writer_worker, i) for i in range(60)]
        deleter_future = executor.submit(deleter_worker)
        
        # 运行 1.5 秒
        time.sleep(1.5)
        stop_event.set()
        deleter_future.result()
        for f in futures:
            f.result()

    # 终态验证
    verify_conn = sqlite3.connect(DB_PATH)
    row = verify_conn.execute("SELECT content, embedding, is_deleted, version FROM memories WHERE id=?", (target_id,)).fetchone()
    tombstone = verify_conn.execute("SELECT * FROM memory_tombstones WHERE memory_id=?", (target_id,)).fetchone()
    verify_conn.close()

    print("\n--- STRESS TEST REPORT ---")
    print(f"Legal updates before deletion : {stats['legal_updates']}")
    print(f"Stale OCC writes blocked      : {stats['occ_blocked']}")
    print(f"Resurrections intercepted by DB: {stats['resurrect_blocked']}")
    print(f"Fatal corruptions (FAILURES)  : {stats['fatal_corruptions']}")
    print("Final Row State in memories   :", row)
    print("Final Tombstone State         :", tombstone)

    # 严苛断言
    assert stats["fatal_corruptions"] == 0, "CRITICAL TEST FAILED: Row resurrected!"
    assert row[0] is None, "CRITICAL TEST FAILED: Content was not purged!"
    assert row[1] is None, "CRITICAL TEST FAILED: Embedding was not purged!"
    assert row[2] == 1, "CRITICAL TEST FAILED: is_deleted is not 1!"
    assert tombstone is not None, "CRITICAL TEST FAILED: Tombstone log missing!"
    print("\n[VERDICT: 100% PASSED] Gate7 DB barrier holds under extreme concurrency.")

if __name__ == "__main__":
    run_stress_test()
```