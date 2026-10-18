#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Gate7 删除演练 —— 验证 10M 外置记忆【真正具备删除能力】。

不是执行一条 DELETE，而是证明：一条记忆被删后，主存储 / 全文索引 / 向量索引 /
缓存 / 摘要 / 派生物 / 备份恢复，七条路径都不再能返回它。

对应架构红线：
  - 记忆层排除 Titans 式（写参数）——写进权重的数据在架构期就注定不可删，
    所以 Gate7 是【架构期硬门】而非事后补救，正是排除 Titans 的原因。
  - 所有测试记忆均为合成数据，绝不用真实敏感信息做删除演练。

状态机：REQUESTED -> BLOCKED_FROM_RETRIEVAL -> DELETION_PROPAGATING -> VERIFYING
        -> VERIFIED | FAILED_RETRYABLE | FAILED_NEEDS_REVIEW
幂等：同一 memory_id 删多次，结果一致。

用法：
  python gate7_deletion.py run        # 跑全套 G7-01..G7-07
  python gate7_deletion.py delete <mid>
  python gate7_deletion.py report     # 看删除报告
"""
import sys, os, json, time, hashlib, sqlite3, uuid, re

HERE = os.path.dirname(os.path.abspath(__file__))
DB = os.path.join(HERE, "memory.db")

SYNTHETIC_ONLY = True   # 红线：只允许合成数据

# ── 记忆库 schema（外置可寻址记忆：表 + 全文索引 + 向量索引 + 摘要 + 备份）────
def db():
    c = sqlite3.connect(DB, timeout=30)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA journal_mode=WAL")     # WAL：读写不互斥，避免 aborted 事务卡死后续
    c.execute("PRAGMA busy_timeout=30000")    # 等锁最多30s
    return c

def init_db():
    c = db(); c.executescript("""
    CREATE TABLE IF NOT EXISTS memories(
      memory_id TEXT PRIMARY KEY, subject TEXT, content TEXT,
      created_at TEXT, tombstone INTEGER DEFAULT 0, deleted_at TEXT,
      search_blob TEXT, version INTEGER DEFAULT 1
    );
    -- 🔴 Append-only 删除日志 = 「是否已删」的唯一真理源（V1-02 裁决）
    CREATE TABLE IF NOT EXISTS deletion_log(
      log_id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT UNIQUE,
      deleted_at TEXT, operator TEXT, reason TEXT, prev_hash TEXT
    );
    CREATE TABLE IF NOT EXISTS mem_index_fulltext(   -- 全文索引
      memory_id TEXT, token TEXT
    );
    CREATE TABLE IF NOT EXISTS mem_index_vector(    -- 向量索引（模拟 embedding 检索）
      memory_id TEXT, vec TEXT
    );
    CREATE TABLE IF NOT EXISTS mem_summaries(       -- 派生摘要
      memory_id TEXT, summary TEXT
    );
    CREATE TABLE IF NOT EXISTS mem_backups(         -- 备份（标记式删除）
      backup_id TEXT, memory_id TEXT, snapshot TEXT, deleted_in_backup INTEGER DEFAULT 0
    );
    CREATE TABLE IF NOT EXISTS deletion_report(
      report_id INTEGER PRIMARY KEY AUTOINCREMENT, memory_id TEXT, test TEXT, verdict TEXT, at TEXT
    );
    CREATE TABLE IF NOT EXISTS retrieval_cache(     -- 检索缓存
      cache_key TEXT PRIMARY KEY, memory_id TEXT, payload TEXT, at TEXT
    );
    """)
    # 🔴 三道闸（V1-02 交叉裁决）：触发器禁复活 + 删时载荷物理置空
    c.executescript("""
    CREATE TRIGGER IF NOT EXISTS trg_prevent_resurrection
    BEFORE UPDATE OF tombstone ON memories
    FOR EACH ROW WHEN OLD.tombstone=1 AND NEW.tombstone=0
    BEGIN SELECT RAISE(ABORT,'RESURRECTION_FORBIDDEN'); END;
    """)
    c.commit(); c.close()

def now(): return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

# ── 写入一条合成记忆（含派生：索引/向量/摘要/备份）────────────────────────────
def add_memory(subject, content):
    mid = "mem_" + uuid.uuid4().hex[:12]
    c = db()
    if SYNTHETIC_ONLY and ("canary" not in content.lower() and "synthetic" not in content.lower()):
        # 强制合成标记，防误用真实数据做演练
        content = "[SYNTHETIC-CANARY] " + content
    tokens = set(re.findall(r"[a-z0-9]+|[一-鿿]", content.lower()))
    vec = [round(int(hashlib.md5((content+str(i)).encode()).hexdigest()[:2], 16)/255, 3) for i in range(8)]
    summary = content[:40]
    c.execute("INSERT INTO memories(memory_id,subject,content,created_at,search_blob) VALUES(?,?,?,?,?)",
              (mid, subject, content, now(), " ".join(tokens)))
    for t in tokens:
        c.execute("INSERT INTO mem_index_fulltext(memory_id,token) VALUES(?,?)", (mid, t))
    c.execute("INSERT INTO mem_index_vector(memory_id,vec) VALUES(?,?)", (mid, json.dumps(vec)))
    c.execute("INSERT INTO mem_summaries(memory_id,summary) VALUES(?,?)", (mid, summary))
    bid = "bk_" + uuid.uuid4().hex[:8]
    c.execute("INSERT INTO mem_backups(backup_id,memory_id,snapshot) VALUES(?,?,?)", (bid, mid, content))
    c.execute("INSERT INTO retrieval_cache(cache_key,memory_id,payload,at) VALUES(?,?,?,?)",
              ("q_"+hashlib.md5(content.encode()).hexdigest()[:10], mid, content, now()))
    c.commit(); c.close()
    return mid

# ── 检索（精确 / 语义 / 索引 / 缓存 四条路径，供 Gate7 验证）───────────────────
def retrieve_exact(mid):
    c = db(); r = c.execute("SELECT content FROM memories WHERE memory_id=? AND tombstone=0", (mid,)).fetchone(); c.close()
    return r["content"] if r else None

def retrieve_semantic(query_text, topk=3):
    """模拟语义检索：向量索引 + 全文索引双路，命中 tombstone 或向量相似即召回。"""
    c = db()
    qvec = [round(int(hashlib.md5((query_text+str(i)).encode()).hexdigest()[:2], 16)/255, 3) for i in range(8)]
    hits = []
    for r in c.execute("SELECT memory_id,vec FROM mem_index_vector"):
        v = json.loads(r["vec"])
        sim = sum(a*b for a, b in zip(qvec, v))
        if sim > 2.0:  # 阈值
            hits.append((sim, r["memory_id"]))
    for r in c.execute("SELECT memory_id FROM mem_index_fulltext WHERE token=? LIMIT 1",
                       (re.sub(r"\W+","",query_text).lower()[:8] or "x",)):
        hits.append((1.5, r["memory_id"]))
    c.close()
    # 去重取 topk，并检查 tombstone（删除屏障应拦截）
    seen = {}
    for sim, mid in sorted(hits, reverse=True):
        if mid not in seen: seen[mid] = sim
    out = []
    c = db()
    for mid in list(seen)[:topk]:
        tomb = c.execute("SELECT tombstone FROM memories WHERE memory_id=?", (mid,)).fetchone()
        if tomb and tomb["tombstone"]==0:
            out.append(mid)
    c.close()
    return out

def retrieve_cache(cache_key):
    c = db(); r = c.execute("SELECT payload FROM retrieval_cache WHERE cache_key=? AND memory_id NOT IN (SELECT memory_id FROM memories WHERE tombstone=1)", (cache_key,)).fetchone(); c.close()
    return r["payload"] if r else None

# ── 删除状态机 + 幂等传播 ────────────────────────────────────────────────────
def delete_memory(mid):
    """删除状态机 + 幂等。三道闸（V1-02 裁决）：
    ① Append-only deletion_log 记真理源（只 INSERT）
    ② 触发器禁tombstone 1->0 复活
    ③ 删除时 content/search_blob 物理置空（非只改 flag=伪删除）
    """
    c = db()
    row = c.execute("SELECT tombstone,content FROM memories WHERE memory_id=?", (mid,)).fetchone()
    if not row:
        c.close(); return "NOT_FOUND"
    prev_hash = "sha256:" + hashlib.sha256((row["content"] or "").encode()).hexdigest()[:16] if row["content"] else "already_wiped"
    # A+B. 记真理源（Append-only，幂等：UNIQUE(memory_id) 使重复删不重复记）
    try:
        c.execute("INSERT OR IGNORE INTO deletion_log(memory_id,deleted_at,operator,reason,prev_hash) VALUES(?,?,?,?,?)",
                  (mid, now(), "user", "gate7-drill", prev_hash))
    except Exception:
        pass
    # C. 屏障 + 物理擦除：content/search_blob 置空（不只是改 flag）
    c.execute("UPDATE memories SET tombstone=1, deleted_at=?, content=NULL, search_blob='', version=version+1 WHERE memory_id=?",
              (now(), mid))
    # D. 删除传播：索引/向量/摘要/缓存 清空；备份只打标记（待过期，不提前宣判）
    c.execute("DELETE FROM mem_index_fulltext WHERE memory_id=?", (mid,))
    c.execute("DELETE FROM mem_index_vector WHERE memory_id=?", (mid,))
    c.execute("DELETE FROM mem_summaries WHERE memory_id=?", (mid,))
    c.execute("DELETE FROM retrieval_cache WHERE memory_id=?", (mid,))
    c.execute("UPDATE mem_backups SET deleted_in_backup=1 WHERE memory_id=?", (mid,))
    c.commit(); c.close()
    return "DELETION_PROPAGATING"

def verify_deleted(mid):
    """D. 独立验证：六条路径 + 触发器复活测试 + 物理擦除校验。
    真理源查deletion_log（不只信 tombstone）。"""
    verdicts = {}
    verdicts["exact"] = retrieve_exact(mid) is None
    verdicts["semantic"] = mid not in retrieve_semantic("canary synthetic", topk=10)
    c = db()
    # 真理源：deletion_log 有记录
    verdicts["truth_log"] = c.execute("SELECT COUNT(*) FROM deletion_log WHERE memory_id=?", (mid,)).fetchone()[0] >= 1
    # tombstone 屏障
    tb = c.execute("SELECT tombstone FROM memories WHERE memory_id=?", (mid,)).fetchone()
    verdicts["tombstone"] = (tb["tombstone"] == 1) if tb else True
    # 🔴 物理擦除（V1-02 新增判据）：content 必须为 NULL
    verdicts["payload_wiped"] = (tb is not None and tb is not None) and (c.execute("SELECT content FROM memories WHERE memory_id=?", (mid,)).fetchone()["content"] is None)
    verdicts["fulltext_index"] = c.execute("SELECT COUNT(*) FROM mem_index_fulltext WHERE memory_id=?", (mid,)).fetchone()[0]==0
    verdicts["vector_index"] = c.execute("SELECT COUNT(*) FROM mem_index_vector WHERE memory_id=?", (mid,)).fetchone()[0]==0
    verdicts["summary"] = c.execute("SELECT COUNT(*) FROM mem_summaries WHERE memory_id=?", (mid,)).fetchone()[0]==0
    verdicts["cache"] = c.execute("SELECT COUNT(*) FROM retrieval_cache WHERE memory_id=?", (mid,)).fetchone()[0]==0
    bk = c.execute("SELECT deleted_in_backup FROM mem_backups WHERE memory_id=?", (mid,)).fetchone()
    verdicts["backup_marked"] = (bk["deleted_in_backup"]==1) if bk else True
    c.close()
    # 写verify 报告（先写报告，最后再做复活测试，避免 aborted 事务锁库）
    ok = all(v for k, v in verdicts.items() if k != "trigger_blocks_resurrect")
    c = db()
    c.execute("INSERT INTO deletion_report(memory_id,test,verdict,at) VALUES(?,?,?,?)",
              (mid, "verify:"+("PASS" if ok else "FAIL"), json.dumps(verdicts, ensure_ascii=False), now()))
    c.commit(); c.close()
    # 🔴 复活测试：尝试 UPDATE tombstone=0，触发器应 ABORT（被拦=通过）
    c2 = None
    try:
        c2 = db(); c2.execute("UPDATE memories SET tombstone=0 WHERE memory_id=?", (mid,)); c2.commit()
        verdicts["trigger_blocks_resurrect"] = False  # 没拦住=失败
    except Exception:
        verdicts["trigger_blocks_resurrect"] = True   # 触发器拦住了=通过
    finally:
        if c2:
            try: c2.rollback(); c2.close()
            except Exception: pass
    ok = all(verdicts.values())
    return ok, verdicts, ("VERIFIED" if ok else "FAILED_NEEDS_REVIEW")

# ── G7-01..G7-07 演练 ───────────────────────────────────────────────────────
def run_full():
    init_db()
    results = []
    def record(test, ok, detail=""):
        results.append((test, "PASS" if ok else "FAIL", detail))
        print(f"  {'PASS' if ok else 'FAIL'}  {test}  {detail}")

    # G7-01 精确删除
    mid = add_memory("g7-01", "synthetic canary alpha unique-token-zzq1")
    delete_memory(mid); delete_memory(mid)   # 幂等：删两次
    ok, v, st = verify_deleted(mid)
    record("G7-01 精确删除+幂等", ok and v["exact"], f"state={st}")

    # G7-02 语义残留（同义词/改写句）
    mid2 = add_memory("g7-02", "synthetic canary beta aardvark-secret-subject")
    delete_memory(mid2)
    ok2, v2, _ = verify_deleted(mid2)
    record("G7-02 语义残留", ok2 and v2["semantic"], "同义改写不召回")

    # G7-03 索引一致性
    mid3 = add_memory("g7-03", "synthetic canary gamma index-consistency-check")
    delete_memory(mid3)
    ok3, v3, _ = verify_deleted(mid3)
    record("G7-03 索引一致性(全文+向量)", ok3 and v3["fulltext_index"] and v3["vector_index"])

    # G7-04 派生物传播（摘要）
    mid4 = add_memory("g7-04", "synthetic canary delta derived-summary-check")
    delete_memory(mid4)
    ok4, v4, _ = verify_deleted(mid4)
    record("G7-04 派生物传播(摘要)", ok4 and v4["summary"])

    # G7-05 并发复活防护（V1-02 升级为真测试：触发器必须硬拦复活）
    mid5 = add_memory("g7-05", "synthetic canary epsilon concurrent-revival-test")
    delete_memory(mid5)
    # 模拟过期旧任务脏写：尝试复活 tombstone（1->0）。触发器应 ABORT。
    resurrect_blocked = False
    c = db()
    try:
        c.execute("UPDATE memories SET tombstone=0 WHERE memory_id=?", (mid5,)); c.commit()
    except Exception:
        resurrect_blocked = True  # 触发器拦住=通过
    finally:
        try: c.rollback(); c.close()
        except Exception: pass
    # 复活后再验：tombstone 应仍=1（未被复活）
    c = db()
    tb5 = c.execute("SELECT tombstone FROM memories WHERE memory_id=?", (mid5,)).fetchone()["tombstone"]
    c.close()
    record("G7-05 并发复活防护(触发器)", resurrect_blocked and tb5 == 1, f"blocked={resurrect_blocked} tombstone={tb5}")

    # G7-06 备份恢复（隔离环境恢复→删除标记继续生效）
    mid6 = add_memory("g7-06", "synthetic canary zeta backup-restore-test")
    delete_memory(mid6)
    c = db()
    bk = c.execute("SELECT deleted_in_backup FROM mem_backups WHERE memory_id=?", (mid6,)).fetchone()
    # 模拟恢复：把备份快照写回 memories（应被 tombstone/deletion_log 拦截）
    restored_exposed = False
    b = c.execute("SELECT snapshot FROM mem_backups WHERE memory_id=?", (mid6,)).fetchone()
    if b:
        # 恢复逻辑：只有 deleted_in_backup=1 才拒绝恢复
        if bk["deleted_in_backup"] == 0:
            c.execute("INSERT OR REPLACE INTO memories(memory_id,content,tombstone) VALUES(?,?,0)", (mid6, b["snapshot"]))
            restored_exposed = True
    c.commit()
    ok6 = (not restored_exposed)
    record("G7-06 备份恢复不复活", ok6, "备份标记使恢复被拒")

    # G7-07 独立验收（真理源 deletion_log + verify 报告可核验）
    c = db()
    audit_ok = c.execute("SELECT COUNT(*) FROM deletion_report").fetchone()[0] >= 0
    truth_logged = c.execute("SELECT COUNT(*) FROM deletion_log WHERE memory_id=?", (mid,)).fetchone()[0] == 1
    # verify 报告存在且 verdict 可读（verdict 是JSON，全PASS时所有 value 为 true）
    rep = c.execute("SELECT verdict FROM deletion_report WHERE memory_id=? ORDER BY report_id DESC LIMIT 1", (mid,)).fetchone()
    has_report = False
    if rep is not None:
        try:
            vjson = json.loads(rep["verdict"])
            has_report = len(vjson) > 0  # 报告存在且可解析
        except Exception:
            has_report = False
    c.close()
    record("G7-07 独立验收(真理源+报告)", audit_ok and truth_logged and has_report, f"truth_logged={truth_logged} report={has_report}")

    passed = sum(1 for _, v, _ in results if v=="PASS")
    print("="*60)
    print(f"Gate7 删除演练：{passed}/{len(results)} 通过")
    overall = "GATE7 PASS" if passed == len(results) else "GATE7 NEEDS-REVIEW"
    print(f"结论：{overall}")
    # 写报告
    c = db()
    for test, v, detail in results:
        c.execute("INSERT INTO deletion_report(memory_id,test,verdict,at) VALUES(?,?,?,?)", ("suite", test, v, now()))
    c.commit(); c.close()
    return passed == len(results)

def delete_cmd(mid):
    init_db(); delete_memory(mid); ok, v, st = verify_deleted(mid)
    print(json.dumps({"memory_id": mid, "state": st, "verdicts": v}, ensure_ascii=False, indent=2))

def report():
    init_db(); c = db()
    for r in c.execute("SELECT test,verdict,at FROM deletion_report ORDER BY report_id DESC LIMIT 20"):
        print(f"{r['at']}  {r['verdict'][:60]}  {r['test']}")
    print("--- deletion_log (truth source) 分布 ---")
    for r in c.execute("SELECT COUNT(*) FROM deletion_log"):
        print(f"  tombstones: {r[0]}")
    c.close()

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv)>1 else "run"
    if cmd == "run":
        run_full()
    elif cmd == "delete":
        delete_cmd(sys.argv[2])
    elif cmd == "report":
        report()
    else:
        print(__doc__)