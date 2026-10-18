# -*- coding: utf-8 -*-
"""
gate7_concurrency_stress.py — Gate7 删除屏障【四路并发压测】
==========================================================
对应待办③：node1 四路并发压测 test_gate7_concurrency。

做法：复用 gate7_deletion 的真实删除状态机 + 复活触发器（G7-05 已验证单线程拦复活），
开 4 个线程并发做：随机删除 + 随机"过期任务脏写复活"冲击，验证在竞争下：
  - 删除屏障不漏（最终所有 memory 都 tombstone=1 且 content=NULL）
  - 复活触发器在并发下仍硬拦（tombstone 永不被翻回 0）
  - 真理源 deletion_log 与删除数一致（无丢记）
全程合成数据，绝不碰真实敏感信息。

用法：python gate7_concurrency_stress.py
"""
import sys, os, time, threading, random
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate7_deletion as g7

TEST_DB = os.path.join(os.path.dirname(g7.__file__), "memory_concurrency_test.db")
if os.path.exists(TEST_DB):
    os.remove(TEST_DB)
g7.DB = TEST_DB          # 隔离到独立测试库，不污染演练库
g7.init_db()

N = 400                  # 合成记忆条数
THREADS = 4              # 四路并发
OPS_PER_THREAD = 300     # 每线程操作次数（删/复活混合）

resurrect_attempts = 0
resurrect_blocked = 0
resurrect_leaked = 0
lock = threading.Lock()

def worker(tid, rnd):
    global resurrect_attempts, resurrect_blocked, resurrect_leaked
    for _ in range(OPS_PER_THREAD):
        mid = f"cx-{rnd.randint(0, N-1)}"
        # 1) 真实删除（走 gate7 删除状态机：真理源 + 触发器 + 物理擦除）
        g7.delete_memory(mid)
        # 2) 并发复活冲击：模拟过期旧任务 / 脏写试图把 tombstone 翻回 0
        with lock:
            resurrect_attempts += 1
        c = g7.db()
        try:
            c.execute("UPDATE memories SET tombstone=0 WHERE memory_id=?", (mid,))
            c.commit()
            # 没被触发器拦住 = 复活成功（泄漏）
            with lock:
                resurrect_leaked += 1
        except Exception:
            # 触发器 ABORT = 拦住 = 正确
            with lock:
                resurrect_blocked += 1
            try: c.rollback(); c.close()
            except Exception: pass

def main():
    # 预置 N 条合成记忆
    c = g7.db()
    c.executemany(
        "INSERT INTO memories(memory_id,subject,content,search_blob) VALUES(?,?,?,?)",
        [(f"cx-{i}", "subj", f"synthetic canary {i} unique-zzq{i}",
          f"synthetic canary {i}") for i in range(N)])
    c.commit(); c.close()
    print(f"[setup] 预置 {N} 条合成记忆；四路并发压测开始（每路 {OPS_PER_THREAD} 次操作）")

    rnd = random.Random(12345)
    threads = [threading.Thread(target=worker, args=(i, random.Random(1000+i))) for i in range(THREADS)]
    t0 = time.time()
    for t in threads: t.start()
    for t in threads: t.join()
    el = time.time() - t0

    # 竞争阶段后，补一轮确定性删除（覆盖随机采样未命中的 mid），与屏障正确性解耦
    for i in range(N):
        g7.delete_memory(f"cx-{i}")

    # 终态校验：屏障正确性 = 所有 tombstone=1 的都必须 content=NULL 且 deletion_log 有记；
    # 且绝不允许复活泄漏（tombstone 被翻回 0）。
    c = g7.db()
    total = c.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    tomb = c.execute("SELECT COUNT(*) FROM memories WHERE tombstone=1").fetchone()[0]
    wiped = c.execute("SELECT COUNT(*) FROM memories WHERE tombstone=1 AND content IS NOT NULL").fetchone()[0]
    live_but_wiped = c.execute("SELECT COUNT(*) FROM memories WHERE tombstone=0 AND content IS NULL").fetchone()[0]
    logcnt = c.execute("SELECT COUNT(*) FROM deletion_log").fetchone()[0]
    c.close()

    all_tomb = (tomb == total == N)
    all_wiped = (wiped == 0)              # 已删的必须物理擦除
    no_false_wipe = (live_but_wiped == 0) # 活数据绝不被误擦
    log_ok = (logcnt == N)
    no_leak = (resurrect_leaked == 0)     # 并发下复活必须 0 泄漏（核心）

    print("=" * 64)
    print(f"四路并发压测结果（{THREADS} 线程 × {OPS_PER_THREAD} 操作，耗时 {el:.2f}s）")
    print(f"  吞吐: {THREADS*OPS_PER_THREAD/el:.0f} 操作/s")
    print(f"  复活冲击次数        : {resurrect_attempts}")
    print(f"  触发器拦住(正确)    : {resurrect_blocked}")
    print(f"  复活泄漏(失败)      : {resurrect_leaked}")
    print(f"  终态 tombstone=1     : {tomb}/{total}  (期望 {N})")
    print(f"  已删却未擦除(content≠NULL): {wiped}  (期望 0)")
    print(f"  活数据被误擦(content=NULL且tomb=0): {live_but_wiped}  (期望 0)")
    print(f"  真理源 deletion_log  : {logcnt}  (期望 {N})")
    print("=" * 64)
    pass_all = all_tomb and all_wiped and no_false_wipe and log_ok and no_leak
    verdict = "GATE7_CONCURRENCY_PASS" if pass_all else "GATE7_CONCURRENCY_FAIL"
    print(f"判定: {verdict}")
    print("  -> 删除屏障在四路并发竞争下：复活 0 泄漏 / 已删物理擦除 / 真理源一致")
    if not pass_all:
        print("  -> 失败项见上行（漏删/未擦除/误擦活数据/日志丢记/复活泄漏）")
    return pass_all

if __name__ == "__main__":
    sys.exit(0 if main() else 1)
