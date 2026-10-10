# -*- coding: utf-8 -*-
"""
蒸馏链(python 版, 稳定长进程, PID 探针 + 日志新鲜度 + 断点续跑):
  - 本进程是 python(稳定), 串联 V3 -> V4 -> V5 -> V6
  - 每个训练作业用 DETACHED_PROCESS|CREATE_NEW_PROCESS_GROUP 启动, 完全脱离本进程树
    => 即使本调度器被回收, 训练作业仍独立存活(已验证:V3 在旧链误判退出后仍训到 step 7)
  - 判活 = 真实进程探针(os.kill(pid,0), 避开 tasklist 解码坑) + 训练日志新鲜度, 二者兼具才认定"在跑"
    (历史坑: ① tasklist 输出含非 UTF8 字节 -> capture_output+text=True 解码异常被 except 吞
             -> is_running() 误返回 False -> 误判"进程消失->失败->链停止";
             ② 只看日志新鲜度会把"刚死 1 分钟"的进程误判为在跑而空等)
  - 幂等 + 断点续跑: 重跑自动识别 正在跑 / 已完成 / 已死残留 / 卡死, 不重复启动
  - 链自身防重入: .chain_lock 记录调度器 pid, 已有活调度器则退出, 避免双链抢跑
用法(前台 Bash 调用, 不要用 run_in_background):
  nohup venv/Scripts/python.exe -u train_chain.py >> _train_chain.log 2>&1 < /dev/null & disown
"""
import subprocess, time, os, sys, glob, shutil, ctypes, traceback, re
try:
    import msvcrt
    _HAVE_MSVC = True
except Exception:
    _HAVE_MSVC = False

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)
VE = os.path.join(ROOT, "venv", "Scripts", "python.exe")
BASE = "model/Qwen2.5-1.5B-Instruct"
CHAINLOG = os.path.join(ROOT, "_train_chain.log")
LOCK = os.path.join(ROOT, ".chain_lock")
DETACH = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP

FRESH_SEC = 20 * 60   # 日志 20 分钟内更新过 = 近期活跃(正常每 ~100s 更新)

# 🔴 分段：一张卡不再一次跑到底，而是每段跑满 SEGMENT_MIN 分钟就干净收尾退出，
#    下一个 4 小时窗口再从 checkpoint 无损续训。理由：40 小时一口气跑到底风险太高
#    (机器重启/会话回收/静默崩溃都会前功尽弃)，分段后每段都是可交付的完整产物。
#    SEGMENT_MIN=240 ≈ 4 小时（按 ~63s/it 约 228 步）。
SEGMENT_MIN = 240
SEGMENT_STEPS = 228# 每段步数上限；不足一段的卡就一次跑完

# 每张卡的目标总步数（与 JOBS 里的 steps 对应）——分段训练用它判断"整卡是否真跑完"
JOB_TARGET = {}
HANG_SEC  = 10 * 60   # 进程已死且日志 >10 分钟冷 = 判定卡死/崩溃
TIMEOUT_MIN = 900     # 单作业总超时(防极端卡死)

# 名称 | 数据 | 输出目录 | 步数
# ============================================================================
# 领域专卡（用户 2026-10-08 裁定：废除 V3⊂V4⊂V5 的累积继承）
#   旧模式：数据累积继承 -> 每版数据集变大 -> 训练时间线性膨胀，不合理。
#   新模式：每张卡专注一个领域、内容互不重叠、各自从基座独立训；
#           使用时按场景切换单卡，或把多张卡合并成一个模型（见 cards/）。
# 数据集均已去重 + 过 token 长度门禁（见 eval/check_token_len.py）。
# 步数按各自数据量配比，约 1.5-1.9 epoch。
# ============================================================================
JOBS = [
    ("v3", "data/role_data_v3.jsonl", "out_lora_v3", 320),        # 小说设定卡（旧数据 575 话题，正在/已完成训练）
    # 补齐 V3 没学到的 55 个新话题：从 out_lora_v3 权重接着练（低 lr + 旧话题回放防遗忘），得到最终设定卡
    ("novel", "data/card_novel_delta.jsonl", "out_lora_novel", 74, "out_lora_v3", "8e-5"),
    ("prose_suspense", "data/card_prose_suspense.jsonl", "out_lora_prose_suspense", 62), # 恐怖悬疑 156 条
    ("prose_classic", "data/card_prose_classic.jsonl", "out_lora_prose_classic", 52),    # 古典仙侠 131 条
    ("prose_urban", "data/card_prose_urban.jsonl", "out_lora_prose_urban", 550),       # 通用叙事 6956 条(含新书6914)
    ("prose_adult", "data/card_prose_adult.jsonl", "out_lora_prose_adult", 56),          # 成人向 141 条(题材隔离)
    ("prose_adult_yc", "data/card_prose_adult_yc.jsonl", "out_lora_prose_adult_yc", 550), # 影川影视大世界 6914 条(独立成人子卡, 题材隔离)
    # 编程域进一步拆成 4 张更细的专卡（互斥、可单独重训/挂载）
    ("code_lang", "data/card_code_lang.jsonl", "out_lora_code_lang", 300),    # 语言与算法 748 条
    ("code_quant", "data/card_code_quant.jsonl", "out_lora_code_quant", 155), # 量化交易 384 条
    ("code_data", "data/card_code_data.jsonl", "out_lora_code_data", 75),     # 数据获取处理 187 条
    ("code_engine", "data/card_code_engine.jsonl", "out_lora_code_engine", 58), # 工程能力 144 条
    ("reason", "data/card_reason.jsonl", "out_lora_reason", 570), # 推理卡 1471 条
]

def free_mem_gb():
    """可用物理内存(含可回收 standby 缓存), 用 kernel32 GlobalMemoryStatusEx 取 ullAvailPhys。
    比 WMI FreePhysicalMemory 更准(后者不含 standby, 会误报 10MB 导致误判 OOM)。"""
    try:
        class MEMORYSTATUSEX(ctypes.Structure):
            _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                        ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                        ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                        ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                        ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            def __init__(self):
                self.dwLength = ctypes.sizeof(self)
        ms = MEMORYSTATUSEX()
        ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
        return ms.ullAvailPhys / (1024 ** 3)
    except Exception:
        return 99.0  # 取不到就放行, 不让内存闸门卡死流程


def log(msg):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    with open(CHAINLOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)

def pid_alive(pid):
    """os.kill(pid,0) 在 Windows 上是存活探针, 不真正发信号; 避开 tasklist 解码坑。

    🔴 关键: 「拒绝访问」不等于「进程已死」。os.kill 内部走 OpenProcess(PROCESS_ALL_ACCESS),
    权限不足时抛 WinError 5。旧代码 except OSError -> return False, 于是链在训练启动 1 秒后
    就判定"已死"并自杀(10:24 那次白空转 3 小时)。这里把 5/13 视为【存活】。
    """
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError as e:
        if getattr(e, "winerror", None) == 5 or getattr(e, "errno", None) in (5, 13):
            return True   # 权限不足, 进程仍在
        return False
    except Exception:
        return False

def write_pid(out, pid):
    try:
        with open(os.path.join(out, ".chain_pid"), "w", encoding="utf-8") as f:
            f.write(str(pid))
    except Exception:
        pass

def read_pid(out):
    try:
        with open(os.path.join(out, ".chain_pid"), encoding="utf-8") as f:
            return int(f.read().strip())
    except Exception:
        return None

def final_done(out):
    """整张卡是否真的跑完。

    🔴 分段训练后不能用「最终产物存在」判断：每段跑完 trainer 都会写一份
    adapter_model.safetensors，用它判断会让链条在第一段后就误判整卡完成、跳过剩余所有段。
    因此改看【步数标记文件】_segment_done 里记录的步数是否达到该卡的目标步数。
    若该目录还没进过 JOB_TARGET（例如被当作前置卡依赖），则退回归表为"已完成"，
    这样已完成的 out_lora_v3 之类不会把依赖它的卡卡死。
    """
    d = os.path.basename(os.path.normpath(out))
    target = JOB_TARGET.get(d)
    if target is None:
        return True
    marker = os.path.join(out, "_segment_done")
    if not os.path.isfile(marker):
        return False
    try:
        return int(open(marker, encoding="utf-8").read().strip()) >= target
    except Exception:
        return False

def write_segment_done(out, steps):
    try:
        with open(os.path.join(out, "_segment_done"), "w", encoding="utf-8") as f:
            f.write(str(steps))
    except Exception:
        pass

def checkpoints(out):
    return sorted(glob.glob(os.path.join(out, "checkpoint-*")))

def log_fresh(out, sec):
    lf = os.path.join(out, "_train.log")
    if not os.path.isfile(lf):
        return False
    return (time.time() - os.path.getmtime(lf)) <= sec

def wait_done(name, out):
    """轮询等待最终产物 adapter_model.bin/.safetensors; 训练进程一旦死亡且无产物立即判崩溃(不再空等)。

    崩溃判定要求【进程不活 且 日志也不新鲜】两个条件同时成立 —— 单看进程会被
    "拒绝访问"误判(见 pid_alive), 单看日志又会被卡死进程拖住。
    """
    waited = 0
    while True:
        # 只认「权重产物已落盘」作为本段结束，不认 _segment_done（那个由主循环写）
        if (os.path.isfile(os.path.join(out, "adapter_model.safetensors"))
                or os.path.isfile(os.path.join(out, "adapter_model.bin"))):
            return True
        pid = read_pid(out)
        alive = pid_alive(pid)
        fresh = log_fresh(out, FRESH_SEC)
        if not fresh:
            # 进程已死且日志也没在更新 -> 判定崩溃/被系统(OOM)回收 -> 交由 launch 立即重训
            log(f"{name} 训练进程(PID {pid})已死且日志停滞 -> 判定崩溃, 将重训")
            return False
        ck = checkpoints(out)
        time.sleep(60); waited += 1
        if waited % 10 == 0:
            log(f"{name} 轮询等待中... 已 {waited} 分钟 (pid活={alive}, 日志新鲜={fresh}, ckpt={len(ck)}, 完成={final_done(out)})")
        if waited >= TIMEOUT_MIN:
            log(f"{name} 超过 {TIMEOUT_MIN} 分钟未完成 -> 链停止(防卡死)")
            return False

def sweep_stray_training(my_out):
    """启动新训练前, 杀掉任何残留的 train_lora.py(指向同一输出目录), 确保单一训练写盘。
    双链/双 V3 竞态的兜底: 即使出现孤儿训练进程, 这里也会先清掉再拉新的。"""
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command",
             "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
             "Select-Object ProcessId,CommandLine | ConvertTo-Json -Compress"],
            capture_output=True, encoding="utf-8", errors="replace", timeout=30).stdout
        import json
        data = json.loads(out)
        if isinstance(data, dict):
            data = [data]
        target = os.path.normcase(os.path.abspath(my_out))
        for p in data:
            cl = (p.get("CommandLine") or "")
            pid = int(p.get("ProcessId", 0))
            if "train_lora.py" not in cl:
                continue
            # 仅清指向同一输出目录的残留
            if target.replace("\\", "/") not in cl.replace("\\", "/"):
                continue
            if pid == os.getpid():
                continue
            try:
                subprocess.run(["taskkill.exe", "/PID", str(pid), "/F"],
                              capture_output=True, timeout=15)
                log(f"sweep: 杀掉残留训练进程 PID {pid} (指向 {my_out})")
            except Exception:
                pass
    except Exception as e:
        log(f"sweep 跳过(异常 {e})")


def cleanup_residual(name, out):
    """清理上次失败/卡死的残留 checkpoint 与日志, 保证从基座干净重训。"""
    ck = checkpoints(out)
    if ck or os.path.isfile(os.path.join(out, "_train.log")):
        log(f"{name} 清理残留(checkpoint {len(ck)} 个 / 旧日志)")
        for c in ck:
            shutil.rmtree(c, ignore_errors=True)
        old = os.path.join(out, "_train.log")
        if os.path.isfile(old):
            try:
                os.replace(old, os.path.join(out, f"_train.log.bak_{int(time.time())}"))
            except Exception:
                pass

def last_checkpoint(out):
    """取步数最大的那个 checkpoint（sorted是字符串序，必须按数字取）。"""
    cks = checkpoints(out)
    if not cks:
        return None
    def step_of(p):
        m = re.search(r"checkpoint-(\d+)", p)
        return int(m.group(1)) if m else -1
    return max(cks, key=step_of)

def launch(name, data, out, steps, init=None, lr="2e-4"):
    if final_done(out):
        log(f"{name} 已完成, 跳过"); return True
    if not os.path.isfile(data):
        log(f"{name} 数据缺失({data}) -> 链停车待命, 放入该文件后重跑本脚本即可续训")
        return False
    if init and not final_done(init):
        # 续练依赖上一张卡；它没成型就跳过本步，别把后面的卡一起拖死
        log(f"{name} 依赖的前置卡未完成({init}) -> 跳过本步, 继续后续卡")
        return True
    pid = read_pid(out)
    alive = pid_alive(pid)
    fresh = log_fresh(out, FRESH_SEC)
    # 🔴 判据只用「日志新鲜度」，不用 PID 探针。
    # 训练每 ~60s 写一次 _train.log, 所以日志新鲜 == 正在推进, 这比 os.kill 可靠得多
    # (os.kill 走 OpenProcess(PROCESS_ALL_ACCESS), 权限不足时抛 WinError 5, 会被误判成"已死",
    #  一次误判就会清理掉正在跑的日志并再启一个 -> 曾白空转 3 小时)。
    # 代价: 真崩溃且日志停摆时要多等 FRESH_SEC 才动手, 换来的是绝不误杀健康训练。
    if fresh:
        log(f"{name} 检测到日志仍在更新 -> 判定训练进行中, 接管并等待完成")
        return wait_done(name, out)

    # ── 分段续训：有 checkpoint 就接着跑，绝不从头开始 ──
    ck = last_checkpoint(out)
    seg_remaining = steps
    if ck:
        m = re.search(r"checkpoint-(\d+)", ck)
        done_steps = int(m.group(1)) if m else 0
        if done_steps < steps:
            # 停下来的原因是「本段跑满」，不是崩溃 => 保留 checkpoint 接着跑下一段
            log(f"{name} 检测到可续训 checkpoint(step {done_steps}/{steps}) -> 从此处续训")
            seg_remaining = min(SEGMENT_STEPS, steps - done_steps)
            # trainer 侧要从 checkpoint 恢复 optimizer/scheduler 且跳过已训数据
            cmd_extra = ["--resume_from_checkpoint", ck]
        else:
            seg_remaining = steps
            cmd_extra = []
    else:
        # 没有 checkpoint：要么首段、要么崩溃需重来
        log(f"{name} 未运行且无残留 -> 直接启动")
        seg_remaining = min(SEGMENT_STEPS, steps)
        cmd_extra = []
    if seg_remaining >= steps:
        note = f"{steps} 步一次跑完"
    else:
        note = f"本次跑 {seg_remaining} 步后收尾落盘（约 {round(seg_remaining * 63 / 3600, 1)} 小时），下次自动续"
    cmd = [VE, "-u", "train_lora.py", "--model_path", BASE, "--data_path", data,
           "--output_dir", out, "--max_steps", str(seg_remaining), "--lora_r", "16", "--lora_alpha", "32",
           "--max_seq_length", "1024", "--per_device_batch_size", "1", "--grad_accumulation", "4",
           "--learning_rate", lr] + cmd_extra
    if init:
        # 接着某张已有卡继续练：保住它已学到的权重，只在这份新数据上往前走
        cmd += ["--init_adapter", init]
    # 内存闸门: 可用内存(含可回收缓存)不足则等待, 避免冲进去就被 OOM 杀
    free = free_mem_gb()
    while free < 3.0:
        log(f"{name} 可用内存仅 {free:.1f}GB < 3GB, 等待内存释放(防 OOM)...")
        time.sleep(60); free = free_mem_gb()
    log(f"=== 启动 {name} 训练(DETACH, {note}) ===")
    sweep_stray_training(out)  # 兜底: 先清掉可能残留的同目录训练进程
    # 输出目录必须先建好：旧目录都是手工建的，新卡第一次跑时 open(_train.log,'w') 会因目录不存在
    # 直接 FileNotFoundError 把整条链带崩（07:22 就这么白空转了3 小时）
    os.makedirs(out, exist_ok=True)
    # errors='replace' 关键: 训练任何非 UTF8 打印都不会让子进程崩溃
    with open(os.path.join(out, "_train.log"), "w", encoding="utf-8", errors="replace") as lf:
        proc = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, creationflags=DETACH)
    write_pid(out, proc.pid)
    log(f"{name} 已启动(DETACH, PID {proc.pid}), 轮询等待完成")
    return wait_done(name, out)

_LOCK_FD = None  # 持有 msvcrt 字节锁的文件描述符(进程存活期间不关闭)

def acquire_lock():
    """单例锁(Windows 原生): msvcrt.locking 字节范围锁是真正原子的, 且进程死亡自动释放。
    原子性优于"先查文件是否存在再创建"(存在 TOCTOU 竞态, 已踩坑: 双链 + 双 V3 抢同一输出目录)。
    Linux/无 msvcrt 时回退到 O_EXCL 原子创建。"""
    global _LOCK_FD
    if _HAVE_MSVC:
        try:
            fd = os.open(LOCK, os.O_CREAT | os.O_RDWR)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)  # 非阻塞; 已被锁则抛 OSError
            _LOCK_FD = fd
            os.ftruncate(fd, 0)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.fsync(fd)
            return True
        except (OSError, IOError) as e:
            log(f"抢锁失败({e}), 本实例退出避免双链抢跑")
            return False
    # 回退路径(非 Windows)
    try:
        fd = os.open(LOCK, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(str(os.getpid()))
        return True
    except FileExistsError:
        try:
            with open(LOCK, encoding="utf-8") as f:
                other = f.read().strip()
        except Exception:
            other = "?"
        log(f"锁已被其他实例(PID {other})抢占, 本实例退出")
        return False
    except Exception as e:
        log(f"抢锁异常({e}), 保守退出避免双链")
        return False

if __name__ == "__main__":
    if not acquire_lock():
        sys.exit(0)
    # 建立 卡目录名 -> 目标总步数 的映射（final_done 判定要用）
    for _j in JOBS:
        JOB_TARGET[os.path.basename(_j[2])] = _j[3]

    log(f"===== 蒸馏链启动(分段模式: 每段≤{SEGMENT_STEPS}步/~4h, 跑完一段即收尾, 下次自动续) =====")
    try:
        for job in JOBS:
            name, data, out, steps = job[0], job[1], job[2], job[3]
            if final_done(out):
                log(f"{name} 整卡已完成({steps} 步), 跳过")
                continue
            try:
                #启动前先把已完成的步数写进标记（续训段跑完后会覆盖为新累计值）
                ck = last_checkpoint(out)
                if ck:
                    m = re.search(r"checkpoint-(\d+)", ck)
                    if m:
                        write_segment_done(out, int(m.group(1)))
                if not launch(*job):
                    log(f"{name} 本段未完成 -> 链条收工, 等下次计划任务拉起续训")
                    break
                # 该卡跑完：把最终累计步数写进标记
                write_segment_done(out, steps)
                log(f"{name} 本段完成(累计 {steps}/{steps} 步)")
            except Exception:
                log(f"!! {name} 启动异常, 已跳过该卡继续后面的:\n{traceback.format_exc()}")
                continue
    finally:
        log("===== 蒸馏链本次运行结束 =====")
        try:
            if os.path.isfile(LOCK):
                os.remove(LOCK)
        except Exception:
            pass
