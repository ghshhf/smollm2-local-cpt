# 记忆层部署手册（换机器照抄这份）

> **先回答"用哪个"**：用 **TinyContext**，PyPI 包名 `tinysuite-context`，**钉死 0.5.1**。
> 不是 MemGPT，不是 Letta，不是 Zep，也不是 `mem0ai`（那个装了但没用）。
>
> 本手册每个命令都在本机实测跑过一遍（2026-10-18 05:5x，干净目录从零重建），
> 不是照抄 README —— 包括三个官方文档没写、但一定会踩的坑。

| 项 | 值 |
|---|---|
| 项目 | TinyContext |
| GitHub | `github.com/TinySuiteHQ/TinyContext` |
| PyPI | `tinysuite-context` |
| **版本（钉死）** | **0.5.1** |
| 许可证 | MIT |
| 存储 | 单文件 SQLite（`memories` 表 + `memories_fts` FTS5 虚表） |
| 检索 | FTS5(BM25) + sqlite-vec(dense) 混合，RRF 融合 |
| 依赖 | 10 个：huggingface_hub / numpy / onnxruntime / openai / opentelemetry-api / platformdirs / sqlite-vec / tiktoken / tokenizers / urllib3 |
| 实测环境 | Windows 11 + **Python 3.13.14** + CPU（无 GPU 也跑） |
| 落盘位置 | `%LOCALAPPDATA%\tinycontext\`（Linux/macOS 走 `platformdirs` 的等价路径） |

---

## 一、装（3 条命令）

**必须独立 venv。** 它的 `huggingface_hub==0.36.0` 依赖与训练环境的 transformers 5.x 互斥，
装进训练 venv 会直接 `ImportError: ResolvedRevision`，本机已踩过两次。

```bash
# 1) 独立环境（不要复用训练 venv）
python -m venv venv_mem
# Windows
venv_mem\Scripts\activate
# Linux/macOS
source venv_mem/bin/activate

# 2) 升级 pip（旧 pip 解析不了它的一些依赖标记）
python -m pip install -U pip

# 3) 钉版本安装
pip install "tinysuite-context==0.5.1"
```

装完校验：

```bash
python -c "import tinycontext, importlib.metadata as m; print('tinycontext ok', m.version('tinysuite-context'))"
# 期望：tinycontext ok 0.5.1
```

> `pip download "tinysuite-context==0.5.1" --no-deps` 实测拿到
> `tinysuite_context-0.5.1-py3-none-any.whl`（63 KB，纯 Python，无编译，任何平台通用）。

---

## 二、中文 embedding（**必做，否则中文检索等于废**）

### 为什么必须换

TinyContext 只内置三个预设，**全是英文模型**：

| 预设名 | 实际模型 | 对中文 |
|---|---|---|
| `fast` | onnx-models/all-MiniLM-L6-v2-onnx | 弱 |
| `balanced`（**默认**） | BAAI/bge-small-en-v1.5 | **弱**：问"主人叫什么名字"召不回中文条目 |
| `quality` | BAAI/bge-base-en-v1.5 | 弱，且更慢 |

所以必须指定自定义模型名 **`BAAI/bge-small-zh-v1.5`**（512 维，24M 参数，BertModel，4 层）。

### 但自定义名不能"一键下全"——要手工拼两半

原因（读源码确认，`services/embedding_service.py`）：自定义模型的 `allow_patterns`
固定为 6 个文件名（`model.onnx / tokenizer.json / tokenizer_config.json /
special_tokens_map.json / vocab.txt / config.json`），而：

- **官方 `BAAI/bge-small-zh-v1.5` 仓库里没有 `model.onnx`**（只有 pytorch_model.bin / safetensors）
- **现成 onnx 仓库 `zc277584121/bge-small-zh-v1.5.onnx` 里只有 `model.onnx`**，没有 tokenizer

⇒ 各取一半，拼进同一个目录。

### 步骤

```bash
# 目标目录名由包自动算：模型名小写、非法字符换 "-"，末尾加 "-onnx"
# BAAI/bge-small-zh-v1.5  ->  baai-bge-small-zh-v1.5-onnx
```

**Windows（PowerShell / cmd 均可，这里是 bash 语法）**

```bash
export MODELS_DIR="$LOCALAPPDATA/tinycontext/models"
export DST="$MODELS_DIR/baai-bge-small-zh-v1.5-onnx"
mkdir -p "$DST"

# 代理（国内必须；走 Karing 的 3067，或你自己的出口）
export HTTPS_PROXY=http://127.0.0.1:3067
export HTTP_PROXY=http://127.0.0.1:3067
unset HF_ENDPOINT          # 重要：不要叠 hf-mirror，两者同时开会 308 卡死

python - <<'PY'
import os
from huggingface_hub import hf_hub_download
dst = os.environ["DST"]
# ① 官方仓库拿 tokenizer + config
for f in ["config.json", "tokenizer.json", "tokenizer_config.json", "vocab.txt"]:
    hf_hub_download("BAAI/bge-small-zh-v1.5", f, local_dir=dst)
    print("ok", f, os.path.getsize(os.path.join(dst, f)))
# ② 第三方仓库拿 onnx 权重
hf_hub_download("zc277584121/bge-small-zh-v1.5.onnx", "model.onnx", local_dir=dst)
print("ok model.onnx", os.path.getsize(os.path.join(dst, "model.onnx")))
PY
```

**实测耗时**：tokenizer 4 件 **1.9 秒**，model.onnx（95.9 MB）**10.0 秒**。

### 校验（尺寸对不上就是拼错了）

目录 `baai-bge-small-zh-v1.5-onnx/` 必须正好这 5 个文件：

| 文件 | 字节数 | 来源 |
|---|---|---|
| `model.onnx` | **95,915,453** | `zc277584121/bge-small-zh-v1.5.onnx` |
| `tokenizer.json` | 439,125 | `BAAI/bge-small-zh-v1.5` |
| `tokenizer_config.json` | 367 | 同上 |
| `vocab.txt` | 109,540 | 同上 |
| `config.json` | 776 | 同上 |

> ⚠️ **`config.json` 别用错的**：必须含 `"hidden_size": 512`、`"num_hidden_layers": 4`、
> `"model_type": "bert"`、`"vocab_size": 21128`。文件里 `_name_or_path` 会是
> `BAAI_bge-small-zh` 之类，不影响使用。
>
> ⚠️ **别用 `onnx-community/bge-small-zh-v1.5-ONNX`**：它把权重拆成 `onnx/model.onnx` +
> `onnx/model.onnx_data` 外挂文件，而自定义模式的下载白名单**不含 `*_data`**
> ⇒ 下到 41 KB 空壳，加载时报 onnx 解析失败。本机踩过。

---

## 三、三个官方文档没写的坑（**必读**）

### 坑 1（最致命）：**Python API 不读环境变量**

`tinycontext.config.resolve_config()` 的 docstring 原文就写着
`"without reading the environment"`。

| 用法 | 环境变量生效吗 |
|---|---|
| `python -m tinycontext doctor / mcp / serve`（CLI） | ✅ 生效（走 `load_context_config`） |
| `tinycontext.save_memories() / recall_memories()`（Python API） | ❌ **完全忽略** |

后果：你设了 `TINYCONTEXT_MEMORY_DB_PATH` 指向新库，API 仍然写进默认库，
而且会"莫名其妙"命中旧数据。本机实测踩到——save 报 `duplicate_of` 指向另一个库的 id。

**正解：API 一律显式传 `config=`**

```python
import os, tinycontext as tc

cfg = tc.TinyContextConfig(
    memory_db_path=os.path.expandvars(r"%LOCALAPPDATA%\tinycontext\memories.db"),
    models_dir=os.path.expandvars(r"%LOCALAPPDATA%\tinycontext\models"),
    embedding_model="BAAI/bge-small-zh-v1.5",
)
```

### 坑 2：`kind` 分两种，只有一种走检索

| `kind` | 行为 | 受哪个参数限 |
|---|---|---|
| `profile`（身份/长期事实） | **不检索，每轮无条件注入** | `profile_max_tokens`（默认 500） |
| `episodic`（**默认值**，事件/片段） | 走 FTS+dense 检索，命中才注入 | `recall_top_k`(10) / `recall_max_tokens`(2000) |

所以把身份写进 `kind="profile"` 后，用 `recall_memories("他是谁")` 查 `memories` 字段是**空的**，
内容在返回值的 **`profile`** 字段里。这不是 bug，是设计。
（本机现在那 7 条身份记忆全是 `profile` ⇒ 每轮常驻。）

### 坑 3：hf-mirror 直连已失效，走代理官方

本机 2026-10-18 实测：

| 路径 | 结果 |
|---|---|
| `HF_ENDPOINT=https://hf-mirror.com` 直连 | ❌ `LocalEntryNotFoundError`（连不上） |
| 代理 `127.0.0.1:3067` → `huggingface.co` 官方 | ✅ 1.9s + 10.0s 下完 |
| 代理 + hf-mirror 同时开 | ❌ 308 循环 |

结论：**上代理走官方，别设 `HF_ENDPOINT`**。以前 hf-mirror 直连更快，现在反了。

---

## 四、自检（一条命令，全部 ok 才算装好）

```bash
# Windows
set TINYCONTEXT_MODELS_DIR=%LOCALAPPDATA%\tinycontext\models
set TINYCONTEXT_EMBEDDING_MODEL=BAAI/bge-small-zh-v1.5
python -m tinycontext doctor
```

期望输出（本机实测原样）：

```
config: ...\context_config.json (not found, using built-in defaults)
database: ...\tinycontext\memories.db
embedding model: BAAI/bge-small-zh-v1.5 (BAAI/bge-small-zh-v1.5, local ONNX)
model directory: ...\models\baai-bge-small-zh-v1.5-onnx
model bundle: ready
embedding drift: none
data directory: ok - ... is writable
sqlite: ok - SQLite 3.53.1 is available
sqlite-vec: ok - sqlite-vec v0.1.9 is available
all checks passed
```

三个关键行，任一不对就是没配好：

- `model bundle: ready` —— 没 ready 就是第二节拼错了
- `embedding drift: none` —— 报 WARNING 说明存量记忆是用别的模型嵌入的，会在下次 recall 时**同步重嵌入（大数据集会卡很久）**
- `sqlite-vec: ok` —— 缺了说明 `sqlite-vec` 没装上，dense 分支失效，只剩字面检索

---

## 五、最小可用脚本（存 + 召回，可直接跑）

```python
import os, tinycontext as tc

DATA = os.path.expandvars(r"%LOCALAPPDATA%\tinycontext")
cfg = tc.TinyContextConfig(
    memory_db_path=os.path.join(DATA, "memories.db"),
    models_dir=os.path.join(DATA, "models"),
    embedding_model="BAAI/bge-small-zh-v1.5",
)

# 身份类：常驻注入
tc.save_memories([
    tc.MemoryInput("模型名叫北极星，是主人起的名字", kind="profile"),
    tc.MemoryInput("称呼用户为主人", kind="profile"),
], config=cfg)

# 事件类：走检索
tc.save_memories([
    tc.MemoryInput("主人 10 月 18 日把 0.3B 权重从魔搭拉回本地，MD5 与远程一致"),
], config=cfg)

res = tc.recall_memories("权重是怎么取回来的", top_k=3, config=cfg)
print("常驻身份:", [m["content"] for m in res["profile"]])
print("检索命中:", [(m["content"][:30], m["relevance"]) for m in res["memories"]])
```

**API 全签名**（实测打印）：

```
save_memories(memories: Sequence[MemoryInput], *, session_id=None, config=None) -> dict
recall_memories(query=None, *, session_id=None, max_tokens=None, top_k=None, config=None) -> dict
list_memories(*, session_id=None, kind=None, since=None, until=None, limit=None, offset=0, sort="recent", config=None) -> dict
get_memory(memory_id, *, config=None) -> dict
update_memory(memory_id, content, *, config=None) -> dict
delete_memory(memory_id, *, config=None) -> dict      ← E-7 的删除就是这一行
MemoryInput(content: str, kind: str = "episodic")
```

返回结构：`recall_memories` 给 `{query, current_time, memories, total_tokens, truncated, matched_count, profile}`，
其中 `memories` 是检索命中、`profile` 是常驻身份，两者都要拼进 prompt。

---

## 六、实测数字（本机，CPU，2026-10-18）

| 项 | 实测值 |
|---|---|
| 装包体积（wheel） | 63 KB |
| 中文 embedding 拼装 | 95.9 MB / 11.9 秒 |
| `doctor` 全项 | all checks passed |
| 存 3 条（含首次加载 onnx） | **2.3 秒** |
| 召回单次（热） | **0.01~0.02 秒** |
| embedding 落盘 | 每条 **2048 字节** = 512 × float32，dim=512 |
| 纯语义召回（字面零重叠） | **2/3 正确**：`迭代次数` → 正确命中"6 epoch 共 2196 步"；`拉取文件的渠道` / `没能记住训练数据` 未命中 |
| `recall_dense_weight` 0.5 / 0.9 / 1.0 | 排序**完全不变**（3 个 query 全一致）⇒ 现阶段别调它，收益为零 |

**老实结论**：中文能用，但别指望惊艳。字面重叠强的查询很准；
纯同义改写只能命中一部分（24M 的小 embedding，属正常水平）。
真正的提升方向是换更大的中文 embedding（如 bge-large-zh-v1.5，需自己导出 onnx），
而不是调融合权重。

---

## 七、迁移到另一台机器：只拷一个文件

记忆全在 **`memories.db`** 一个文件里（65 KB 量级）。换机流程：

```bash
# 源机
copy "%LOCALAPPDATA%\tinycontext\memories.db"  memories_backup.db

# 目标机（先按第一、二节装好，再覆盖）
copy memories_backup.db "%LOCALAPPDATA%\tinycontext\memories.db"
python -m tinycontext doctor     # 确认 embedding drift: none
```

如果目标机的 `embedding_model` 与源机不同，`doctor` 会报 drift，
下次 recall 时**全量同步重嵌入**——记忆条数上千时会卡几十秒到几分钟，属正常。

---

## 八、为什么是它（选型依据，一句话版）

**E-7 硬门槛**：个人数据必须**逐条可寻址、可删除**。
`memories.db` 是单文件 SQLite，删一条就是 `delete_memory(id)` 或一行 `DELETE`，
删完可 `VACUUM`，无残留副本；FTS 虚表随主表联动。
⇒ Metis（arXiv 2607.26760）那种把记忆压进潜状态/权重的方案，**删不干净，直接出局**；
Titans 式 persistent memory 同理。

**为什么不选 mem0**：`mem0ai 2.2.1`（Apache-2.0）要 LLM 参与记忆抽取 + 独立向量库，
16 GB 无 GPU 的机器上是过度工程。**留着当对照组，不当主线。**

---

## 九、卸载

```bash
pip uninstall -y tinysuite-context
# 数据（记忆本体 + embedding 模型）
rmdir /s /q "%LOCALAPPDATA%\tinycontext"     # Linux: rm -rf ~/.local/share/tinycontext
```

记忆与模型都在用户目录，不散落在 site-packages，卸载无残留。

---

## 十、出处（本手册的数据从哪来）

| 结论 | 依据 |
|---|---|
| 版本 / 依赖 / 许可证 | `pip show tinysuite-context` 实测 |
| 三个预设全是英文 | 读 `services/embedding_service.py` 的 `_PRESET_MODELS` |
| 自定义模型只下 6 个文件名 | 同文件 `_COMMON_ONNX_ALLOW_PATTERNS` |
| API 不读环境变量 | 读 `config.py:resolve_config` docstring |
| profile 不走检索 | 读 `pipelines/memory_recall.py` + 实测返回结构 |
| 召回正确率 2/3、0.01s | 干净目录实测（详见 §六） |
| onnx 仓库只有 model.onnx | `huggingface.co/api/models/zc277584121/bge-small-zh-v1.5.onnx/tree/main` |
