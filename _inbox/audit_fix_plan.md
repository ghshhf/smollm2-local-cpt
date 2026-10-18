# millionFund 审计修复方案（委员会终裁版）

> 48 项审计问题的处置总纲。每题经「六席讨论（提案甲/乙 + 前沿 + 审计甲/乙 + 收口）→ 主席终裁（代码实证 + 方案纠偏）」两阶段产出。  
> 终裁原则：**磁盘实证优先于任何模型陈述**（本题单中 5 例委员会事实错误、4 例「已修」伪状态均由主席 grep/GitHub API 推翻）。  
> 版本：v1.1（2026-10-16 03:2x）——已终裁 29/48；剩余 19 题：H8 六席已收口待裁（1）；M10-M16 + L1-L11 需补跑六席再裁（18，流水线已停待重启；M10 上次停在审计甲 403 降级，需从 S4 续跑）。

---

## 一、总览

### 1.1 处置状态

| 分区                      | 题数     | 已终裁              | 必改     | 伪报/关闭               | 待处理            |
| ----------------------- | ------ | ---------------- | ------ | ------------------- | -------------- |
| CRITICAL + CI（C/H9/H10） | 10     | 10               | 8      | 2（H10、H2 计入 HIGH 区） | 0              |
| HIGH                    | 13     | 12（H1-H7/H9-H13） | 11     | 1（H2）               | 1（H8 六席已收口待裁）  |
| MEDIUM                  | 16     | 9（M1-M9）         | 9      | 0                   | 7（M10-M16 需补跑） |
| LOW                     | 11     | 0                | —      | —                   | 11（L1-L11 需补跑） |
| **合计**                  | **48** | **29**           | **27** | **2**               | **19**         |

### 1.2 已终裁题目结论速查

| 题       | 结论             | 一句话方案                                                                                                                               |
| ------- | -------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| C1      | 必改             | 缓存 TTL 秒/毫秒混用 4 处；PR-2 最小修，缓存基建重构列后续                                                                                                |
| C2      | 必改             | ESM 中 require() 改 static import + async 签名 + Composable 形态                                                                          |
| C3      | 必改             | Trade Store 异步函数同步调用；代际守卫正确写法（utils/storage.ts）                                                                                     |
| C4      | 必改             | useTimer cleanup 存在但零注册；补 onScopeDispose（注释撒谎实锤）                                                                                    |
| C5      | 必改             | `::description` → `:description`；records 是 computed                                                                                 |
| C6      | 必改             | Alerts.vue `</span>/span>`、`</p>/p>` 垃圾文本删行                                                                                         |
| C7      | 必改·**P0 渲染崩溃** | `placeholders.value` 对 computed 内部局部 const 取 .value ⇒ 页面打不开；一行修 + 测试重写                                                              |
| C8      | 必改             | Trades.vue i18n 绑定缺冒号仅 L78/L94/L95 三处（netValue 字段已被外部修过）                                                                            |
| H9      | 必改·**CI 阻塞**   | actions 空号版本 11 处：checkout@v5→v7、setup-node@v5→v7、setup-java@v5→v6、download-artifact@v7→v8、github-script@v7→v8；版本号经 GitHub API 实时核验 |
| H10     | 伪报·关闭          | rename 目标与 upload path 一致，无代码改动                                                                                                     |
| H11+H12 | 必改             | manualChunks 分包 + tesseract.js 动态化，合并为一个构建优化 PR                                                                                     |
| H13     | 必改             | localStorage 写入 try/catch 降级                                                                                                        |
| H1      | 必改             | 5 组件 searchTimer 补 onUnmounted 清理（HoldingEdit 插在 onSearchInput 后，非收口所说 watch 后）                                                     |
| H2      | **伪报·关闭**      | beforeinstallprompt/appinstalled 全项目零命中，监听器不存在                                                                                      |
| H3      | 必改             | refreshEstimates 补 `if (isRefreshing) return` 守卫                                                                                    |
| H4      | 必改             | finish() 闭包内 settled 幂等标记（收口的模块级 guard 方案无效，已纠正）                                                                                    |
| H5      | 必改             | initHolidayData 改 Promise 缓存单飞（收口「标记后移」方案会引入重入，已否）                                                                                  |
| H6      | 必改             | useDefaultWebSocket 引用计数；清理责任收拢到单例函数（onUnmounted 真实注册点收口说错）                                                                         |
| H7      | 必改             | 4 个 API 文件串行 for-await 改 Promise.allSettled（bond.ts 仅升级；不引 p-limit）                                                                 |
| M1      | 必改             | newsMap 全文件零消费 = 死字段，删（不是响应式问题——委员会定性错误）                                                                                            |
| M2      | 必改             | 虚拟滚动死代码整段删 + exportToCSV 补 setTimeout revoke                                                                                        |
| M3      | 必改             | FundInfo 无 return/rating/risk 字段 ⇒ 收口数值方案编译不过；改 TODO 诚实化                                                                            |
| M4      | 必改             | 删两条硬编码假行情，改 futuresError 错误态 + 重试                                                                                                   |
| M5      | 必改             | apiKeyVisible ref 新增 + 取反 + :type 绑定 + 图标联动                                                                                         |
| M6      | 必改             | 三元两分支同值修复 + 补 switch_success i18n 键（收口漏了键）                                                                                          |
| M7      | 必改             | 删 `v-if="true"`                                                                                                                     |
| M8      | 必改             | 三死代码文件（734 行）直接删；「合并回 holding.ts」「require bug」系收口臆想                                                                                 |
| M9      | 必改             | fetch → http 封装（文件内已 import）；axios 新建方案作废                                                                                           |

### 1.3 委员会质量问题台账（供后续复盘）

| 病症      | 实例                             | 成因                      |
| ------- | ------------------------------ | ----------------------- |
| 「已修」伪状态 | H1/H3/M6/M7/M8/M9 六题收口标已修，磁盘原样 | openai 通道收口的通病：爱下结论不问磁盘 |
| 定性错误    | M1「响应式失效」实为死代码                 | 不 grep 消费方就编影响链         |
| 方案缺一半   | M6 引用不存在的 i18n 键               | 不核 locales 文件           |
| 字段臆想    | M3 对 FundInfo 不存在的字段写数值比较      | 不读 types/fund.ts        |
| 锁设计错误   | H4 用模块级布尔做单次守卫                 | 把锁和被保护数据混为一谈            |
| 凭空施工    | H2 给不存在的监听器写移除补丁               | 审计基线过期，不重新 grep         |
| 版本号凭记忆  | H9 一度称 upload-artifact@v7 不存在  | 不查 GitHub API           |

**流程修正（已生效）**：终裁前主席对每题的「需核验点」逐一 grep/API 取证；凡收口标「已修」的一律先证伪再采信。

---

## 二、PR 切分与合入顺序

遵循「一个功能一个 PR」；按风险排优先级：

### 第一批 · 阻塞级（先合）

| PR    | 题  | 内容                                     | 理由            |
| ----- | -- | -------------------------------------- | ------------- |
| PR-01 | H9 | CI actions 版本 11 处替换（sed 一行版见终裁稿）      | CI 全挂，挡所有后续验证 |
| PR-02 | C7 | HoldingEdit.vue placeholders.value 一行修 | 渲染即崩，用户可见 P0  |

### 第二批 · CRITICAL 修复（C1-C6、C8）

| PR    | 题        | 内容                                           |
| ----- | -------- | -------------------------------------------- |
| PR-03 | C2       | ESM require → static import                  |
| PR-04 | C3       | 代际守卫                                         |
| PR-05 | C4       | onScopeDispose 注册                            |
| PR-06 | C1       | TTL 单位最小修                                    |
| PR-07 | C5/C6/C8 | 三个一行级文本/语法修（同属「笔误类」，合并无耦合风险——如需严格一功能一 PR 可拆） |

### 第三批 · HIGH 修复

| PR    | 题        | 内容                                                      |
| ----- | -------- | ------------------------------------------------------- |
| PR-08 | H11+H12  | 构建优化（manualChunks + tesseract 动态化）                      |
| PR-09 | H13      | localStorage 降级                                         |
| PR-10 | H1/H6    | 清理责任类：5 组件防抖定时器清理 + WebSocket 单例引用计数                    |
| PR-11 | H3/H4/H5 | 三个单飞守卫（同属竞态类：isRefreshing 标志 / settled 标记 / Promise 缓存） |
| PR-12 | H7       | 4 个 API 文件串行 N+1 → Promise.allSettled（不引 p-limit）       |
| PR-13 | 覆盖率      | vitest 阈值 6%→60% 目标 + 各 PR 测试稿归位（M16 终裁后并入）             |

### 第四批 · MEDIUM 清理

| PR    | 题        | 内容                                   |
| ----- | -------- | ------------------------------------ |
| PR-14 | M2/M8    | 死代码删除（虚拟滚动段 + 三 store 文件）            |
| PR-15 | M4/M5/M6 | 三个 UI 行为修复（错误态/可见性/语言切换）             |
| PR-16 | M1/M3/M7 | 三个诚实化清理（死字段/TODO/死指令）                |
| PR-17 | M9       | http 封装替换（前置：核验 utils/http.ts 文本方法名） |

### 第五批 · LOW（流水线终裁后定）

---

## 三、遗留清单（不在本周期修，登记造册）

| 编号   | 内容                                                         | 来源 |
| ---- | ---------------------------------------------------------- | -- |
| L-01 | crossValidation 整对象退役（News.vue L69-72 定义 + L513 写入行）       | M1 |
| L-02 | FundInfo 补 return/rating/risk 字段 + FundScreener 真排序        | M3 |
| L-03 | M9 若 utils/http.ts 无文本型方法：qt.gtimg.cn 维持 fetch + 注释说明      | M9 |
| L-04 | 迟到请求 AbortController 取消（http.text resolve 后掐断 extract 副作用） | H4 |
| L-05 | C1 缓存基建重构（PR-2 只做最小修）                                      | C1 |
| L-06 | PWA 安装提示若重加：onMounted/onUnmounted 配对注册                     | H2 |
| L-07 | initHolidayData API 全失败时允许重试（catch 置空 holidayInitPromise）  | H5 |
| L-08 | 若目标站观测到 429：自制 ≤10 行并发池，不引三方包                              | H7 |

---

## 四、附：终裁稿索引

每题完整终裁（含代码 diff、测试稿、核验记录）在 `_产出/Q_<题号>_S7_主席终裁.md`：

C1-C8、H1-H7、H9、H10、H11_H12（合并稿 Q_H11_H12_S7）、H13、M1-M9。

待裁：H8（六席已收口，随时可裁）；M10-M16、L1-L11 需先补跑六席（流水线 03:0x 停止，M10 停在审计甲 403 降级，需从 S4 续跑）。收口后同目录追加，本方案升 v2 时更新总览表。
