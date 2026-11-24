# 任务与 Skill 使用指南

> **本文用途**：提供任务、操作文档与 Skill 的对应关系。选择与当前目标匹配的入口，按操作文档中的调用示例执行。

## 统一入口与输入补全

不确定该选哪个 Skill，或希望从一个问题开始推进时，先读取 [`hcu-plugin-assistant`](../skills/hcu-plugin-assistant/SKILL.md)。它会检索本仓现有 Skill、操作指南与经验，选择匹配入口并继续专项任务。它不替代七个专项 Skill，也不要求所有任务按同一流程执行。

**调用方式**：Agent 已打开方法库时，发送下面的请求；否则补充方法库的实际位置。

```text
读取 skills/hcu-plugin-assistant/SKILL.md。
我的目标：<描述要开发、安装、排查或交付的内容>。
请检索相关 Skill 和方法；优先读取已有信息，缺少必要输入时向我提问后继续。
```

| 用户输入 | 匹配方式 | 可能需要补充的信息 |
|---|---|---|
| “怎么开始开发新插件？” | 检索工程初始化指南，解释最小案例 | 方法咨询不要求先提供机器路径 |
| “帮我创建新插件工程。” | `hcu-plugin-scaffold`，必要时再衔接开发 Skill | 官方源码与目标版本、外部新目录、首项功能与执行环境 |
| “wheel 装上了，但找不到入口。” | `hcu-plugin-install`，核对解释器、metadata 和实际来源 | 产物和环境可先检查；不可访问的信息再补问 |
| “已加载，但 TP4 Graph 输出异常。” | `hcu-plugin-triage`，按症状检索经验 | 触发配置、实现仓、已有日志、预期结果；实际 GPU 验证前再确认资源 |
| “生成内嵌源码并比较 HCU 差异。” | `hcu-plugin-backtrack`，结合逻辑对照章节 | 两个生成输入、新输出位置及独立 HCU 比较来源 |

需要补问时，助手说明问题将影响哪个执行步骤。能从上下文或可访问环境读取的路径、HEAD、dirty 和日志不重复索取；非必需的命名等常规选择由专项 Skill 处理。等待必要回答时继续独立的只读检查，依赖该信息的写入或服务启动保持待补充。信息齐备后直接继续已授权工作。

人工阅读可直接使用下表；已经知道专项 Skill 时直接调用它即可。统一入口本身不会自动安装依赖、修改客户端 Skill 配置或启动后台任务。完整案例见 [根 README](../README.md#4-开发使用案例)。

## 1. 任务分类与文档入口

| 任务类别 | 推荐顺序 |
|---|---|
| 运行时使用 | [安装与启用](installation.md) → 安装与入口核验 → 设备及加载验证 → 要求的服务验收 |
| 插件开发 | [工程初始化](first-plugin.md)（新工程）或 [能力开发](plugins.md)（已有工程）→ 实现与针对性验证 |
| 故障诊断 | [故障诊断](troubleshooting.md) → 按症状选择参考 → 按需查询 [历史经验](development-methodology.md) |
| 迁移与交付 | [三方迁移](migration.md)、[版本升级](upgrading.md) 或 [源码回溯](backtracking.md)，按目标选择 |

[架构](architecture.md) 用于接口与组件设计，[治理](governance.md) 用于实现归属和 adapter 交互，[验证](validation.md) 用于确定验收范围。完整文档职责见 [文档中心](README.md)。

## 2. Skill 适用范围与交付要求

| 任务 | Skill | 执行说明与调用示例 | 主要交付 |
|---|---|---|---|
| 创建独立插件工程 | [hcu-plugin-scaffold](../skills/hcu-plugin-scaffold/SKILL.md) | [工程初始化](first-plugin.md#使用-skill-创建新插件) | 外部工程、入口映射、实际调用与启停结果 |
| 安装已有插件源码或 wheel | [hcu-plugin-install](../skills/hcu-plugin-install/SKILL.md) | [安装与验证](installation.md#使用-skill-安装与验证插件) | 安装与环境记录、入口及加载报告 |
| 新增或修改已有插件能力 | [hcu-plugin-dev](../skills/hcu-plugin-dev/SKILL.md) | [能力开发](plugins.md#使用-skill-开发插件能力) | 实现、注册/回退契约、相关测试及证据 |
| 定位加载后的运行回归 | [hcu-plugin-triage](../skills/hcu-plugin-triage/SKILL.md) | [故障诊断](troubleshooting.md#使用-skill-执行故障诊断) | 最小复现、首次分叉、修复及回归结果 |
| official / fork / plugin 差异审计 | [sglang-hcu-plugin-migration](../skills/sglang-hcu-plugin-migration/SKILL.md) | [三方迁移](migration.md#使用-skill-执行三方迁移) | 固定基线、差异清单、处置台账 |
| 指定官方目标升级与发版 | [upgrading-sglang-hcu-upstream-first](../skills/upgrading-sglang-hcu-upstream-first/SKILL.md) | [版本升级](upgrading.md#使用-skill-执行版本升级) | 补丁处置、验收矩阵、产物来源与 last-good |
| 生成内嵌源码产物 | [hcu-plugin-backtrack](../skills/hcu-plugin-backtrack/SKILL.md) | [源码回溯](backtracking.md#使用-skill-生成与核验内嵌源码) | 新源码树、来源清单、哈希及运行验证记录 |

## 3. 调用规则与任务衔接

向 Agent 提供方法库绝对路径，并使用对应文档中的请求模板。路径方式不依赖自动发现；`$skill-name` 仅适用于客户端已发现的 Skill。保留完整仓库，避免共享相对路径失效。

工程初始化 Skill 负责包与接入链；真实 HCU 功能由开发 Skill 负责。安装 Skill 区分安装、入口、设备和加载；加载后出现数值、状态、Graph 或性能问题时转诊断 Skill。三方迁移负责差异处置，指定官方升级与发版由升级 Skill 组织，要求 embedded 交付时再调用回溯 Skill。

普通安装不要求迁移或回溯；已有插件新增能力不要求重新创建工程。Skill 的任务输入和验收标准分别定义在对应操作文档与 `SKILL.md` 中，缺少本次必需输入时先检查已有上下文，再补充必要信息。

## 4. 参考资料与历史证据

- 方法工具：[环境准备](bootstrap.md)、[三方迁移](migration.md)、[源码回溯](backtracking.md)。
- 工程规范：[配置管理](configuration.md)、[验证规范](validation.md)、[协作流程](contributing.md)。
- 历史故障：[开发经验与故障案例索引](development-methodology.md)、[V3 实践记录](v3-runtime-validation.md)。
- 方法来源与交付：[vLLM Skill 迁移分析](vllm-skill-migration.md)、[交付记录](delivery-validation.md)、[等价性验证报告](equivalence-validation-report.md)。

当前需求的实现范围、验证结果和 HCU 逻辑差异见 [V3 2026-10-26 实施记录](hcu-sglang-development.md#2026-10-26-v3-2026-10-26-实施与验证记录)。历史经验用于定位相似问题，当前支持范围以对应固定版本的实施记录为准。

团队需要将 DAS 按模块转为插件时，先查看 [模块盘点与开发计划网页](../migration-plan.html) 和 [维护说明](sglang-das-migration-plan.md)，再用三方迁移 Skill 评审本轮范围。网页的 78 条建议项需认领和裁剪，不能按静态映射计数推定已迁移或未支持。

按阶段实施全量 DAS 迁移时，三方迁移 Skill 维护 [执行台账](migration.md#全量差异执行台账)，开发 Skill 实施本仓 `hcu_sglang`；每个阶段的实现与验证结果写入 [开发记录](hcu-sglang-development.md)。

## 5. 与 adapter 仓的文档分工

插件运行在 `adapter-das` 仓（影子库 + `hcu_adapter` 运行时桥）之上，两仓
文档不互为备份：**机制层事实以 adapter 仓为准**（加载时序、影子库、平台
语义、算子档位与目录、环境键、后端选择）；**插件策略层与开发流程以本仓
为准**（Hook 注册契约与 `original_fn` 语义、工程与迁移台账、组合验证
矩阵）。跨仓引用写"仓坐标"（仓 + 文档路径 + 节号），不放跨仓相对链接。

同名文档分工对照（6 对）：

| 文档 | adapter-das 仓 | 本仓 |
|---|---|---|
| architecture.md | 机制分层：影子库 / is_cuda 语义 / 三档路由 / 桥结构 | 插件架构 + 未来运行时目录规划 |
| governance.md | 治理宪法：双仓判别（真相源）、裁决链、§7 共建 | 组合兼容契约：目标所有权、验证矩阵（本仓 §2-§4） |
| bootstrap.md | **宿主**新容器部署手册（依赖钉 / tvm-ffi 预热 / 自检） | 方法库开发环境准备 |
| ci.md | L0/L1/L2 宿主门禁体系 | 方法仓 CI 阶段约束 |
| plugins.md | 需求登记表 P1-P10：**哪些项必须走 plugin** | 能力开发指南：**拿到项之后怎么做** |
| reading-guide.md | 机制侧导航 | 插件方法侧导航（本页） |

归仓判据（一项工作归哪仓）以 adapter-das `docs/governance.md` §0 判别
问句为准；本仓 [治理 §1](governance.md) 与 adapter 仓 `docs/plugins.md`
§0 是它的两个应用视角，三处任一改动同笔同步。典型路由：行为还原 / 内核
/ 函数换绑 / 配置与环境键类问题去 adapter 仓（从其 `docs/reading-guide.md`
进入）；新增能力经 §0 判定落本仓后回到 [工程初始化](first-plugin.md) 或
[能力开发](plugins.md)。

GLM5.2 Channel FP8 配方、MQA planner 接线和固定组合验证见 [FP8 适配记录](glm52-fp8-adaptation.md)。
