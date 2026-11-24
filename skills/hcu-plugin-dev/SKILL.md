---
name: hcu-plugin-dev
description: 设计或开发 SGLang 在 HCU 上的新增融合、后端、专项内核或调度能力，规划 seam、注册、依赖与回退，并验证独立或 adapter 组合运行。用于插件能力开发与设计评审，已有行为适配部分按职责拆分给 adapter。
---

# HCU 插件能力开发


## 适用场景与任务衔接

[阅读指南](../../docs/reading-guide.md) 给出本 Skill 的输入、产出和可复制请求。各项 Skill 按任务选用，不要求顺序执行；安装现有插件使用 [安装 Skill](../hcu-plugin-install/SKILL.md) 和 [安装指南](../../docs/installation.md)。

新建独立包使用 [工程初始化 Skill](../hcu-plugin-scaffold/SKILL.md) 和 [初始化指南](../../docs/first-plugin.md)；给现有 hcu 新增能力则在本仓 `hcu_sglang/` 修改对应模块、沿现有 register 接入，并更新开发迁移记录。出现异常时读 [排障入口](../../docs/troubleshooting.md)，需要定位根因再用 triage。历史经验按症状查询，不作为额外必读流程。

## 用途与输入

用于一项新能力从设计到验证的执行。输入：需求与支持范围、official/plugin SHA 和路径；使用 adapter 时加其 SHA/能力依赖；可选 DAS 来源、产物目录和执行资源。当前实现位于本仓 `hcu_sglang/`，开发步骤、提交与验收记录在 [开发迁移记录](../../docs/hcu-sglang-development.md)。

设计接口或处理跨仓交互时读 [架构](../../docs/architecture.md) 与 [治理](../../docs/governance.md)：adapter 和 plugin 独立开发、可共同部署。判别“上游是否已有行为”用于实现拆分，不把部署画成二选一。

生成或修改源码遵守[生成代码约定](../../docs/contributing.md#生成代码约定)：不新增说明性注释、生成标记、来源 docstring 或 lint 抑制注释；保留来源已有内容与许可声明，说明写入独立文档和清单。

## 执行流程

1. 固定基线与 dirty，定义新增行为、shape/dtype/mode、模型/并行和接受条件；保留用户已有工作。
2. 按 [开发指南](../../docs/plugins.md) 选择正式 seam 或窄 Hook。沿公共 caller→实际 readout/consumer→出口确定参数、状态、视图和 collective 所有权，再说明与 adapter 的交互、包装顺序、依赖与关闭后的有效基线。
3. 将映射/门控与实现分开，顶层入口沿唯一聚合链到预期目标集合。有意多 AROUND 组合时测试顺序；AST 候选不代表已注册或已应用。
4. `original_fn` 可能已经过 adapter 或合法前序包装；回退应保留当前有效基线，不重新 import “原版”绕过适配。可选依赖懒加载，不吞 kernel/状态错误。
5. 按 [测试契约](references/test-contract.md) 建立修复前反例：保留真实构造/公共分发/消费者或 FFN 边界，按影响面覆盖实际 TP>1、空/非空与状态。mock 昂贵计算，不 mock 掉要验证的边界。
6. 回到原接受配置做 Graph、实际生成和本次产物复验，记录已执行范围与清理；短确定性回答需完整答案和正常结束，不能搜索关键词充当语义验收。正式部署不从 DAS fork 动态 import 实现。

## 按需参考

| 任务 | 读取 |
|---|---|
| 查询已有接口/状态故障的排查经验 | [开发经验与故障案例索引](../../docs/development-methodology.md) |
| 包结构、阶段或首次运行时落地 | [架构规划](../../docs/architecture.md#4-未来运行时目录规划)、[CI](../../docs/ci.md) |
| 环境变量、配置、全局/per-forward 状态 | [runtime-and-env](references/runtime-and-env.md)、[配置手册](../../docs/configuration.md) |
| 方法搬移、数值/graph 测试或 bench | [test-contract](references/test-contract.md)、[验证矩阵](../../docs/validation.md) |
| 跨仓协作或评审 | [协作流程](../../docs/contributing.md) |

## 输出与完成条件

输出能力方案、涉及仓库/版本/目标表、实现和文档变更、[验证记录](../../templates/validation-record.md)。区分单仓/组合与 CPU/graph/服务/指标结论；未执行 lane 保持 pending 或有理由的 not-applicable。不把未来包名/entry point 当作当前已实现接口。

## 实践补充

V3 实践要点：按官方 is_record 支持 msgspec/dataclass 输入，额外选项不能补写冻结 seed；已存在的具名 resolution seam 优先于旧私有方法 Hook。Python 启动兼容层不导入 vendor/torch，真实 HIP 能力与框架合成语义分开。见 [实践手册](../../docs/v3-runtime-validation.md)。

## 升级中的能力取舍

迁移既有优化前按 [上游接管规则](../upgrading-sglang-hcu-upstream-first/references/upgrade-workflow.md) 核实固定目标，保留最小 HCU 差异与临时 patch 退出条件。涉及 vendor/FLA/KV/collective 时按 [原生与状态契约](../upgrading-sglang-hcu-upstream-first/references/native-contracts.md) 核对完整数据流和转换成本，不按算子名称认定能力缺失。

## 方法抽取的执行语义

抽取/迁移类方法时，按 [测试契约](references/test-contract.md#抽取方法与模块别名) 验证 `super`/receiver/装饰器、常量限定及预导入消费者。语法和 apply 通过不能代替构造与实际分支；AST 重写应能重复执行。来源类拆分或 descriptor 绑定不明确时，先审查当前类与 caller，再更新 profile。


Attention 类图、CP 空查询或 native/JIT 迁移按 [测试契约](references/test-contract.md#attention-类图与原生构建边界) 选择针对性回归：核对实际继承和通信顺序、最终编译输入及 provider 的数据载体契约；局部原语通过不能代替完整模型验收。

分阶段 DAS 迁移时关联 [逐路径执行台账](../../docs/migration.md#全量差异执行台账)，新增 Hook 同步 [当前目标合同](../../tests/contracts/hcu_source_alignment_targets.json)。保留迁入 manifest 的历史语义；消费者反例、Hook 加载、CPU 张量测试与 GPU/Graph/服务分别验收。

归一化、residual 原地更新或 FP8 输出迁移时，按[归一化契约](references/test-contract.md#归一化的-residual-与量化返回契约)检查实际生产/消费与别名，分开记录 CPU 数据流、vendor 数学和完整服务结果。

FileIO key/LRU 接口迁移按 [分片状态与淘汰归属](references/test-contract.md#fileio-分片状态与淘汰归属) 成对验证生产、读回、metadata 和容量所有权；共享 MLA 数据可读与本 rank 可淘汰范围分别断言。

KV 事件/LoRA 名称迁移按 [namespace 的生产与消费](references/test-contract.md#kv-事件-namespace-的生产与消费) 验证最终 namespace、延迟写回和事件合并；Python、Rust binding 与发布端分别提供证据。

Rust crate 或 token hash 迁移按 [provider、事件 ABI 与宽 token hash](references/test-contract.md#rust-provider事件-abi-与宽-token-hash) 配对预检/loader、编译资产与真实 tuple 生产消费，并验证跨语言编码和 eviction ownership。

GLM5.2 INT4 的专项复现输入、脚本与验证边界见 [适配与0518对照](../../docs/glm52-int4-adaptation.md)。历史组合性能不作为新分支或其他硬件的默认验收结果。

GLM5.2 Channel FP8 的专项输入、已有 RoPE 配方和本次证据范围见 [FP8 适配记录](../../docs/glm52-fp8-adaptation.md)。
