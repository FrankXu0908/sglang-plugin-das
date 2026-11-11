# CI 与质量门禁：方法工具与当前运行时

> **本文用途**：给出当前可执行的检查命令、它们能证明的范围、失败处理方式，以及未来运行时和跨仓组合的门禁规划。验收断言见 [验证手册](validation.md)，提交清单见 [协作流程](contributing.md)。

## 1. 当前已实现的检查

Python 3.10+，仓库根目录执行：

```bash
python3 ci/l0_gate.py
python3 -m unittest discover -s tests -v
```

| 检查 | 实际覆盖 | 不覆盖 |
|---|---|---|
| L0 Python | 仓库 Python AST 语法，不导入运行时 | 运行语义、外部依赖或数值结果 |
| L0 运行时处置快照 | 从 triage 文件范围与当前 decisions 重算 closure，缺失或过期时失败；含全量 decisions 摘要，范围外决定变化也需重建 | 实现正确性、GPU/模型等价；implemented 不自动晋升 validated |
| L0 迁移文件头 | 按两份来源合同检查来源路径、SHA256、声明状态及原始头部；迁移/抽取副本逐行保留来源声明，来源无声明者只登记、不补造 | 原创接线、未登记迁移文件、来源仓本身的许可完整性、法律审查 |
| L0 Skill | 名称/描述、名称与目录一致、agents prompt 与资源文件链接 | 完整 YAML 语义或 Agent 行为质量 |
| L0 文档 | Markdown 相对文件链接存在性 | 锚点有效性、HTML 链接、Mermaid 渲染或可读性 |
| L0 阶段边界 | 校验 hcu_sglang/pyproject 布局、禁止旧根包与全局启动文件、检查旧 namespace import | 所有可能包名的通用业务代码检测 |
| unittest | 隔离 Git fixture、loader 与确定性响应验收工具契约 | 外部 plugin 的模型与 GPU 可用性 |

新增文档除执行 L0 外，还要人工/工具核对标题锚点、图文、HTML 导览和当前/规划标注。修改 Skill 后使用维护环境的 Skill validator 核对 frontmatter 与元信息；这些额外工具不成为本库的运行依赖。

## 2. 自动化接入现状

GitHub Actions 已配置以下 workflow（配置文件存在不等于真实触发已验证）：

| 文件 | 触发 | 内容 | 依赖 |
|---|---|---|---|
| [.github/workflows/quality-gate.yml](../.github/workflows/quality-gate.yml) | PR | 复用 `HYGON-AI/quality-gate` 的 PR 门禁 | 组织复用 workflow |
| [.github/workflows/nightly-wheel.yml](../.github/workflows/nightly-wheel.yml) | push `main` / 手动 dispatch | [packaging/build.sh](../packaging/build.sh) 构建 wheel；push 时经 CIUpload 上传 `das_nightly/dtk2610`，dispatch 只保留 artifact | 自托管 runner `hygon-build`、`vars.HARBOR_SITE`、`vars.PYPI_URL`、`secrets.PYPI_PASSWORD` |

nightly 在构建期把版本打成 `<pyproject 版本>+hcu.<YYMMDDHHMM>.g<commit6>`，产物即最终包，不做上传后改写。[.workflow/l0-gate.yml](../.workflow/l0-gate.yml) 仍只是 runner-neutral 命令配方，nightly 不执行 L0 与单测；workflow 生效前需完成一次真实 push 触发核对并记录结果。

## 3. 验证分层与运行时门禁规划

| 层 | 内容 | 环境依赖 | 责任归属（红了怪谁） | 当前状态 |
|---|---|---|---|---|
| L0 | 纯 CPU 结构/合约单测：AST、Skill/资源、链接、阶段边界、工具契约 | 无 GPU、无 sglang | 本仓结构 | 已实现（`ci/l0_gate.py` + unittest discover） |
| L1 | vendor 算子库准入单测：直调 lightop/aiter 等内部算子库**原生 API** 的数值与支持面准入 | GPU（或 mock），无 sglang 树 | 算子库/准入清单 | 规划；散见 adapter 仓 tests/kernels 的对拍资产待按边界收编 |
| L1.5 | 对应 sglang 版本的 layer 层单测：tiny Engine model smoke（5 家族 × eager/graph）+ 官方 `test/registered` 选测；adapter 格与 adapter+plugin 格复用同一清单，nodeid 级 diff | GPU + sglang 树（+adapter） | sglang 集成；叠加差异归 plugin | 工具已备（adapter 仓 `tools/l15_model_smoke`，三树 adapter 格全绿）；adapter+plugin 格接入中 |
| L1.6 | plugin 补充的 layer 层单测：特有模型 tiny 家族（bailing/dspark/hunyuan/glm4_moe/minimax_m2/qwen3_5）、layer hook 直调断言 | GPU + sglang 树 + adapter + plugin | plugin | 规划；雏形为 `tests/runtime/` 单点，待正名收编（vendor 原生 API 直调的归 L1） |
| L2 | 端到端模型单测：真实权重、请求/服务、gsm8k 数值、serving 冒烟 | GPU + 权重 + 组合部署 | 部署组合整体 | 按需执行（gsm8k 工具已有） |

边界口径与放行顺序：

- **L1/L1.5 按「被测对象归属」划**：测 vendor 库原生 API → L1；测 sglang 封装后的算子路径（会被 adapter seam 替换）→ L1.5。
- **L1.6 与 L1.5 平行**（同层深度的 plugin 专属补充），不是更深；编号不表示深度递进。
- **放行顺序**：adapter 格 L1.5 绿 → adapter+plugin 格 L1.5 diff 干净 → L1.6 绿 → L2；每级失败定位到唯一责任仓，不得假定任何两个最新 HEAD 兼容。
- 「官方 + plugin（无 adapter）」不是验收组合，仅诊断（见 [governance](governance.md#3-组合验证矩阵)）；L2 天然只有组合格。

首次引入 `pyproject.toml` 或运行时目录时，要先更新 [架构阶段](architecture.md#4-未来运行时目录规划)、L0 的当前排除约束与相应测试。不能只删除禁止规则而不增加运行时质量约束。

## 4. 失败与缺失条件的处理

| 情况 | 处理方式 |
|---|---|
| L0 文件链接/语法失败 | 修复对应路径/语法；不靠移除文件规避 |
| fixture 测试失败 | 保留失败日志和命令，核查输入保护或工具行为变化 |
| 缺外部依赖或 GPU | lane 记 pending，说明解除条件；不记 passed |
| 能力明确不支持某组合 | 记 not-applicable，关联设计/支持范围 |
| 历史已知失败 | 固定版本和签名、记录风险；不自动增加宽松阈值 |

工具测试仅在临时目录创建小型 Git 仓库。运行服务或探针时按 [共享资源约定](contributing.md#5-共享节点与产物管理) 限定本任务资源，不能为自动门禁无条件抢占设备。

## 5. 证据留存

短期日志放 `artifacts/<task>/`；交付文档记录实际命令、解释器/镜像、SHA、断言和汇总。长期证据需要稳定存放位置，因为 ignored 日志不会随 Git 提交。当前及历史结果见 [交付记录](delivery-validation.md)，记录格式见 [validation-record](../templates/validation-record.md)。

## 版本升级与发版门槛

按 [契约与影响](../skills/upgrading-sglang-hcu-upstream-first/references/contracts-and-impact.md) 记录选测与失效条件；选测/CI 本身变化用独立清单核对，必测 skip/xfail 不作通过。实际 wheel/image/embedded 验收及候选/last-good 分离见 [升级发版 Skill](../skills/upgrading-sglang-hcu-upstream-first/SKILL.md)。新增这些方法不代表本仓已经接入硬件 runner 或发布流水线。

## 网页快照维护

[离线导览](../architecture.html) 内嵌当前文档与 Skill 内容，不在浏览时读取外部网络。修改共享文件或 [网页模板](../templates/architecture.html) 后，运行 `python3 tools/build_guide.py` 重新生成，随后运行 `python3 tools/build_guide.py --check`；后者只读比较，快照过期时非零退出。完整原文仍在 docs/skills/tools 等目录维护，不直接手改生成页。

浏览器或 DOM 验证覆盖页面内跳转、文档章节链接、搜索与无结果状态、请求复制反馈、移动导航和历史返回；实际视觉渲染与 DOM 测试分别记录，不以语法或链接检查代替渲染结论。当前 runner 配方包含快照检查，接入平台后才形成自动 CI。

模块迁移规划页另由 `python3 tools/build_migration_plan.py --check` 检查计划定义、保存的源码盘点与 HTML 是否一致；该命令不重新访问输入仓库。三份来源版本变化后按 [规划维护指南](sglang-das-migration-plan.md#4-重新采集与维护) 显式重新采集，不能把缓存页面通过当作当前源码已审计。

当前 `hcu_sglang` 的入口、轻量导入、命名空间和显式 launcher 有仓库测试；真实依赖下的严格加载、wheel 与服务分别记录在 [开发迁移记录](hcu-sglang-development.md)。不得将仓库测试总数理解为全部 GPU/模型已验。
