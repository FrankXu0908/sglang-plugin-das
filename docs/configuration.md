# 配置管理：路径、开关与运行时状态

> **本文用途**：区分工具参数、SGLang/adapter/plugin 配置的所有权，说明组合运行时的覆盖与记录规则。当前 `hcu_sglang/config.py` 与 `hcu_sglang/environ.py` 已含插件配置和注册逻辑；所有权及状态规范继续适用于当前实现。

## 1. 当前工具的路径配置

| 名称 | 含义 | 要求 |
|---|---|---|
| `REPO` | 当前方法库 | 在仓库根目录用 `pwd -P` 获取；执行 L0、工具测试和脚本的位置 |
| `OFFICIAL` | 指定 SGLang 官方工作树 | 固定 SHA；与清单工具的 official-ref 匹配 |
| `FORK` | DAS 来源树 | 审计保留 dirty；不能作为正式运行时 import 来源 |
| `PLUGIN` | 含业务实现的 plugin 树；当前实现可与 `REPO` 相同 | 当前包为 `hcu_sglang`；支持旧布局的工具按需显式选择 `--plugin-package` |
| `ARTIFACTS` | 本次审计/日志输出目录 | 建议位于 `REPO/artifacts/<task>`，不要落到来源树内 |
| `OUTPUT` | 源码回溯新输出树 | 不与任一输入相同、互为父目录或子目录 |
| `WHEEL` | 本次安装的 wheel 文件 | 安装前核对交付方 SHA256；不从旧任务目录自动选择 |
| `MODEL` / `SERVED_MODEL_NAME` | 模型目录 / 实际服务模型标识 | 按配方填写；响应校验使用服务返回的模型标识 |
| `CONTAINER` / `CONTAINER_USER` | 执行容器 / 合法的 UID:GID | 仅容器方式需要；镜像挂载内路径与宿主路径分别确认 |
| `SOURCE_ROOT` / `VLLM_SKILL_SOURCE` | 历史抽取来源根 / 原 vLLM Skill 根 | 仅来源溯源需要；JSON 中以变量占位符记录，不是工具默认路径 |

在仓库根目录准备本地配置：新建 `.env`，按上表填写本次任务需要的变量（如 `OFFICIAL`、`PLUGIN`、`ARTIFACTS`）及路径；已有 `.env` 时保留并编辑它。来源、产物和模型路径可位于任意目录，不要求互为同级。外部路径填写本机实际绝对路径，含空格的值使用引号。进入容器后按容器内可见位置配置，不能直接沿用宿主路径。

```bash
# 从当前方法库根目录执行；.env 只保存本机配置，已被 Git 忽略。
export REPO="$(pwd -P)"
test -e .env || touch .env
# 编辑 .env 后再加载；仅 source 自己维护、可信的配置文件。
set -a
. ./.env
set +a
# 按任务检查必需项，例如插件能力开发：
: "${OFFICIAL:?请在 .env 中设置官方源码路径}"
: "${PLUGIN:?请在 .env 中设置外部插件源码路径}"
: "${ARTIFACTS:?请设置本次证据目录}"
```

这些是手册中的 shell 变量，不是本库自动读取的环境变量接口。加载 `.env` 是显式操作；实际 CLI 参数以工具 `--help` 为准，设置 `PLUGIN` 不会自动启用运行时插件。只阅读文档或运行离线工具测试时无需配置外部路径。首次初始化插件只需现有官方源码与外部新工程目录；按对应指南检查所需输入。

临时安装或示例工程使用 `mktemp -d`，由系统及 `TMPDIR` 决定位置，不固定临时文件名。需要持久部署时显式指定可写安装目录。历史记录中的 `${REPO}`、`${SOURCE_ROOT}` 等是脱敏位置标识；来源清单仍保留原始版本及文件哈希，不自动解析占位符，也不把当前路径冒充历史路径。原始 ignored 运行日志保持原样，按任务单独保管。

共享文档、Skill、模板与代码不得包含个人目录、私人节点名称或固定任务临时目录。目录结构示意使用相对路径；外部输入用显式变量或 CLI 参数。操作系统接口（如 `/proc`）和标准 shebang 不属于用户目录配置。

## 2. 运行时配置所有权

| 配置来源 | 所有者 | plugin 接入原则 |
|---|---|---|
| `SGLANG_PLUGINS` 与框架参数 | pinned SGLang | 按该版本 loader/CLI 语义使用 |
| SGLang 自有 `Envs` / typed descriptor | pinned SGLang | 有对应 API 时使用 `.get()` / `.override()`，不在 Hook 中散落重复解析 |
| adapter 的配置/快关 | adapter | 引用其固定版本手册；不复制同名注册表或改写它的默认值 |
| plugin 自有能力开关 | plugin 实现 | 集中定义、记录类型/默认/支持条件，独立命名并可诊断 |
| NCCL / driver / launcher / OS 变量 | 外部依赖或部署 | 按所有者的接口消费，不擅自改变语义 |

本仓 `hcu_sglang` 已包含来源插件的配置开关和 `hcu` 入口；具体以当前代码为准，不假定存在统一的 `HCU_DISABLE_OPS` 或额外配置服务。

### 当前 Qwen4 运行时配置

以下字段由 `hcu_sglang/environ.py` 注册到官方 typed `Envs`，在服务启动前设置并继承给 worker；更改后重启服务。默认关闭的性能路径需按支持条件显式启用，checkpoint/INT8 PLE 支持不依赖这些性能开关。

| 名称 | 类型 / 默认 | 行为与支持条件 |
|---|---|---|
| `SGLANG_QWEN4_COMPACT_PREFILL` | bool / false | 纯 EXTEND/IDLE 使用 SUM_LEN，idle rank 保留一行；TP=DP=EP、attention TP/CP=1、Gatherv、无 A2A/TBO、disabled prefill Graph |
| `SGLANG_QWEN4_LOCAL_ROUTING` | bool / false | compact prefill 在量化前的本 rank BF16 输入上选专家；要求非融合 shared expert |
| `SGLANG_QWEN4_ROUTED_INT8` | bool / false | 局部 routing 后传输 INT8 codes/scales，直接交给支持预量化的 AITER moe_c 一次 |
| `SGLANG_ENABLE_DP_GATHER_INT8` | bool / false | Qwen4 FFN 的独立 INT8 transport 入口；SUM_LEN/Gatherv、支持预量化的 quant method |
| `SGLANG_MOE_C_NATIVE_SORT` | bool / false | E129/top-k10、HIP、支持的 block size 使用保留 drop-route 的 native sort；依赖 AITER sorting/conversion API |
| `SGLANG_QWEN4_SPLIT_HC_MIX` | bool / false | gfx936/BF16、1–16 行、HC4/hidden2560/lowrank320 的分阶段 HC mix；保留确定性模式守卫 |
| `SGLANG_QWEN4_FLAT_STATE_SCATTER` | bool / false | gfx936、36 层 FP32 48×128×128、1–16 条 metadata 的连续状态回写；保留 scratch 源行映射 |
| `SGLANG_QWEN4_PREFILL_DIAGNOSTICS` | bool / false | 首次 compact 布局输出 rank/counts，和 routing 日志一起诊断实际路径 |
| `SGLANG_QWEN4_PREFILL_QSA_IMPL` | str / `triton` | 实验性 QSA P 端 reader：`native-packed` 使用容器 FlashAttention 的 packed BF16 API；`native-paged` 在单位 scale、FP8 E5M2/E4M3、NHD 或 HCUMHA BHSD 分页缓存且各请求 prefix 不短于 top-k 槽位数时直接读分页缓存（该 paged provider 仅支持 GQA group12），其余受支持输入用 packed API。BF16 Q、head_dim=256、GQA group 1–16、CP=1；其他形状保留 Triton 路径。冷 prefill 保留 BF16 当前 K/V，不改 D 或 MTP。需包含这些专项 API 的 HCU FlashAttention；首次命中记录 provider/shape，缺依赖或内核错误直接传播。当前验证目标为 disabled prefill Graph，不能据此宣称 Graph 可用。`triton` 保留原有效路径；环境字段在 metadata 阶段读取，修改需重启 worker。 |
| `SGLANG_HCU_QSA_DECODE_IMPL` | str / `triton` | `native-paged` 在 gfx936、CP1、page64、BF16 Q、D256、GQA group12、官方连续 NHD 或 HCUMHA BHSD 缓存及 BF16/E5M2/E4M3FN KV 下直接调用 HCU FA sparse paged GQA，省去 compact 和 BF16 scratch；支持 decode / MTP 的 eager 与 Graph。按当前 row request 从真实 `req_to_token` 生成页表，不能使用 Graph 的占位 token table；按每行 sequence length 屏蔽未来索引，保留空选择的 NaN。FP8 使用原 `k_scale_float/v_scale_float`（None 为 1）的设备 FP32 scalar，scale 在 Graph 捕获前固定。其他布局/dtype/形状/CP 保留现有 compact + Triton 路径；CPU 交给传入的 original。缺少 FA API 或 provider 报错直接传播，不伪装回退成功。`triton` 可显式回退；匹配下文 Qwen4 W4A8 配方时缺省设为 `native-paged`，其他模型默认不变。需要与 Torch ABI 匹配的 FA 构建。 |
| `SGLANG_QWEN4_PLE_PREFILL_INPLACE_SILU` | bool / `False` | 实验性 PLE 普通 EXTEND 临时激活复用：仅 HCU CUDA 张量、CP1、关闭 autograd 且非 decode fast path 时，对原 `_short_conv` 的独占 advanced-index 临时张量执行原生 inplace SiLU。卷积、状态和追踪写回保持社区方法；使用私有 functional facade，不改全局 Torch、社区文件或 D/MTP。首次启用验证固定目标方法的 AST 指纹，漂移直接拒绝并要求重新审查别名所有权；关闭或不满足条件保留传入的有效基线。验证边界见开发记录。 |
| `SGLANG_PLE_ETP_REDUCE_SCATTER` | bool / true | 支持的 PLE 分片布局采用 reduce-scatter；其他布局保留 all-reduce + scatter |
| `SGLANG_PLE_ETP_REDUCE_SCATTER_MIN_TOKENS` | int / 2048 | 小于阈值时使用 all-reduce + scatter |

关闭相应性能开关或超出窄形状门控时调用当前有效基线，不绕过 adapter。显式启用 INT8 transport 而缺少必要量化/通信合同，或启用 native sort 却缺少 vendor API 时明确报错。`SGLANG_DP_USE_GATHERV`、Graph、并行与 MoE 后端属于官方配置，按其接口设置；adapter 的 `qsa_mqa` 快关参见 `adapter-das` 仓 `docs/env-vars.md`，不由插件重复定义。完整支持范围与已测组合见 [开发记录](hcu-sglang-development.md#2026-11-04-qwen4-exp-w4a8-与-int8-ple-适配)。

## 3. 新增配置的设计检查

| 检查项 | 要给出的答案 |
|---|---|
| 名称与类型 | 谁拥有；布尔/枚举/数值/字符串，取值范围是什么 |
| 默认与覆盖 | 未设置、显式设置和不支持环境下分别采用什么；显式用户值优先，非法或不支持值给出可诊断结果 |
| 生效阶段 | import 前、进程初始化、请求级还是 forward 级；修改后是否需重启 |
| 依赖与组合 | 需要哪个 adapter/vendor/official 版本；另一个仓是否影响同一状态 |
| 观察与回退 | 日志如何显示最终值、来源和选中路径；停用后回到哪条有效基线 |
| 测试 | 默认、显式覆盖、不支持/缺依赖、进程传播与恢复是否覆盖 |

优先用一个含义明确的配置，不增加彼此冲突的多个布尔项。使用目标版本实际存在的 API；不能因为 DAS Skill 讲了 RuntimeContext 就假定旧 official 也具备同名接口。

## 4. 配置与状态分层

```mermaid
flowchart LR
    E[启动参数与环境] --> C[解析后的进程配置]
    C --> R[资源/stream/buffer 生命周期]
    R --> F[请求与 per-forward 状态]
```

配置、进程资源和每步状态分开。融合 KV 写入标记属于 batch/layer 的生命周期，不放模块全局跨请求复用；Graph 的动态输入不能只在 Python capture 时赋值。若 pinned 版本有 publish/override 机制，按其生命周期写入并记录来源。

平台判定也有所有权：adapter 可能向 SGLang 暴露 CUDA 语义，而真实 torch/编译器仍是 HIP。plugin 的硬件能力判定应使用足够直接且被验证的设备/依赖信息，不能只凭某个 SGLang `is_cuda()` 推断物理设备与 vendor 符号。

V3 的外部运行时输出实际引入 `HCU_PLUGIN_MINIMAX_OPT`，由 `--minimax-opt` 在构造配置前设置并继承给 worker；在 ServerArgs 上仅提供兼容读取，不写冻结记录。该接口最初属于外部 runtime 0.3.0，当前迁入实现见 `hcu_sglang/config.py`；历史行为与验收边界见 [V3 手册](v3-runtime-validation.md)。

官方配置记录可能是 msgspec Struct，不要仅靠 dataclasses.is_dataclass 判断；使用 pinned 版本的 `arg_utils.is_record`/record_fields。CLI 解析后还要 resolve_once 才算完成配置解析。

### 当前 custom all-reduce 参数

当前 HCU 插件提供 `--custom-all-reduce-backend {auto,native,aiter,off}`，默认 `auto`。官方0.5.22的原始 ServerArgs/comm记录没有该字段，插件在 CLI 构造原始记录之前消费参数，并用内部 `HCU_PLUGIN_CUSTOM_ALL_REDUCE_BACKEND` 继承到 worker；不向冻结输入或 `get_exec().comm` 补写属性。直接构造官方 `ServerArgs(custom_all_reduce_backend=...)` 不是已支持接口；程序化调用应在初始化 engine 前使用 `hcu_sglang.config.configure_custom_all_reduce_backend(value)`。

| 规则 | 实际行为 |
|---|---|
| 优先级 | 官方解析后的 `disable_custom_all_reduce=True` 强制 `off`；否则保持显式后端；HCU上 `auto` 且 legacy `SGLANG_USE_AITER_AR=1` 按DAS提升为 `aiter` |
| 生效阶段 | 每个 worker 的官方 distributed bootstrap 先设置原有 flags，再将插件解析结果交给组工厂；一个进程对应一个活动 engine 配置 |
| 组覆盖 | world组始终 `off`；model-parallel组默认取解析结果；显式组级参数优先，不修改其他组成员定义 |
| strict-Fabric | `aiter` + `AITER_AR_TRANSPORT=fabric` 时构造异常原样抛出，disabled communicator 明确拒绝；不静默回落成 RCCL |
| QuickAllReduce | 保留DAS的独立 `SGLANG_USE_QUICK_CUSTOM_ALLREDUCE` 门控；Fabric/auto transport 不并建 IPC QuickAllReduce |
| 原配方别名 | `--nsa-prefill-backend` / `--nsa-decode-backend` 使用原 deprecated alias action，写入当前 DSA 字段，未指定时不覆盖 canonical 默认值 |

后端初始化日志记录 requested/selected transport、disabled、ranks与world size。参数解析通过不代表 communicator 初始化或模型正确；IPC、Fabric、Graph及各组拓扑分别验收，见[开发记录](hcu-sglang-development.md)。

## 5. 变更与排障记录

每次配置改动同步默认值说明、支持范围、测试与启动日志示例。发生冲突时先列出“原始值 → 解析值 → 最终值 → 实际分支命中”，再修改实现。验证模板见 [validation-record](../templates/validation-record.md)，详细状态约束见开发 Skill 的 [runtime-and-env](../skills/hcu-plugin-dev/references/runtime-and-env.md)。

## 6. GLM CT W4A8 的原生 BF16 RoPE 选择

插件开关 `SGLANG_HCU_USE_LIGHTOP_ROPE` 为布尔型，默认 `False`；在 GLM-5.3 CT W4A8 配方中显式设置为 `1`。它在 HCU 的 BF16 `RotaryEmbedding` 构造结束后准备镜像内公共 LightOp `rotary_embedding_fuse`，由正式 `forward_cuda` AROUND Hook 消费；修改后需重新构造服务。默认关闭时保留既有分发，非 HCU、非 BF16 或融合 KV-store 调用保留当前有效原实现。

该路径使用已有 `HcuRotaryFallback` 的 typed cache memo，为 LightOp 提供与 query 同 dtype/device 的缓存视图；框架持有的 FP32 `cos_sin_cache` 及其他 indexer 消费者保持原所有权。LightOp 的缺失或执行错误直接暴露，不在数值失败后静默切换其他算法。此选择不改变 CT W4 权重、动态 INT8 激活或 GEMM 分发。

LightOp 自行选择 launch，不套用旧 sgl_kernel 的 256-thread head 分块；真实 64-head 输入完整调用与拆成 8-head 调用的数值并不相同。offsets 在调用前按原语义加到 positions，返回原地更新的 query/key。

适用证据和待验证范围见[开发记录](hcu-sglang-development.md#2026-11-05-glm-ct-w4a8-的-bf16-rope-选择)。两组八 rank 的真实 Q/K 及更换输入后的组件 Graph 已逐元素对齐原生，完整服务仍待复测。

## 7. GLM CT W4A8 的公共 W8A8 GEMM 分发

`SGLANG_HCU_W8A8_USE_LIGHTOP` 为插件布尔开关，默认 `False`。GLM CT W4A8 配方显式设置为 `1`，使既有 `triton_scaled_mm` 接口使用镜像内公共 LightOp 分发，跳过插件自行选择的 smooth ASM 快路径；公共接口内部仍可能使用 ASM。开关不改变输入 INT8 量化值、token scale、权重或输出 dtype；默认关闭及非 HCU 的行为保持原样。应在服务启动前设置，Graph 捕获后不要切换后端。

真实首层 Q-b 的两组八 rank 输入、权重和 scale 对照中，两套镜像的公共接口均逐元素重建原生输出；插件的 ASM 快路径有四个元素不同。该有限组件证据不证明所有 shape 或完整模型通过，后续分发和模型复测见[开发记录](hcu-sglang-development.md#2026-11-05-glm-ct-w4a8-的-w8a8-gemm-分发)。

## 8. GLM-5.3 FP8 prefill 的 K1 indirect-A 候选

`SGLANG_HCU_GLM53_FP8_K1_AUTO_BUILD=1` 单独启用 K1 启动编译，默认关闭。
符合下述模型、并行和设备条件的 prefill worker 在初始化 Graph 前准备扩展。
同节点 rank 使用同一个缓存目录时，由一个进程编译并发布，其余进程复用。
缓存位于 `${XDG_CACHE_HOME:-~/.cache}/hcu_sglang/glm53-fp8-k1`，
由源码、Python ABI、Torch、HIP、DTK 和目标架构等构建条件区分。
更换 Python 或 DTK 后，首次启动会为新环境编译一次。缓存未命中时，
服务环境需提供 DTK 的 `aicc`、Ninja 和配套头文件，编译失败会中止启动并输出错误。
已显式配置扩展路径时沿用该路径。

`SGLANG_HCU_GLM53_FP8_K1_INDIRECT` 为布尔开关，默认 `False`。`SGLANG_HCU_GLM53_FP8_K1_INDIRECT_SO` 指定独立扩展的绝对路径；仅在开关打开且当前请求满足全部门控条件时才加载。未配置路径时记录告警并使用现有路径；显式配置的扩展或其相邻 `build-manifest.json` 不匹配时抛出错误，已进入扩展后的编译、加载与 kernel 错误不会被吞掉。

当前候选只覆盖固定 GLM-5.3 Channel FP8 配置、main-model 普通带前缀 EXTEND、DeepEP normal、TP8/CP8/EP8、CP interleave 和 `SGLANG_ENABLE_CP_V2=1` 的非 LayerSplit eager 路径；还要求 gfx938、固定 W8A8 channel/token 量化、E32/Top8/K6144/N4096 与规定的权重和 scale 布局。非目标模型、非前缀、LayerSplit、decode、MTP、graph、其他布局及未启用开关均使用已有分发。LayerSplit 是本候选尚未验证的范围限制，未来是否支持取决于实际路由与缓存对照，并非架构上不能共存。

扩展由独立构建器生成，不覆盖公共 DeepGEMM，构建命令和来源见[开发记录](hcu-sglang-development.md#2026-11-10-glm-53-fp8-prefill-k1-indirect-a-候选)。新路径验证路由容量并同步传播错误。普通无 clamp 量化且 expert inverse 关闭时使用未初始化输出分配；其他路径仍清零。该调整依据八 rank padding 对拍，修改后的耗时仍需复测。模型实际请求、安装包、同输入数值、全部 rank 与端到端均未验证；不得以开关可配置或构建成功称为算子对齐。


## 9. GLM-5.3 FP8 prefill 的 activation inverse 与 K8 gather 候选

本节是两项默认关闭的 plugin 新能力配置，开发流程见 [hcu-plugin-dev](../skills/hcu-plugin-dev/SKILL.md)。输入包括固定 GLM-5.3 FP8 checkpoint、官方 SGLang v0.5.22、adapter/plugin 版本，以及与运行环境 ABI 一致的独立扩展及其构建清单。

| 配置 | 默认值 | 用途 |
|---|---|---|
| `SGLANG_HCU_GLM53_FP8_EXPERT_INVERSE` | `False` | 无 swiglu clamp 时的限定形状 expert activation FP8 量化 |
| `SGLANG_HCU_GLM53_FP8_EXPERT_INVERSE_SO` | 未设置 | activation 独立扩展的绝对路径 |
| `SGLANG_HCU_GLM53_FP8_EP_GATHER_K8` | `False` | FP8 normal MoE 原有 LightOp no-map 分支中的 K8 gather |
| `SGLANG_HCU_GLM53_FP8_EP_GATHER_K8_SO` | 未设置 | gather 独立扩展的绝对路径 |

两项独立于 K1 开关，在相同单次 MoE forward 内绑定 owner 并在退出时恢复作用域。只允许固定模型主模型、TP/CP/EP8、CP interleave v2、普通有前缀 EXTEND、非 LayerSplit/MTP/graph；与原生一致，批内至少一项 prefix 非零即可，混合批次的实际 target 调用仍待验证。K1 和 token inverse 使用相同的 any-prefix 条件。W4A8、masked MoE、其他模型和原有 clamp 分支不进入候选。

activation 仅接受连续 BF16 `[M,4096]`，M 为 `14336/84224/84992`，返回 FP8 E4M3 `[M,2048]` 与 FP32 `[M,1]` scale；不扩展白名单。gather 保留同一次 dispatch/K1 产生的 int64 expert IDs、FP32 weights 和 int32 output index，输出 BF16 `[recv_rows,6144]`，按 K0..7 顺序做 FP32 FMA 后转换；供给的 output buffer 不被替换。

构建入口为 `python -m hcu_sglang.layers.moe.build_glm53_fp8_prefill_ops --help`；构建输入须显式提供固定外部源码及所需头文件，清单记录其哈希、编译 ABI 和二进制。共享配置不包含本机或历史机器绝对路径。未命中门控使用原有效路径；显式启用且配置的扩展不匹配应报错，不吞进入扩展后的错误。

当前仅根据原生真实 profile 和源码建立接线；plugin 实际命中、同输入输出、全部 rank、MTP/graph 回退及服务组合均未验收。独立扩展构建成功或原生侧对拍结果不能替代这些验证。


## 10. GLM-5.3 FP8 prefill 的 token inverse 候选

本项使用 [hcu-plugin-dev](../skills/hcu-plugin-dev/SKILL.md)，输入为固定模型、官方框架与 adapter/plugin 版本，以及独立扩展和构建清单。`SGLANG_HCU_GLM53_FP8_TOKEN_INVERSE` 默认 `False`；`SGLANG_HCU_GLM53_FP8_TOKEN_INVERSE_SO` 指定扩展绝对路径。构建入口为 `python -m hcu_sglang.layers.quantization.build_glm53_fp8_token_inverse --help`。

候选在 attention `forward_core` 中绑定当前 owner，在其 `o_proj` 的 dynamic channel FP8 分支绑定实际输入，只有量化入口拿到同一 storage、视图和设备时才调用。当前限于固定模型主模型 IFB CP8/EP8、interleave v2、带前缀普通 EXTEND、FlashMLA sparse 和非 LayerSplit eager；不覆盖 decode、MTP 或 graph。连续 BF16 输入为 `[4096,16384]`，输出 E4M3FN 与 FP32 `[4096,1]` scale；按固定原生实现计算 absmax/448 及 448/absmax，不添加 epsilon、clamp 或更换量化策略。

开关关闭或门控未命中时保留当前有效分发；没有扩展路径时告警回退，显式配置的扩展及清单不匹配时失败。attention/投影的字段与权重 stride 已静态核对，真实 target 请求命中、各活跃 rank 的同输入输出和性能仍待验证。加载期函数绑定证据不等于请求执行证据。


## 11. GLM5.3 channel FP8 的 KV B 加载兼容

本项由 [hcu-plugin-dev](../skills/hcu-plugin-dev/SKILL.md) 维护，输入为固定模型配置和 checkpoint、官方及 adapter/plugin 版本。已有 KV B channel→tensor 权重量化分支在匹配 GLM5.3 FP8 配置身份与 CT scheme 后，使用原生计算顺序及零维 scale，避免 scale 维度引起 BF16 权重提升为 FP32；未匹配继续现有路径，不增加用户开关。来源差异和验证边界见[开发记录](hcu-sglang-development.md#2026-11-11-glm53-channel-fp8-的-kv-b-权重加载对齐)。这不是通用量化策略调整，不覆盖 W4A8/W4A16；layer 3 同输入组件已逐字节对齐，真实模型加载和其他路径仍需验收。


## 12. GLM5.3 FP8 prefill physical KV compact 候选

本项使用 [hcu-plugin-dev](../skills/hcu-plugin-dev/SKILL.md)，输入为固定 GLM5.3 FP8 模型配置、官方和 adapter/plugin 版本及配套 FlashMLA。`SGLANG_HCU_GLM53_FP8_PHYSICAL_KV_COMPACT` 是 plugin 布尔开关，默认 `False`；模型和设备范围在 backend 初始化时绑定，设置后需重启进程。

候选仅适用于 TP/CP/EP8 主模型、interleave CP v2 的普通 EXTEND 和至少一条请求有 prefix 的 batch，要求现有 FlashMLA sparse、FP8 656 KV、kpool=1、TopK=2048，且不使用 LayerSplit/DCP/TBO/LoRA 或 graph。逐 forward 缓存的只有 physical-row plan，逐层使用现有 paged KV dequant；保序重映射后由 raw FlashMLA 消费。范围不符或 raw 接口不可用时保留原 full dequant 和现有 FlashMLA 路径；进入 compact 后如接口消失则报错，避免改变 TopK 顺序。

当前实现只完成源码和调用合同检查；真实请求命中、混合 prefix、各 rank 同输入数值及模型组合尚未验收。LayerSplit、MTP、D decode 和 graph 未纳入该候选。来源与待验证边界见[开发记录](hcu-sglang-development.md#2026-11-11-glm53-fp8-prefill-physical-kv-compact-候选)。


## 13. GLM5.3 FP8 add-RMS align16 候选

`SGLANG_HCU_GLM53_FP8_ADD_RMS_ALIGN16=1` 选择专项 prefill 候选，默认关闭；必须在 decoder 构造前设置。`SGLANG_HCU_GLM53_FP8_ADD_RMS_ALIGN16_SO` 指向固定原生扩展的绝对路径。未提供路径时告警并保留当前有效回退；提供路径但 SHA 或 ABI 不匹配时明确报错，不静默换库。

仅已识别的主模型 decoder 两个 norm、普通 CP8/EP8 prefix eager、规定 shape/layout 和数学条件生效；下游 fused quant、MTP、graph、LayerSplit 和其他模型保持回退。它与 physical KV compact 的开关独立，复用 per-forward metadata scope。普通 norm 可复用 adapter 既有 `HCU_RMS_NORM_IMPL=lightop` 和 `HCU_FUSED_ADD_RMS_IMPL=lightop`，须核对配置文件优先级及实际解析结果，不能把配置声明当作 kernel 命中。

当前是未完成同输入验证的候选；范围、固定来源和剩余证据见 [开发记录](hcu-sglang-development.md#2026-11-11-glm53-fp8-add-rms-align16-候选)。


## 14. GLM5.3 FP8 prefix KV producer 候选

本项使用 [hcu-plugin-dev](../skills/hcu-plugin-dev/SKILL.md)，输入为固定 Channel FP8 checkpoint、官方 SGLang 0.5.22 与 adapter/plugin 版本、配套 LightOp 融合 API，以及实际 CP 输入和缓存布局证据。`SGLANG_HCU_GLM53_FP8_KV_ROPE_STORE` 为 plugin 布尔开关，默认 `False`，须在模型和 backend 构造前设置；它独立于 physical compact 和 add-RMS，初始化共用模型身份判定。

只在精确识别的 GLM5.3 FP8 主模型、TP/CP/EP8、interleave CP v2、普通带前缀 EXTEND eager、FP8 656-byte KV cache 和 FlashMLA sparse 路径尝试。要求 canonical CPU lengths、`gather_index is None`、实际 BF16 latent/norm/RotaryEmbedding 与规定布局；MTP、decode、graph/compile、LayerSplit、DCP、LoRA、born-Q、BMM fusion plan、非标准 norm/RoPE 和其他模型保留当前有效路径。开关关闭不选择新 producer。

prepare 在 KV norm 前判定资格，保留 Q norm、projection、indexer 和 absorbed BMM；选中后使用配套 LightOp `fused_rms_norm_rope_contiguous_q_cat` 生成本地 packed KV 与 Q，沿原 CP group、对称内存和 all-gather 交换原始字节，再写 allocator 提供的 physical locations。core 消费单次返回的私有 Q 标记，显式关闭重复 KV 写入；既有 Qconcat 和 FlashMLA consumer 不另改算法。

显式选中路径要求 LightOp 同时提供包装层、对应原生符号和 Q head stride 256 支持。缺失时明确报错；进入融合后若 Q geometry 或交换合同不符，不改用另一算法掩盖错误。每行保持 656 byte/164 int32 word，FP8 KV scale 固定沿原生参数，禁止浮点搬运、重复量化或改变 TopK 顺序。来源与分阶段验证边界见 [开发记录](hcu-sglang-development.md#2026-11-12-glm53-fp8-prefix-kv-producer-本地候选)。

目前为未部署的本地接线候选。已有真实目标输入的八 rank 融合叶子与独立 store 数学证据，不等于候选 helper、实际八 rank collective、正式模型分发或完整精度通过；这些验收分别记录。

## 15. GLM-5.3 Channel FP8 的 prefill 组合开关

`SGLANG_GLM_OPT_PREFILL=1` 在满足模型、输入和供应商扩展条件时启用
packed KV、physical KV compact、Add RMS align16、K1 indirect 和 EP K8
gather。各项显式开关和构建产物要求见前面的对应小节。

`SGLANG_GLM_PREFILL_SHARED_OVERLAP=1` 独立启用 eager prefill 的共享专家旁流。
Normal DeepEP 提交 routed FP8 量化后记录事件，旁流等待该事件后执行共享专家，
主流在合并输出前等待旁流结束。两个开关默认关闭。

## Qwen4 W4A8 启动默认值

`hcu_sglang.launch_server` 在导入服务之前调用
`serve_defaults.apply_qwen4_serve_defaults()`；plugin 注册入口在加载 vendor
与模型模块之前补调。仅匹配本地 `config.json` 的
`model_type=qwen4_exp/qwen4_exp_text`、BF16 和
`slimquant_w4a8/slimquant_w4a8_marlin`，不依赖权重目录名。
显式 CLI dtype/quantization 优先于 checkpoint。未匹配、非 HCU、远程模型 ID
及没有 argv 的程序化 Engine 不应用本配方。

默认项由 `hcu_sglang/serve_defaults.py` 集中维护：QSA 4096 key span /
512 MiB logits / native-paged，compact prefill、local routing、routed INT8、
SiLU INT8 融合、split HC mix、flat scatter、LightOp、fused top-k、causal
conv、AITER linear attention / AR、W8A8 method 3、AITER W4A8 MoE C / native
sort、DP gatherv、Python radix core 和 PD prefill DP rank 查询。
`AITER_AR_MAX_SIZE_MB` 默认 64。`SGLANG_HCU_QWEN_SILU_INT8_FUSION`
通过官方 typed env descriptor 读取，缺省关闭，匹配配方时默认启用。

只有 prefill 角色默认 drain=wait；只有显式启用 prefill delayer 且
`--prefill-delayer-queue-min-ratio 0` 时默认 mixed timeout。避免影响默认队列
参数下的启动。已有内核的 dtype/shape/parallel/架构回退不变；尤其 split HC
和 flat scatter 保留 gfx936 准入，不声称在 gfx938 上实际命中。

所有项均使用 `setdefault`，显式环境变量（包括 `0`）优先；模块以 INFO 记录最终
值及来源。该调用早于上游日志初始化时，默认启动日志可能不显示此记录；
可检查 spawn worker 的继承环境及实际后端日志。可以用对应原值逐项回退，例如
`SGLANG_QWEN4_ROUTED_INT8=0`、`SGLANG_QWEN4_PREFILL_QSA_IMPL=triton`
或 `SGLANG_HCU_QSA_DECODE_IMPL=triton`。
这只是验证过的模型配方，不修改所有模型的 SGLang 全局默认值。

NUMA 两项和 AITER AG 在固定上游已默认开启，PLE inplace SiLU 已默认关闭，
部署可以省略这些重复 export。`GPU_MAX_HW_QUEUES`、网络/RDMA、缓存路径、
插件启用、`HCU_SLIMQUANT`、bootstrap timeout 与超长上下文许可保留在部署层；
不能把本节点的驱动绕行和资源选择作为所有模型的默认值。
