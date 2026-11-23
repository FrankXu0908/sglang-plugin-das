# MoriDAS 插件后端开发导读

本文面向首次接触 MoriDAS 的 SGLang 开发者。当前实现基于 HCU Plugin `708a332`，对齐官方 SGLang v0.5.22 `baa9befa38`。组合验证使用 Adapter `69955d8`。逐次变更见[开发迁移记录](hcu-sglang-development.md)。

## 支持范围

- 通过 `--moe-a2a-backend mori-das --moe-runner-backend deep_gemm` 选择后端。当前覆盖 DeepSeek V4 Flash W8A8 INT8 的低延迟 EP Decode，支持 eager 空输入、Full CUDA Graph padding 和 DSpark 的 target/draft MoE。
- 默认 kernel 为 `InterNodeV1LLChunkedSync`。显式 kernel 名由已安装 Mori 的枚举解析。`SGLANG_MORI_DAS_CHUNKED_SYNC_IMPL` 会在 worker 构造 op 前传给 Mori，`0`、`1`、`2` 分别表示 SingleChunk、SingleChunk Dispatch Overlap、MultiChunk。当前实测使用 `0`。
- Dispatch 与专家计算使用 INT8，GEMM2 写出 BF16。FP8/BF16 Dispatch、其他 MoE runner 和 TBO/SBO 当前未支持。`dispatch_a/b` 和 `combine_a/b` 提供分阶段接口，常规调用仍顺序执行。

## 功能实现路径

1. SGLang 发现 `sglang.srt.plugins` entry point，调用 `hcu_sglang.register()` 并应用 Hook。插件扩展 target 和 speculative 两个 CLI 字段，使它们都能接受 `mori-das`。
2. `MoeA2ABackend._missing_` 将 `mori-das` 映射为 `CUSTOMIZED`，并设置进程内标记。带该标记的 `is_deepep()` 兼容判定让模型进入已有的 Low-Latency EP 路径。`create_moe_dispatcher` Hook 创建 `MoriDasDispatcher`。
3. Dispatcher 使用 EP 的 CPU 进程组初始化 Mori SHMEM，按配置缓存 `EpDispatchCombineOp`。TopK 后处理生成 INT32 路由，padding ID 为 `-1`。`dispatch_standard_moe()` 返回 INT8 激活、scale 和专家计数，封装为 `DeepEPLLDispatchOutput`。eager 空输入用一行无效路由参与通信。
4. MoE core Hook 按层上的实际 dispatcher 选择 `mori_das.run_moe_core()`，依次执行 masked GEMM1、SwiGLU clamp/量化和 masked GEMM2。ChunkedSync 的 GEMM2 直接写入 Mori 注册的 Combine buffer，供 `combine_standard_moe()` 消费。Combine 最后恢复真实 token 数及输入 dtype。

## 改动入口

| 文件 | 修改职责 |
| --- | --- |
| `hcu_sglang/__init__.py`、`hcu_sglang/server_args.py` | 幂等注册 MoriDAS Hook，扩展 target/draft 后端参数 |
| `hcu_sglang/layers/moe/mori_das.py` | 后端映射、dispatcher、SHMEM/op 缓存、INT8 MoE core 和 Combine buffer |
| `hcu_sglang/layers/moe/topk_source.py` | 在已有 HCU TopK 后处理中选择路由 dtype 和 padding 规则 |
| `hcu_sglang/layers/moe/deepep_fp8.py`、`hcu_sglang/layers/quantization/compressed_moe_marlin.py` | 按实际 dispatcher 分流计算和权重准备。HCU DeepEP W8A8 的 Normal/Low-Latency Dispatch 显式配置为 INT8 |
| `hcu_sglang/speculative_dspark.py` | 允许 DSpark target 和 draft 选择 MoriDAS，沿用现有 DP Attention 校验 |
| `hcu_sglang/disagg/decode.py`、`hcu_sglang/disagg/prefill.py` | DeepSeek V4 PD 两端统一使用插件提供的 KV transfer 元数据接口 |
| `hcu_sglang/layers/moe/dsv4_ops.py`、`hcu_sglang/models/deepseek/current_v4/hcu_backend/sparse_prefill_utils.py` | V4 通用兼容。HCU 非 EP SwiGLU clamp 可走 Triton，旧 LightOp 缺少 SWA helper 时使用已有 Triton 实现 |
| `tests/test_mori_das.py`、`tests/test_hcu_runtime.py`、`tests/test_w8a8_runner_dispatch.py` | 覆盖后端路由、CLI/Hook 注册、padding、权重准备和 Combine buffer 等契约 |

## 与 DeepEP 的关系和设计取舍

- **复用模型入口和协议。** `CUSTOMIZED` 标记与 `DeepEPLLDispatchOutput / DeepEPLLCombineInput` 让两种后端接入同一模型主链，官方枚举和模型 forward 继续复用。其他 `CUSTOMIZED` 后端按原接口处理。
- **各自持有数据契约。** MoriDAS 使用原有权重布局、独立 INT8 core 和注册 Combine buffer。DeepEP 使用自身的 Dispatch、权重准备和 core。MoriDAS 也读取 `SGLANG_USE_FUSED_SILU_MUL_CLAMP_QUANT`，按模型的 `swiglu_limit` 选择融合或显式 clamp。
- **在第一次 TopK 后处理完成分流。** MoriDAS 使用 INT32 和 `ID=-1`，DeepEP 的 Low-Latency DeepGEMM 路径使用 INT64 和 `ID=-1`，通用 HIP 路径继续使用 `ID=0` 和 `weight=0`。DeepEP 专用的 INT64 LightOp 融合入口跳过 MoriDAS。
- **沿用 Plugin 默认调度。** DP 元数据同步和 Graph 投票走通用实现。Dispatcher 的 A/B 边界可供未来调度器接入，当前交付为同步路径。

## 验证入口

开发时从 `tests/test_mori_das.py` 核对 dispatcher 分流、空输入、pad 和 GEMM2 buffer，再用 `tests/test_hcu_runtime.py`、`tests/test_w8a8_runner_dispatch.py` 检查 CLI/Hook 和权重准备。升级 SGLang 时用[插件开发 Skill](../skills/hcu-plugin-dev/SKILL.md) 与[能力开发指南](plugins.md) 核实 Hook 目标、调用顺序及两后端数据契约。

2026-11-19 双节点 TP16 / Attention DP16 / EP16、BS1、SingleChunk、L2 flush、HBM DBR 条件下，关闭与开启 DSpark 的 FakeDecode 各完成 32/32 条。实测加载 Mori TCP 注册重试版 `0.1.dev292+g2ce4cc7.tcpretry.20261119`，服务设置 `SGLANG_HCU_JIT_OVERLAY=1` 选用 Plugin 自带的 HCU JIT 源。DSpark 使用 checkpoint 内置 draft，16 个 rank 均完成 target 和 draft verify 图捕获。该结果覆盖这组 FakeDecode 配置，模型精度及其他模型和并行组合另行验证。
