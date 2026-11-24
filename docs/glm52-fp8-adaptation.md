# GLM5.2 Channel FP8 TP8/EP8/CP8 适配

> 本文用途：记录 GLM5.2 FP8 在固定 SGLang 0.5.22、adapter/plugin 组合上的启动配方和本次验证边界。范围为单节点八张 gfx938，BF16 执行、FP8 KV、DeepEP 与 EAGLE；其他设备和量化格式未外推。

## 输入与版本

- 模型：hygon/GLM-5.2-Channel-FP8-w8a8，GlmMoeDsaForCausalLM，compressed-tensors token/channel FP8。
- 官方框架：SGLang 0.5.22 PyPI cp312 wheel，DTK2610/Torch2.13；官方文件逐字节核验。
- adapter 22 分支：bcec3e046cb5835410d0c8ab5503acbfdf0a3e9c，加本次同卡数外来 PCI 拓扑 guard 修复。
- plugin 22 分支：1d51ae54dd8146709cb436c2ebc3dfbfedfd3e5b，加本次 MQA planner 接线及配方。该分支包含用户提交 [3872dc0 / #89](https://github.com/HYGON-AI/sglang-plugin-das/commit/3872dc075b48bbf05e68b96afb8f13f68cf62e90)，不重复实现 RoPE。
- BW1101 的实际 HIP 身份为 gfx938；保持真实硬件身份。

## 执行与所有权

使用 [hcu-plugin-dev](../skills/hcu-plugin-dev/SKILL.md)：输入上述版本、模型、native 环境和外部证据目录，交付配方、最窄修复和真实验证。安装两个仓库并保留其源码 SHA、dirty patch 与未跟踪文件清单，不从 DAS fork 动态导入实现，不修改官方 Python。

```bash
HCU_ADAPTER_HOST=none python3 -m pip install --no-deps --no-build-isolation -e "$ADAPTER_ROOT" -e "$PLUGIN_ROOT"
MODEL_PATH="$LOCAL_MODEL" ADAPTER_ROOT="$ADAPTER_ROOT" PLUGIN_ROOT="$PLUGIN_ROOT" \
  bash "$PLUGIN_ROOT/examples/glm52-fp8/launch.sh"
```

入口为 python3 -m hcu_sglang，使用 --dsa-prefill-backend/--dsa-decode-backend。--attn-cp-size 8 的实际 DP、prefill CP 与 Graph 状态以 server-info 为准。当前保持用户 FP8/0518 配方，不擅自带入 INT4 的量化参数、chunk/request 数量或 prefill-CP 开关。

SGLANG_USE_LIGHTOP_CHANNEL_FP8=1 选择动态 token/channel Linear。SGLANG_HCU_USE_LIGHTOP_ROPE=1 复用 #89 已验证的已有 RoPE 路径；主缓存为 BF16 时不可遗漏此开关，否则会进入要求 FP32 cache 的社区 JIT。无需新增 RoPE 数值实现。#102 的特定 Python3.10/Torch2.11 AOT 兜底不能替代当前 Python3.12/Torch2.13 的真实 provider 验证。

## 修复与反例

adapter 的 NCCL_TOPO_FILE guard 原来只检查 GPU 数量，八卡外来 XML 可穿过检查并导致拓扑计算崩溃。现在进一步检查 XML GPU PCI 地址是否存在于本机 DRM render-device；未知地址仍保持保守，显式关闭守卫优先。修复前 CPU 反例失败，修复后 18 项通过，实际八 rank 初始化已通过。

插件的 current_dsa backend/indexer 使用官方 resolver 时，adapter 会返回 Torch fallback 占位 schedule，而实际 consumer 是 LightOP paged MQA。mqa_schedule 在真实 HCU 上选用配套 LightOP planner；非 HCU 委托原 resolver。初始化、刷新和 indexer 按需规划共用此入口。无新增 Hook，不修改 adapter Torch provider 或全局 resolver。LightOP schedule 行数由内部调度决定，不能按物理 CU 数猜测。

旧插件 c47683e 上先保留三个实际失败反例，修复后连同既有 indexer 测试 9 项通过；真实 gfx938 FP8 MQA 的 8/32/64 heads、跨页变长与 Q/gate 变输入 graph replay 3 项通过。上述旧基点证据不自动转授到最新版本，最新组合复测独立记录。

## 本轮验证与对照

本次完整模型 Graph、正常生成、性能及 trace 采集已完成，验证范围见下表。先前因其他任务占卡而 OOM 的记录，以及本轮旧基点 MQA/漏配 RoPE 开关的失败记录均保留。

0518 对照复用固定 input IDs、128/1024/4096 输入长度、128 输出 token、并发 1/8、三次重复；清缓存并预热，性能测试与 CPU/GPU profiler 分开。保留完整回答、结束原因、逐请求时序、有效配置、逐 rank trace 和算子汇总。合成输出或有限算术题仅作 smoke，不代表完整量化模型精度验收。

两仓旧基点 L0/导览检查已通过；旧插件 CPU suite 有 3 项隐藏 GPU 导入错误和 13 项跳过，不能记作全绿。最新基点检查另行保存。实际浏览器交互和窄屏视觉检查未执行，不把导览构建当作截图验收。


最新 1d51ae5 组合组件复测：4 项 MQA 接线、3 项 FP8 MQA GPU、2 项已有 LightOP RoPE GPU、1 项已有融合 TopK GPU 通过，均包含相应变输入 graph replay。TopK 路径按 #89 启用 SGLANG_DSA_FUSE_TOPK=1 与 SGLANG_USE_LIGHTOP_TOPK_IDS_POSTPROCESS=1；保留 SGLANG_HCU_GLM_CT_BF16_ROUTER=1 数值策略，未套用 INT4 量化参数。未融合 Torch topk 曾出现 no-kernel-image，原失败保留。最新 CPU suite 195 项，原三项 HY4/mRoPE 导入错误、13 skips；L0/导览检查通过。

## 2026-11-21 最终同协议结果

用户明确选择保持原 FP8 启动脚本和既有 0518 的实际配置：TP8/EP8、attn_cp_size=8、enable_prefill_cp=False、DP=1、page64、FP8 KV、mem fraction0.8、chunk8192、16 requests、EAGLE3/1/4、DeepEP auto。未套用 INT4 配方的量化参数、chunk16384、32 requests 或 prefill CP。仅修改本地模型路径、0.5.22 backend 参数名称与正式插件入口；新增适配环境选择单独记录。

候选运行 adapter-plugin22-tp8ep8cp8-20261121-140706，基线运行 baseline-0518-recheck-tp8ep8cp8-20261121-142102。两版均完成 target verify、draft decode/extend Graph 与正常服务请求。三个限定算术/抽取题均答案一致且正常 stop；原始 81 请求及独立稳定测试 81 请求的完整文本、六项 EAGLE 接受统计全部逐请求一致。合成输入和三个短题不构成数据集精度或完整概率分布验收。

两版各保存 16 份 CPU/GPU Chrome trace 和 16 份逐 rank 算子 CSV；C1/C8 各 rank0..7 覆盖已核对，原始 trace 绑定 SHA256。内核中可见 gfx938 grouped FP8 GEMM、DeepEP、FlashMLA sparse 和 channelwise GEMM，确认真实 FP8 执行路径。

稳定测试每个输入长度/并发先完整预热，清缓存后测三次。每请求输出128，温度0，ignore_eos=True。下表为三次组吞吐均值；C8 组内相同提示可共享前缀。

| 输入 | 并发 | 0518 token/s | 组合 token/s | 比值 |
|---|---:|---:|---:|---:|
|128|1|56.22|56.69|1.008|
|128|8|262.04|265.65|1.014|
|1024|1|47.72|49.39|1.035|
|1024|8|140.53|144.13|1.026|
|4096|1|26.76|27.72|1.036|
|4096|8|74.28|76.69|1.033|

短输入 TTFT 并非完全对齐：C1为0.2455→0.2973秒，C8为0.7611→0.8017秒；1024/4096组下降约1.6%～3.5%。样本有限，不能据此断言统计显著性或归因单个补丁。此比较包含 Torch/Python/native 镜像差异，是整体部署版本对照。

0518 在 profiler 完成后的补测中观察到 flush/health 不响应，原始证据保留，未修改基线实现；同配置干净重启后完成独立稳定采集。候选稳定采集在其 profiler 完成后执行；两个服务实例和预热边界均记录。两版测试服务随后停止并释放显存。

SGLang官方5283文件运行后核验仍无差异。完整CPU suite195项，保留原3项导入错误与13 skips，不记为全绿。全仓Hook静态合同结论以本次原报告为准，不以模型通过覆盖其他模型/多模态目标。

最终补验：最新分支 FP8 Linear/量化/bias/回退/变输入 Graph 的10项GPU回归通过，结合MQA3、RoPE2、融合TopK1，共16项GPU组件验证通过；MQA接线4项CPU与adapter拓扑18项CPU通过。最新注册审计681/681 applied、unapplied为空，但全仓目标集合存在31项缺失及1项额外，保留failed原结论。当前adapter220个和plugin496个运行时Python文件与成功运行启动快照逐字节一致。

## 2026-11-23 三项数据集评测

沿用上述固定源码与镜像组合，仅评估 adapter + plugin；thinking=false，max_tokens=8192，temperature=1，top_p=0.95，seed=42，客户端并发64，全量1983题。HumanEval Pass@1 为153/164（93.29%），GSM8K Accuracy 为1269/1319（96.21%），MATH500 Accuracy 为474/500（94.80%）。API及判分错误均为0；MATH500有3题触及输出上限，包含在结果内。评测完成后模型服务自动停止。
这是固定组合的一次采样结果，不是单独 MQA 修复的精度增益证明；本次没有重跑0518数据集对照。此前CPU全套及其他模型静态合同的失败边界仍然保留。
