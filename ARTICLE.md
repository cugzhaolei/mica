# deepseek4-strata：把 284B 的 DeepSeek V4 Flash 塞进两张 2080Ti 的工程化路线

> 一句话：用 Strata 式"专家分层卸载"（GPU→RAM→SSD 三层缓存 + 投机解码）的思路，
> 把 DeepSeek V4 Flash（284B 总参 / 13B 激活 / 256 专家选 6 / 43 层 / 1M 上下文）
> 适配到 2×2080Ti 22G + 157G RAM 的老工作站上。本文含首批 Turing sm_75 公开实测数据。

## 为什么要做这件事

2080Ti（Turing sm_75）是 2018 年的卡：无 FP4/BF16 tensor core、PCIe gen3、单卡 22G。
业界跑 DeepSeek V4 Flash 的方案（vLLM/SGLang/KTransformers）验证硬件都是 SM_89/120，
**sm_75 上没有任何社区实测帖**。而 MoE 架构（每 token 只激活 13B）恰好是
"显存不够、内存/分层来凑"的最佳场景——这正是 Strata 引擎在 Qwen 125B 上验证过的路：
单张 2080Ti 跑 125B MoE，decode 48-60 tok/s。

## 架构侦察：V4 Flash 与 Qwen MoE 的异同

对 Strata 引擎源码（13K 行 C++ 核心）+ llama.cpp DEEPSEEK4 支持的交叉侦察结论：

| 组件 | V4 Flash | 适配策略 |
|---|---|---|
| 专家张量命名 | `blk.N.ffn_{gate,up,down}_exps.weight` | **与 Qwen 完全同款**，pack 工具链零改动复用 |
| 注意力 | MLA 变体 + CSA/HCA 混合压缩（43 层里 20 层 4×、20 层 128× 压缩） | 需新增压缩段算子 |
| 稀疏检索 | Lightning Indexer，top_k=512 | **与 Strata QSA 内核同构，大部分可复用** |
| 残差 | mHC 超连接（Sinkhorn） | 纯新增小算子 |
| 投机解码 | DSpark 模块 | 对齐 Strata 现有 MTP 通路 |
| KV 大小 | kv_heads=1、head_dim=512 + 压缩红利 | **1M 上下文 KV 仅 ~16.7G fp16 / 8.4G q8**，上下文近乎免费 |

真正要新写的只有两个小算子（CSA/HCA 压缩段、mHC-Sinkhorn），其余全是复用。

## P0 基线实测（首批 sm_75 数据）

模型：unsloth DeepSeek-V4-Flash-GGUF UD-IQ2_XXS（90.9G）
环境：llama.cpp（含 deepseek4.cpp），`-ngl 99 -ncmoe 99`，非专家 14.7G 进 GPU，专家全驻 RAM

| 配置 | prefill | decode |
|---|---|---|
| 基线 `-b 512 -ub 256` | 6.5-8 tok/s | 3.2→4.8 tok/s（页缓存热后 +50%） |
| `+ -cram 6 -b 2048 -ub 512`（大 batch + 压缩注意力模式） | **73 tok/s（×11）** | — |
| `+ -fa 1`（flash-attn） | — | **变慢，Turing 上有害** |

三组数据讲了三件事：

1. **prefill 可救**：大 batch + 压缩注意力直接 10 倍。V4 的 CSA/HCA 层在 prefill 时是真实算力节省
2. **decode 救不动**：瓶颈是 76G 专家权重在 CPU 做 GEMM（Cascade Lake 无 AMX），
   llama.cpp 静态张量放置没有 LRU 缓存，天花板 ~5 tok/s——任何 attention 参数都无效
3. **上下文豪华**：1M 原生烘焙（YaRN factor 16 写死在 GGUF 里），KV 极小，
   1M 填满也只要 8.4G（对比稠密模型 224K 就要 3-6G）

## 目标架构（P2 完成后的形态）

```
GPU0: Qwen 125B（Strata 现役主力）
GPU1: V4 Flash 非专家 14.7G + 专家 LRU 缓存 ~6G（按热度命中 60-80%）
RAM:  V4 专家 76G 全量（第二层）+ Qwen 50G
SSD:  仅冷存储，不在运行时路径
```

预期收益：**decode 4.8 → 25-45 tok/s，prefill 73 → 300-800 tok/s**
（Turing GDDR6 640GB/s 上热专家 GEMM vs PCIe gen3 12GB/s 冷传输，Strata 已验证过这套账）。

llama.cpp 为什么做不到：张量放置是静态的，GPU1 只剩 6.5G 只能放 7% 专家，
静态放置命中率极低；必须走带 LRU 预取的 ExpertSource 分层缓存——这就是这个框架的存在理由。

## 五阶段路线

- **P0** llama.cpp 基线跑通 + sm_75 首份跑分 ✅（本文数据）
- **P1** pack builder：融合专家张量 → 每专家 [gate|up|down] blob 交错重排
- **P2** 引擎架构分支：路由→ExpertSource（复用）、MLA+CSA/HCA→QSA 变体、mHC 小算子
- **P3** DSpark 接入投机解码通路
- **P4** 调优：GPU1 二级专家缓存、1M 上下文、会话缓存

## 风险台账

- Turing 无 FP4/BF16 tensor core → 必须走 GGUF IQ 量化路径（已验证 IQ2_XXS 可跑）
- llama.cpp V4 支持仍在活跃修复期：chat 模板/reasoning 解析有 bug、`-sm tensor` 段错误 #28826
- QSA kernel 的 KV 布局与 MLA 压缩段可能不一致 → 预留"先 dense 注意力跑通再优化"退路

## 代码与复现

仓库：github.com/（见 iot-pub.top 文章页链接）
复现命令：`llama-server -m DeepSeek-V4-Flash-UD-IQ2_XXS.gguf -ngl 99 -ncmoe 99 -cram 6 -b 2048 -ub 512 -fa 0`
注意：`-fa 1` 在 Turing 上对 decode 有害；8080 端口易与现有服务冲突。

*实测与框架开发进行中，数据均为本机真实跑分。*
