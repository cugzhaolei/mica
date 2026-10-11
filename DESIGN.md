# deepseek4-strata — DeepSeek V4 Flash 适配 Strata 分层卸载框架

> 目标：把 Strata（MoE 专家分层卸载 GPU→RAM→SSD + 投机解码）从 qwen4exp 硬编码
> 扩展到 DeepSeek V4 Flash（284B/13B激活，256专家选6，43层，1M ctx）。
> 本机：2×2080Ti 22G（Turing sm_75，PCIe gen3），157G RAM。

## 源码侦察结论（2026-10-11，/mnt/windows/Strata）

| 发现 | 位置 | 对适配的意义 |
|---|---|---|
| 引擎架构键硬编码 `qwen4exp.*` | src/program/generate.cpp（仅 10 处引用） | 加 `deepseek4.*` 分支改动面小 |
| 注意力内核 = QSA（indexer top-k 稀疏 + int8/q4 KV + rotate） | src/kernels/qsa.hpp, native_qsa* | 与 V4 的 Lightning Indexer top-512 同构，**大部分可复用** |
| `ExpertSource` 抽象类 + FileExpertSource(mmap) | include/strata/core/expert_source.hpp | 专家池与主循环解耦，**卸载层无需改**，只需喂新 router 输出 |
| vendored llama.cpp 已含 LLM_ARCH_DEEPSEEK4 | third_party/llama.cpp/src/llama-arch.cpp:83 | GGUF 键名/转换表现成（conversion/deepseek.py） |
| pack 格式：dense.bin + index.txt + native_experts.txt + conversions.json | /mnt/shared/Strata-data-iq3/packs/iq3_s/ | 写一个 deepseek4 pack builder 即可复用整个加载链 |
| layer.cpp 权重名：attn_q/k/v/o + indexer.k_proj/q_proj + *_norm | src/core/layer.cpp:1041 | 已是"压缩注意力+indexer"命名，V4 的 q_a/q_b/kv_a 需映射 |

## V4 Flash 与 Qwen4exp 的架构差异（要新增的三块）

1. **MLA 变体 + CSA/HCA 混合压缩注意力**（逐层 compress_ratio 4×/128×，滑窗 128）
   → QSA 的 top-k 稀疏读取骨架可复用；需新增"压缩段"（把 4/128 个 token 压成 1 个 KV 单元）
   → 好消息：KV 极小（1M ctx ~180MB/序列），KV streaming 层反而变简单
2. **mHC 超连接（Sinkhorn）** — 残差连接变体，纯新增小算子（43层每层一次小矩阵 Sinkhorn）
3. **DSpark 投机解码** — Strata 已有 MTP 通路（--mtp rt），V4 官方带 DSpark 模块，接口对齐后可替换

## 分层路线（每阶段独立可验收）

- **P0 基线（不碰 Strata）**：llama.cpp（Strata vendored fork 或上游最新）+ unsloth GGUF
  IQ2_XXS(91G) `--n-cpu-moe` 跑通，实测 sm_75 decode/prefill → 决定值不值得深挖
  **P0 阻塞：磁盘**。/mnt/shared 剩 22G、/mnt/windows 剩 91G，91G 模型放不进任何盘
  （RAM 157G 也不够当盘）。候选：删 /mnt/windows/Strata-data（72G IQ2_XS 旧模型数据，
  现跑 IQ3_S 用 /mnt/shared/Strata-data-iq3，iq2xs 档若弃用即可删）→ /mnt/windows 腾出 163G
- **P1 pack builder**：`ds4_pack.py`（本目录骨架已写）— 关键发现：llama.cpp DEEPSEEK4
  专家张量名 `blk.N.ffn_{gate,up,down}_exps.weight` 与 Strata iq_pack.py 的
  is_expert() 后缀判据**完全一致**，pack 工具链几乎零改动复用；
  需新增的是三个融合张量→每专家 [gate|up|down] blob 的交错重排
- **P2 架构分支**：generate.cpp 加 `deepseek4exp` 分支：读 deepseek4.* 键、
  路由→ExpertSource（复用）、MLA+CSA/HCA→QSA 变体 kernel、mHC 小算子
- **P3 投机**：DSpark 接入 MTP 通路（或先用 ngram 保底）
- **P4 调优**：GPU1 二级专家缓存（remote-expert-opt）、YaRN@1M、conversation cache

## 风险台账

- Turing 无 FP4/BF16 tensor core：官方 FP4 专家权重必须走 GGUF IQ 量化路径（Strata 已支持 IQ3_S/IQ2_XS）
- llama.cpp V4 open bug：-sm tensor 段错误 #28826 → P0 用默认 schedule 绕开
- sm_75 无社区实测帖：P0 就是来填这个空（也是视频素材）
- QSA kernel 假设 KV 布局与 MLA 压缩段不完全一致 → P2 最大不确定点，预留"先 dense 注意力跑通再优化"退路

## P0 基线实测结果（2026-10-11，首次公开级 sm_75 数据）

模型：unsloth/DeepSeek-V4-Flash-GGUF UD-IQ2_XXS（90.9G，三件套，architecture=deepseek4）
命令：`llama-server -ngl 99 -ncmoe 99 -b 512 -ub 256 -t 24 -c 8192`，CUDA_VISIBLE_DEVICES=1（与 Strata GPU0 共存）

| 指标 | 实测 | 备注 |
|---|---|---|
| 加载 | 36s（90.9G mmap） | 非专家 14.7G 进 GPU1，专家全驻 RAM |
| decode | 3.2→4.8 tok/s | 第2请求快50%（RAM 页缓存热）；落在调研预估下沿 |
| prefill | 6.5-8 tok/s | CPU MoE GEMM 吃掉全部算力，不可用于长 prompt |
| 内存 | 74G used + 55G shared | 157G 内富余 |
| 质量 | 能答但 chat 模板/reasoning 解析有 bug（内容为空/思考链截断） | llama.cpp V4 支持仍在活跃修复期 |

**P0 结论**：可行性确认（跑通+能推理）；性能天花板确认（CPU 专家 GEMM 在
Cascade Lake 无 AMX 上 = 4.8 tok/s）。对比 Strata GPU 专家缓存跑 Qwen125B = 48-60 tok/s，
**P2 适配的预期收益 = 一个数量级（4.8 → 30-50 tok/s），值得做**。
复现配方在 /mnt/windows/Dev/deepseek4-strata/（dl_v4flash.py + 本表命令）。
下一步（新会话）：P1 ds4_pack 专家 blob 交错重排 → P2 generate.cpp deepseek4exp 分支。
