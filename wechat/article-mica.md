# 把 284B 的 DeepSeek V4 Flash 塞进两张 2080Ti：一次真实的架构侦察与跑分

> 一句话：MoE 时代，2018 年的 2080Ti 还没退役。我们把 DeepSeek V4 Flash（284B 总参 / 13B 激活 / 256 专家选 6 / 1M 上下文）在 2×2080Ti 22G + 157G RAM 的老工作站上真跑了起来，并拿到了这批卡的第一手实测数据。

## 一、为什么要跟一张 2018 年的卡过不去

2080Ti（Turing sm_75）在今天跑大模型的处境很尴尬：没有 FP4/BF16 tensor core、PCIe 只有 gen3、单卡显存 22G。业界跑 DeepSeek V4 Flash 的方案（vLLM / SGLang / KTransformers）验证硬件清一色 SM_89/120，**sm_75 上找不到任何社区实测帖**。

但 MoE 架构给了老卡一条活路：V4 Flash 每个 token 只激活 13B 参数，284B 的体量里绝大部分是"用不上的专家"。显存装不下，就用内存装；内存只是中转，就分层调度。这正是我们在 Strata 引擎上已经验证过的路——同一台机器，单张 2080Ti 用"专家分层卸载（GPU→RAM→SSD）+ 投机解码"跑 Qwen 125B MoE，decode 能到 48-60 tok/s。

这个项目的目标就是把这套思路扩展到 DeepSeek V4 Flash。项目代号 **mica**，代码在 github.com/cugzhaolei/mica。

## 二、架构侦察：V4 Flash 到底需要新写多少代码

动手前先做了交叉侦察：Strata 引擎源码（13K 行 C++ 核心）× llama.cpp 的 DEEPSEEK4 支持。结论比预想的乐观：

| 组件 | V4 Flash 的情况 | 适配策略 |
|---|---|---|
| 专家张量命名 | `blk.N.ffn_{gate,up,down}_exps.weight` | **与 Qwen 完全同款**，pack 工具链零改动复用 |
| 注意力 | MLA 变体 + CSA/HCA 混合压缩（43 层里 20 层 4×、20 层 128× 压缩） | 需新增"压缩段"算子 |
| 稀疏检索 | Lightning Indexer，top_k=512 | **与 Strata 现有 QSA 内核同构，大部分可复用** |
| 残差连接 | mHC 超连接（Sinkhorn） | 纯新增小算子，每层一次小矩阵运算 |
| 投机解码 | 官方 DSpark 模块 | 对齐 Strata 现有 MTP 通路即可 |
| KV 大小 | kv_heads=1、head_dim=512，叠加压缩红利 | **1M 上下文 KV 仅 ~16.7G fp16 / 8.4G int8**，长上下文近乎免费 |

真正要从零写的只有两个小算子（CSA/HCA 压缩段、mHC-Sinkhorn），其余全是复用。Strata 的 `ExpertSource` 抽象层把"专家从哪来"和主循环解耦得很干净——喂新的 router 输出就行，卸载层一行不用改。

## 三、P0 基线：先把 llama.cpp 路线跑到极限，拿第一手数据

在动 Strata 引擎之前，先用最现成的路线（llama.cpp + unsloth GGUF）把天花板量出来。模型是 UD-IQ2_XXS 量化，90.9G，三个分片 mmap 加载 36 秒，非专家权重 14.7G 进 GPU，专家全部驻内存。

以下全部是本机真实日志数据（不是估算）：

**跑分汇总**

| 配置 | prefill | decode |
|---|---|---|
| 基线 `-ngl 99 -ncmoe 99 -b 512 -ub 256` | 6.5-8 tok/s | 3.2→4.8 tok/s（页缓存热后 +50%） |
| 大 batch + 压缩注意力 `-cram 6 -b 2048 -ub 512` | **73 tok/s（×11）** | 2.1-2.8 tok/s |
| 上面基础上开 flash-attn `-fa 1` | — | **变慢，Turing 上有害** |

**单条日志摘录**（run2，2408 token 的 prompt）：

```
prompt eval time = 33008.45 ms / 2408 tokens ( 72.95 tokens per second)
       eval time = 18946.85 ms /   40 tokens (  2.06 tokens per second)
```

三条结论：

1. **prefill 可救**：大 batch + 压缩注意力直接 10 倍提升。V4 的 CSA/HCA 层在 prefill 阶段是真实的算力节省，不是纸面设计。
2. **decode 救不动**：瓶颈是 76G 专家权重在 CPU 上做 GEMM（Cascade Lake 没有 AMX），llama.cpp 的张量放置是静态的、没有 LRU 缓存，天花板 ~5 tok/s——任何 attention 参数都改变不了这个数。
3. **上下文是真豪华**：1M 原生烘焙在 GGUF 里（YaRN factor 16 写死），KV 极小，填满 1M 也只要 8.4G。对比稠密模型 224K 上下文就要吃 3-6G，这是压缩注意力的红利。

顺带记录两个坑：llama.cpp 的 V4 支持还在活跃修复期，chat 模板/reasoning 解析有 bug（出现过回复内容为空、思考链截断）；`-sm tensor` 模式有段错误（上游 issue #28826），用默认 schedule 绕开。

## 四、为什么要自研分层引擎：一张图看懂

llama.cpp 静态放置为什么到不了高速？因为 GPU 显存只剩 ~6.5G，只能静态放下 7% 的专家，命中率极低；而 MoE 的访问是有热度的，需要的是**带 LRU 预取的专家分层缓存**——这就是这个框架存在的理由。

目标架构（P2 完成后的形态）：

```
GPU0: Qwen 125B（Strata 现役主力）
GPU1: V4 Flash 非专家 14.7G + 专家 LRU 缓存 ~6G（按热度命中 60-80%）
RAM:  V4 专家 76G 全量（第二层）+ Qwen 专家 ~50G
SSD:  仅冷存储，不在运行时路径上
```

预期收益：**decode 4.8 → 25-45 tok/s，prefill 73 → 300-800 tok/s**。账很好算：Turing GDDR6 带宽 640GB/s，热专家在显存里做 GEMM；PCIe gen3 只有 ~12GB/s，冷传输靠预取藏在计算后面。这套账 Strata 在 Qwen 125B 上已经验证过。

## 五、五阶段路线（每阶段独立可验收）

- **P0** llama.cpp 基线跑通 + sm_75 首份跑分 —— ✅ 完成，就是本文数据
- **P1** pack builder：把 llama.cpp 的融合专家张量重排成每专家 [gate|up|down] blob，接入 Strata 加载链 —— 骨架已写
- **P2** 引擎架构分支：路由→ExpertSource（复用）、MLA+CSA/HCA→QSA 变体 kernel、mHC 小算子
- **P3** DSpark 接入投机解码通路（保底方案 ngram）
- **P4** 调优：GPU1 二级专家缓存、1M 上下文、会话缓存

最大的不确定点在 P2：QSA kernel 假设的 KV 布局与 MLA 压缩段可能不完全一致。退路已经预留——先用 dense 注意力跑通正确性，再做稀疏优化。

## 六、复现

仓库：https://github.com/cugzhaolei/mica （含下载脚本、pack 骨架、全部实验日志）

```
llama-server -m DeepSeek-V4-Flash-UD-IQ2_XXS.gguf \
  -ngl 99 -ncmoe 99 -cram 6 -b 2048 -ub 512 -fa 0
```

注意两点：`-fa 1` 在 Turing 上对 decode 有害；`-sm tensor` 会段错误，别开。

*老卡不死，只是分层。P1/P2 进行中，后续数据继续在这更新。*
