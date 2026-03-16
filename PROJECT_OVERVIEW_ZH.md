# Pruned Attention 项目总览

## 1. 项目一句话

这是一个围绕 **Llama 长上下文推理加速** 搭建的研究型仓库，核心思路是：

- 用一个更小、更便宜的“代理注意力”先预测哪些 KV token 最重要；
- 再只对这些候选位置做高精度注意力计算；
- 以此降低长文本推理时的 KV Cache 访存、计算和显存开销。

从仓库内容看，它不只是一个单点算法实现，而是由三部分组成：

1. **主算法实现**：`src/pruned_attention/`，实现剪枝注意力、校准、蒸馏、推理接入；
2. **评测与对比**：`LongBench/` 与 `scripts/`，用于长文本任务、PPL、LoRA 微调和基线对比；
3. **硬件协同模拟**：`simulator/`，用 C++ 建模 HBM / DDR / 稀疏流水线，验证软硬件协同设计收益。

---

## 2. 仓库想解决什么问题

### 2.1 背景

在自回归解码时，Llama 一类模型会不断把历史 token 的 Key / Value 缓存在 KV Cache 中。上下文越长：

- KV Cache 占用越大；
- 每一步注意力都要读取更长的历史序列；
- 推理瓶颈会越来越偏向 **访存带宽**，而不是纯算力。

这个仓库的核心判断是：

> 注意力分数通常具有明显稀疏性，不需要每次都精确访问全部历史 KV。

因此它采用两阶段方案：

1. 先用压缩后的 `q_small / k_small` 粗筛；
2. 再对筛出的 Top-K、Sink Token、Local Window 做精确计算。

这本质上是一个 **“预测 → 精算”** 的稀疏注意力框架。

---

## 3. 仓库整体结构

## 3.1 顶层目录含义

### `src/pruned_attention/`

主算法代码，项目最核心的目录。

- `attention.py`：压缩注意力和自定义 cache 的实现；
- `calibration.py`：做敏感度分析，生成每层保留维度索引；
- `distillation.py`：逐层蒸馏 `q_proj_small / k_proj_small`；
- `modeling_llama.py`：对 Hugging Face Llama 的定制封装；
- `utils.py`：伪量化模块。

### `src/h2o_attention/`

另一个稀疏注意力基线实现，核心思想是 **Heavy Hitter + Recent Window**。

这个目录更像对比方法或实验分支，用来与 pruned attention 做比较。

### `scripts/`

一组直接可运行的实验脚本，包括：

- 校准；
- 蒸馏；
- 流式推理；
- LoRA 微调；
- PPL 测试；
- LoRA + 压缩模型推理。

### `LongBench/`

集成的长文本评测基准及其扩展脚本。

包含：

- LongBench / LongBench-E 评测代码；
- retrieval-based context compression；
- summarization-based compression；
- 部分预测结果、配置文件和样例权重。

### `simulator/`

一个独立的 C++ 硬件模拟器，目标不是训练模型，而是模拟：

- HBM 扫描压缩 KV；
- DDR 稀疏抓取 full KV；
- BF16 / Int4 计算阵列；
- 多阶段流水线重叠。

它服务于“算法 + 硬件协同”的论文式验证。

### `data/`

主要是实验产物，而不是训练源码：

- `indices.json`；
- `small_attn_weights.pt`；
- 各类 pruning / distillation / PPL 图表与 JSON 结果。

---

## 4. Python 主线：压缩注意力是怎么工作的

这一部分是整个仓库的主线。

### 4.1 核心对象：`CompressedLlamaAttention`

`src/pruned_attention/attention.py` 中的 `CompressedLlamaAttention` 是核心模块。

它接收原始 Llama attention layer，并做三件事：

1. 保留原始 `q_proj / k_proj / v_proj / o_proj`；
2. 根据校准得到的 `keep_indices`，裁出更小的 `q_proj_small / k_proj_small`；
3. 在前向过程中，用 `small q/k` 做代理打分，用 `full q/k/v` 做精确输出。

可以把它理解为：

- **Small Path**：负责便宜地估计“哪些 token 值得看”；
- **Full Path**：只在少量候选位置上做真正的注意力。

### 4.2 为什么是“按维度剪枝”

这里不是直接删 token，也不是删整个头，而是先删 **每个 KV 头内部的部分特征维度**。

校准阶段会分析：

- 哪些 RoPE 后的维度对 Top-K 注意力排序最敏感；
- 然后按头保留最重要的维度对；
- 最终形成每层一个 `indices.json`。

所以这个项目的“压缩”有两个层次：

1. **特征维度压缩**：构造 small q/k；
2. **时序 token 稀疏化**：运行时只精算部分 token。

### 4.3 前向流程

在 `CompressedLlamaAttention.forward()` 中，大致流程是：

1. 用全量投影计算 `q_full / k_full / v_full`；
2. 用小投影计算 `q_small / k_small`；
3. 对 full 和 small 路径都应用 RoPE；
4. 如有 cache，则同时缓存 full key/value 和 `small_key_states`；
5. 用 `q_small @ k_small^T` 计算代理分数 `proxy_scores`；
6. 从代理分数中选 Top-K；
7. 加上 `sink_size` 和 `local_window` 形成保留集合；
8. 仅对保留位置计算真实注意力输出。

其中它分两种场景：

- **Prefill / 多 token 场景**：先算完整 `real_scores`，再对非保留位置打 mask；
- **Decode / 单 token 场景**：直接按索引 gather 所需 `k_full / v_full`，再精算。

这说明项目已经针对推理阶段最关键的单步解码做了专门优化。

### 4.4 为什么保留 Sink + Local

只保留动态 Top-K，容易导致数值稳定性和局部语法一致性问题，所以这里又叠加了两类固定保留策略：

- **Sink Token**：始终保留最前面的少量 token，当作全局锚点；
- **Local Window**：始终保留最近窗口，保证局部上下文连续性。

这也是整个项目最重要的稳定性设计之一。

### 4.5 “逃逸层”机制

`CompressedLlamaAttention` 中有 `escaped_layer = [0, 1, 30, 31]`。

含义是：这些层不做压缩，直接走原始全注意力。

这反映出作者的经验判断：

- 最前层和最后层通常更敏感；
- 压缩它们的收益未必划算；
- 保留全精度有助于稳定整体性能。

这属于一个很典型的研究型工程取舍。

---

## 5. Cache 设计：为什么这里要自定义 KV Cache

标准 Hugging Face Cache 只缓存 `key/value`，但这个项目还要缓存 **small key**，因为代理打分在 decode 阶段也要继续用到历史的 `k_small`。

因此 `src/pruned_attention/attention.py` 里实现了：

- `KVWithSmallKLayer`：每层缓存 `keys / values / small_keys`；
- `CustomCache` / `CustomDynamicCache`：兼容 HF cache 体系；
- `KVWithSmallKCache`：实际在生成时传入的 cache 类型。

这让模型在流式生成时可以同时维护：

- full KV，用于最终精算；
- small K，用于下一步快速 Top-K 预测。

也就是说，这个项目的压缩注意力不是一次性的离线计算，而是 **真正支持增量解码** 的。

---

## 6. 校准：如何得到 `indices.json`

对应文件：`src/pruned_attention/calibration.py`

### 6.1 核心思想

校准阶段不是训练，而是做一次 **敏感度分析**。

函数 `get_compression_indices()` 的逻辑可以概括为：

1. 对某层 attention 输入 hidden states；
2. 计算 full attention scores；
3. 只关注高分的 query-key 对；
4. 逐个分析每一对维度对这些高分位置的贡献；
5. 为每个 KV head 选出最重要的维度对；
6. 生成该层的保留维度索引。

这里保留的是 **成对维度**，因为 RoPE 旋转本身以二维配对形式工作。

### 6.2 数据来源

`get_c4_simple()` 会从 C4 中抽取固定长度样本，作为校准数据。

也就是说，校准并不依赖下游监督标签，而是基于无监督文本分布完成。

### 6.3 输出结果

`run_calibration_and_save_indices()` 会遍历所有层，输出一个 JSON：

```json
{
  "0": [[...], [...]],
  "1": [[...], [...]],
  "2": [[...], [...]]
}
```

每个 layer 对应一组 `[num_kv_heads, compressed_dim]` 的保留索引。

随后 `apply_compression_to_model()` 会读取这个文件，把原始 attention 替换成 `CompressedLlamaAttention`。

---

## 7. 蒸馏：为什么还要训练 `small q/k`

对应文件：`src/pruned_attention/distillation.py`

### 7.1 为什么需要蒸馏

仅仅通过切片原始权重来构造 `q_proj_small / k_proj_small`，虽然可以运行，但它们未必足够善于恢复“谁是 Top-K”。

所以项目又做了一个轻量蒸馏步骤：

- 冻结原始主干；
- 只训练 `q_proj_small.weight` 和 `k_proj_small.weight`；
- 让 small path 更好地逼近 full path 的排序能力。

### 7.2 当前实现的损失函数

已实现并在主脚本中使用的是：

- `CompressedLlamaAttentionTopKDistillWrapper`
- 损失：`MarginRankingLoss`

做法是：

1. teacher 用 full q/k 得到真实重要位置；
2. student 用 small q/k 产生代理分数；
3. 强化“重要位置分数 > 不重要位置分数”。

这和论文式“排序蒸馏”思路是一致的。

### 7.3 逐层蒸馏策略

`train_layer_wise_distillation()` 采用的是 **layer-wise distillation**：

1. 先截获第 0 层输入；
2. 蒸馏第 0 层 small attention；
3. 用蒸馏后的第 0 层前向，生成第 1 层输入；
4. 再蒸馏第 1 层；
5. 依次推进到最后一层。

好处是：

- 显存更友好；
- 每次只优化一个局部模块；
- 对研究实验而言更容易观察层间效果。

蒸馏结束后，通常会导出 `small_attn_weights.pt`。

---

## 8. 自定义 `modeling_llama.py` 的作用

如果只改 attention 层，很多事情仍然不顺：

- 生成时需要自定义 cache；
- layer forward 需要透传 `position_embeddings`；
- 需要确保 decoder layer 可替换；
- 需要和 HF 输出结构保持兼容。

因此 `src/pruned_attention/modeling_llama.py` 做了一个轻量重封装：

- `LlamaDecoderLayer`：把 self attention 保持为可替换结构；
- `LlamaModel`：默认在 `use_cache=True` 时创建 `KVWithSmallKCache`；
- `LlamaForCausalLM`：维持 Hugging Face 风格的 forward / loss / logits 接口。

可以理解为：

> 这个文件是“把 pruned attention 无缝接入 HF Llama 运行时”的胶水层。

---

## 9. 量化模块：`PseudoQuantizer`

对应文件：`src/pruned_attention/utils.py`

该模块提供伪量化支持，用于模拟：

- INT4 量化；
- Binary 量化。

特点：

- 使用 STE（Straight-Through Estimator）；
- 前向用 fake quantized 值；
- 反向仍向原始张量传播梯度。

当前接入点在 `CompressedLlamaAttention` 中：

- 如果传入 `mode`，会对 `q_small / k_small` 做伪量化；
- 主要目的是模拟更低比特的代理路径，进一步降低成本。

这说明作者不仅考虑“剪枝”，也在尝试 **剪枝 + 量化** 的联合压缩。

---

## 10. H2O 分支：仓库中的对比方法

对应目录：`src/h2o_attention/`

这里实现了另一个思路：

- 累积历史注意力得分；
- 保留 heavy hitter token；
- 再叠加 recent window；
- 下一步通过 mask 控制可见范围。

和 `pruned_attention` 的差别在于：

- `pruned_attention` 更偏“代理预测 + 精算”；
- `h2o_attention` 更偏“历史得分驱动的 cache 淘汰 / 保留”。

因此这个仓库实际上内置了至少两种长上下文稀疏推理方案。

---

## 11. 脚本层：仓库的实际使用方式

### 11.1 `scripts/run_calibration.py`

作用：

- 载入基础 Llama；
- 从 C4 取样；
- 运行校准；
- 输出 `indices.json`。

这是整个压缩流程的起点。

### 11.2 `scripts/run_distillation.py`

作用：

- 载入定制版 `LlamaForCausalLM`；
- 读取 `indices.json`；
- 逐层训练 `q_proj_small / k_proj_small`；
- 导出蒸馏后的 `small_attn_weights.pt`。

### 11.3 `scripts/run_streaming_inference.py`

这是一个统一推理入口，支持三种模式：

- `base`：原始模型；
- `compress`：pruned attention；
- `h2o`：heavy hitter attention。

它的意义是把三种方法放到同一个生成脚本里做对比。

### 11.4 `scripts/run_perplexity_test.py`

用于离线算 PPL，支持：

- WikiText-2；
- WikiText-103；
- PTB；
- C4；
- LAMBADA。

常用于比较：

- 原模型；
- 压缩模型；
- 加载蒸馏权重后的压缩模型。

### 11.5 `scripts/run_lora_finetuning.py`

作用：

- 先把模型压缩；
- 再加载蒸馏得到的 small-attention 权重；
- 冻结 small 模块；
- 在混合数据集上做 LoRA 微调。

训练数据混合了：

- Tulu-3 SFT mixture；
- Orca-AgentInstruct；
- C4；
- PTB；
- WikiText-2。

这说明仓库不只关心“压缩后能不能跑”，还关心“压缩后能否继续对齐 / 指令微调”。

### 11.6 `scripts/run_inference_with_lora.py`

作用：

- 加载基础模型；
- 应用压缩 attention；
- 加载 small attention 权重；
- 再配合 LoRA adapter 做流式生成。

这个脚本更像实验阶段的 end-to-end demo。

---

## 12. LongBench：项目里的评测体系

`LongBench/` 基本上是一个被纳入仓库的完整评测子系统。

### 12.1 它在本项目中的角色

主算法做完后，需要回答两个问题：

1. 长文本任务上性能损失有多大？
2. 与其他 context compression / sparse attention 方法相比效果如何？

LongBench 就承担这个角色。

### 12.2 目录功能

- `LongBench/pred.py`：对各个任务生成预测；
- `LongBench/eval.py`：按任务指标打分；
- `LongBench/metrics.py`：F1、ROUGE、检索、分类等指标实现；
- `LongBench/config/`：模型路径、上下文长度、prompt 模板、生成长度配置；
- `LongBench/task_zh.md`：任务定义；
- `LongBench/data/`：数据和样例权重。

从 `pred.py` 里的模型枚举可以看出，这里已经集成了：

- 原始 Llama；
- `llama3-8b-compress`；
- `llama3-8b-h2o`；
- 其他长上下文模型基线。

也就是说，LongBench 在这个仓库中已经不只是“第三方 benchmark”，而是作者实验闭环的一部分。

### 12.3 LongBench 的两个扩展方向

#### 检索式压缩：`LongBench/retrieval/`

这里提供三种 retriever：

- BM25；
- Contriever；
- OpenAI Embedding。

流程是：

1. 先对长上下文检索；
2. 把检索片段拼进输入；
3. 再做下游问答或推理；
4. 用 `pred.py / eval.py` 打分。

这是一类典型的 **context compression baseline**。

#### 摘要式压缩：`LongBench/summ/`

`compress.py` 会：

- 把超长文本切段；
- 用外部模型分别摘要；
- 再拼接成压缩后的上下文。

它更像另一种 baseline：

- 不做 attention 级别压缩；
- 而是直接改写输入文本。

所以整个仓库在评测维度上其实覆盖了：

- 模型内部压缩；
- 检索式压缩；
- 摘要式压缩。

---

## 13. simulator：软硬件协同部分在做什么

这是仓库里第二条非常重要的主线。

### 13.1 目标

`simulator/doc/abstract.md` 表达得很明确：

- 长文本推理越来越受制于 memory wall；
- 算法上可以先预测少量关键 token；
- 硬件上需要分级存储和稀疏流水线把这种预测机制吃干榨净。

换句话说：

> Python 代码负责证明“算法上可行”，C++ simulator 负责证明“硬件上值得做”。

### 13.2 主要模块

#### Memory 子系统

- `HBMController`：模拟高带宽、突发流式读取，适合 small KV；
- `DDRController`：模拟多通道、多 bank、row hit / row conflict，适合 full KV 稀疏抓取。

其中 `DDRController` 明确实现了：

- `LINEAR` 映射；
- `HASH_XOR` 映射。

这对应论文里“通过映射减少随机访问 bank 冲突”的思路。

#### PE / 计算阵列

`simulator/include/PE/` 与 `simulator/src/PE/` 下有一套较完整的算子级建模：

- `MacUnit`、`AddUnit`、`MultiplyUnit`；
- `ReduceMacUnit`、`ReduceMacArrayUnit`；
- `AddTreeUnit`；
- `GemvScheduler`。

这套组件用来搭出：

- Int4 预测阵列；
- BF16 精算阵列；
- GEMV 级别的调度和流水线。

#### System 层

`PipelineSimulator` 把整个过程抽象为三阶段：

- `PREDICT_HBM`：HBM 上扫描 small KV；
- `FETCH_DDR`：DDR 上稀疏抓取 full KV；
- `COMPUTE_BF16`：高精度 attention 计算。

它还会记录：

- 每个 head group 的开始 / 结束周期；
- 整体 bubbles；
- 阶段重叠情况；
- 总加速比。

这已经不是“代码风格 demo”，而是较完整的体系结构实验框架。

### 13.3 实验程序

`simulator/experiments/` 中至少有两个实验：

- `experiment_memory_efficiency.cpp`：比较不同 DDR 映射 / 请求合并策略；
- `experiment_pipeline_overlap.cpp`：分析多阶段流水线重叠和系统瓶颈。

### 13.4 测试体系

`simulator/test/` 覆盖了：

- bf16 基础运算；
- int4 pipeline；
- GEMV scheduler；
- 多种 PE 单元。

说明 simulator 部分的工程完成度相对高于普通原型脚本。

---

## 14. 这个项目的完整工作流

如果从“第一次接手这个仓库”视角看，推荐把它理解成下面的流水线：

### 阶段 A：准备基础模型

- 准备 Hugging Face Llama 权重；
- 确保使用 `attn_implementation="eager"`。

### 阶段 B：校准

- 用 `scripts/run_calibration.py`；
- 在 C4 样本上分析每层敏感维度；
- 生成 `indices.json`。

### 阶段 C：替换 attention

- 调用 `apply_compression_to_model()`；
- 把原始 self attention 替换成 `CompressedLlamaAttention`。

### 阶段 D：蒸馏

- 用 `scripts/run_distillation.py`；
- 逐层训练 small q/k；
- 输出 `small_attn_weights.pt`。

### 阶段 E：推理与评估

- `run_streaming_inference.py` 做在线生成；
- `run_perplexity_test.py` 做语言建模质量评估；
- `LongBench/` 做长文本任务评测；
- 与 `h2o_attention`、retrieval compression、summary compression 做对比。

### 阶段 F：继续适配下游

- `run_lora_finetuning.py` 对压缩模型做指令微调；
- `run_inference_with_lora.py` 做最终推理验证。

### 阶段 G：硬件侧验证

- 用 `simulator/` 验证 HBM / DDR / pipeline 设计是否真正带来吞吐收益。

---

## 15. 我对当前仓库状态的判断

从代码组织和实验痕迹来看，这个仓库处于 **“研究原型 + 实验积累 + 部分工程化”** 的状态。

### 15.1 优点

- 主算法链路完整：校准 → 替换 → 蒸馏 → 推理 → 评测；
- 兼顾算法与硬件两条主线；
- 已纳入 LongBench 等评测体系；
- 支持增量解码，不只是离线分析；
- 还延伸到了 LoRA 微调和 PPL 评估。

### 15.2 明显的原型特征

- README 仍偏简略，部分说明与当前仓库不完全同步；
- 根目录 README 提到了 `requirements.txt`，但根目录当前没有该文件；
- `attention.py` 中 `KVWithSmallKCache` 定义重复出现了一次；
- 一些脚本仍写死了本地绝对路径，说明主要按作者环境运行；
- `src/slide_attention/` 只剩缓存文件，没有可读源码；
- 部分脚本和目录更像实验记录，而不是面向发布的产品接口。

### 15.3 适合如何理解

最合适的定位不是“现成库”，而是：

> 一个围绕“长上下文稀疏注意力 + 软硬件协同”展开的研究仓库。

如果后续要继续维护，建议把它拆成三层看：

1. **算法核心层**：`src/pruned_attention/`；
2. **实验与评测层**：`scripts/` + `LongBench/` + `data/`；
3. **体系结构验证层**：`simulator/`。

---

## 16. 对每个目录的“阅读优先级”建议

如果后续还要继续深入，我建议按这个顺序读：

1. `README.md`
2. `src/pruned_attention/attention.py`
3. `src/pruned_attention/calibration.py`
4. `src/pruned_attention/distillation.py`
5. `src/pruned_attention/modeling_llama.py`
6. `scripts/run_calibration.py`
7. `scripts/run_distillation.py`
8. `scripts/run_streaming_inference.py`
9. `LongBench/pred.py` + `LongBench/eval.py`
10. `simulator/doc/abstract.md`
11. `simulator/include/System/PipelineSimulator.h`
12. `simulator/src/System/PipelineSimulator.cpp`

这样会先建立“算法怎么跑”，再补上“怎么评估”和“为什么硬件上也成立”。

---

## 17. 一句话总结

**这是一个以 Llama 为对象、以 KV Cache 压缩为核心、同时覆盖算法实现、评测对比和硬件模拟的长文本推理研究仓库。**

如果要再压缩成一句更直白的话：

> 它的核心是在 Llama 中先用“小注意力”预测重要 token，再用“真注意力”只算关键部分，并用 LongBench 和硬件模拟器验证这件事到底值不值得。
