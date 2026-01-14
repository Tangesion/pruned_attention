这是一篇关于**大语言模型（LLM）长文本推理加速**的硕士学位论文。

论文的核心痛点在于：随着LLM上下文窗口（Context Window）变长，键值缓存（KV Cache）呈线性甚至超线性增长，导致推理过程从“计算密集型”转变为“访存密集型”，受限于显存容量和带宽（Memory Wall）。

论文提出了一种**软硬件协同（Co-design）**的解决方案：在算法侧，提出“预测-检索”范式，利用低精度、剪枝后的模型预测关键Token；在硬件侧，设计异构FPGA加速器，利用HBM和DDR的分级存储来解决非结构化稀疏访存的带宽瓶颈。

以下是该论文的详细提纲及各部分内容总结：

---

### 1. 绪论 (Chapter 1)

**主要内容：** 阐述研究背景、挑战及现状，定义了长文本推理的核心瓶颈。

* **背景与挑战：**
* LLM向长文本发展，KV Cache显存占用和带宽需求爆炸式增长 。


* 
**存储墙瓶颈：** 在解码（Decoding）阶段，算术强度极低，硬件性能受限于内存带宽而非计算峰值 。


* 
**通用硬件劣势：** CPU并行能力弱；GPU虽然并行强，但在稀疏注意力机制下，面临线程束发散和非合并内存访问的问题，效率低下 。




* **现有方案局限：**
* ASIC（如TPU）缺乏灵活性，难以适应新算法 。


* 存内计算（IMC）精度和工艺尚不成熟 。


* 现有的稀疏策略（如静态窗口、淘汰法）要么损失长距离依赖精度，要么难以硬件化 。




* 
**本文切入点：** 利用FPGA的“软件定义硬件”特性，设计专门针对**动态稀疏矩阵向量乘法（Sparse/Dynamic GEMV）**的加速器 。



### 2. 整体架构 (Chapter 2)

**主要内容：** 介绍了“预测-检索”计算范式和加速器的顶层设计。

* 
**理论基础：** 注意力矩阵具有稀疏性和重尾分布，绝大多数分数趋近于零。无需访问全量KV，只需检索Top-K关键键值对 。


* **计算范式：**
* 
**预测阶段（Predict）：** 使用剪枝且量化（Int4）的“Small KV”进行全量扫描，快速计算粗粒度分数并生成Top-K索引 。


* 
**检索阶段（Retrieval）：** 根据索引，从片外DRAM读取全精度的“Full KV”数据 。




* **硬件架构概览：**
* 
**分级存储：** Small KV存放在高带宽的HBM中；Full KV存放在大容量的DDR中 。


* 
**流水线：** 预填充（Prefill）阶段并行写入双路数据；解码阶段执行“压缩流式预测 -> 稀疏检索 -> 精确计算”的流水线 。





### 3. 算法优化：基于权重敏感度的预取框架 (Chapter 3)

**主要内容：** 解决如何“精准且低代价”地预测出Top-K索引，防止精度下降。

* **权重敏感度列剪枝：**
* 
**发现：** 注意力权重在特征维度分布不均匀，少数关键列主导结果 。


* 
**方法：** 基于激活贡献度分析，计算特征对Top-K的敏感度，剪除冗余列，构建轻量级预测矩阵 。




* **混合稀疏掩码策略（解决“注意力崩塌”）：**
* 单纯依赖动态Top-K会导致数值不稳定。
* 
**策略：** 最终索引 = 动态Top-K（捕捉长距离依赖）+ **Sink Token**（锚点，保证数值稳定）+ **Local Window**（局部窗口，保证语法连贯）。




* **层间蒸馏优化：**
* **问题：** 剪枝会导致特征空间漂移，排序不准。
* 
**方法：** 冻结主干，仅微调预测模块。使用**Ranking Loss（排序损失）**而非MSE，强迫学生模型对正样本的打分高于负样本 。




* **混合精度量化：**
* 预测层使用**Int4非对称量化**，直接在低比特下计算点积，以计算换带宽 。





### 4. 硬件设计：面向稀疏感知的加速器 (Chapter 4)

**主要内容：** 解决算法落地到硬件时的“非连续访存”和“流水线停顿”问题。

* **异构存储与数据布局：**
* 
**HBM（存放索引）：** 采用通道连续流式布局，打包存储Int4索引，匹配HBM突发传输粒度，实现满速扫描 。


* 
**DDR（存放载荷）：** 采用**基于哈希的Bank交错映射**。将逻辑上连续的Token打散到不同Bank，避免随机访问时的行缓冲冲突（Row Conflict），提升有效带宽 。




* **细粒度流水线设计：**
* 将层计算解耦为：P-Stage（预测，HBM密集读取）、F-Stage（抓取，DDR随机读取）、C-Stage（计算）。


* 利用多头注意力的天然并行性，在不同头组（Head Group）间重叠这三个阶段，掩盖DDR的高延迟 。




* **关键硬件模块：**
* 
**HSE (HBM Stream Engine)：** 绕过缓存直接流式读取HBM 。


* 
**TFU (Top-K Filtering Unit)：** 摒弃全排序，采用**动态阈值过滤 + URAM缓冲**，实现O(1)复杂度的实时筛选 。


* 
**SGU (Sparse Gather Unit)：** 乱序稀疏收集单元。包含请求重排序缓冲区（ROB），合并相邻请求，并使用非阻塞MSHR机制掩盖访存延迟 。


* 
**异构计算阵列：** 分离Int4预测引擎（高吞吐扫描）和BF16注意力引擎（高精度计算）。





### 5. 总结 (Conclusion)

* 本文提出了一套完整的长文本推理加速方案。通过算法层的剪枝、蒸馏、混合掩码，实现了高精度的稀疏预测；通过硬件层的异构存储映射和乱序访存控制器，解决了稀疏访问的带宽效率问题。最终实现了在大规模长文本场景下的高性能推理。



 1. 异构存储子系统 (Heterogeneous Memory Subsystem)
  论文的核心在于利用 HBM 存索引（流式）和 DDR 存载荷（随机），你需要建立能够体现两者行为差异的模型。

   * 新增模块：`HBMController` (面向 Int4 Small KV)
       * 功能： 模拟高带宽、突发传输（Burst Access）。
       * 关键特性： 简单的线性延迟模型，假设数据是连续存储的。
       * 验证点： 验证 P-Stage（预测阶段）能否在极短时间内喂饱 Int4 计算阵列。
   * 新增模块：`DDRController` (面向 BF16 Full KV)
       * 功能： 模拟高延迟、Bank 冲突和随机访问。
       * 关键逻辑：
           * Bank Mapping： 实现论文提到的“基于哈希的 Bank 交错映射”。
           * Row Buffer 模型： 区分 Row Hit（行缓冲命中，低延迟）和 Row Conflict（行冲突，高延迟 + 预充电时间）。
       * 验证点： 验证论文中的映射策略是否真的减少了 Bank 冲突，提升了有效带宽。

  2. 稀疏收集单元 (Sparse Gather Unit - SGU)
  这是连接“预测”和“计算”的桥梁，也是论文硬件设计章节的重点（ROB, MSHR）。

   * 新增模块：`SGU` (Sparse Gather Unit)
       * 输入： Top-K 索引列表（来自 TFU）。
       * 输出： BF16 数据块（送往 BF16 计算阵列）。
       * 内部逻辑：
           * Reorder Buffer (ROB)： 模拟请求重排序。
           * Request Merging： 检查索引是否相邻，如果相邻则合并成一个长的 DDR Burst 请求。
           * Non-blocking Cache/Buffer： 模拟 MSHR (Miss Status Holding Register)，允许在等待 DDR 数据返回时处理后续请求。
       * 验证点： 证明乱序收集和请求合并机制能有效掩盖 DDR 的长延迟。

  3. Top-K 过滤单元 (Top-K Filtering Unit - TFU)
  目前的 Int4 GEMV 只输出了分数，论文提出需要硬件化的筛选模块。

   * 新增模块：`TopKUnit`
       * 位置： 接在 Int4 ReduceMacArrayUnit 的输出之后。
       * 逻辑：
           * 实现论文提到的 “动态阈值过滤” 算法（而非全排序）。
           * 模拟 URAM 缓冲区的读写延迟。
       * 验证点： 验证该模块是否能在 O(1) 或极低延迟内产出索引，而不阻塞后续流水线。

  4. 顶层流水线控制器 (Co-design Orchestrator)
  目前的 GemvScheduler 只是调度单一的矩阵乘法。你需要一个更高层级的调度器来模拟完整的“预测-检索”流水线。

   * 新增模块：`SystemController` 或 `DualPhaseScheduler`
       * 职责： 协调 P-Stage, F-Stage, C-Stage 的并行。
       * 逻辑：
           1. P-Stage: 启动 Int4 阵列，读取 HBM。
           2. Filter: 拿 Int4 结果进 TFU，得索引。
           3. F-Stage: 将索引送入 SGU，从 DDR 抓取数据。
           4. C-Stage: 数据回来后，启动 BF16 阵列进行计算。
       * 关键特性： 多头并行（Head Group Parallelism）。即 Head 0 在做 C-Stage 时，Head 1 可能在做 F-Stage。
       * 验证点： 统计各阶段的 Bubble（气泡/停顿），验证三级流水线是否完美重叠。

  建议的代码重构/添加路径

  为了实现上述功能，建议在 src/PE 外层新建一个 src/System 或 src/Memory 目录：

   1. `src/Memory/DRAM.h`: 定义通用的内存接口。
   2. `src/Memory/DDR_Banked.cpp`: 实现带有 Bank 状态机（Idle, Active, Precharge）的内存模型。
   3. `src/Hardware/SGU.cpp`: 实现请求合并逻辑。
   4. `src/System/PipelineOrchestrator.cpp`: 实例化两个 GemvScheduler（一个 Int4，一个 BF16），连接 Memory 和 SGU，模拟时钟周期的流转。

  总结
  目前的模拟器只做了大脑（Compute）。
  为了验证论文，你需要给它加上手（SGU/TFU）和复杂的书桌（HBM/DDR 存储层次），并测量从大脑发出指令到书拿到手里的时间效率。



  ✦ 这是一个非常关键的问题。对于硕士论文，尤其是涉及“软硬件协同”的架构类论文，你不需要实现一个功能完全正确（Bit-true）、能跑通真实 LLM 输出“Hello World”的模拟器。

  你的模拟器核心目标是 Performance Modeling（性能建模），而不是 Functional Emulation（功能仿真）。

  以下是针对第五章实验的建议，以及为了完成这些实验，你需要将之前的模块“简化”到什么程度。

  核心原则：Cycle-Accurate（周期精确）但 Data-Agnostic（数据无关）

  你不需要真的去读写 DDR 里的数据，也不需要真的算出矩阵乘法的结果（目前的 int4/bf16 算子只要能消耗正确的周期数即可）。

  你需要模拟的是“时间”和“流量”，而不是“数值”。

  ---

  第五章可以包含的 4 类实验（及其对应的实现要求）

  实验 1：端到端性能对比（End-to-End Latency & Speedup）
  目的： 展示你的 FPGA 加速器在处理长文本（如 32K, 64K, 128K 长度）时，推理速度比 GPU 快多少。

   * 对比对象：
       * Baseline (GPU): NVIDIA A100 或 RTX 4090。你可以写一个简单的 Python/PyTorch 脚本，运行 FlashAttention 或标准的 Attention，测量不同长度下的推理延迟（Latency）。
       * Ours (FPGA Simulator): 你的模拟器跑出来的 Cycle 数 $\times$ FPGA 频率（如 200MHz）。
   * 模拟器需求：
       * 需要实现： SystemController（顶层流水线调度）。你需要它来统计 P-Stage、F-Stage、C-Stage 是否在正确地重叠（Overlap）。
       * 简化实现： 不需要真的做矩阵乘法。
           * GemvScheduler 可以简化为：接收指令 -> 等待 (K * N / Num_PEs) 个周期 -> 返回完成信号。
           * 重点： 必须准确模拟 DDR 延迟。当你的 SGU 发出稀疏读请求时，模拟器必须根据请求的离散程度，返回一个准确的等待周期数。

  实验 2：存储子系统有效性验证（Memory Efficiency / Effective Bandwidth）
  目的： 证明你的“基于哈希的 Bank 交错映射”和“SGU 乱序合并”真的解决了稀疏访问的带宽瓶颈。

   * 实验设计：
       * 横轴：稀疏度（Top-K Ratio，例如 1%, 5%, 10%）。
       * 纵轴：DDR 有效带宽利用率（Effective Bandwidth Utilization）。
       * 对比：
           1. Naive 实现： 模拟普通的线性映射，遇到随机索引就产生 Row Conflict。
           2. Ours： 你的哈希映射 + 请求合并。
   * 模拟器需求：
       * 需要实现： 一个比较精细的 DRAM_Model。
       * 怎么做： 这个模块不需要真的存数据。它只需要维护每个 Bank 当前的 Row_ID。
           * 当请求来了，计算它属于哪个 Bank、哪个 Row。
           * 如果 Row_ID 变了，延迟 = tPRE + tACT + tCAS。
           * 如果 Row_ID 没变，延迟 = tCAS。
       * SGU 模块： 需要实现“将相邻索引合并”的逻辑，因为这直接减少了 DRAM 的请求次数。

  实验 3：流水线效率与消融实验（Pipeline Efficiency & Ablation Study）
  目的： 展示三级流水线（P-F-C）是如何掩盖 DDR 延迟的。

   * 实验设计：
       * 可视化图表： 类似于甘特图（Gantt Chart），展示随着时间推移，HBM（预测）、DDR（抓取）、BF16 Array（计算）的忙碌状态。
       * 消融：
           * Case A: 串行执行（Predict -> Gather -> Compute）。
           * Case B: 你的完整流水线。
       * 指标： 流水线气泡（Bubble）占比。
   * 模拟器需求：
       * 依靠 SystemController 记录日志。只要你的调度器逻辑是对的，这个图也就是生成的 Log 可视化而已。

  实验 4：算法精度验证（Algorithmic Accuracy）
  注意： 这个实验不要用 C++ Simulator 做！

   * 目的： 证明剪枝和 Int4 量化没有把模型“脑子搞坏”。
   * 实验内容： PPL (Perplexity) vs Sparsity。
   * 做法： 使用 Python/PyTorch。加载一个开源模型（如 LLaMA-2-7B），用代码模拟你的剪枝逻辑，跑数据集（如 Wikitext-2）。
   * 结论： “在保持 PPL 损失小于 0.1 的情况下，我们实现了 95% 的稀疏度。” —— 然后把这个 95% 稀疏度参数喂给 C++ Simulator 跑性能。

  ---

  总结：你需要做什么？

  为了搞定 Chapter 5，你需要对现在的 Simulator 做如下调整（按优先级排序）：

   1. DRAM Latency Model (优先级：高)：
       * 写一个类，输入是一堆随机的地址（模拟 Top-K 索引），输出是读取这些数据所需的总周期数。
       * 逻辑里要包含 Bank Conflict 的惩罚计算。

   2. Pipeline Scheduler (优先级：高)：
       * 写一个主循环，每一轮通过简单的状态机（State Machine）决定 Int4 单元、DDR 接口、BF16 单元谁在干活。
       * 统计总周期数。

   3. SGU Request Merging (优先级：中)：
       * 输入一堆索引，输出合并后的请求包。这会影响上面 DRAM Model 的输入数量。

   4. Top-K Unit (优先级：低)：
       * 可以直接硬编码一个固定的延迟（比如 20 个周期），假设硬件能在这个时间内完成筛选。不需要真的去写排序算法。

  一句话建议：
  把 C++ Simulator 当作一个复杂的“计算器”。输入是配置参数（PE 数量、频率、DDR
  参数）和工作负载（序列长度、稀疏度），输出是时间（Latency）。不要在里面跑真实的数据流，那样工作量太大且对论文产出性价比低。
