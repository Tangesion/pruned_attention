# `simulator/experiments` 实验说明与 FPGA 仿真模型

本文档说明 `simulator/experiments` 目录下三个实验是怎么做的，并解释这个模拟器是如何抽象 FPGA 实现的。

对应源码：

- `simulator/experiments/experiment_memory_efficiency.cpp`
- `simulator/experiments/experiment_pipeline_overlap.cpp`
- `simulator/experiments/experiment_end_to_end_decode.cpp`
- `simulator/src/System/PipelineSimulator.cpp`
- `simulator/src/Memory/DDRController.cpp`
- `simulator/src/Memory/HBMController.cpp`

## 1. 先说结论：这个模拟器在模拟什么

这个模拟器不是 RTL 级、也不是 HLS 级的逐拍硬件仿真。它更接近一个**面向体系结构设计空间探索的 cycle-level / latency-level simulator**，核心目的不是验证功能正确性，而是回答下面几类问题：

- 稀疏访问下，DDR 的 `bank mapping` 是否真的能减少冲突。
- 稀疏收集单元是否能通过请求合并和重排提高有效带宽。
- `P/F/C` 三段流水能否把预测、抓取、计算重叠起来。
- 在不同上下文长度和稀疏率下，端到端 decode attention 的瓶颈落在哪一段。

它抽象了以下 FPGA 硬件特征：

- `HBM` 的高带宽顺序扫描能力。
- `DDR` 的高延迟、`channel/bank/row buffer` 行为。
- Int4 预测阵列和 BF16 计算阵列的吞吐上限。
- `TFU/URAM` 这类片上筛选与缓存模块的吞吐/带宽限制。
- `SGU` 风格的请求合并、乱序重排、有限 in-flight 请求数。
- 多个 head group 在 `Predict -> Fetch -> Compute` 三段流水之间的排队和回压。

它没有细化到以下层面：

- 没有建模真实 RTL 信号、握手时序、AXI 通道协议。
- 没有建模布线、时钟收敛、资源放置、仲裁器微结构。
- HBM 控制器在 `PipelineSimulator` 里目前主要以解析模型参与，不是完整事件驱动的逐请求仿真。
- `P` 阶段和 `C` 阶段很多时候用的是解析公式，不是把每一次访存都送进控制器。

所以更准确地说，这个工程是在做**“FPGA 架构行为级性能模拟”**，而不是“真实 FPGA bitstream 行为复现”。

## 2. 整体架构：代码如何对应论文里的硬件

从代码结构上看，模拟器把论文里的硬件系统拆成三层：

### 2.1 存储子系统

- `HBMController`
  负责模拟 Int4 `small KV` 的连续流式读取。
- `DDRController`
  负责模拟 BF16 `full KV` 的随机访问、bank 冲突和 burst 传输。

### 2.2 计算子系统

- `GemvScheduler`
  负责驱动 `ReduceMacArrayUnit` 等 PE 阵列，模拟 Int4 / BF16 GEMV 的吞吐和流水深度。

### 2.3 顶层流水控制

- `PipelineSimulator`
  把一次 head group 的处理拆成三段：
  - `P-Stage`: HBM 上的 Int4 扫描与 Top-K 预测
  - `F-Stage`: DDR 上的 sparse gather
  - `C-Stage`: BF16 attention 计算

整个 decode attention 的性能评估，基本就是围绕这三层展开的。

## 3. 模拟器如何模拟 FPGA 实现

这一节是重点。

### 3.1 HBM 是怎么模拟的

`HBMController` 的模型非常简单，本质上是一个**固定带宽 + 固定流水延迟**模型。

它的配置项只有三个：

- `bandwidth_GBps`
- `frequency_GHz`
- `latency_cycles`

控制器先把带宽换算成每周期可传输字节数：

```text
bytes_per_cycle = bandwidth_GBps / frequency_GHz
```

随后每个请求的完成时间按下面思路计算：

```text
burst_cycles = ceil(size_bytes / bytes_per_cycle)
start_cycle  = max(arrival_cycle, bus_next_free_cycle)
ready_cycle  = start_cycle + latency_cycles + burst_cycles
```

这相当于把 HBM 当成一个顺序流口：

- 数据连续。
- 无 bank 冲突细节。
- 无复杂仲裁。
- 主要体现“带宽足够高、适合做流式扫描”。

这和论文里的设定是一致的：`small KV` 放在 HBM，重点不是随机访问，而是高吞吐全量扫描。

### 3.2 DDR 是怎么模拟的

`DDRController` 才是整个模拟器最关键的硬件近似。

它的核心配置包括：

- `tCL`: 列访问延迟
- `tRCD`: 行激活到列访问延迟
- `tRP`: 预充电延迟
- `tBURST`: burst 传输时长
- `num_channels`
- `num_banks_per_channel`
- `row_size_bytes`
- `mapping_strategy`

#### 3.2.1 地址映射

控制器先把字节地址按 64B cache line 切块，然后解析成：

- `channel`
- `bank`
- `row`
- `col`

支持两种映射方式：

- `LINEAR`
  连续地址按常规方式落到连续 `channel/bank`。
- `HASH_XOR`
  用 `row` 的部分比特对 `channel` 和 `bank` 做异或扰动，把逻辑上规律的访问打散。

`HASH_XOR` 的目的很明确：

- 降低多个流同时打到同一 bank 的概率。
- 减少行冲突。
- 让多路 head stream 更均匀地铺到不同 bank/channel。

#### 3.2.2 Row Buffer 模型

每个 bank 维护两个状态：

- 当前打开的 `open_row_id`
- 该 bank 下一次可接收命令的 `bank_next_free_cycle`

每个 burst 到来时会判断三种情况：

- `row miss`
  bank 当前没有打开行，需要 `tRCD + tCL`
- `row hit`
  访问命中当前打开行，只需要 `tCL`
- `row conflict`
  访问落到另一行，需要 `tRP + tRCD + tCL`

然后再和该 channel 的数据通路占用做仲裁：

```text
data_start_cycle = max(bank_ready_cycle, channel_next_data_free_cycle[channel])
data_end_cycle   = data_start_cycle + tBURST
```

这一步非常像真实 DRAM 的关键瓶颈：

- bank 决定命令是否能发。
- channel 决定数据总线何时空闲。

最终一个请求的 `ready_cycle` 取其所有 burst 的最晚完成时间。

#### 3.2.3 为什么这能近似 FPGA 上的 DDR 访存

因为对这类稀疏 attention 加速器来说，真正重要的不是精确到 JEDEC 全部状态机，而是下面三件事：

- 多个随机请求是否集中打到少数 bank。
- 请求是否命中同一行，从而复用 row buffer。
- burst 是否在 channel 上排队，形成总线瓶颈。

`DDRController` 恰好把这三件事都显式建模了，所以足以支撑架构层比较。

### 3.3 SGU 是怎么模拟的

这里没有单独的 `SGU` 类，但 SGU 的关键行为被拆进了实验代码和 `PipelineSimulator` 中。

主要有三类行为。

#### 3.3.1 请求合并

`build_request_stream()` 会检查地址是否连续：

- 如果相邻 token 的地址正好相差一个 token 大小，并且合并后不超过 `max_merge_bytes`
- 就把多个 token 合并成一个更大的 DDR burst

这对应论文里的 SGU request merge / burst coalescing。

#### 3.3.2 多流交织

`interleave_request_streams()` 会把多个 head stream 轮询交织成一个总请求流。

这对应：

- 多头并发访问同一片 DDR。
- SGU 前端不断从多个 head stream 发请求。

#### 3.3.3 有限 in-flight 请求

在 `run_simulation()` 和 `run_ddr_simulation()` 中，请求不是无限发送的，而是受到：

- `max_in_flight_bursts`
- `kMaxInflightRequests`

这类阈值限制。

代码逻辑是：

- 先尽量发请求。
- 如果 in-flight 已满，就停发，等待部分请求返回。
- 每个周期驱动一次 DDR，并回收已完成请求。

这就是一个很典型的**MSHR / ROB 容量受限**近似。

虽然没有显式写成 “ROB entry” 结构体，但行为上已经在模拟：

- 非阻塞访存
- 请求悬挂
- 控制器回压

### 3.4 TFU / URAM 是怎么模拟的

`TFU` 和 `URAM` 也没有独立模块，而是以**吞吐上界**的方式并入 `P-Stage` 延迟模型。

`P-Stage` 延迟由四部分取最大值得到：

- HBM 扫描时间
- Int4 GEMV 计算时间
- TFU 过滤时间
- URAM 写入候选索引时间

公式思想如下：

```text
P_latency =
    max(
        hbm_cycles,
        int4_compute_cycles,
        tfu_cycles,
        uram_write_cycles
    ) + pipeline_overhead
```

其中：

- `tfu_cycles ~= seq_len / tfu_throughput_per_cycle`
- `uram_write_cycles ~= candidate_bytes / uram_bytes_per_cycle`

这意味着模拟器把 TFU/URAM 看成一个与 GEMV 并行的流式后处理单元，最终由最慢者决定阶段时长。

### 3.5 Int4 / BF16 计算阵列是怎么模拟的

有两种粒度。

#### 3.5.1 解析式近似

默认最常用的是解析式：

- `P-Stage` 计算量近似为 `context_length * head_dim * heads_per_group / predict_pes`
- `C-Stage` 计算量近似为 `2 * kv_tokens * head_dim * heads_per_group / compute_pes`

然后再加固定流水额外开销。

优点是快，适合大 sweep。

#### 3.5.2 更细一点的阵列级模拟

`PipelineSimulator` 还支持 `use_realistic_stage_models = true`。

这时会调用 `GemvScheduler`：

- `calculate_p_latency_realistic()`
  用随机 Int4 `Q/K` 跑一遍 Int4 GEMV 调度器，得到更真实的阵列周期数。
- `calculate_c_latency_realistic()`
  用随机 BF16 `Q/K/V` 跑两次 GEMV，近似模拟 `QK^T` 与 `Score * V`。

这里的 “realistic” 仍然不是完整系统级硬件仿真，而是：

- 计算阵列逐拍
- 访存仍然是高层抽象

### 3.6 `P/F/C` 流水线是怎么模拟的

`PipelineSimulator` 是整个顶层控制器。

它会为每个 head group 创建一个 `HeadGroupTask`，并维护三类队列：

- `pending_start_queue`
- `pending_fetch_queue`
- `pending_compute_queue`

以及三段共享资源：

- `busy_p_stage_task`
- `busy_f_stage_task`
- `busy_c_stage_task`

每个周期 `step()` 都做同样的事情：

1. 检查 `C` 段是否完成，完成就把任务标记为 `FINISHED`
2. 如果 `C` 段空闲且 compute queue 非空，就启动新的 `C` 任务
3. 检查 `F` 段是否完成，完成后把任务推进到 compute queue
4. 如果 `F` 段空闲且 fetch queue 非空，就启动新的 `F` 任务
5. 检查 `P` 段是否完成，完成后把任务推进到 fetch queue
6. 如果 `P` 段空闲且 start queue 非空，就启动新的 `P` 任务

额外还模拟了两件事：

- `max_queue_size`
  队列深度有限，表示片上 FIFO / buffer 容量有限，可能产生回压。
- `total_bubbles`
  如果某个周期 `P/F/C` 三段都没有活动，就记为 bubble。

这就是论文里“多 head group 级联流水”的软件实现。

### 3.7 一个非常重要的现实含义

这个模拟器并不是在说：

> FPGA 真实每个周期都恰好会发生这些操作

它在说的是：

> 如果这个 FPGA 架构由这些资源约束主导，那么整体吞吐和时延大概率会落在这个量级，瓶颈切换关系也会类似。

所以它最适合做：

- 架构比较
- 参数 sweep
- 趋势分析
- 论文图表生成

而不是：

- RTL 签核
- 控制逻辑正确性验证
- 和板级实测逐周期对齐

## 4. 三个实验分别是怎么做的

## 4.1 `experiment_memory_efficiency.cpp`

### 4.1.1 实验目的

这个实验只聚焦 `F-Stage`，也就是：

- 稀疏检索访问 DDR 时
- 不同数据布局和 SGU 策略
- 对有效带宽、行命中率、冲突率到底有什么影响

它不关心 `P-Stage` 和 `C-Stage`，因此是一个**纯内存子系统实验**。

### 4.1.2 固定参数

实验固定：

- 保留比例 `FIXED_SELECTION_RATIO = 0.10`
  也就是只取 10% token，等价于 90% 稀疏。
- `HIDDEN_SIZE = 128`
- `BYTES_PER_ELEMENT = 2`
- 每个 token 的 `K/V` 载荷大小是 `128 * 2 = 256B`
- `NUM_HEAD_STREAMS = 32`
- DDR 为 `8 channel, 16 bank/channel`

扫描的上下文长度是：

- `16K`
- `32K`
- `64K`
- `128K`
- `256K`
- `512K`

### 4.1.3 工作负载怎么构造

#### 第一步：生成稀疏 token 索引

`generate_indices()` 不会完全均匀随机采样，而是生成**带簇状局部性**的索引。

做法是：

- 每次随机选一个 cluster 起点
- 再连续加入若干 token
- 直到凑够 `10%` 的 token 数量

这比纯随机采样更贴近 attention 中“热点片段成簇出现”的情况。

#### 第二步：为每个 head 生成地址流

`build_per_head_addresses()` 会把同一组 token 索引映射到 32 个 head stream 上。

关键点有两个：

- 每个 head 都访问自己独立的 KV 区域。
- 当关闭 merge 时，会把 token 顺序打乱，制造更强的非顺序访问。

这一步等价于：

- 同一批选中的 token，需要在多个 head 上分别 gather 对应的 full KV。

#### 第三步：SGU 请求合并

`build_request_stream()` 根据 `max_merge_bytes` 决定是否把连续 token 合成一个大请求。

四种方法分别对应：

- `Baseline`
  `LINEAR` 映射，不合并请求
- `Hash`
  `HASH_XOR` 映射，不合并请求
- `SGU`
  `LINEAR` 映射，允许最多 `4KB` 合并
- `Hash+SGU`
  `HASH_XOR` 映射，允许最多 `8KB` 合并，并开启 hash-aware 调度

#### 第四步：多头流交织

`interleave_request_streams()` 用 round-robin 把 32 个 head 的请求交织起来，模拟多头并发访问 DDR。

#### 第五步：hash-aware 请求重排

只有 `Hash+SGU` 会走 `rebalance_requests_by_channel_bank()`：

- 先根据地址估计它会映射到哪个 `channel/bank`
- 按 `channel-bank` 或仅 `bank` 分桶
- 桶内按地址排序
- 再分块轮转吐出请求

这一步模拟的是：

- SGU 或调度器知道 hash 映射规则
- 主动把请求整理成更利于 row locality、也更均匀分散的顺序

### 4.1.4 时延怎么统计

`run_simulation()` 是一个简单的逐周期驱动器：

1. 只要 in-flight burst 没超上限，就继续发请求
2. 调一次 `ddr.step(current_cycle)`
3. 回收所有已完成请求
4. 进入下一个周期

直到全部请求完成为止。

输出指标包括：

- `Cycles`
- `BW (GB/s)`
- `Requests`
- `row hit rate`
- `row conflict rate`
- `channel skew`
- `bank skew`

### 4.1.5 这个实验回答什么问题

这个实验的本质是在验证两件事：

- 单纯换成 hash bank mapping，能否降低 bank 热点与冲突
- 在 hash mapping 基础上，再做 SGU merge 与调度重排，是否能进一步提升有效带宽

如果论文要证明“稀疏 gather 的瓶颈主要在 DDR 访问组织方式”，这个实验就是最直接的证据。

## 4.2 `experiment_pipeline_overlap.cpp`

### 4.2.1 实验目的

这个实验聚焦三级流水：

- `P-Stage`: Int4 预测
- `F-Stage`: DDR 抓取
- `C-Stage`: BF16 计算

它想回答的是：

- 这三段能重叠多少
- 和串行执行相比能加速多少
- 在不同上下文长度和稀疏率下，瓶颈如何变化

### 4.2.2 固定硬件配置

这个实验把硬件固定在一组相对激进的参数上：

- DDR 通道数固定 `16`
- `head_groups = 8`
- `heads_per_group = 4`
- `frequency = 200MHz`
- `P_stage_pe_nums = 2048`
- `C_stage_pe_nums = 256`
- `HBM bandwidth = 400 GB/s`
- `URAM bandwidth = 1000 GB/s`
- `use_hash_aware_fetch = true`
- `use_dual_fetch = true`
- `fetch_num_lanes = 2`
- `max_sgu_merge_bytes = 16KB`

其中 `fetch_head_stride_padding_bytes = 64 * channels * 16` 很关键。

它的作用是人为在 head 之间插入一段 padding，让相邻 head 的基地址旋转到不同 bank/channel，减少多 head 同时访问时的规律性冲突。

### 4.2.3 工作负载怎么扫

实验扫描：

- 上下文长度：`16K` 到 `512K`
- 稀疏率：`1%`、`5%`、`10%`、`20%`

每个点跑一次完整 `PipelineSimulator`。

### 4.2.4 `PipelineSimulator` 里发生了什么

每个 head group 被看成一项任务。

对每项任务：

- `P` 阶段时长由 `calculate_p_latency()` 估算
- `F` 阶段时长由 `calculate_f_latency()` 通过 DDR 仿真得到
- `C` 阶段时长由 `calculate_c_latency()` 估算

然后顶层按单资源三级流水调度这些任务。

#### `P` 阶段

取下列四者最大值：

- HBM 扫描时间
- Int4 GEMV 时间
- TFU 过滤时间
- URAM 写候选索引时间

再加一个固定流水开销。

#### `F` 阶段

这是这个实验最像“硬件”的部分。

每个 group 会：

- 生成共享的 sparse token 索引
- 按 head 构造地址流
- 按 lane 划分 fetch 负载
- 对每个 lane 独立做 DDR 时延仿真
- 最终取最慢 lane 作为该 group 的 `F` 阶段时长

如果启用了双 fetch lane，最后还会额外乘一个约 `1.05` 的协调开销。

#### `C` 阶段

按稀疏后的 `selected_kv_len` 估算 BF16 attention 的算力消耗。

### 4.2.5 指标如何计算

实验输出：

- `total_cycles`
  完整流水执行总周期
- `serial_cycles`
  把每个 group 的 `P+F+C` 直接相加得到的串行基线
- `speedup_vs_serial`
  `serial / pipeline`
- `avg_p_cycles`
- `avg_f_cycles`
- `avg_c_cycles`
- 代表性 workload 的阶段时间线 CSV

因此它非常适合拿来画：

- workload sweep 的加速比曲线
- 代表性 case 的 Gantt 图

### 4.2.6 这个实验回答什么问题

它主要回答：

- 稀疏 attention 不只是“单段变快”，而是可以通过 `P/F/C overlap` 进一步提升系统吞吐。
- 当 `F` 段太长时，流水线会被 DDR 卡住。
- 当稀疏度更高时，`C` 段和 `F` 段都会缩短，`P` 段可能开始接近瓶颈。

## 4.3 `experiment_end_to_end_decode.cpp`

### 4.3.1 实验目的

这是三个实验里最完整、最接近论文主结果的一个。

它把五种方法放在同一套硬件假设下比较：

- `Dense-FullKV`
- `Sparse-LinearNaive`
- `Sparse-HashNoMerge`
- `Sparse-HashMerge`
- `Sparse-HashMergePipeline`

也就是说，它同时比较：

- 稠密 vs 稀疏
- 线性映射 vs hash 映射
- 不合并 vs 合并
- 串行阶段执行 vs 完整流水执行

### 4.3.2 扫描参数

默认扫描：

- `context_lengths = {8192, 16384, 32768, 65536, 131072}`
- `sparsities = {0.01, 0.05, 0.10}`

另外可以从命令行改：

- `--ddr-channels`
- `--head-groups`
- `--heads-per-group`
- `--queue-depth`

### 4.3.3 五种方法分别怎么建模

#### `Dense-FullKV`

这个基线表示：

- 不做预测
- 直接读全量 KV
- 然后做 BF16 attention

实现上：

- `compute_cycles` 直接按 full context 估算
- `fetch_cycles` 不会真的把所有 token 全部逐个模拟到最大 context，而是先对最多 `2048` 个 token 做一次样本仿真，再按上下文长度线性放大

这里是一个很重要的近似：

- 这样可以显著降低仿真时间
- 但默认假设 dense 顺序访问的 DDR 行为可以近似按长度线性扩展

#### `Sparse-LinearNaive`

表示：

- 稀疏检索
- DDR 线性映射
- 不做请求合并
- `P/F/C` 不重叠

它的总时间是：

```text
total = total_predict_cycles + total_fetch_cycles + total_compute_cycles
```

其中：

- `P` 与 `C` 按每 group 时长乘 head group 数
- `F` 用所有 group 交织后的共享 DDR 工作负载直接仿真

#### `Sparse-HashNoMerge`

和上面相同，但 DDR 改成 `HASH_XOR`。

它衡量的是“只改 bank mapping，不改 SGU”的收益。

#### `Sparse-HashMerge`

在 `HASH_XOR` 基础上再允许相邻 token 合并成大 burst。

它衡量的是“bank mapping + 请求合并”的组合收益。

#### `Sparse-HashMergePipeline`

这是最完整的方案：

- `HASH_XOR`
- merge
- 每个 group 独立估计 `F` 阶段时长
- 再用软件流水调度器做 `P/F/C` overlap

这里要注意一个建模细节：

- 这个版本里的 `F` 阶段时长是**按 group 单独估算**的
- 不会让两个 group 的 `F` 阶段在同一时刻共同争用 DDR
- 顶层用“单个 F-stage 资源串行服务不同 group”的方式近似整体行为

这是一种合理但偏乐观的抽象：

- 它更像“一个共享 SGU/DDR fetch engine 依次服务各 group”
- 而不是“多个 group 同时在 F 阶段竞争同一片 DDR”

### 4.3.4 指标怎么统计

最终输出指标包括：

- `Cycles`
- `Latency(us)`
- `Tok/s`
- `Speedup vs Dense`
- `DDR GB/s`
- `Bubble%`
- `P/F/C util`
- `avg_p_cycles`
- `avg_f_cycles`
- `avg_c_cycles`

其中：

- `Bubble%` 只有完整流水方案最有意义
- `P/F/C util` 反映三段在总周期里的占比

### 4.3.5 这个实验回答什么问题

它回答的是论文最核心的问题：

- 只做算法稀疏是否足够
- 只换 bank mapping 是否足够
- 只做 merge 是否足够
- 最终是否必须把“预测、访存组织、流水线重叠”一起做，才能得到明显系统级收益

## 5. 这三个实验之间的关系

可以把三个实验理解成逐层递进：

### 第 1 层：`experiment_memory_efficiency`

只看 `F-Stage`。

目的是证明：

- DDR mapping
- SGU merge
- request reordering

确实能改善访存效率。

### 第 2 层：`experiment_pipeline_overlap`

把 `P/F/C` 三段串起来。

目的是证明：

- 即使每段单独已经优化，系统级性能还取决于流水线能否重叠。

### 第 3 层：`experiment_end_to_end_decode`

把不同方法放在一张表里统一比较。

目的是证明：

- 完整软硬件协同方案相对 dense 基线到底能快多少。

所以从论文叙事上，最顺的讲法通常是：

1. 先用 memory efficiency 证明稀疏 gather 端的硬件优化必要性。
2. 再用 pipeline overlap 证明系统级流水重叠的收益。
3. 最后用 end-to-end decode 给出统一总结果。

## 6. 如何运行

在仓库根目录下：

```bash
cmake --build simulator/build --target \
  experiment_memory_efficiency \
  experiment_pipeline_overlap \
  experiment_end_to_end_decode -j
```

运行三个实验：

```bash
./simulator/build/experiment_memory_efficiency
./simulator/build/experiment_pipeline_overlap
./simulator/build/experiment_end_to_end_decode
```

`end_to_end_decode` 支持额外参数，例如：

```bash
./simulator/build/experiment_end_to_end_decode \
  --contexts 8192,32768,131072 \
  --sparsities 0.01,0.05,0.10 \
  --ddr-channels 8 \
  --head-groups 8 \
  --heads-per-group 4 \
  --queue-depth 8 \
  --csv simulator/results/end_to_end_decode.csv
```

结果文件主要会写到：

- `simulator/results/`

相关绘图脚本：

- `simulator/plot_pipeline_overlap.py`
- `simulator/plot_end_to_end_decode.py`

## 7. 如何在论文里解释“模拟器是如何模拟 FPGA 的”

如果要写进论文或答辩，建议直接用下面这套表述。

### 7.1 一句话版本

该模拟器采用面向体系结构的周期级性能模型，对 HBM 顺序扫描、DDR 多 bank 随机访问、SGU 请求合并与重排，以及 `P/F/C` 多阶段流水线进行联合建模，用于评估长上下文稀疏 attention 加速器在 FPGA 上的性能趋势与瓶颈分布。

### 7.2 展开版

可以分四点讲：

1. `P-Stage` 用 HBM 带宽、Int4 阵列吞吐、TFU 吞吐和 URAM 写带宽联合决定预测时延。
2. `F-Stage` 用 `channel/bank/row-buffer` DDR 模型显式估计稀疏 gather 的访问代价，并支持 hash bank mapping、burst merge 和有限 in-flight 请求。
3. `C-Stage` 用 BF16 阵列吞吐估计精确 attention 的计算时延。
4. 顶层用 head-group 级别的三级流水调度器模拟阶段重叠、FIFO 深度限制和 bubble 产生。

### 7.3 需要主动说明的边界

为了让评审或老师不误解，最好主动说明：

- 这不是 RTL 级仿真。
- 这是性能建模，不是功能验证。
- 结果更适合说明趋势、瓶颈和相对收益，而不是替代板级实测绝对值。

## 8. 当前模型的优点与局限

### 优点

- 分层清晰，能把算法、存储、流水线三种因素拆开分析。
- DDR 模型抓住了稀疏访存最关键的 bank/row/burst 行为。
- 能快速做大规模参数 sweep，适合论文图表。

### 局限

- HBM 模型相对粗。
- 顶层流水里 `F-Stage` 的 group 间竞争是近似建模，不是全局统一事件调度。
- `P/C` 阶段大量使用解析式，未建模片上缓存、数据搬移和精细控制开销。
- 未和真实 FPGA 板卡实测做自动校准的话，绝对数值只能视作估计值。

## 9. 建议你在汇报时怎么讲

如果你要基于这份代码向别人解释，最顺的顺序是：

1. 先说 `memory_efficiency` 证明 DDR 访问组织很关键。
2. 再说 `pipeline_overlap` 证明三级流水能进一步隐藏延迟。
3. 最后说 `end_to_end_decode` 把这些硬件优化和稀疏算法统一起来比较。

这样叙事上会比较完整，也最符合这套代码当前的结构。
