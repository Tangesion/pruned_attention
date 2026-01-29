# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

### Build
```bash
mkdir -p build
cd build
cmake ..
make
```

### Run Experiments
Executables are located in `build/` after compiling.
```bash
./build/experiment_memory_efficiency
./build/experiment_pipeline_overlap
```

### Run Tests
Unit tests are compiled as separate executables in `build/`.
```bash
./build/test_bf16_basic_ops
./build/test_add_sim
./build/test_mac_unit
./build/test_gemv_scheduler
./build/test_int4_pipeline
```

## Architecture

This project is a C++ performance simulator for an FPGA-based accelerator designed for Long-Context LLM inference ("Predict-Retrieval" paradigm).

- **System (`src/System`)**:
  - `PipelineSimulator`: Top-level orchestrator modeling the 3-stage pipeline (Predict, Fetch, Compute). Tracks cycles, bubbles, and overlap efficiency.
- **Memory (`src/Memory`)**:
  - `DDRController`: Simulates DDR DRAM behavior (latency, bank conflicts, row buffer hits/misses). Uses hash-based bank mapping for sparse retrieval.
  - `HBMController`: Simulates High Bandwidth Memory for streaming access (used in Prediction stage).
- **Processing Elements (`src/PE`)**:
  - Models computational units (MacUnit, AddTreeUnit, etc.).
  - "Cycle-Accurate" throughput but often "Data-Agnostic" (calculates latency based on workload size).
  - Contains schedulers like `GemvScheduler`.
- **BF16/Int4 (`src/bf16`, `src/int4`)**:
  - Basic arithmetic and simulation logic for reduced precision formats.

## Development Conventions

- **Language**: C++17
- **Build System**: CMake (min 3.10)
- **Namespaces**: `System`, `PE`, `Memory`, `bf16`
- **Testing**: Custom `assert`-based test files in `test/` (no external framework like GTest).
- **Structure**: Separation of interface (`include/`) and implementation (`src/`). Header-only preferred for simple utilities.


# 相关组件论文里的实现细节


### 1.1 关键模块微架构设计与混合精度并行策略

为消除DDR非确定性延迟、HBM高吞吐要求及FPGA资源受限带来的物理瓶颈，本节详细阐述了三大关键模块的微架构实现。

#### 1.1.1 乱序稀疏收集单元 (SGU)

**目标：** 将逻辑层面的Top-K索引稀疏性转化为底层物理存储的突发访问，解决DDR随机访问效率低下的问题。

* **地址变换流水线 (Address Translation Pipeline)：**
* **并行映射：** 采用基于哈希的Bank交错映射策略，集成异或哈希逻辑阵列。
* **极低延迟：** 利用预计算查表法消除模运算开销，单周期完成逻辑ID到物理地址（Rank/Bank/Row/Col）解码。
* **时序保护：** 物理地址携带原始时间戳标签，确保乱序执行后能还原逻辑时序。


* **请求重排序缓冲区 (ROB)：**
* **局部性感知：** 基于全相联CAM结构，维护数十个未决请求。
* **请求合并：** 自动检测并融合同一行且列相邻的读请求，分摊行激活延迟。
* **动态调度：** 优先发射目标Bank处于“已激活”状态的请求，最小化时序惩罚。


* **非阻塞MSHR机制 (Non-blocking MSHR)：**
* **乱序容忍：** 借鉴CPU设计，通过事务ID管理DDR交互，支持64个并发请求。
* **发射即忘：** 读请求发出后不挂起流水线，数据返回后精准回填至计算缓冲区，实现逼近顺序扫描的吞吐性能。



#### 1.1.2 基于动态阈值与URAM的混合筛选器

**目标：** 解决长上下文（128k+）下大值（+）导致的寄存器资源耗尽与时序收敛难题，实现筛选逻辑与稀疏度的解耦。

* **统计-过滤机制 (Statistics & Filtering)：**
* **直方图统计：** 前端单次扫描将Int4分数离散化为粗粒度区间，通过前缀和逻辑寻找动态截止阈值$\tau$。
* **时间平滑：** 复用上一轮推理阈值作为初值，利用Attention的时间局部性，在入口处丢弃>90%非关键Token。


* **URAM候选缓冲 (URAM Buffering)：**
* **资源解耦：** 放弃寄存器堆，利用高密度UltraRAM构建环形FIFO缓冲区，存储未完全排序的候选元组。
* **溢出驱逐：** 缓冲区满时触发“最小值替换”机制并同步上调阈值$\tau$，确保保留全局最优子集。


* **精确校正：**
* 在扫描结束阶段，通过快速选择逻辑（Quick Select）对URAM数据进行最终修剪，确保输出严格符合值。
* **收益：** 将硬件复杂度从线性增长 + 存储资源。



#### 1.1.3 异构计算阵列的空间分区策略

**目标：** 在单芯片上同时支持高吞吐Int4预测与高精度BF16计算，通过空间物理隔离避免流水线阻塞。

* **统一微架构 (ReduceMacUnit)：**
* **交织累加：** 为应对DSP高频下的固定延迟（如4周期），MacUnit采用深度为的交织累加队列，切断数据依赖，实现1 MAC/Cycle峰值吞吐。
* **级联结构：** 由负责计算的MacUnit和负责规约的AddTreeUnit组成。


* **Int4 预测引擎 (P-Stage)：**
* **精度保障：** 输入Int4，内部采用Int32累加器防止长序列溢出。
* **高密度计算：** 利用FPGA SIMD特性，单DSP并行执行多路Int4乘加，适配HBM高带宽。


* **BF16 注意力引擎 (C-Stage)：**
* **标准兼容：** 完整模拟BFloat16硬件行为（对阶/舍入）。
* **动态映射：** 当矩阵宽度小于阵列大小时，自动开启Batch维度（多Head/多Request）并行，最大化PE利用率。


* **协同机制：** Int4引擎与BF16引擎通过SRAM双缓连接，形成“生产-消费”闭环，实现全流水线无缝衔接。
