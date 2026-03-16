# End-to-End Decode Attention Experiment

这个实验面向论文中的**硬件侧端到端 decode attention 子系统评估**，统一比较以下 5 类方法：

- `Dense-FullKV`：不做预测，直接读取 full KV 并做 BF16 attention
- `Sparse-LinearNaive`：稀疏 attention，但 DDR 使用线性映射、无请求合并、无流水线重叠
- `Sparse-HashNoMerge`：稀疏 attention，DDR 使用哈希映射，但不合并请求
- `Sparse-HashMerge`：稀疏 attention，DDR 使用哈希映射并合并相邻请求
- `Sparse-HashMergePipeline`：在 `Sparse-HashMerge` 基础上启用完整的 `P/F/C` 三阶段流水线

## 运行方式

在仓库根目录下：

```bash
cmake --build simulator/build --target experiment_end_to_end_decode -j
./simulator/build/experiment_end_to_end_decode
```

可选参数：

```bash
./simulator/build/experiment_end_to_end_decode \
  --contexts 8192,32768,131072 \
  --sparsities 0.01,0.05,0.10 \
  --ddr-channels 8 \
  --head-groups 8 \
  --heads-per-group 4 \
  --csv simulator/results/end_to_end_decode.csv
```

## 输出指标

程序会在终端输出并可选导出 CSV，主要指标包括：

- `Cycles`：单次 decode attention 子系统总时延
- `Latency(us)`：按 200MHz 折算的时延
- `Tok/s`：理论每 token 吞吐
- `Speedup`：相对 `Dense-FullKV` 的加速比
- `DDR GB/s`：按总时延折算的 DDR 有效带宽
- `Bubble%`：流水线空泡率，仅完整流水线方案非零或可观测
- `P util / F util / C util`：预测 / 抓取 / 计算三阶段利用率

## 适合论文里的图表

建议直接基于 CSV 画以下图：

1. 不同 `context length` 下的总时延对比图
2. 不同 `sparsity` 下的 speedup 曲线
3. 不同方法的 DDR 有效带宽柱状图
4. `Sparse-HashMergePipeline` 的阶段利用率堆叠图
5. 不同 `DDR channel` 配置下的总时延折线图
