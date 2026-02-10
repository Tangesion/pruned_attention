# 第五章 实验结果与分析 — 书写指南

## 目录
1. [数据源与文件位置](#数据源与文件位置)
2. [各节书写要点](#各节书写要点)
3. [图表规范](#图表规范)
4. [语言风格要求](#语言风格要求)
5. [与第三章的对应关系](#与第三章的对应关系)

---

## 数据源与文件位置

实验数据存储在项目根目录下：

| 数据类型 | 文件路径 | 说明 |
|---------|---------|------|
| PPL结果 | `experiment_ppl_results/ppl_results.json` | 困惑度测试数据 |
| PPL测试结果 | `experiment_ppl_results_test/` | 测试阶段PPL数据 |
| 实验4结果 | `experiment_4_results.json` | 剪枝策略对比 |
| 实验5结果 | `experiment_5_results/` | 蒸馏效果验证 |
| 实验5测试 | `experiment_5_results_test/` | 蒸馏测试数据 |
| 原始PPL基线 | `test_original_ppl.json` | Full Attention基线 |
| 论文第三章内容 | `paper.md` | 算法描述原文 |
| 实验设计方案 | `.claude/skills/experiment/experiment.md` | 实验总体设计 |

### 关键代码文件

| 文件 | 用途 |
|------|------|
| `src/pruned_attention/attention.py` | CompressedLlamaAttention 核心实现 |
| `src/pruned_attention/calibration.py` | 敏感度分析与剪枝 |
| `src/pruned_attention/distillation.py` | 层间蒸馏逻辑 |
| `scripts/run_pg16_ppl_test.py` | PG19 PPL评测脚本 |
| `scripts/run_calibration.py` | 校准运行脚本 |
| `scripts/run_distillation.py` | 蒸馏运行脚本 |

---

## 各节书写要点

### 5.1 实验环境与配置
- 列出模型、数据集、硬件环境、超参数
- 明确对比基线（Full Attention, Sliding Window, H2O, StreamingLLM）
- 给出压缩比等关键配置参数
- 用表格形式呈现

### 5.2 核心性能验证

#### 5.2.1 长文本建模能力（Perplexity）
- 读取 `experiment_ppl_results/` 数据
- 对比方法：Full Attention / Ours (BF16) / Ours (Int4) / Sliding Window / H2O
- X轴=序列长度(4k→128k)，Y轴=PPL
- 重点分析：压缩后PPL与Full Attention的差距，Sliding Window在长序列上的崩塌

#### 5.2.2 大海捞针测试
- Heatmap 可视化
- X轴=文本长度，Y轴=needle插入深度，颜色=准确率
- 验证混合掩码的长距离检索能力

#### 5.2.3 下游任务评测（LongBench）
- 参考 `LongBench/` 目录下的评测框架
- 雷达图或表格展示多任务性能
- 分析不同任务类型（QA、摘要、检索）的表现差异

### 5.3 组件有效性分析

#### 5.3.1 剪枝策略有效性（对应3.2.1 & 3.2.2）
- 对比：Sensitivity-based (Ours) vs Magnitude-based vs Random
- 指标：Top-K Recall
- 读取 `experiment_4_results.json`

#### 5.3.2 蒸馏效果验证（对应3.3）
- 对比：Before vs After Distillation / MSE vs Ranking Loss
- 指标：Recall、PPL
- 读取 `experiment_5_results/`
- 展示训练曲线和Recall柱状图

#### 5.3.3 混合掩码必要性（对应3.2.3）
- 对比：Only Dynamic / Dynamic+Sink / Dynamic+Sink+Local (完整)
- 指标：PPL
- 分析各组件的独立贡献

#### 5.3.4 量化影响分析（对应3.4）
- 对比：FP16 Predict vs Int4 Predict
- 指标：Recall@K、PPL
- 读取PPL结果中BF16与Int4的对比数据

### 5.4 系统效率分析
- 理论计算显存节省
- 公式推导 + 数据表格
- 带宽节省分析

### 5.5 本章小结
- 总结核心发现（3-4句话）
- 强调关键数据点

---

## 图表规范

- **编号**：图5.1、图5.2... 表5.1、表5.2...
- **标题位置**：图标题在下方，表标题在上方
- **标题格式**：中文描述 + 英文补充（如需要）
- **坐标轴**：标签清晰，含单位
- **颜色**：对比方法间颜色区分明显
- **数据精度**：PPL保留2位小数，Recall保留1位小数(百分比)

---

## 语言风格要求

- **主体语言**：中文
- **术语处理**：首次出现给出英文全称和缩写，后续可直接用缩写
  - 例："困惑度（Perplexity, PPL）"
- **公式**：LaTeX格式，编号为 (5.x)
- **时态**：过去时描述实验过程，现在时描述结果和结论
- **人称**：使用"本文"而非"我们"
- **避免口语化**：使用"表明"而非"说明了"，"显著"而非"很大"

---

## 与第三章的对应关系

| 第三章内容 | 第五章实验 | 核心验证目标 |
|-----------|-----------|-------------|
| 3.2.1-3.2.2 敏感度剪枝 | 5.3.1 剪枝策略有效性 | Sensitivity > Magnitude > Random |
| 3.2.3 混合稀疏掩码 | 5.3.3 混合掩码必要性 | Sink+Local的不可或缺性 |
| 3.3 层间蒸馏 | 5.3.2 蒸馏效果 | Ranking Loss > MSE Loss |
| 3.4 Int4量化 | 5.3.4 量化影响 | Int4对排序鲁棒 |
| 整体框架 | 5.2 核心性能 | 接近Full Attention、优于H2O |
