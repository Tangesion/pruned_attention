# 第五章实验设计（基于第三章算法）

本章重点验证算法的**有效性、准确性与理论效率提升**。由于第四章硬件设计尚未完成，本章主要分为两部分：  
- **算法验证实验（Pure Algorithm Evaluation）**  
- **硬件联合仿真/实测（Hardware Evaluation，后续补充）**

---

## 一、实验环境设置（前提）

- **模型选择**
  - Llama-3-8B-Instruct
  - Llama-2-7B / 13B（根据显存条件）
  - 长文本微调模型：LongChat / Vicuna-128k

- **数据集**
  - **PPL 测试**：PG-19（长篇书籍）、GovReport（长篇报告）
  - **下游任务**：LongBench、L-Eval
  - **长文本检索**：Needle In A Haystack（大海捞针）

- **对比基线（Baselines）**
  - Full Attention（理论上限）
  - Sliding Window（局部窗口，如 1024）
  - H2O / StreamingLLM（动态稀疏）
  - Uniform / Random Pruning（验证敏感度剪枝必要性）

---

## 二、核心性能验证（Accuracy & Robustness）

### 实验 1：长文本建模能力（Perplexity）
- **目的**：验证压缩后 PPL 不显著上升，且优于稀疏对比方法  
- **操作**：在 PG-19 上逐步增加序列长度（4k → 32k → 128k）  
- **图表**：折线图（X=长度，Y=PPL）  
- **预期**：
  - Full Attention 最低
  - Sliding Window 超过窗口后崩塌
  - **你的方法接近 Full Attention，明显优于 H2O/Random**

---

### 实验 2：大海捞针（Passkey Retrieval）
- **目的**：验证混合稀疏掩码捕捉长距离依赖能力  
- **操作**：在不同深度、不同长度插入 Key，测回忆准确率  
- **图表**：Heatmap  
- **预期**：准确率高、红区少，Top-K 检索有效

---

### 实验 3：下游任务评测（LongBench）
- **目的**：验证真实任务能力  
- **操作**：运行 LongBench  
- **图表**：雷达图或表格  
- **对比**：full，h2o，ours，在不同压缩比例下的效果

---

## 三、组件有效性分析（Ablation Studies）

### 实验 4：剪枝策略有效性（对应 1.2.1 & 1.2.2）
- **对比**：
  - Ours（Sensitivity-based）
  - Magnitude-based
  - Random
- **指标**：Top-K Recall  
- **预期**：你的方法 Recall 最高

---

### 实验 5：蒸馏效果（对应 1.3）
- **对比**：
  - Before Distillation
  - After Distillation（你的排序蒸馏）
  - MSE Distillation
- **指标**：Recall 或 PPL  
- **图表**：训练曲线 + Recall 柱状图  
- **预期**：Ranking Loss 最优

---

### 实验 6：混合掩码必要性（对应 1.2.3）
- **对比**：
  - Only Dynamic Top-K
  - Dynamic + Sink
  - Dynamic + Sink + Local（完整版）
- **指标**：PPL  
- **预期**：无 Sink 时 PPL 高；加入 Local 后进一步下降

---

### 实验 7：量化影响（对应 1.4）
- **对比**：
  - FP16 Predict
  - Int4 Predict
- **指标**：Recall@K、PPL  
- **预期**：Recall 小幅下降（<1–2%），PPL 基本持平

---

## 四、系统效率分析（Theoretical Efficiency）

### 实验 8：显存占用分析
- **公式**：
  - Baseline：`2 × L × D × 2 Bytes (FP16)`
  - Ours：`2 × L × (d_compressed × 0.5 Bytes + D × Sparsity × 2 Bytes)`
- **图表**：堆叠面积图  
- **预期**：显存增长斜率显著降低，支持更长上下文

---

### 实验 9：IO 访问量/带宽节省
- **指标**：Memory Access Volume (GB/token)  
- **对比**：全量 KV vs 稀疏 KV  
- **预期**：稀疏度 5% 时带宽减少 >90%

---

## 五、论文写作建议（研三）

- **章节过渡**：
  - “本章首先基于 PyTorch 框架对第三章算法进行全精度验证（5.1–5.3），随后结合第四章硬件加速器进行仿真评估（5.4）。”
- **Highlight 亮点**：
  - “Int4 下 Recall 仍 >95%，验证注意力排序对噪声鲁棒”
- **图表规范**：
  - 轴标签清晰、颜色对比明显、标题统一
- **失败案例分析（可选）**：
  - 分析极端密集指代场景的失败原因

---

## 总结

只要把 **PPL（保精度）**、**Needle（长文本能力）**、**Recall（检索准确率）**、**Memory Savings（存算开销）** 这四个维度讲清楚，就是一篇高质量硕士论文。