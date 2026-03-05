# 实验步骤
## 1. PPL实验
* 目标： 对比不同压缩比下的PPL变化曲线
* 对比对象： Full Attention，H2O，slidewindow，ours(bf16)，ours(int4)
* 实验参考脚本：[run_pg16_ppl_test](../scripts/run_pg16_ppl_test.py)
* 实验配置（命令）：
```bash
python scripts/run_pg16_ppl_test.py \
      --base-model /home/tgx/models/Llama-3.2-1B \
      --eval-all \
      --topk-ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 \
      --num-eval-tokens 1024 \
      --sink-size 4 --local-window 16 \
      --escaped-layers 0 1 14 15 \
      --output-dir ./experiment_ppl_results
```

## 2. 剪枝策略有效性实验
* 目标：对比不同剪枝标准对Top-K召回率的影响
* 对比对象：随机选择，基于幅值选择，ours（校准敏感度选择）
* 实验参考脚本：[run_pruning_experiment](../scripts/run_pruning_experiment.py)
* 实验配置（命令）：
```bash
python scripts/run_pruning_experiment.py \
--model_path /home/tgx/models/Llama-3.2-1B
```

## 3. 蒸馏效果验证
* 实验参考脚本：[run_distillation_experiment](../scripts/run_distillation_experiment.py) 

### 3.1 蒸馏训练收敛曲线
* 目标：对比不同方法的蒸馏训练曲线loss图
* 对比对象：传统均方误差损失（MSE Distillation），本文提出的排序一致性损失（Ranking Loss）

### 3.2 蒸馏效果召回率对比
* 目标：对比蒸馏前及不同蒸馏方法的Recal
* 对比对象：每层蒸馏前Recall，每层MSE蒸馏后Recall，每层Ranking Loss蒸馏后Recall

### 3.3 蒸馏效果PPL对比
* 目标：对比蒸馏前及不同蒸馏方法的PPL
* 对比对象：蒸馏前PPL，MSE蒸馏后PPL，Ranking Loss蒸馏后PPL（可以多来点数据集，然后画折线）

## 4. 下游任务评测
