import torch
import torch.nn as nn
import torch.nn.functional as F

class PseudoQuantizer(nn.Module):
    def __init__(self, bits=4, mode="int", group_size=-1):
        """
        bits: 量化位数 (1 for binary, 4 for int4)
        mode: 'int' or 'binary'
        """
        super().__init__()
        self.bits = bits
        self.mode = mode
    
    def forward(self, x):
        if self.mode == "binary" or self.bits == 1:
            return self.quantize_binary(x)
        else:
            return self.quantize_int(x)

    def quantize_int(self, x):
        # === INT4 量化 (Symmetric) ===
        # 范围: [-7, 7] (为了对称通常放弃 -8)
        qmax = 2**(self.bits - 1) - 1
        qmin = -qmax
        
        # 1. 计算 Scale (Per-Token / Per-Head 量化)
        # x shape: [Batch, Heads, Seq, Dim]
        # 我们在最后一个维度 (Dim) 上找最大值作为缩放基准
        # 加上 1e-5 防止除零
        scale = x.abs().amax(dim=-1, keepdim=True) / qmax
        scale = torch.clamp(scale, min=1e-5)
        
        # 2. 量化 (Quantize) -> Round -> Clamp
        # detached scale 确保梯度只流经 x
        x_int = torch.clamp(torch.round(x / scale), qmin, qmax)
        
        # 3. 反量化 (De-quantize) 模拟计算时的数值
        x_fake_quant = x_int * scale
        
        # 4. STE (Straight-Through Estimator)
        # 前向传播用 x_fake_quant，反向传播梯度直接传给 x
        return x + (x_fake_quant - x).detach()

    def quantize_binary(self, x):
        # === Binary (1-bit) 量化 ===
        # XNOR-Net 风格: Sign(x) * Mean(|x|)
        # 纯 Sign 效果通常不好，加上 Mean(|x|) 作为缩放因子能大幅提升精度
        
        # 1. 计算缩放因子 E(|x|)
        scale = x.abs().mean(dim=-1, keepdim=True)
        
        # 2. 二值化: x >= 0 -> 1, x < 0 -> -1
        x_bin = torch.sign(x)
        # torch.sign(0) 是 0，硬件上通常映射为 1，这里为了严谨做个处理
        x_bin[x_bin == 0] = 1.0
        
        # 3. 重建
        x_fake_bin = x_bin * scale
        
        # 4. STE
        return x + (x_fake_bin - x).detach()