import torch
import torch.nn as nn
import torch.nn.functional as F

class PseudoQuantizer(nn.Module):
    """
    A module for simulating quantization (INT4 or binary) using a Straight-Through Estimator (STE).
    This allows the model to learn how to handle quantized weights during training,
    while the actual forward pass uses fake-quantized values.
    """
    def __init__(self, bits=4, mode="int"):
        """
        Args:
            bits (int): The number of bits for quantization (e.g., 4 for INT4, 1 for binary).
            mode (str): The quantization mode, either 'int' or 'binary'.
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
        # === Symmetric INT4 Quantization ===
        # The range is [-7, 7]. -8 is typically omitted for symmetry.
        qmax = 2**(self.bits - 1) - 1
        qmin = -qmax
        
        # 1. Calculate Scale (Per-Token / Per-Head quantization)
        # x shape: [Batch, Heads, Seq, Dim]
        # We find the max value along the last dimension (Dim) as the scaling factor.
        # Add 1e-5 to prevent division by zero.
        scale = x.abs().amax(dim=-1, keepdim=True) / qmax
        scale = torch.clamp(scale, min=1e-5)
        
        # 2. Quantize -> Round -> Clamp
        # Detaching the scale ensures gradients only flow through x.
        x_int = torch.clamp(torch.round(x / scale), qmin, qmax)
        
        # 3. De-quantize to simulate the values used in computation.
        x_fake_quant = x_int * scale
        
        # 4. STE (Straight-Through Estimator)
        # In the forward pass, use the fake-quantized value.
        # In the backward pass, the gradient is passed directly to the original `x`.
        return x + (x_fake_quant - x).detach()

    def quantize_binary(self, x):
        # === Binary (1-bit) Quantization ===
        # Style inspired by XNOR-Net: Sign(x) * Mean(|x|)
        # Using a scaling factor (mean of absolute values) significantly improves
        # precision compared to using Sign alone.
        
        # 1. Calculate the scaling factor E(|x|)
        scale = x.abs().mean(dim=-1, keepdim=True)
        
        # 2. Binarize: x >= 0 -> 1, x < 0 -> -1
        x_bin = torch.sign(x)
        # torch.sign(0) is 0, but in hardware it's often mapped to 1.
        # We handle this explicitly for correctness.
        x_bin[x_bin == 0] = 1.0
        
        # 3. Reconstruct the fake-binary value
        x_fake_bin = x_bin * scale
        
        # 4. STE
        return x + (x_fake_bin - x).detach()