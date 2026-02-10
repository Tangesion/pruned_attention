import torch
import os
import random
import argparse

# It's good practice to add the project root to the python path
# to make imports work seamlessly for scripts.
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from transformers import AutoTokenizer, LlamaForCausalLM
#from src.pruned_attention.modeling_llama import LlamaForCausalLM # Use our custom model
from src.pruned_attention.distillation import train_layer_wise_distillation


def main():
    """
    Main script to run layer-wise distillation to fine-tune the small
    projection matrices in the CompressedLlamaAttention layers.
    """
    parser = argparse.ArgumentParser(description="Run layer-wise distillation for a compressed Llama model.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the pretrained Llama model.")
    parser.add_argument("--seq_len", type=int, default=512, help="Sequence length for calibration and distillation data.")
    parser.add_argument("--num_samples", type=int, default=512, help="Number of samples to use for distillation.")
    parser.add_argument("--train_steps", type=int, default=300, help="Number of training steps per layer.")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for distillation.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for the optimizer.")
    args = parser.parse_args()

    print(f"Loading model: {args.model_path}")
    # We must use our own LlamaForCausalLM implementation to ensure the custom
    # decoder layers and cache are used correctly.
    # Using 'eager' attention is necessary for the distillation process, as we need
    # access to the original attention scores.
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    indices_path = f"data/{os.path.basename(args.model_path)}/indices.json"

    if not os.path.exists(indices_path):
        print(f"Error: Indices file not found at {args.indices_path}. Please run calibration first.")
        return

    print("Starting Layer-wise Distillation...")
    distilled_model = train_layer_wise_distillation(
        model=model,
        tokenizer=tokenizer,
        indices_path=indices_path,
        train_steps_per_layer=args.train_steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seq_len=args.seq_len,
        num_calibration_samples=args.num_samples
    )
    
    save_path = "data/" + os.path.basename(args.model_path)
    
    print(f"Saving distilled small attention weights to {save_path}...")
    small_state = {}
    for i, layer in enumerate(distilled_model.model.layers):
        attn = layer.self_attn
        if hasattr(attn, "q_proj_small") and hasattr(attn, "k_proj_small"):
            small_state[f"model.layers.{i}.self_attn.q_proj_small.weight"] = attn.q_proj_small.weight.detach().cpu()
            small_state[f"model.layers.{i}.self_attn.k_proj_small.weight"] = attn.k_proj_small.weight.detach().cpu()

    small_path = os.path.join(save_path, "small_attn_weights.pt")
    torch.save(small_state, small_path)
    print(f"Saved small attention weights to {small_path}")
    print("Done!")

if __name__ == "__main__":
    torch.manual_seed(42)
    random.seed(42)
    main()
