import os
print("HF_ENDPOINT =", os.getenv("HF_ENDPOINT"))
import torch
from transformers import AutoTokenizer, LlamaForCausalLM
import argparse

# It's good practice to add the project root to the python path
# to make imports work seamlessly for scripts.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.pruned_attention.calibration import run_calibration_and_save_indices, get_c4_simple


def main():
    """
    Main script to run the calibration process for the Llama model attention heads
    and save the resulting dimension indices.
    """
    parser = argparse.ArgumentParser(description="Run attention calibration for a Llama model.")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained Llama model."
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of calibration samples to use from the C4 dataset."
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=512,
        help="Sequence length for the calibration data."
    )
    parser.add_argument(
        "--use_modelscope",
        action="store_true",
        help="Whether to use ModelScope's dataset loading instead of Hugging Face's datasets library."
    )
    
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {args.model_path}...")
    # Ensure eager attention is used for the calibration process
    model = LlamaForCausalLM.from_pretrained(
        args.model_path, 
        torch_dtype=torch.bfloat16, 
        attn_implementation="eager"
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    print(f"Preparing {args.num_samples} calibration samples of length {args.seq_len}...")
    calibration_data = get_c4_simple(
        tokenizer, 
        n_samples=args.num_samples, 
        seq_len=args.seq_len,
    )
    
    model_name = os.path.basename(args.model_path)
    
    run_calibration_and_save_indices(
        model, 
        calibration_data, 
        device, 
        model_name=model_name
    )

if __name__ == "__main__":
    main()
