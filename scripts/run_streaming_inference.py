import torch
import os
import sys
import argparse
from transformers import AutoTokenizer, StoppingCriteria, StoppingCriteriaList, LlamaForCausalLM
# Add project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.pruned_attention.attention import KVWithSmallKCache

class StreamTokens(StoppingCriteria):
    def __init__(self, tokenizer, input_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.input_len = input_len
        self.last_printed = input_len  

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        seq = input_ids[0]
        cur_len = seq.shape[0]

        if cur_len > self.last_printed:
            new_tokens = seq[self.last_printed:cur_len]
            text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            print(text, end="", flush=True)
            self.last_printed = cur_len

        return False

def main():
    parser = argparse.ArgumentParser(description="Unified Streaming Inference for Llama")
    parser.add_argument("--method", type=str, default="base", choices=["base", "compress", "h2o"], 
                        help="Inference method: 'base' (original), 'compress' (pruned), or 'h2o' (heavy hitter)")
    parser.add_argument("--model_path", type=str, default="/home/tgx/data/models/Llama-3-8B-Instruct",
                        help="Path to the model")
    parser.add_argument("--prompt_file", type=str, default="prompt1.txt", help="Path to the prompt file")
    
    # H2O specific arguments
    parser.add_argument("--h2o_heavy", type=float, default=0.1, help="H2O: Heavy hitter ratio")
    parser.add_argument("--h2o_recent", type=float, default=0.1, help="H2O: Recent window ratio")
    
    # Compress specific arguments (optional, adjust defaults as needed)
    parser.add_argument("--indices_path", type=str, default="../indices.json", help="Compress: Path to indices.json")

    args = parser.parse_args()

    # --- 1. Dynamic Imports & Model Class Selection ---
    if args.method == "compress":
        print("Mode: Compress (Pruned Attention)")
        try:
            from src.pruned_attention.calibration import apply_compression_to_model
        except ImportError as e:
            print(f"Error importing pruned_attention modules: {e}")
            sys.exit(1)
    else:
        # Base or H2O uses standard Transformers class initially
        print(f"Mode: {args.method.capitalize()}")
        
        if args.method == "h2o":
            try:
                from src.h2o_attention.attention import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
            except ImportError as e:
                print(f"Error importing H2O attention modules: {e}")
                sys.exit(1)

    # --- 2. Load Tokenizer & Model ---
    print(f"Loading model from {args.model_path}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16, # Llama 3 usually prefers bfloat16
        attn_implementation="eager" # Required for both H2O and Pruned modifications
    )

    # --- 3. Apply Method-Specific Logic ---
    if args.method == "compress":
        print("Applying Compression to model...")
        # 根据你的 apply_compression_to_model 签名调整参数
        # 如果它需要 indices_path，请传入 args.indices_path
        # 这里假设它接受 model 作为第一个参数
        model = apply_compression_to_model(model) 

    elif args.method == "h2o":
        print(f"Applying H2O Attention (Heavy: {args.h2o_heavy}, Recent: {args.h2o_recent})...")
        model.config.heavy_ratio = args.h2o_heavy
        model.config.recent_ratio = args.h2o_recent
        
        model = convert_kvcache_llama_heavy_recent(model, model.config)
        
        ## Patch: Ensure num_heads exists (fix for potential bug in H2O class)
        #for module in model.modules():
        #    if isinstance(module, LlamaAttention_heavy_hitter):
        #        if not hasattr(module, 'num_heads'):
        #            module.num_heads = model.config.num_attention_heads
        #
        ## Reset internal state
        #for module in model.modules():
        #    if hasattr(module, '_reset_masks'):
        #        module._reset_masks()

    model.eval()

    # --- 4. Prepare Input ---
    if not os.path.exists(args.prompt_file):
        print(f"Error: Prompt file '{args.prompt_file}' not found.")
        sys.exit(1)

    with open(args.prompt_file, "r", encoding="utf-8") as f:
        prompt = f.read()

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    input_len = inputs["input_ids"].shape[1]
    streamer = StreamTokens(tokenizer, input_len)

    # --- 5. Generate ---
    print("\n--- Start Generation ---\n")
    print(prompt, end="", flush=True) # Print prompt first
    
    output = model.generate(
        **inputs,
        max_new_tokens=256,
        use_cache=True,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        repetition_penalty=1.2,
        past_key_values=KVWithSmallKCache(config=model.config) if args.method == "compress" else None,
        stopping_criteria=StoppingCriteriaList([streamer]),
    )

    print("\n\n--- End Generation ---")

if __name__ == "__main__":
    main()