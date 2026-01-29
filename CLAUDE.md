# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

- **Install Dependencies**: `pip install -r requirements.txt`
- **Install Package (Editable)**: `pip install -e .`
- **Calibration**: Analyze base model to find sensitive dimensions.
  `python scripts/run_calibration.py --model_path <llama_path> --output_path indices.json`
- **Distillation**: Fine-tune small projection weights.
  `python scripts/run_distillation.py --model_path <llama_path> --indices_path indices.json --save_path ./distilled_model`
- **Perplexity Testing**:
  `python scripts/run_perplexity_test.py --model_path <llama_path> --indices_path indices.json`
- **LoRA Fine-tuning**:
  `python scripts/run_lora_finetuning.py` (check script for args)
- **Streaming Inference**:
  `python scripts/run_streaming_inference.py`

## Architecture

This project implements a compressed Key-Value (KV) cache for Llama models using a proxy-based "scout" attention mechanism.

- **Source (`src/pruned_attention/`)**:
  - `attention.py`: Implements `CompressedLlamaAttention`. The core logic where "scout" attention identifies important KV pairs before full attention.
  - `modeling_llama.py`: Modified Hugging Face Llama model wrapper to support the custom attention mechanism.
  - `calibration.py`: Logic for analyzing attention sensitivity to generate `indices.json`.
  - `distillation.py`: Logic for layer-wise distillation to align compressed attention with original behavior.
  - `utils.py`: Helpers, including `PseudoQuantizer`.

- **Scripts (`scripts/`)**:
  - Operational entry points for the two-stage process (Calibration -> Distillation) and subsequent inference/testing.
  - `indices.json`: The artifact produced by calibration, dictating which KV dimensions to retain.

- **Simulator (`simulator/`)**:
  - Contains build artifacts or C++ extensions for simulation (if applicable).
