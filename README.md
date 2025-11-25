# Pruned Attention for Llama Models

This project implements a technique for compressing the Key-Value (KV) cache in large language models like Llama. It uses a proxy-based attention mechanism where a smaller, "scout" attention calculation is used to identify the most important key-value pairs. The full attention calculation is then performed only on this selected subset, reducing computational cost and memory usage during inference.

The project includes scripts for:
1.  **Calibration**: Analyzing a base model to determine the most sensitive dimensions in the attention mechanism to keep.
2.  **Distillation**: Fine-tuning the compressed attention projections to better mimic the behavior of the original, full attention.

## Project Structure

This project has been refactored into a more structured Python package format.

-   `src/pruned_attention/`: Contains the core Python package source code.
    -   `attention.py`: Implements the `CompressedLlamaAttention` module.
    -   `calibration.py`: Contains functions for the calibration process.
    -   `distillation.py`: Contains the logic for layer-wise distillation.
    -   `modeling_llama.py`: A modified version of the Hugging Face Llama model implementation to support the custom attention mechanism.
    -   `utils.py`: Utility functions, including the `PseudoQuantizer`.
-   `scripts/`: Contains runnable scripts for performing key tasks.
    -   `run_calibration.py`: Script to generate the `indices.json` file.
    -   `run_distillation.py`: Script to perform layer-wise distillation and save the fine-tuned small projection weights.
-   `requirements.txt`: A list of Python dependencies.
-   `pyproject.toml`: Project definition file for packaging and installation.

## Setup

1.  **Install Dependencies**: It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

2.  **Install the Package**: Install the project in editable mode. This allows you to run the scripts and modify the source code.
    ```bash
    pip install -e .
    ```

## Usage

The process involves two main steps: Calibration and optional Distillation.

### 1. Calibration

This step analyzes a base Llama model and generates an `indices.json` file, which specifies which dimensions of the KV cache to keep.

-   **Command**:
    ```bash
    python scripts/run_calibration.py --model_path /path/to/your/llama/model --output_path indices.json
    ```
-   **Arguments**:
    -   `--model_path`: (Required) Path to the Hugging Face Llama model directory.
    -   `--output_path`: (Optional) Path to save the generated indices file. Defaults to `indices.json`.
    -   `--num_samples`: (Optional) Number of calibration samples to use. Defaults to 16.
    -   `--seq_len`: (Optional) Sequence length of calibration data. Defaults to 512.

### 2. Distillation (Optional)

This step fine-tunes the small projection matrices (`q_proj_small` and `k_proj_small`) created during the compression process to better match the original attention distribution. This can improve the performance of the compressed model.

-   **Command**:
    ```bash
    python scripts/run_distillation.py --model_path /path/to/your/llama/model --indices_path indices.json --save_path ./distilled_model
    ```
-   **Arguments**:
    -   `--model_path`: (Required) Path to the Hugging Face Llama model directory.
    -   `--indices_path`: (Required) Path to the `indices.json` file generated during calibration.
    -   `--save_path`: (Optional) Directory to save the `small_attn_weights.pt` file. Defaults to `./compressed_distilled_model`.
    -   Other arguments like `--lr`, `--batch_size`, etc., are available to control the training process.

### 3. Using the Compressed Model

To use the final compressed model, you would typically:
1.  Load the base Llama model.
2.  Use `apply_compression_to_model` from `pruned_attention.calibration`.
3.  If you performed distillation, load the `small_attn_weights.pt` and apply them to the `CompressedLlamaAttention` layers.

*(Example usage script to be added)*
