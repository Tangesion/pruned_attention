# BF16 Long-Context LLM Accelerator Simulator

## Project Overview

This project is a C++ performance simulator for a custom FPGA-based hardware accelerator designed to optimize Long-Context Large Language Model (LLM) inference. It implements the architecture proposed in a Master's thesis focusing on a "Predict-Retrieval" paradigm to overcome the Memory Wall bottleneck in long-context scenarios.

### Core Concept: Predict-Retrieval Paradigm
The simulator models a hardware architecture that addresses the linear/super-linear growth of KV Cache in LLMs by splitting the inference process:
1.  **Predict (P-Stage):** Uses compressed Int4 "Small KV" cache stored in HBM (High Bandwidth Memory) to quickly scan and identify the Top-K most relevant tokens.
2.  **Retrieve (F-Stage):** Fetches only the necessary high-precision (BF16) "Full KV" data from larger, slower DDR memory based on the prediction.
3.  **Compute (C-Stage):** Performs the final Attention computation using the sparsely retrieved data.

### Key Objectives
*   **Performance Modeling:** To accurately model the latency and throughput of the proposed FPGA architecture.
*   **Memory Efficiency:** To validate the effectiveness of the proposed memory hierarchy (HBM for indices, DDR for payload) and access patterns (Stream vs. Random).
*   **Pipeline Optimization:** To demonstrate the speedup achieved by overlapping the Predict, Fetch, and Compute stages.

## Directory Structure

*   **`src/`**: Source code for the simulator library (`bf16_sim`).
    *   `bf16/`: Basic BF16 arithmetic and simulation logic.
    *   `Memory/`: Memory controller models (`DDRController`, `HBMController`) simulating latency, bank conflicts, etc.
    *   `PE/`: Processing Element logic (`MacUnit`, `AddUnit`, etc.) and schedulers.
    *   `System/`: System-level components like `PipelineSimulator` that orchestrate the entire process.
*   **`include/`**: Header files mirroring the `src` structure.
*   **`experiments/`**: High-level simulation scripts used to generate results for the thesis (e.g., memory efficiency, pipeline overlap).
*   **`test/`**: Unit tests for individual components.
*   **`doc/`**: Documentation, including `abstract.md` describing the thesis.
*   **`CMakeLists.txt`**: CMake build configuration.

## Build and Run

The project uses CMake (minimum version 3.10) and requires a C++17 compliant compiler.

### Building
```bash
mkdir build
cd build
cmake ..
make
```

### Running Experiments
After building, executables will be available in the `build/` directory.

*   **Memory Efficiency Experiment:**
    ```bash
    ./build/experiment_memory_efficiency
    ```
*   **Pipeline Overlap Experiment:**
    ```bash
    ./build/experiment_pipeline_overlap
    ```

### Running Tests
Unit tests are compiled as separate executables.
```bash
./build/test_add_sim
./build/test_bf16_basic_ops
./build/test_pipeline_simulator # (Example name, check build dir for exact target)
```

## Architecture Details

### System Simulation (`System/PipelineSimulator`)
The `PipelineSimulator` is the top-level orchestrator. It models the three-stage pipeline (Predict, Fetch, Compute) and tracks:
*   **Cycle Counts:** Total execution time.
*   **Bubbles:** Pipeline stalls due to dependencies or resource contention.
*   **Stage Overlap:** Efficiency of the pipelining strategy.

### Memory Models (`Memory/`)
*   **`DDRController`:** Simulates DDR DRAM behavior, including Bank Mapping (Hash-based), Row Buffer hits/misses, and timing parameters (tCL, tRCD, tRP). It is crucial for validating the sparse retrieval latency.
*   **`HBMController`:** Simulates High Bandwidth Memory, primarily modeled for high-throughput streaming access used in the Prediction stage.

### Processing Elements (`PE/`)
Models the computational units of the FPGA, such as Matrix-Multiply Units (MacUnit) and Adder Trees. These modules are "Cycle-Accurate" in terms of throughput but "Data-Agnostic" in many high-level simulations (calculating latency based on workload size rather than actual data values).

## Development Conventions

*   **Language:** C++17.
*   **Build System:** CMake.
*   **Testing:** Custom `assert`-based test files in `test/`. No external testing framework (like GTest) is currently used.
*   **Code Style:**
    *   Namespaces: `System`, `PE`, `Memory`, `bf16`.
    *   Header-only dependencies are preferred for simple utilities.
    *   Separation of interface (`.h`) and implementation (`.cpp`).

## Experiment Guide (Thesis)

The `experiments/` directory contains the logic to reproduce the thesis results:
1.  **End-to-End Latency:** Comparing FPGA simulator cycles vs. GPU baselines.
2.  **Memory Efficiency:** validating the Hash-based Bank Mapping against naive linear mapping.
3.  **Ablation Studies:** Quantifying the benefit of the 3-stage pipeline vs. serial execution.
