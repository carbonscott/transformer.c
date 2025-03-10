# CUDA Transformer Block Implementation

A single-script distillation example of llm.c that focuses exclusively on the inference pipeline for easy incorporation into other projects.

## Features

- Self-attention mechanism with flash attention optimization
- Multi-head attention support
- Layer normalization
- Feed-forward network with GELU activation
- Residual connections
- Support for different floating-point precisions (FP32, FP16, BF16)

## Prerequisites

- CUDA Toolkit (11.0+)
- cuBLAS and cuBLASLt libraries
- A compatible NVIDIA GPU (Volta architecture or newer recommended)
- C++ compiler with C++11 support

## Compilation

### For different GPU architectures

```bash
# For Volta GPUs (Titan V, V100)
nvcc -o example_transformer example_transformer.cu transformer_block.cu -lcublas -lcublasLt -arch=compute_70 -code=sm_70

# For Turing GPUs (GTX 16xx, RTX 20xx)
nvcc -o example_transformer example_transformer.cu transformer_block.cu -lcublas -lcublasLt -arch=compute_75 -code=sm_75

# For Ampere GPUs (A100)
nvcc -o example_transformer example_transformer.cu transformer_block.cu -lcublas -lcublasLt -arch=compute_80 -code=sm_80

# For Ada Lovelace GPUs (RTX 40xx, L40S)
nvcc -o example_transformer example_transformer.cu transformer_block.cu -lcublas -lcublasLt -arch=compute_89 -code=sm_89
```

### To specify precision

```bash
# For FP32 precision
nvcc -o example_transformer example_transformer.cu transformer_block.cu -lcublas -lcublasLt -DENABLE_FP32 -arch=compute_89 -code=sm_89

# For FP16 precision
nvcc -o example_transformer example_transformer.cu transformer_block.cu -lcublas -lcublasLt -DENABLE_FP16 -arch=compute_89 -code=sm_89

# For BF16 precision (default if not specified)
nvcc -o example_transformer example_transformer.cu transformer_block.cu -lcublas -lcublasLt -arch=compute_89 -code=sm_89
```

## Usage

The provided `example_transformer.cu` demonstrates how to use the transformer block implementation:

- Define the transformer dimensions (batch size, sequence length, embedding dimension, etc.)
- Initialize the transformer block
- Load weights into the transformer block
- Run the forward pass with input data
- Clean up resources

Example:
```cpp
// Define transformer dimensions
TransformerDims dims = {
    .B = 2,   // Batch size
    .T = 32,  // Sequence length
    .C = 768, // Embedding dimension
    .NH = 12, // Number of attention heads
    .HS = 64  // Head size (C / NH = 768 / 12 = 64)
};

// Initialize transformer block
TransformerBlock block;
transformer_block_init(&block, dims);

// Load weights
transformer_block_load_weights(&block, weights_data);

// Run forward pass
transformer_block_forward(&block, output, input, stream);

// Free resources
transformer_block_free(&block);
```

## Implementation Details

- The implementation uses vectorized 128-bit loads/stores for efficiency
- Flash attention algorithm is used for memory-efficient attention calculation
- cuBLAS is used for matrix multiplications
- GELU activation is implemented for feed-forward networks
- Memory is managed efficiently with proper allocation and deallocation
