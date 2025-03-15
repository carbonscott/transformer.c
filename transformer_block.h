#ifndef TRANSFORMER_BLOCK_H
#define TRANSFORMER_BLOCK_H

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <stdio.h>
#include <assert.h>
#include "cuda_common.h"

// Set default precision to BF16 if not specified
#ifndef ENABLE_FP32
#ifndef ENABLE_FP16
#define ENABLE_BF16
#endif
#endif

// WARP_SIZE is not a compile time constant, defining here
#define WARP_SIZE 32U
#define CEIL_DIV(M, N) (((M) + (N)-1) / (N))

// ----------------------------------------------------------------------------
// Structure to hold tensor dimensions
typedef struct {
    int B;  // batch size
    int T;  // sequence length
    int C;  // embedding dimension
    int NH; // number of attention heads
    int HS; // head size (C / NH)
} TransformerDims;

// ----------------------------------------------------------------------------
// Structure to hold transformer block parameters
typedef struct {
    // Attention weights
    floatX* query_weight;
    floatX* key_weight;
    floatX* value_weight;
    floatX* attn_output_weight;

    // Layer norms
    floatX* ln1_weight;
    floatX* ln1_bias;
    floatX* ln2_weight;
    floatX* ln2_bias;

    // MLP weights
    floatX* mlp_fc_weight;
    floatX* mlp_fc_bias;
    floatX* mlp_proj_weight;
    floatX* mlp_proj_bias;

    // Dimensions
    TransformerDims dims;

    // cuBLAS handles
    cublasHandle_t cublas_handle;
    cublasLtHandle_t cublaslt_handle;
    void* cublaslt_workspace;
    size_t cublaslt_workspace_size;
} TransformerBlock;

// Forward declarations for key functions
void transformer_block_init(TransformerBlock* block, TransformerDims dims);
void transformer_block_forward(TransformerBlock* block, floatX* output, floatX* input, cudaStream_t stream);
void transformer_block_free(TransformerBlock* block);
void transformer_block_load_weights(TransformerBlock* block, const float* weights_data);

#endif // TRANSFORMER_BLOCK_H
