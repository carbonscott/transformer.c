#include "transformer_block.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// Warp/Block reduction functions
__device__ inline float warpReduceSum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_xor_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

__device__ inline float warpReduceMax(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_xor_sync(0xFFFFFFFF, val, offset));
    }
    return val;
}

// ----------------------------------------------------------------------------
// GELU activation function
__global__ void gelu_forward_kernel(floatX* out, const floatX* inp, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= N) return;

    x128 packed_out;
    x128 packed_inp = load128(inp + idx);
    for(int k = 0; k < packed_inp.size; ++k) {
        float xi = (float)packed_inp[k];
        float cube = 0.044715f * xi * xi * xi;
        packed_out[k] = (floatX)(0.5f * xi * (1.0f + tanhf(0.7978845608f * (xi + cube))));
    }
    store128(out + idx, packed_out);
}

void gelu_forward(floatX* out, const floatX* inp, int N, cudaStream_t stream) {
    const int block_size = 512;
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    gelu_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp, N);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Layer normalization
__global__ void layernorm_forward_kernel(floatX* out, float* mean, float* rstd,
                                 const floatX* inp, const floatX* weight,
                                 const floatX* bias, int N, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= N) return;

    // The row of input that this thread is responsible for
    const floatX* x = inp + idx * C;

    // Calculate mean
    float sum = 0.0f;
    for (int i = 0; i < C; i++) {
        sum += (float)x[i];
    }
    float m = sum / C;
    if(mean != nullptr) {
        mean[idx] = m;
    }

    // Calculate standard deviation
    sum = 0.0f;
    for (int i = 0; i < C; i++) {
        float diff = (float)x[i] - m;
        sum += diff * diff;
    }
    float s = rsqrtf(sum / C + 1e-5f);
    if(rstd != nullptr) {
        rstd[idx] = s;
    }

    // Final normalization and scaling by weight/bias
    floatX* o = out + idx * C;
    for (int c = 0; c < C; c++) {
        float n = s * ((float)x[c] - m);
        o[c] = (floatX)(n * (float)weight[c] + (float)bias[c]);
    }
}

void layernorm_forward(floatX* out, float* mean, float* rstd,
                       floatX* inp, const floatX* weight, const floatX* bias,
                       int B, int T, int C, cudaStream_t stream) {
    const int block_size = 256;
    const int N = B * T;
    const int grid_size = CEIL_DIV(N, block_size);
    layernorm_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, mean, rstd, inp, weight, bias, N, C);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Residual connection
__global__ void residual_forward_kernel(floatX* out, const floatX* inp1, const floatX* inp2, int N) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    if (idx >= N) return;

    x128 packed_out;
    x128 packed_inp1 = load128(inp1 + idx);
    x128 packed_inp2 = load128(inp2 + idx);
    for (int k = 0; k < packed_inp1.size; k++) {
        packed_out[k] = (floatX)((float)packed_inp1[k] + (float)packed_inp2[k]);
    }
    store128(out + idx, packed_out);
}

void residual_forward(floatX* out, const floatX* inp1, const floatX* inp2, int N, cudaStream_t stream) {
    const int block_size = 256;
    const int grid_size = CEIL_DIV(N, block_size * x128::size);
    residual_forward_kernel<<<grid_size, block_size, 0, stream>>>(out, inp1, inp2, N);
    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Self-attention mechanism

// Query, Key, Value transformation kernel
__global__ void qkv_transform_kernel(floatX* q, floatX* k, floatX* v,
                                    const floatX* inp,
                                    const floatX* q_weight, const floatX* k_weight, const floatX* v_weight,
                                    int B, int T, int C, int NH, int HS) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * T * NH * HS) return;

    int h = (idx / (T * HS)) % NH;
    int t = (idx / HS) % T;
    int b = idx / (NH * T * HS);
    int hs = idx % HS;

    float sum_q = 0.0f, sum_k = 0.0f, sum_v = 0.0f;
    for (int c = 0; c < C; c++) {
        float x = (float)inp[b * T * C + t * C + c];
        sum_q += x * (float)q_weight[h * HS * C + hs * C + c];
        sum_k += x * (float)k_weight[h * HS * C + hs * C + c];
        sum_v += x * (float)v_weight[h * HS * C + hs * C + c];
    }

    q[b * NH * T * HS + h * T * HS + t * HS + hs] = (floatX)sum_q;
    k[b * NH * T * HS + h * T * HS + t * HS + hs] = (floatX)sum_k;
    v[b * NH * T * HS + h * T * HS + t * HS + hs] = (floatX)sum_v;
}

// Simplified flash attention kernel (causal attention)
__global__ void flash_attention_kernel(floatX* out, const floatX* q, const floatX* k, const floatX* v,
                                      int B, int NH, int T, int HS) {
    int b = blockIdx.z / NH;
    int h = blockIdx.z % NH;
    int t = blockIdx.y;

    extern __shared__ float shmem[];
    float* k_cache = shmem;
    float* v_cache = &shmem[T * HS];

    // Load keys and values into shared memory
    for (int i = threadIdx.x; i < T * HS; i += blockDim.x) {
        int t_idx = i / HS;
        int hs_idx = i % HS;
        if (t_idx <= t) { // Causal mask
            k_cache[i] = (float)k[b * NH * T * HS + h * T * HS + t_idx * HS + hs_idx];
            v_cache[i] = (float)v[b * NH * T * HS + h * T * HS + t_idx * HS + hs_idx];
        }
    }
    __syncthreads();

    // Compute attention for each head dimension
    for (int hs = threadIdx.x; hs < HS; hs += blockDim.x) {
        float q_val = (float)q[b * NH * T * HS + h * T * HS + t * HS + hs];
        float max_score = -1e9f;
        float sum_exp = 0.0f;
        float output = 0.0f;

        // Attention calculation
        for (int s = 0; s <= t; s++) {
            float score = 0.0f;
            for (int d = 0; d < HS; d++) {
                score += q_val * k_cache[s * HS + d] / sqrtf(HS);
            }

            // Update max_score for numerical stability
            max_score = fmaxf(max_score, score);
        }

        // Second pass to calculate softmax
        for (int s = 0; s <= t; s++) {
            float score = 0.0f;
            for (int d = 0; d < HS; d++) {
                score += q_val * k_cache[s * HS + d] / sqrtf(HS);
            }

            float exp_score = expf(score - max_score);
            sum_exp += exp_score;
            output += exp_score * v_cache[s * HS + hs];
        }

        // Apply scaling and write output
        output /= sum_exp;
        out[b * NH * T * HS + h * T * HS + t * HS + hs] = (floatX)output;
    }
}

// Matrix multiplication wrapper
void matmul(floatX* C, const floatX* A, const floatX* B,
            int M, int N, int K, cublasHandle_t handle, cudaStream_t stream) {
    cublasSetStream(handle, stream);

    #if defined(ENABLE_FP32)
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N);
    #elif defined(ENABLE_FP16)
    const __half alpha = __float2half(1.0f);
    const __half beta = __float2half(0.0f);
    cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, N,
                A, K,
                &beta,
                C, N);
    #else // BF16
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,
                &alpha,
                B, CUDA_R_16BF, N,
                A, CUDA_R_16BF, K,
                &beta,
                C, CUDA_R_16BF, N,
                CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    #endif

    cudaCheck(cudaGetLastError());
}

// ----------------------------------------------------------------------------
// Initialize the transformer block
void transformer_block_init(TransformerBlock* block, TransformerDims dims) {
    block->dims = dims;

    // Initialize cuBLAS handles
    cublasCreate(&block->cublas_handle);
    cublasLtCreate(&block->cublaslt_handle);

    // Allocate workspace for cublasLt
    block->cublaslt_workspace_size = 32 * 1024 * 1024; // 32 MiB
    cudaMalloc(&block->cublaslt_workspace, block->cublaslt_workspace_size);

    // Allocate memory for weights
    size_t C = dims.C;
    size_t NH = dims.NH;
    size_t HS = dims.HS;

    // Attention weights
    cudaMalloc(&block->query_weight, NH * HS * C * sizeof(floatX));
    cudaMalloc(&block->key_weight, NH * HS * C * sizeof(floatX));
    cudaMalloc(&block->value_weight, NH * HS * C * sizeof(floatX));
    cudaMalloc(&block->attn_output_weight, NH * HS * C * sizeof(floatX));

    // Layer norm weights
    cudaMalloc(&block->ln1_weight, C * sizeof(floatX));
    cudaMalloc(&block->ln1_bias, C * sizeof(floatX));
    cudaMalloc(&block->ln2_weight, C * sizeof(floatX));
    cudaMalloc(&block->ln2_bias, C * sizeof(floatX));

    // MLP weights
    int mlp_hidden = 4 * C; // Standard GPT-2 uses 4x expansion factor
    cudaMalloc(&block->mlp_fc_weight, C * mlp_hidden * sizeof(floatX));
    cudaMalloc(&block->mlp_fc_bias, mlp_hidden * sizeof(floatX));
    cudaMalloc(&block->mlp_proj_weight, mlp_hidden * C * sizeof(floatX));
    cudaMalloc(&block->mlp_proj_bias, C * sizeof(floatX));
}

// Forward pass through the transformer block
void transformer_block_forward(TransformerBlock* block, floatX* output, floatX* input, cudaStream_t stream) {
    TransformerDims dims = block->dims;
    int B = dims.B;
    int T = dims.T;
    int C = dims.C;
    int NH = dims.NH;
    int HS = dims.HS;

    // Allocate temporary buffers
    floatX *x1, *x2, *q, *k, *v, *attn_out, *mlp_hidden, *mlp_out;
    float *ln1_mean, *ln1_rstd, *ln2_mean, *ln2_rstd;

    cudaMalloc(&x1, B * T * C * sizeof(floatX));
    cudaMalloc(&x2, B * T * C * sizeof(floatX));
    cudaMalloc(&q, B * NH * T * HS * sizeof(floatX));
    cudaMalloc(&k, B * NH * T * HS * sizeof(floatX));
    cudaMalloc(&v, B * NH * T * HS * sizeof(floatX));
    cudaMalloc(&attn_out, B * NH * T * HS * sizeof(floatX));
    cudaMalloc(&mlp_hidden, B * T * 4 * C * sizeof(floatX));
    cudaMalloc(&mlp_out, B * T * C * sizeof(floatX));
    cudaMalloc(&ln1_mean, B * T * sizeof(float));
    cudaMalloc(&ln1_rstd, B * T * sizeof(float));
    cudaMalloc(&ln2_mean, B * T * sizeof(float));
    cudaMalloc(&ln2_rstd, B * T * sizeof(float));

    // 1. Layer Norm 1
    layernorm_forward(x1, ln1_mean, ln1_rstd, input, block->ln1_weight, block->ln1_bias, B, T, C, stream);

    // 2. Self-attention
    int block_size = 256;
    int grid_size = CEIL_DIV(B * T * NH * HS, block_size);
    qkv_transform_kernel<<<grid_size, block_size, 0, stream>>>(
        q, k, v, x1, 
        block->query_weight, block->key_weight, block->value_weight,
        B, T, C, NH, HS);

    // Flash attention (simplified)
    size_t shared_mem_size = 2 * T * HS * sizeof(float); // For k and v cache
    dim3 grid(1, T, B * NH);
    flash_attention_kernel<<<grid, block_size, shared_mem_size, stream>>>(attn_out, q, k, v, B, NH, T, HS);

    // Project attention output back to original dimension
    matmul(x2, attn_out, block->attn_output_weight, B * T, C, NH * HS, block->cublas_handle, stream);

    // 3. Residual connection
    residual_forward(x1, input, x2, B * T * C, stream);

    // 4. Layer Norm 2
    layernorm_forward(x2, ln2_mean, ln2_rstd, x1, block->ln2_weight, block->ln2_bias, B, T, C, stream);

    // 5. Feed-forward network
    int mlp_hidden_dim = 4 * C; // Standard GPT-2 uses 4x expansion factor
    matmul(mlp_hidden, x2, block->mlp_fc_weight, B * T, mlp_hidden_dim, C, block->cublas_handle, stream);

    // Add bias to hidden layer
    grid_size = CEIL_DIV(B * T * mlp_hidden_dim, block_size);
    // Simple bias add kernel would go here

    // Apply GeLU activation
    gelu_forward(mlp_hidden, mlp_hidden, B * T * mlp_hidden_dim, stream);

    // Project back to original dimension
    matmul(mlp_out, mlp_hidden, block->mlp_proj_weight, B * T, C, mlp_hidden_dim, block->cublas_handle, stream);

    // 6. Final residual connection
    residual_forward(output, x1, mlp_out, B * T * C, stream);

    // Free temporary buffers
    cudaFree(x1);
    cudaFree(x2);
    cudaFree(q);
    cudaFree(k);
    cudaFree(v);
    cudaFree(attn_out);
    cudaFree(mlp_hidden);
    cudaFree(mlp_out);
    cudaFree(ln1_mean);
    cudaFree(ln1_rstd);
    cudaFree(ln2_mean);
    cudaFree(ln2_rstd);
}

// Free transformer block resources
void transformer_block_free(TransformerBlock* block) {
    // Free all allocated weights
    cudaFree(block->query_weight);
    cudaFree(block->key_weight);
    cudaFree(block->value_weight);
    cudaFree(block->attn_output_weight);

    cudaFree(block->ln1_weight);
    cudaFree(block->ln1_bias);
    cudaFree(block->ln2_weight);
    cudaFree(block->ln2_bias);

    cudaFree(block->mlp_fc_weight);
    cudaFree(block->mlp_fc_bias);
    cudaFree(block->mlp_proj_weight);
    cudaFree(block->mlp_proj_bias);

    // Free cuBLAS resources
    cudaFree(block->cublaslt_workspace);
    cublasLtDestroy(block->cublaslt_handle);
    cublasDestroy(block->cublas_handle);
}

// Function to load weights from file or memory
void transformer_block_load_weights(TransformerBlock* block, const float* weights_data) {
    TransformerDims dims = block->dims;
    int C = dims.C;
    int NH = dims.NH;
    int HS = dims.HS;

    size_t offset = 0;

    // Helper function to copy and convert weights if needed
    auto copy_weights = [](floatX* dst, const float* src, size_t count) {
        #if defined(ENABLE_FP32)
        cudaMemcpy(dst, src, count * sizeof(float), cudaMemcpyHostToDevice);
        #else
        // Convert from FP32 to FP16/BF16
        float* host_buffer = new float[count];
        memcpy(host_buffer, src, count * sizeof(float));

        floatX* device_buffer;
        cudaMalloc(&device_buffer, count * sizeof(floatX));

        // Convert from float to floatX on GPU
        // You'd need a proper conversion kernel here

        delete[] host_buffer;
        cudaFree(device_buffer);
        #endif
    };

    // Copy weights for attention
    copy_weights(block->query_weight, weights_data + offset, NH * HS * C);
    offset += NH * HS * C;

    copy_weights(block->key_weight, weights_data + offset, NH * HS * C);
    offset += NH * HS * C;

    copy_weights(block->value_weight, weights_data + offset, NH * HS * C);
    offset += NH * HS * C;

    copy_weights(block->attn_output_weight, weights_data + offset, NH * HS * C);
    offset += NH * HS * C;

    // Copy weights for layer norm 1
    copy_weights(block->ln1_weight, weights_data + offset, C);
    offset += C;

    copy_weights(block->ln1_bias, weights_data + offset, C);
    offset += C;

    // Copy weights for layer norm 2
    copy_weights(block->ln2_weight, weights_data + offset, C);
    offset += C;

    copy_weights(block->ln2_bias, weights_data + offset, C);
    offset += C;

    // Copy weights for MLP
    int mlp_hidden = 4 * C;

    copy_weights(block->mlp_fc_weight, weights_data + offset, C * mlp_hidden);
    offset += C * mlp_hidden;

    copy_weights(block->mlp_fc_bias, weights_data + offset, mlp_hidden);
    offset += mlp_hidden;

    copy_weights(block->mlp_proj_weight, weights_data + offset, mlp_hidden * C);
    offset += mlp_hidden * C;

    copy_weights(block->mlp_proj_bias, weights_data + offset, C);
    offset += C;
}
