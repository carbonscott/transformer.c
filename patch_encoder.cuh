/*
Patch encoder for vision transformers.
This file implements CUDA kernels for projecting image patches to the embedding dimension.
*/
#ifndef PATCH_ENCODER_CUH
#define PATCH_ENCODER_CUH

#include "cuda_common.h"
#include "cuda_utils.cuh"

// Linear projection of patches to embedding dimension
__global__ void patch_embedding_kernel(floatX* out,
                                     const floatX* inp, const floatX* patch_proj, 
                                     int B, int N_patches, int P, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int total_size = B * N_patches * C;
    if (idx >= total_size) { return; }

    // Calculate indices
    int bn = idx / C;
    int b = bn / N_patches;
    int n = bn % N_patches;
    int c = idx % C;

    // Linear projection: sum(inp[n,p] * patch_proj[p,c])
    float sum = 0.0f;
    for (int p = 0; p < P; p++) {
        sum += (float)inp[b * N_patches * P + n * P + p] * 
               (float)patch_proj[p * C + c];
    }

    // Store result
    out[b * N_patches * C + n * C + c] = (floatX)sum;
}

// Host function for patch embedding
void patch_embedding_forward(floatX* out, const floatX* patches, const floatX* patch_proj,
                           int B, int N_patches, int P, int C, cudaStream_t stream = 0) {
    const int block_size = 256;
    const int total_size = B * N_patches * C;
    const int grid_size = CEIL_DIV(total_size, (int)(block_size * x128::size));

    patch_embedding_kernel<<<grid_size, block_size, 0, stream>>>(
        out, patches, patch_proj, B, N_patches, P, C);
    cudaCheck(cudaGetLastError());
}

// Backward pass for patch embedding (gradient w.r.t patches)
__global__ void patch_embedding_backward_patches_kernel(floatX* d_patches,
                                                     const floatX* d_out, 
                                                     const floatX* patch_proj,
                                                     int B, int N_patches, int P, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = B * N_patches * P;
    if (idx >= total_size) { return; }

    // Calculate indices
    int b = idx / (N_patches * P);
    int n = (idx / P) % N_patches;
    int p = idx % P;

    // Calculate gradient for this patch element
    float grad = 0.0f;
    for (int c = 0; c < C; c++) {
        grad += (float)d_out[b * N_patches * C + n * C + c] * 
                (float)patch_proj[p * C + c];
    }

    // Store gradient
    d_patches[b * N_patches * P + n * P + p] = (floatX)grad;
}

// Backward pass for patch embedding (gradient w.r.t projection matrix)
__global__ void patch_embedding_backward_proj_kernel(floatX* d_proj,
                                                  const floatX* d_out,
                                                  const floatX* patches,
                                                  int B, int N_patches, int P, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = P * C;
    if (idx >= total_size) { return; }

    // Calculate indices
    int p = idx / C;
    int c = idx % C;

    // Calculate gradient for this projection weight
    float grad = 0.0f;
    for (int b = 0; b < B; b++) {
        for (int n = 0; n < N_patches; n++) {
            grad += (float)d_out[b * N_patches * C + n * C + c] * 
                    (float)patches[b * N_patches * P + n * P + p];
        }
    }

    // Store gradient (accumulate, don't overwrite)
    atomicAdd(&d_proj[p * C + c], (floatX)grad);
}

// Host function for patch embedding backward pass
void patch_embedding_backward(floatX* d_patches, floatX* d_proj,
                            const floatX* d_out, const floatX* patches, 
                            const floatX* patch_proj,
                            int B, int N_patches, int P, int C,
                            cudaStream_t stream = 0) {
    const int block_size = 256;

    // Compute gradients w.r.t patches
    const int total_patches = B * N_patches * P;
    const int grid_size_patches = CEIL_DIV(total_patches, block_size);

    patch_embedding_backward_patches_kernel<<<grid_size_patches, block_size, 0, stream>>>(
        d_patches, d_out, patch_proj, B, N_patches, P, C);
    cudaCheck(cudaGetLastError());

    // Compute gradients w.r.t projection matrix
    const int total_proj = P * C;
    const int grid_size_proj = CEIL_DIV(total_proj, block_size);

    patch_embedding_backward_proj_kernel<<<grid_size_proj, block_size, 0, stream>>>(
        d_proj, d_out, patches, B, N_patches, P, C);
    cudaCheck(cudaGetLastError());
}

#endif // PATCH_ENCODER_CUH
