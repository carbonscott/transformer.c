/*
Patch-based processing for vision transformers.
This file implements CUDA kernels for:
1. Reshaping 1D detector signals to 2D images
2. Extracting patches from 2D images
3. Normalizing patches
*/
#ifndef PATCH_PROCESSOR_CUH
#define PATCH_PROCESSOR_CUH

#include "cuda_common.h"
#include "cuda_utils.cuh"

// Reshape 1D signal into batched 2D images
// Input: 1D signal with dimensions (num_fpga * num_asic * height * width)
// Output: Batched 2D images with dimensions (num_fpga * num_asic, height, width)
__global__ void reshape_signal_kernel(floatX* out, const floatX* inp,
                                    int num_fpga, int num_asic,
                                    int height, int width) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_size = num_fpga * num_asic * height * width;

    if (idx >= total_size) return;

    // Calculate source and destination indices
    int batch = idx / (height * width);
    int h = (idx % (height * width)) / width;
    int w = idx % width;

    // Copy data from flattened 1D to batched 2D format
    out[batch * height * width + h * width + w] = inp[idx];
}

// Extract patches from 2D images
// Input: Batched 2D images with dimensions (B, H, W)
// Output: Patches with dimensions (B, num_patches, patch_size*patch_size)
__global__ void patchify_kernel(floatX* out, const floatX* inp,
                              int B, int H, int W, 
                              int patch_size, int patches_h, int patches_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_patches = B * patches_h * patches_w;
    int patch_area = patch_size * patch_size;

    if (idx >= total_patches) return;

    // Calculate indices
    int b = idx / (patches_h * patches_w);
    int patch_idx = idx % (patches_h * patches_w);
    int ph = patch_idx / patches_w;
    int pw = patch_idx % patches_w;

    // Calculate top-left corner of this patch
    int h_start = ph * patch_size;
    int w_start = pw * patch_size;

    // Extract the patch
    for (int i = 0; i < patch_size; i++) {
        for (int j = 0; j < patch_size; j++) {
            int h_pos = h_start + i;
            int w_pos = w_start + j;

            // Handle boundary conditions (if image doesn't divide evenly by patch_size)
            if (h_pos < H && w_pos < W) {
                out[b * patches_h * patches_w * patch_area + 
                    patch_idx * patch_area + 
                    i * patch_size + j] = 
                    inp[b * H * W + h_pos * W + w_pos];
            } else {
                // Pad with zeros for patches that go beyond the image boundary
                out[b * patches_h * patches_w * patch_area + 
                    patch_idx * patch_area + 
                    i * patch_size + j] = 0.0f;
            }
        }
    }
}

// Normalize patches independently (mean=0, var=1)
// Input: Patches with dimensions (B, num_patches, patch_dim)
// Output: Normalized patches with the same dimensions
// Also outputs the mean and variance for each patch for later denormalization
__global__ void normalize_patches_kernel(floatX* out, float* means, float* vars,
                                       const floatX* inp,
                                       int B, int num_patches, int patch_dim) {
    // Each thread handles one patch
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * num_patches) return;

    int b = idx / num_patches;
    int p = idx % num_patches;

    // Calculate mean
    float mean = 0.0f;
    for (int i = 0; i < patch_dim; i++) {
        mean += (float)inp[b * num_patches * patch_dim + p * patch_dim + i];
    }
    mean /= patch_dim;

    // Calculate variance
    float var = 0.0f;
    for (int i = 0; i < patch_dim; i++) {
        float diff = (float)inp[b * num_patches * patch_dim + p * patch_dim + i] - mean;
        var += diff * diff;
    }
    var /= patch_dim;

    // Store mean and variance for later denormalization
    means[b * num_patches + p] = mean;
    vars[b * num_patches + p] = var;

    // Normalize the patch
    for (int i = 0; i < patch_dim; i++) {
        float normalized = ((float)inp[b * num_patches * patch_dim + p * patch_dim + i] - mean) / 
                           sqrtf(var + 1e-6f);
        out[b * num_patches * patch_dim + p * patch_dim + i] = (floatX)normalized;
    }
}

// Denormalize patches (for reconstruction)
__global__ void denormalize_patches_kernel(floatX* out, const floatX* inp,
                                         const float* means, const float* vars,
                                         int B, int num_patches, int patch_dim) {
    // Each thread handles one patch
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * num_patches) return;

    int b = idx / num_patches;
    int p = idx % num_patches;

    float mean = means[b * num_patches + p];
    float var = vars[b * num_patches + p];

    // Denormalize the patch
    for (int i = 0; i < patch_dim; i++) {
        float denormalized = (float)inp[b * num_patches * patch_dim + p * patch_dim + i] * 
                             sqrtf(var + 1e-6f) + mean;
        out[b * num_patches * patch_dim + p * patch_dim + i] = (floatX)denormalized;
    }
}

// Add position embeddings to patch embeddings
__global__ void add_position_embeddings_kernel(floatX* out, 
                                             const floatX* patch_embeddings,
                                             const floatX* pos_embeddings,
                                             int B, int num_patches, int C) {
    // Each thread handles x128::size channels for one patch in one batch
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int total_elements = B * num_patches * C;

    if (idx >= total_elements) return;

    int bn = idx / C;
    int b = bn / num_patches;
    int n = bn % num_patches;
    int c = idx % C;

    // Load patch embeddings and position embeddings
    x128 patch_emb = load128(patch_embeddings + b * num_patches * C + n * C + c);
    x128 pos_emb = load128(pos_embeddings + n * C + c);

    // Add them together
    x128 result;
    for (int i = 0; i < x128::size; i++) {
        result[i] = (floatX)((float)patch_emb[i] + (float)pos_emb[i]);
    }

    // Store the result
    store128(out + b * num_patches * C + n * C + c, result);
}

// Host function for reshaping 1D signal to batched 2D
void reshape_signal(floatX* out, const floatX* inp,
                   int num_fpga, int num_asic, int height, int width,
                   cudaStream_t stream = 0) {
    int total_elements = num_fpga * num_asic * height * width;
    int block_size = 256;
    int grid_size = CEIL_DIV(total_elements, block_size);

    reshape_signal_kernel<<<grid_size, block_size, 0, stream>>>(
        out, inp, num_fpga, num_asic, height, width);
    cudaCheck(cudaGetLastError());
}

// Host function for patchifying images
void patchify(floatX* out, const floatX* inp,
             int B, int H, int W, int patch_size,
             cudaStream_t stream = 0) {
    int patches_h = CEIL_DIV(H, patch_size);
    int patches_w = CEIL_DIV(W, patch_size);
    int total_patches = B * patches_h * patches_w;
    int block_size = 256;
    int grid_size = CEIL_DIV(total_patches, block_size);

    patchify_kernel<<<grid_size, block_size, 0, stream>>>(
        out, inp, B, H, W, patch_size, patches_h, patches_w);
    cudaCheck(cudaGetLastError());
}

// Host function for normalizing patches
void normalize_patches(floatX* out, float* means, float* vars, const floatX* inp,
                      int B, int num_patches, int patch_dim,
                      cudaStream_t stream = 0) {
    int block_size = 256;
    int grid_size = CEIL_DIV(B * num_patches, block_size);

    normalize_patches_kernel<<<grid_size, block_size, 0, stream>>>(
        out, means, vars, inp, B, num_patches, patch_dim);
    cudaCheck(cudaGetLastError());
}

// Host function for denormalizing patches
void denormalize_patches(floatX* out, const floatX* inp,
                        const float* means, const float* vars,
                        int B, int num_patches, int patch_dim,
                        cudaStream_t stream = 0) {
    int block_size = 256;
    int grid_size = CEIL_DIV(B * num_patches, block_size);

    denormalize_patches_kernel<<<grid_size, block_size, 0, stream>>>(
        out, inp, means, vars, B, num_patches, patch_dim);
    cudaCheck(cudaGetLastError());
}

// Host function for adding position embeddings
void add_position_embeddings(floatX* out, const floatX* patch_embeddings,
                            const floatX* pos_embeddings,
                            int B, int num_patches, int C,
                            cudaStream_t stream = 0) {
    int block_size = 256;
    int total_elements = B * num_patches * C;
    int grid_size = CEIL_DIV(total_elements, block_size * x128::size);

    add_position_embeddings_kernel<<<grid_size, block_size, 0, stream>>>(
        out, patch_embeddings, pos_embeddings, B, num_patches, C);
    cudaCheck(cudaGetLastError());
}

#endif // PATCH_PROCESSOR_CUH
