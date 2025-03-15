#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "transformer_block.h"
#include "patch_processor.cuh"
#include "patch_encoder.cuh"

int main() {
    // Image dimensions
    const int B = 30;            // Batch size
    const int H = 192;           // Height
    const int W = 168;           // Width
    const int patch_size = 12;   // Size of each patch

    // Calculate derived dimensions
    const int patches_h = CEIL_DIV(H, patch_size);
    const int patches_w = CEIL_DIV(W, patch_size);
    const int num_patches = patches_h * patches_w;
    const int patch_dim = patch_size * patch_size;

    // Transformer embedding dimension
    const int C = 768;
    const int num_heads = 32;
    const int head_size = C / num_heads;

    printf("Configuration:\n");
    printf("  Batch size: %d\n", B);
    printf("  Image dimensions: %d x %d\n", H, W);
    printf("  Patch size: %d x %d\n", patch_size, patch_size);
    printf("  Patches per image: %d x %d = %d\n", patches_h, patches_w, num_patches);
    printf("  Patch dimension: %d\n", patch_dim);
    printf("  Embedding dimension: %d\n", C);
    printf("  Estimated shared memory needed: %d bytes\n", 2 * num_patches * head_size * 4);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Allocate memory for input 1D signal
    float* h_raw_signal = (float*)malloc(B * H * W * sizeof(float));

    // Initialize with random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < B * H * W; i++) {
        h_raw_signal[i] = dist(gen);
    }

    // Allocate device memory for all stages of processing
    floatX* d_raw_signal;
    floatX* d_images;
    floatX* d_patches;
    floatX* d_norm_patches;
    float* d_patch_means;
    float* d_patch_vars;
    floatX* d_embedded_patches;
    floatX* d_patch_position_embeddings;
    floatX* d_transformer_input;
    floatX* d_transformer_output;

    // Patch projection weights
    floatX* d_patch_proj;

    // Allocate memory on device
    cudaMalloc(&d_raw_signal, B * H * W * sizeof(floatX));
    cudaMalloc(&d_images, B * H * W * sizeof(floatX));
    cudaMalloc(&d_patches, B * num_patches * patch_dim * sizeof(floatX));
    cudaMalloc(&d_norm_patches, B * num_patches * patch_dim * sizeof(floatX));
    cudaMalloc(&d_patch_means, B * num_patches * sizeof(float));
    cudaMalloc(&d_patch_vars, B * num_patches * sizeof(float));
    cudaMalloc(&d_embedded_patches, B * num_patches * C * sizeof(floatX));
    cudaMalloc(&d_patch_position_embeddings, num_patches * C * sizeof(floatX));
    cudaMalloc(&d_transformer_input, B * num_patches * C * sizeof(floatX));
    cudaMalloc(&d_transformer_output, B * num_patches * C * sizeof(floatX));

    // Patch projection matrix (from patch_dim to C)
    cudaMalloc(&d_patch_proj, patch_dim * C * sizeof(floatX));

    // Initialize patch projection with small random values
    floatX* h_patch_proj = (floatX*)malloc(patch_dim * C * sizeof(floatX));
    std::normal_distribution<float> proj_dist(0.0f, 0.02f); // Small initialization
    for (int i = 0; i < patch_dim * C; i++) {
        h_patch_proj[i] = (floatX)proj_dist(gen);
    }
    cudaMemcpy(d_patch_proj, h_patch_proj, patch_dim * C * sizeof(floatX), cudaMemcpyHostToDevice);

    // Initialize position embeddings
    floatX* h_pos_embeddings = (floatX*)malloc(num_patches * C * sizeof(floatX));
    for (int i = 0; i < num_patches * C; i++) {
        h_pos_embeddings[i] = (floatX)proj_dist(gen);
    }
    cudaMemcpy(d_patch_position_embeddings, h_pos_embeddings, num_patches * C * sizeof(floatX), cudaMemcpyHostToDevice);

    // Copy input signal to device
    for (int i = 0; i < B * H * W; i++) {
        ((floatX*)h_raw_signal)[i] = (floatX)h_raw_signal[i]; // Convert to floatX type
    }
    cudaMemcpy(d_raw_signal, h_raw_signal, B * H * W * sizeof(floatX), cudaMemcpyHostToDevice);

    printf("Processing pipeline:\n");
    printf("  1. Reshaping 1D signal to batched 2D images...\n");
    // Reshape 1D signal to batched 2D images
    reshape_signal(d_images, d_raw_signal, 1, B, H, W, stream);

    printf("  2. Extracting patches from images...\n");
    // Extract patches from images
    patchify(d_patches, d_images, B, H, W, patch_size, stream);

    printf("  3. Normalizing patches...\n");
    // Normalize patches
    normalize_patches(d_norm_patches, d_patch_means, d_patch_vars, d_patches, B, num_patches, patch_dim, stream);

    printf("  4. Embedding patches to dimension %d...\n", C);
    // Project patches to embedding dimension
    patch_embedding_forward(d_embedded_patches, d_norm_patches, d_patch_proj, B, num_patches, patch_dim, C, stream);

    printf("  5. Adding position embeddings...\n");
    // Add position embeddings
    add_position_embeddings(d_transformer_input, d_embedded_patches, d_patch_position_embeddings, B, num_patches, C, stream);

    // Initialize transformer with updated num_patches
    printf("  6. Initializing transformer with sequence length = %d (num_patches)...\n", num_patches);
    TransformerDims transformer_dims = {
        .B = B,               // Batch size
        .T = num_patches,     // Sequence length (number of patches)
        .C = C,               // Embedding dimension
        .NH = num_heads,      // Number of attention heads
        .HS = head_size       // Head size
    };

    TransformerBlock transformer;
    transformer_block_init(&transformer, transformer_dims);

    // Initialize transformer weights
    size_t total_weight_params = 0;
    total_weight_params += 4 * transformer_dims.NH * transformer_dims.HS * C; // Attention weights
    total_weight_params += 4 * C; // LayerNorm weights
    total_weight_params += C * 4 * C + 4 * C + 4 * C * C + C; // MLP weights

    printf("     Transformer parameters: %zu\n", total_weight_params);
    float* h_weights = (float*)malloc(total_weight_params * sizeof(float));
    for (size_t i = 0; i < total_weight_params; i++) {
        h_weights[i] = proj_dist(gen);
    }

    // Load weights into transformer
    transformer_block_load_weights(&transformer, h_weights);

    // Debug prints before forward pass
    printf("  7. Running transformer forward pass...\n");
    printf("     Input dimensions: B=%d, T=%d, C=%d\n", B, num_patches, C);

    // Run transformer forward pass
    transformer_block_forward(&transformer, d_transformer_output, d_transformer_input, stream);

    // Wait for all operations to complete
    cudaStreamSynchronize(stream);

    // Copy back a small sample of the output for verification
    floatX* h_output_sample = (floatX*)malloc(10 * sizeof(floatX));
    cudaMemcpy(h_output_sample, d_transformer_output, 10 * sizeof(floatX), cudaMemcpyDeviceToHost);

    // Report output size information
    size_t total_elements = B * num_patches * C;
    size_t total_bytes = total_elements * sizeof(floatX);
    printf("\nTransformer output: [%d, %d, %d] shape with %zu total elements (%.2f MB)\n",
           B, num_patches, C, total_elements, total_bytes / (1024.0f * 1024.0f));

    printf("\nSample output (first 10 values of first patch):\n");
    for (int i = 0; i < 10; i++) {
        printf("  %f\n", (float)h_output_sample[i]);
    }

    // Clean up
    printf("\nCleaning up resources...\n");
    free(h_raw_signal);
    free(h_patch_proj);
    free(h_pos_embeddings);
    free(h_weights);
    free(h_output_sample);

    cudaFree(d_raw_signal);
    cudaFree(d_images);
    cudaFree(d_patches);
    cudaFree(d_norm_patches);
    cudaFree(d_patch_means);
    cudaFree(d_patch_vars);
    cudaFree(d_embedded_patches);
    cudaFree(d_patch_position_embeddings);
    cudaFree(d_transformer_input);
    cudaFree(d_transformer_output);
    cudaFree(d_patch_proj);

    transformer_block_free(&transformer);
    cudaStreamDestroy(stream);

    printf("Vision transformer processing completed successfully.\n");
    return 0;
}
