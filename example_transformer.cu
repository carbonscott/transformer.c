#include <stdio.h>
#include <stdlib.h>
#include <random>
#include "transformer_block.h"

// Helper function to initialize random weights
void initialize_random_weights(float* weights, size_t size) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, 0.02f); // Small standard deviation for stable initialization

    for (size_t i = 0; i < size; i++) {
        weights[i] = dist(gen);
    }
}

// Helper function to calculate the total number of weight parameters
size_t calculate_weight_size(const TransformerDims& dims) {
    int C = dims.C;
    int NH = dims.NH;
    int HS = dims.HS;
    int mlp_hidden = 4 * C;

    size_t total = 0;

    // Attention weights
    total += NH * HS * C; // query
    total += NH * HS * C; // key
    total += NH * HS * C; // value
    total += NH * HS * C; // output projection

    // Layer norms
    total += C; // ln1 weight
    total += C; // ln1 bias
    total += C; // ln2 weight
    total += C; // ln2 bias

    // MLP weights
    total += C * mlp_hidden; // fc weight
    total += mlp_hidden;     // fc bias
    total += mlp_hidden * C; // proj weight
    total += C;              // proj bias

    return total;
}

int main() {
    // Define transformer dimensions
    TransformerDims dims = {
        .B = 2,   // Batch size
        .T = 32,  // Sequence length
        .C = 768, // Embedding dimension
        .NH = 12, // Number of attention heads
        .HS = 64  // Head size (C / NH = 768 / 12 = 64)
    };

    printf("Initializing transformer block with dimensions:\n");
    printf("  Batch size: %d\n", dims.B);
    printf("  Sequence length: %d\n", dims.T);
    printf("  Embedding dimension: %d\n", dims.C);
    printf("  Number of attention heads: %d\n", dims.NH);
    printf("  Head size: %d\n", dims.HS);

    // Initialize transformer block
    TransformerBlock block;
    transformer_block_init(&block, dims);

    // Prepare random weights
    size_t weight_size = calculate_weight_size(dims);
    printf("Total number of weight parameters: %zu\n", weight_size);

    float* random_weights = (float*)malloc(weight_size * sizeof(float));
    initialize_random_weights(random_weights, weight_size);

    // Load weights into transformer block
    transformer_block_load_weights(&block, random_weights);
    free(random_weights);

    // Allocate and initialize input tensor with random data
    floatX* h_input = (floatX*)malloc(dims.B * dims.T * dims.C * sizeof(floatX));
    for (int i = 0; i < dims.B * dims.T * dims.C; i++) {
        h_input[i] = (floatX)(((float)rand() / RAND_MAX) * 0.1f); // Small random values
    }

    // Allocate device memory for input and output
    floatX *d_input, *d_output;
    cudaMalloc(&d_input, dims.B * dims.T * dims.C * sizeof(floatX));
    cudaMalloc(&d_output, dims.B * dims.T * dims.C * sizeof(floatX));

    // Copy input to device
    cudaMemcpy(d_input, h_input, dims.B * dims.T * dims.C * sizeof(floatX), cudaMemcpyHostToDevice);

    // Create CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Run forward pass
    printf("Running transformer block forward pass...\n");
    transformer_block_forward(&block, d_output, d_input, stream);

    // Wait for completion
    cudaStreamSynchronize(stream);

    // Allocate host memory for output and copy results back
    floatX* h_output = (floatX*)malloc(dims.B * dims.T * dims.C * sizeof(floatX));
    cudaMemcpy(h_output, d_output, dims.B * dims.T * dims.C * sizeof(floatX), cudaMemcpyDeviceToHost);

    // Print a sample of the output (first few values)
    printf("Output sample (first 10 values):\n");
    for (int i = 0; i < 10; i++) {
        printf("  %f\n", (float)h_output[i]);
    }

    // Clean up
    free(h_input);
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaStreamDestroy(stream);
    transformer_block_free(&block);

    printf("Transformer block execution completed successfully.\n");
    return 0;
}
