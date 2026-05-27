#include <cuda_runtime.h>

__global__ void rgb_to_grayscale_kernel(const float* input, float* output, int width, int height) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = width * height;
    if (idx < total) {
        int baz = 3 * idx;
        output[idx] = 0.299f * input[baz] + 0.587f * input[baz+1] + 0.114 * input[baz+2];
    }
}

// Note: rgb_image, grayscale_output are device pointers
extern "C" void solution(const float* rgb_image, float* grayscale_output, size_t height, size_t width, size_t channels) {
    int total_pixels = width * height;
    int threadsPerBlock = 256;
    int blocksPerGrid = (total_pixels + threadsPerBlock - 1) / threadsPerBlock;

    rgb_to_grayscale_kernel<<<blocksPerGrid, threadsPerBlock>>>(rgb_image, grayscale_output, width, height);
    cudaDeviceSynchronize();
}
