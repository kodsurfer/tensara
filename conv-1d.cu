#include <cuda_runtime.h>

__global__ void conv_1d_kernel(const float* A, const float* B, float* C, size_t N, size_t K) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float sum = 0.0f;
        int ofst = (K - 1) / 2;
        for (int j = 0; j < K; j++) {
            int i = idx + j - ofst;
            if (i >= 0 && i < N) {
                sum += A[i] * B[j];
            }
        }
        C[idx] = sum;
    }
}
// Note: A, B, C are device pointers
extern "C" void solution(const float* A, const float* B, float* C, size_t N, size_t K) {
    int threads_per_block = 256;
    int blocks_per_grid = (N + threads_per_block - 1) / threads_per_block;
    conv_1d_kernel<<<blocks_per_grid, threads_per_block>>>(A, B, C, N, K);
    cudaDeviceSynchronize();
}
