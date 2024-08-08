#include <stdio.h>
#include <cuda_runtime.h>

// CUDA kernel to initialize the array
__global__ void initArray(float *arr, int NX, int NY, int NQ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int q = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = x * (NY * NQ) + y * NQ + q;

    if (x < NX && y < NY && q < NQ) {
        arr[idx] = static_cast<float>(x + y + q);  // Example initialization
    }
}

// CUDA kernel to perform an operation on the array
__global__ void processArray(float *arr, int NX, int NY, int NQ) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int q = blockIdx.z * blockDim.z + threadIdx.z;
    int idx = x * (NY * NQ) + y * NQ + q;

    if (x < NX && y < NY && q < NQ) {
        arr[idx] += 1.0f;  // Example operation: increment each element by 1
    }
}

int main() {
    int NX = 4;  // Number of grid points in the x direction
    int NY = 3;  // Number of grid points in the y direction
    int NQ = 2;  // Number of variables being stored
    size_t size = NX * NY * NQ * sizeof(float);

    // Allocate memory on the host
    float *h_arr = (float*)malloc(size);

    // Allocate memory on the device
    float *d_arr;
    cudaMalloc(&d_arr, size);

    // Define grid and block dimensions
    dim3 blockDim(8, 8, 8);
    dim3 gridDim((NX + blockDim.x - 1) / blockDim.x, 
                 (NY + blockDim.y - 1) / blockDim.y, 
                 (NQ + blockDim.z - 1) / blockDim.z);

    // Initialize the array on the device
    initArray<<<gridDim, blockDim>>>(d_arr, NX, NY, NQ);
    cudaDeviceSynchronize();

    // Perform an operation on the array
    processArray<<<gridDim, blockDim>>>(d_arr, NX, NY, NQ);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(h_arr, d_arr, size, cudaMemcpyDeviceToHost);

    // Print the result
    for (int x = 0; x < NX; x++) {
        for (int y = 0; y < NY; y++) {
            for (int q = 0; q < NQ; q++) {
                int idx = x * (NY * NQ) + y * NQ + q;
                printf("h_arr[%d][%d][%d] = %f\n", x, y, q, h_arr[idx]);
            }
        }
    }

    // Free device memory
    cudaFree(d_arr);

    // Free host memory
    free(h_arr);

    return 0;
}