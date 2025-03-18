#include "Tools.h"

#include <iomanip>
#include <iostream>

#include <cuda_runtime.h>

using namespace std;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

//#define VERBOSE // Prints input matrix and results. Only uncomment for small matrix sizes!
#define RUN_CPU // Runs CPU code for reference (slow!!!)
#define N 1024 // Must be a multiple of THREADS_PER_BLOCK
#define THREADS_PER_BLOCK 16 // per axis -> block has this value squared threads.
void multiplyMatrix(float* result, const float* a, const float* b, const int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            result[i * n + j] = 0.0f;
            for (unsigned int k = 0; k < n; k++)
            {
                result[i * n + j] += a[i * n + k] * b[k * n + j];
            }
        }
    }
}

void dumpMatrix(const float* m, const int n)
{
    for (unsigned int i = 0; i < n; i++)
    {
        for (unsigned int j = 0; j < n; j++)
        {
            cout << setw(3) << setprecision(3) << m[i * n + j] << " ";
        }
        cout << endl;
    }
}

float randF(const float min = 0.0f, const float max = 1.0f)
{
    int randI = rand();
    float randF = (float)randI / (float)RAND_MAX;
    float result = min + randF * (max - min);

    return result;
}

__global__ void multiplyMatrixGpu1(float* result, const float* a, const float* b, const int n)
{
    // TODO: Implement a trivial GPU square matrix multiplication.
    // Use one thread per output element.
    // MY CODE
    // Compute row and column index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure the thread is within bounds
    if (row < n && col < n)
    {
        float sum = 0.0f;       // Initialize sum to store the result cell

        // Perform dot product for row of A and column of B
        for (int k = 0; k < n; k++)
        {
            sum += a[row * n + k] * b[k * n + col];
        }

        // Store the computed value in the result matrix
        result[row * n + col] = sum;
    }
    //////////////////
}

__global__ void multiplyMatrixGpu2(float* result, const float* a, const float* b, const int n)
{
    // TODO: Implement a more sophisticated GPU square matrix multiplication.
    // Compute square submatrices per block. Load the common input
    // data of all threads of a block into shared memory cooperatively.
    //// MY CODE
    // Allocate shared memory for the submatrices
    __shared__ float sharedA[THREADS_PER_BLOCK][THREADS_PER_BLOCK];
    __shared__ float sharedB[THREADS_PER_BLOCK][THREADS_PER_BLOCK];

    // Compute row and column index for this thread
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Initialize sum for storing this thread's result element
    float sum = 0.0f;

    // Iterate over tiles of the input matrices
    for (int tile = 0; tile < n / THREADS_PER_BLOCK; tile++)
    {
        // Load data into shared memory
        if (row < n && (tile * THREADS_PER_BLOCK + threadIdx.x) < n)
            sharedA[threadIdx.y][threadIdx.x] = a[row * n + tile * THREADS_PER_BLOCK + threadIdx.x];
        else
            sharedA[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < n && (tile * THREADS_PER_BLOCK + threadIdx.y) < n)
            sharedB[threadIdx.y][threadIdx.x] = b[(tile * THREADS_PER_BLOCK + threadIdx.y) * n + col];
        else
            sharedB[threadIdx.y][threadIdx.x] = 0.0f;

        // Now synchronize to ensure all threads have loaded their data
        __syncthreads();

        // Compute partial sum for this tile
        for (int k = 0; k < THREADS_PER_BLOCK; k++)
        {
            sum += sharedA[threadIdx.y][k] * sharedB[k][threadIdx.x];
        }

        // Now synchronize again to ensure computation is done before loading next tile
        __syncthreads();
    }

    // Store final result in the output matrix, this shall take less than 2 ms runtime hopefully!!
    if (row < n && col < n)
    {
        result[row * n + col] = sum;
    }
    ////////////////////////////////
}

int main(int argc, char** argv)
{
    __int64_t startTime;
    __int64_t endTime;

    // Allocate all memory
    float* hM1 = new float[N * N];
    float* hM2 = new float[N * N];
    float* hMR = new float[N * N];
    float* gM1;
    cudaMalloc(&gM1, sizeof(float) * N * N);
    float* gM2;
    cudaMalloc(&gM2, sizeof(float) * N * N);
    float* gMR;
    cudaMalloc(&gMR, sizeof(float) * N * N);

    // Initialize matrices and upload to CUDA
    for (unsigned int n = 0; n < N * N; n++)
    {
        hM1[n] = randF(-1.0, 1.0);
        hM2[n] = randF(-1.0, 1.0);
    }
    cudaMemcpy(gM1, hM1, sizeof(int) * N * N, cudaMemcpyHostToDevice);
    cudaMemcpy(gM2, hM2, sizeof(int) * N * N, cudaMemcpyHostToDevice);
#ifdef VERBOSE
    cout << "Input Matrices:" << endl;
    dumpMatrix(hM1, N);
    cout << endl;
    dumpMatrix(hM2, N);
    cout << endl << endl;
#endif

#ifdef RUN_CPU
    // Calculations on CPU
    startTime = continuousTimeNs();
    multiplyMatrix(hMR, hM1, hM2, N);
    endTime = continuousTimeNs();
#ifdef VERBOSE
    cout << "CPU:" << endl;
    dumpMatrix(hMR, N);
    cout << endl;
#endif
    cout << "CPU time: " << (endTime - startTime) << "ns" << endl;
#endif

    // Calculations on GPU
    int blocksPerGridX =
        N % THREADS_PER_BLOCK == 0 ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
    int blocksPerGridY =
        N % THREADS_PER_BLOCK == 0 ? N / THREADS_PER_BLOCK : N / THREADS_PER_BLOCK + 1;
    startTime = continuousTimeNs();
    multiplyMatrixGpu1<<<dim3(blocksPerGridX, blocksPerGridY, 1),
                         dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)>>>(gMR, gM1, gM2, N);
    cudaDeviceSynchronize();
    endTime = continuousTimeNs();
    cudaMemcpy(hMR, gMR, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
#ifdef VERBOSE
    cout << "GPU simple:" << endl;
    dumpMatrix(hMR, N);
    cout << endl;
#endif
    cout << "GPU simple time: " << (endTime - startTime) << "ns" << endl;
    startTime = continuousTimeNs();
    multiplyMatrixGpu2<<<dim3(blocksPerGridX, blocksPerGridY, 1),
                         dim3(THREADS_PER_BLOCK, THREADS_PER_BLOCK, 1)>>>(gMR, gM1, gM2, N);
    cudaDeviceSynchronize();
    endTime = continuousTimeNs();
    cudaMemcpy(hMR, gMR, sizeof(float) * N * N, cudaMemcpyDeviceToHost);
#ifdef VERBOSE
    cout << "GPU advanced:" << endl;
    dumpMatrix(hMR, N);
    cout << endl;
#endif
    cout << "GPU advanced time: " << (endTime - startTime) << "ns" << endl;

    // Free all memory
    cudaFree(gM1);
    cudaFree(gM2);
    cudaFree(gMR);
    delete[] hM1;
    delete[] hM2;
    delete[] hMR;

    checkCUDAError("end of program");
}

void checkCUDAError(const char* msg)
{
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess != err)
    {
        fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString(err));
        exit(-1);
    }
}
