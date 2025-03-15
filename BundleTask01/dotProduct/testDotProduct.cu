// ==========================================================================
// $Id$
// ==========================================================================
// (C)opyright: 2009-2010
//
//   Ulm University
//
// Creator: Hendrik Lensch
// Email:   {hendrik.lensch,johannes.hanika}@uni-ulm.de
// ==========================================================================
// $Log$
// ==========================================================================

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

using namespace std;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

#define MAX_BLOCKS 256
#define MAX_THREADS 128

#define RTEST // use random initialization of array

/* compute the dot product between a1 and a2. a1 and a2 are of size
 dim. The result of each thread should be stored in _dst[blockIdx.x *
 blockDim.x + threadIdx.x]. Each thread should accumulate the dot
 product of a subset of elements.
 */
__global__ void dotProdKernel(float* _dst, const float* _a1, const float* _a2, int _dim)
{

    // program your kernel here
    // Declare shared memory to store partial sums within a block
    __shared__ float sharedMem[MAX_THREADS];

    // Compute the global thread index (unique across all threads)
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // Compute the stride to allow threads to process more elements
    int stride = gridDim.x * blockDim.x;

    // Initialize a sum variable to accumulate dot product results
    float sum = 0.0f;

    // Each thread computes the partial dot product for its assigned elements
    for (int i = tid; i < _dim; i += stride)
    {
        sum += _a1[i] * _a2[i]; // Multiply corresponding elements and accumulate
    }

    // Store the partial sum in shared memory at the corresponding thread index
    sharedMem[threadIdx.x] = sum;

    // Ensure all threads finish writing to shared memory before proceeding
    __syncthreads();

    // Perform parallel reduction to sum up values within the block
    for (int s = blockDim.x / 2; s > 0; s >>= 1)
    {
        // Only the first half of threads participate in reduction at each step
        if (threadIdx.x < s)
        {
            sharedMem[threadIdx.x] += sharedMem[threadIdx.x + s]; // Sum two elements
        }

        // Synchronize threads before next reduction step
        __syncthreads();
    }

    // The first thread of each block writes the final block sum to global memory
    if (threadIdx.x == 0)
    {
        _dst[blockIdx.x] = sharedMem[0]; // Store block's total sum in _dst
    }

    //!!!!!!!!! missing  !!!!!!!!!!!!!!!!!!!!!!!!
    // No additional code is required here since the function already completes the task.
    // The kernel completes by writing the sum of each block into global memory via _dst.
}

/* This program sets up two large arrays of size dim and computes the
dot product of both arrays.

The arrays are uploaded only once and the dot product is computed
multiple times. While this does not make too much sense it
demonstrates the possible speedup.  */
int main(int argc, char* argv[])
{
    // parse command line
    int acount = 1;

    if (argc < 3)
    {
        printf("usage: testDotProduct <dim> <GPU-flag [0,1]>\n");
        exit(1);
    }

    // number of elements in both vectors
    int dim = atoi(argv[acount++]);

    // flag indicating whether the CPU or the GPU version should be executed
    bool gpuVersion = atoi(argv[acount++]);

    printf("dim: %d\n", dim);

    float* cpuArray1 = new float[dim];
    float* cpuArray2 = new float[dim];

    // initialize the two arrays (either random or deterministic)
    for (int i = 0; i < dim; ++i)
    {
#ifdef RTEST
        cpuArray1[i] = drand48();
        cpuArray2[i] = drand48();
#else
        cpuArray1[i] = 2.0;
        cpuArray2[i] = i % 10;
#endif
    }

    // now the gpu stuff
    float* gpuArray1;
    float* gpuArray2;
    float* gpuResult;

    float* h;

    if (gpuVersion)
    {
        // allocate two gpuArray1 and gpuArray2 and gpuResult array on GPU
        cudaMalloc((void**)&gpuArray1, dim * sizeof(float)); // Allocating memory for array 1 on GPU
        cudaMalloc((void**)&gpuArray2, dim * sizeof(float)); // Allocating memory for array 2 on GPU
        cudaMalloc((void**)&gpuResult, MAX_BLOCKS * sizeof(float)); // Allocating memory for the result array on GPU

        //!!!!!!!!! missing  !!!!!!!!!!!!!!!!!!!!!!!!
        // Additional memory allocation for a result storage array on the host.
        // We need to allocate an array on the host to collect results from the GPU.
        // We will copy the results back to the host after kernel execution.
        // Allocate a float array on the host to hold the results of each block's dot product sum.
        // This array will store partial sums calculated by the GPU kernel.
        // The number of elements in the array is equal to MAX_BLOCKS since we are using MAX_BLOCKS blocks for the kernel.
        
        // Example of allocation:
        
        h = new float[MAX_BLOCKS];  // Allocating memory to store partial results from each block
        
        //!!!!!!!!! missing  !!!!!!!!!!!!!!!!!!!!!!!!
        // Here, you need to ensure that the allocated memory on the device is properly initialized
        // This is handled in the cudaMemcpy step below.
        
        // copy the array once to the device
        cudaMemcpy(gpuArray1, cpuArray1, dim * sizeof(float), cudaMemcpyHostToDevice); // Copying data from host to device
        cudaMemcpy(gpuArray2, cpuArray2, dim * sizeof(float), cudaMemcpyHostToDevice); // Copying data from host to device
    }

    const int num_iters = 100;
    double finalDotProduct;

    if (!gpuVersion)
    {
        printf("cpu: ");
        for (int iter = 0; iter < num_iters; ++iter)
        {
            finalDotProduct = 0.0;
            for (int i = 0; i < dim; ++i)
            {
                finalDotProduct += cpuArray1[i] * cpuArray2[i];
            }
        }
    }
    else
    {

        // CUDA version here
        printf("gpu: ");
        // Define the number of blocks in the grid (MAX_BLOCKS) 
        dim3 blockGrid(MAX_BLOCKS);

        // Define the number of threads per block (MAX_THREADS)
        dim3 threadBlock(MAX_THREADS);

        // Repeat the dot product computation multiple times for benchmarking
        for (int iter = 0; iter < num_iters; ++iter)
        {
            // Launch the CUDA kernel with the configured grid and thread block sizes
            dotProdKernel<<<blockGrid, threadBlock>>>(gpuResult, gpuArray1, gpuArray2, dim);
        }

        // Copy the partial sums from GPU memory back to the CPU
        cudaMemcpy(h, gpuResult, MAX_BLOCKS * sizeof(float), cudaMemcpyDeviceToHost);

        // Initialize the final dot product result to 0
        finalDotProduct = 0.0;

        // Sum up the partial dot products computed by each block
        for (int i = 0; i < MAX_BLOCKS; ++i)
        {
            finalDotProduct += h[i]; // Accumulate the partial results to get the final dot product
        }

        //!!!!!!!!! missing  !!!!!!!!!!!!!!!!!!!!!!!!
        // After completing the kernel execution and gathering the results from all blocks,
        // we would finalize the result and compute the total dot product. 
        // This allow us to aggregate partial results from each block.
        // At this point, we have collected partial sums in 'h' (host array),
        // and we will sum these partial results to get the final dot product.
    }

    printf("Result: %f\n", finalDotProduct);

    if (gpuVersion)
    {

        // cleanup GPU memory
        cudaFree(gpuArray1); // Free memory for array 1 on GPU
        cudaFree(gpuArray2); // Free memory for array 2 on GPU
        cudaFree(gpuResult); // Free memory for result on GPU

        //!!!!!!!!! missing  !!!!!!!!!!!!!!!!!!!!!!!!
        // This cleanup ensures that we free up memory on both the GPU and host to avoid memory leaks.
        // Always remember to clean up allocated resources after the computations are complete.

        delete[] h; // Free the host memory that was allocated for storing results from the GPU
    }

    delete[] cpuArray2; // Free CPU array 2
    delete[] cpuArray1; // Free CPU array 1

    checkCUDAError("end of program");

    printf("done\n");
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
