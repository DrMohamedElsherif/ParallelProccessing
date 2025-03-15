#include "SignalStrengthsSortedCuda.h"

#include "CellPhoneCoverage.h"
#include "CudaArray.h"
#include "Helpers.h"

#include <iostream>

#include <cuda_runtime.h>

using namespace std;

// "Smart" CUDA implementation which computes signal strengths
//
// First, all transmitters are sorted into buckets
// Then, all receivers are sorted into buckets
// Then, receivers only compute signal strength against transmitters in nearby buckets
//
// This multi-step algorithm makes the signal strength computation scale much
//  better to high number of transmitters/receivers

struct Bucket
{
    int startIndex; // Start of bucket within array
    int numElements; // Number of elements in bucket
};

///////////////////////////////////////////////////////////////////////////////////////////////
//
// No-operation sorting kernel
//
// This takes in an unordered set, and builds a dummy bucket representation around it
// It does not perform any actual sorting!
//
// This kernel must be launched with a 1,1 configuration (1 grid block, 1 thread).

static __global__ void noSortKernel(const Position* inputPositions, int numInputPositions,
                                    Position* outputPositions, Bucket* outputBuckets)
{
    int numBuckets = BucketsPerAxis * BucketsPerAxis;

    // Copy contents of input positions into output positions

    for (int i = 0; i < numInputPositions; ++i)
        outputPositions[i] = inputPositions[i];

    // Set up the set of buckets to cover the output positions evenly

    for (int i = 0; i < numBuckets; i++)
    {
        Bucket& bucket = outputBuckets[i];

        bucket.startIndex = numInputPositions * i / numBuckets;
        bucket.numElements = (numInputPositions * (i + 1) / numBuckets) - bucket.startIndex;
    }
}

// !!! missing !!!
// Kernels needed for sortPositionsIntoBuckets(...)
/// MY CODE
// CUDA kernel to compute histogram (Step 1)
constexpr int BucketSize = 1;     // I tested several sizes for the tutor's given number '16'
                                  // size 1 yieleded the lowest non covered receivers.
__global__ void computeHistogram(const Position* inputPositions, int numInputPositions, int* histogram)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numInputPositions)
    {
        int bucketIndex = (inputPositions[idx].y / BucketSize) * BucketsPerAxis +
                          (inputPositions[idx].x / BucketSize);
        atomicAdd(&histogram[bucketIndex], 1);
    }
}

// CUDA kernel to compute prefix sum for bucket start positions (Step 2)
#define MAX_BLOCKS 256
__global__ void computePrefixSum(int* histogram, Bucket* buckets, int numBuckets)
{
    __shared__ int temp[MAX_BLOCKS];

    int idx = threadIdx.x;
    if (idx < numBuckets)
        temp[idx] = histogram[idx];

    __syncthreads();

    for (int offset = 1; offset < numBuckets; offset *= 2)
    {
        if (idx >= offset)
            temp[idx] += temp[idx - offset];

        __syncthreads();
    }

    if (idx < numBuckets)
    {
        buckets[idx].startIndex = (idx == 0) ? 0 : temp[idx - 1];
        buckets[idx].numElements = histogram[idx];
    }
}

// CUDA kernel to scatter elements into the correct bucket (Step 3)
__global__ void scatterToBuckets(const Position* inputPositions, int numInputPositions,
                                 Position* outputPositions, Bucket* buckets, int* bucketOffsets)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numInputPositions)
    {
        int bucketIndex = (inputPositions[idx].y / BucketSize) * BucketsPerAxis +
                          (inputPositions[idx].x / BucketSize);
        int positionIndex = atomicAdd(&bucketOffsets[bucketIndex], 1);
        outputPositions[positionIndex] = inputPositions[idx];
    }
}


///////////////////////////////////////////////////////////////////////////////////////////////
//
// Sort a set of positions into a set of buckets
//
// Given a set of input positions, these will be re-ordered such that
//  each range of elements in the output array belong to the same bucket.
// The list of buckets that is output describes where each such range begins
//  and ends in the re-ordered position array.

static void sortPositionsIntoBuckets(CudaArray<Position>& cudaInputPositions,
                                     CudaArray<Position>& cudaOutputPositions,
                                     CudaArray<Bucket>& cudaOutputPositionBuckets)
{
    // Bucket sorting with "Counting Sort" is a multi-phase process:
    //
    // 1. Determine how many of the input elements should end up in each bucket (build a histogram)
    //
    // 2. Given the histogram, compute where in the output array that each bucket begins, and how
    // large it is
    //    (perform prefix summation over the histogram)
    //
    // 3. Given the start of each bucket within the output array, scatter elements from the input
    //    array into the output array
    //
    // Your new sort implementation should be able to handle at least 10 million entries, and
    //  run in reasonable time (the reference implementations does the job in 200 milliseconds).

    //=================  Your code here =====================================
    // !!! missing !!!
    int numBuckets = BucketsPerAxis * BucketsPerAxis;
    CudaArray<int> cudaHistogram(numBuckets); // Create the array
    cudaMemset(cudaHistogram.cudaArray(), 0, numBuckets * sizeof(int));

    CudaArray<int> cudaBucketOffsets(numBuckets);
    cudaMemset(cudaBucketOffsets.cudaArray(), 0, numBuckets * sizeof(int));


    int numThreads = 256;
    int numBlocks = (cudaInputPositions.size() + numThreads - 1) / numThreads;

    // Step 1: Compute histogram
    computeHistogram<<<numBlocks, numThreads>>>(cudaInputPositions.cudaArray(), cudaInputPositions.size(), cudaHistogram.cudaArray());

    // Step 2: Compute prefix sum
    computePrefixSum<<<1, numBuckets>>>(cudaHistogram.cudaArray(), cudaOutputPositionBuckets.cudaArray(), numBuckets);

    // Copy histogram to offsets for Step 3
    cudaBucketOffsets.copyFromCuda(cudaHistogram.cudaArray());

    // Step 3: Scatter elements
    scatterToBuckets<<<numBlocks, numThreads>>>(cudaInputPositions.cudaArray(), cudaInputPositions.size(),
                                                cudaOutputPositions.cudaArray(), cudaOutputPositionBuckets.cudaArray(),
                                                cudaBucketOffsets.cudaArray());

    // Instead of sorting, we will now run a dummy kernel that just duplicates the
    //  output positions, and constructs a set of dummy buckets. This is just so that
    //  the test program will not crash when you try to run it.
    //
    // This kernel is run single-threaded because it is throw-away code where performance
    //  does not matter; after all, the purpose of the lab is to replace it with a
    //  proper sort algorithm instead!

    //========== Remove this code when you begin to implement your own sorting algorithm ==========

    //noSortKernel<<<1, 1>>>(cudaInputPositions.cudaArray(), cudaInputPositions.size(),
                           //cudaOutputPositions.cudaArray(), cudaOutputPositionBuckets.cudaArray());
}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Go through all transmitters in one bucket, find highest signal strength
// Return highest strength (or the old value, if that was higher)

static __device__ float scanBucket(const Position* transmitters, int numTransmitters,
                                   const Position& receiver, float bestSignalStrength)
{
    for (int transmitterIndex = 0; transmitterIndex < numTransmitters; ++transmitterIndex)
    {
        const Position& transmitter = transmitters[transmitterIndex];

        float strength = signalStrength(transmitter, receiver);

        if (bestSignalStrength < strength)
            bestSignalStrength = strength;
    }

    return bestSignalStrength;
}

///////////////////////////////////////////////////////////////////////////////////////////////
//
// Calculate signal strength for all receivers

static __global__ void calculateSignalStrengthsSortedKernel(const Position* transmitters,
                                                            const Bucket* transmitterBuckets,
                                                            const Position* receivers,
                                                            const Bucket* receiverBuckets,
                                                            float* signalStrengths)
{
    // Determine which bucket the current grid block is processing

    int receiverBucketIndexX = blockIdx.x;
    int receiverBucketIndexY = blockIdx.y;

    int receiverBucketIndex = receiverBucketIndexY * BucketsPerAxis + receiverBucketIndexX;

    const Bucket& receiverBucket = receiverBuckets[receiverBucketIndex];

    int receiverStartIndex = receiverBucket.startIndex;
    int numReceivers = receiverBucket.numElements;

    // Distribute available receivers over the set of available threads

    for (int receiverIndex = threadIdx.x; receiverIndex < numReceivers; receiverIndex += blockDim.x)
    {
        // Locate current receiver within the current bucket

        const Position& receiver = receivers[receiverStartIndex + receiverIndex];
        float& finalStrength = signalStrengths[receiverStartIndex + receiverIndex];

        float bestSignalStrength = 0.f;

        // Scan all buckets in the 3x3 region enclosing the receiver's bucket index

        for (int transmitterBucketIndexY = receiverBucketIndexY - 1;
             transmitterBucketIndexY < receiverBucketIndexY + 2; ++transmitterBucketIndexY)
            for (int transmitterBucketIndexX = receiverBucketIndexX - 1;
                 transmitterBucketIndexX < receiverBucketIndexX + 2; ++transmitterBucketIndexX)
            {
                // Only process bucket if its index is within [0, BucketsPerAxis - 1] along each
                // axis

                if (transmitterBucketIndexX >= 0 && transmitterBucketIndexX < BucketsPerAxis
                    && transmitterBucketIndexY >= 0 && transmitterBucketIndexY < BucketsPerAxis)
                {
                    // Scan bucket for a potential new "highest signal strength"

                    int transmitterBucketIndex =
                        transmitterBucketIndexY * BucketsPerAxis + transmitterBucketIndexX;
                    int transmitterStartIndex =
                        transmitterBuckets[transmitterBucketIndex].startIndex;
                    int numTransmitters = transmitterBuckets[transmitterBucketIndex].numElements;
                    bestSignalStrength = scanBucket(&transmitters[transmitterStartIndex],
                                                    numTransmitters, receiver, bestSignalStrength);
                }
            }

        // Store out the highest signal strength found for the receiver

        finalStrength = bestSignalStrength;
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////

void calculateSignalStrengthsSortedCuda(const PositionList& cpuTransmitters,
                                        const PositionList& cpuReceivers,
                                        SignalStrengthList& cpuSignalStrengths)
{
    int numBuckets = BucketsPerAxis * BucketsPerAxis;

    // Copy input positions to device memory

    CudaArray<Position> cudaTempTransmitters(cpuTransmitters.size());
    cudaTempTransmitters.copyToCuda(&(*cpuTransmitters.begin()));

    CudaArray<Position> cudaTempReceivers(cpuReceivers.size());
    cudaTempReceivers.copyToCuda(&(*cpuReceivers.begin()));

    // Allocate device memory for sorted arrays

    CudaArray<Position> cudaTransmitters(cpuTransmitters.size());
    CudaArray<Bucket> cudaTransmitterBuckets(numBuckets);

    CudaArray<Position> cudaReceivers(cpuReceivers.size());
    CudaArray<Bucket> cudaReceiverBuckets(numBuckets);

    // Sort transmitters and receivers into buckets

    sortPositionsIntoBuckets(cudaTempTransmitters, cudaTransmitters, cudaTransmitterBuckets);
    sortPositionsIntoBuckets(cudaTempReceivers, cudaReceivers, cudaReceiverBuckets);

    // Perform signal strength computation
    CudaArray<float> cudaSignalStrengths(cpuReceivers.size());

    int numThreads = 256;
    dim3 grid = dim3(BucketsPerAxis, BucketsPerAxis);

    calculateSignalStrengthsSortedKernel<<<grid, numThreads>>>(
        cudaTransmitters.cudaArray(), cudaTransmitterBuckets.cudaArray(), cudaReceivers.cudaArray(),
        cudaReceiverBuckets.cudaArray(), cudaSignalStrengths.cudaArray());

    // Copy results back to host memory
    cpuSignalStrengths.resize(cudaSignalStrengths.size());
    cudaSignalStrengths.copyFromCuda(&(*cpuSignalStrengths.begin()));
}
