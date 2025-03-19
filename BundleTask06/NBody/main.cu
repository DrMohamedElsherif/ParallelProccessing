
#include "Tools.h"
#include "gltools.h"

#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <unistd.h>

#include <cuda_runtime.h>

using namespace std;

// Simple utility function to check for CUDA runtime errors
void checkCUDAError(const char* msg);

#define GUI
#define NUM_FRAMES 250     /*I replace 250 with 100 as recommended in the sheet*/

#define THREADS_PER_BLOCK 128
#define EPS_2 0.00001f
#define GRAVITY 0.00000001f

float randF(const float min = 0.0f, const float max = 1.0f)
{
    int randI = rand();
    float randF = (float)randI / (float)RAND_MAX;
    float result = min + randF * (max - min);

    return result;
}

inline __device__ float2 operator+(const float2 op1, const float2 op2)
{
    return make_float2(op1.x + op2.x, op1.y + op2.y);
}

inline __device__ float2 operator-(const float2 op1, const float2 op2)
{
    return make_float2(op1.x - op2.x, op1.y - op2.y);
}

inline __device__ float2 operator*(const float2 op1, const float op2)
{
    return make_float2(op1.x * op2, op1.y * op2);
}

inline __device__ float2 operator/(const float2 op1, const float op2)
{
    return make_float2(op1.x / op2, op1.y / op2);
}

inline __device__ void operator+=(float2& a, const float2 b)
{
    a.x += b.x;
    a.y += b.y;
}
///////// MY KERNELS //////////
__global__ void updateAccelerations(float2* positions, float2* accelerations, float* masses, int numBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numBodies) {
        float2 acc = make_float2(0.0f, 0.0f);
        float2 myPos = positions[i];
        for (int j = 0; j < numBodies; j++) {
            if (i != j) {
                float2 r = positions[j] - myPos;
                float distSqr = r.x * r.x + r.y * r.y + EPS_2;
                float invDist = rsqrtf(distSqr);
                float invDist3 = invDist * invDist * invDist;
                acc += r * (masses[j] * invDist3);
                
            }
        }
        accelerations[i] = acc;
    }
}

__global__ void updateVelocitiesAndPositions(float2* positions, float2* velocities, float2* accelerations, int numBodies) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numBodies) {
        velocities[i] += accelerations[i] * GRAVITY;
        positions[i] += velocities[i];

    }
}
//////////////////////////////
int main(int argc, char** argv)
{
    if (argc != 2)
    {
        cout << "Usage: " << argv[0] << " <numBodies>" << endl;
        return 1;
    }
    unsigned int numBodies = atoi(argv[1]);
    unsigned int numBlocks = numBodies / THREADS_PER_BLOCK;
    numBodies = numBlocks * THREADS_PER_BLOCK;

    // allocate memory
    float2* hPositions = new float2[numBodies];
    float2* hVelocities = new float2[numBodies];
    float* hMasses = new float[numBodies];

    // Initialize Positions and speed
    for (unsigned int i = 0; i < numBodies; i++)
    {
        hPositions[i].x = randF(-1.0, 1.0);
        hPositions[i].y = randF(-1.0, 1.0);
        hVelocities[i].x = hPositions[i].y * 0.007f + randF(0.001f, -0.001f);
        hVelocities[i].y = -hPositions[i].x * 0.007f + randF(0.001f, -0.001f);
        hMasses[i] = randF(0.0f, 1.0f) * 10000.0f / (float)numBodies;
    }

    // TODO 1: Allocate GPU memory for
    // - Positions,
    // - Velocities,
    // - Accelerations and
    // - Masses
    // of all bodies and initialize them from the CPU arrays (where available).
    //// MY CODE
    float2* dPositions;
    float2* dVelocities;
    float2* dAccelerations;
    float* dMasses;

    cudaMalloc((void**)&dPositions, numBodies * sizeof(float2));
    cudaMalloc((void**)&dVelocities, numBodies * sizeof(float2));
    cudaMalloc((void**)&dAccelerations, numBodies * sizeof(float2));
    cudaMalloc((void**)&dMasses, numBodies * sizeof(float));

    cudaMemcpy(dPositions, hPositions, numBodies * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(dVelocities, hVelocities, numBodies * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(dMasses, hMasses, numBodies * sizeof(float), cudaMemcpyHostToDevice);
    ///////////

    // Free host memory not needed again
    delete[] hVelocities;
    delete[] hMasses;

    // Initialize OpenGL rendering
#ifdef GUI
    initGL();
    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    GLuint sp = createShaderProgram("white.vs", 0, 0, 0, "white.fs");

    GLuint vb;
    glGenBuffers(1, &vb);
    GL_CHECK_ERROR;
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    GL_CHECK_ERROR;
    glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * numBodies, hPositions, GL_STATIC_DRAW);
    GL_CHECK_ERROR;
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    GL_CHECK_ERROR;

    GLuint va;
    glGenVertexArrays(1, &va);
    GL_CHECK_ERROR;
    glBindVertexArray(va);
    GL_CHECK_ERROR;
    glBindBuffer(GL_ARRAY_BUFFER, vb);
    GL_CHECK_ERROR;
    glEnableVertexAttribArray(glGetAttribLocation(sp, "inPosition"));
    GL_CHECK_ERROR;
    glVertexAttribPointer(glGetAttribLocation(sp, "inPosition"), 2, GL_FLOAT, GL_FALSE, 0, 0);
    GL_CHECK_ERROR;
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    GL_CHECK_ERROR;
    glBindVertexArray(0);
    GL_CHECK_ERROR;
#endif

    // Calculate
    for (unsigned int t = 0; t < NUM_FRAMES; t++)
    {
        __int64_t computeStart = continuousTimeNs();

        // TODO 3: Update accelerations of all bodies here.
        updateAccelerations<<<numBlocks, THREADS_PER_BLOCK>>>(dPositions, dAccelerations, dMasses, numBodies);

        // TODO 4: Update velocities and positions of all bodies here.
        updateVelocitiesAndPositions<<<numBlocks, THREADS_PER_BLOCK>>>(dPositions, dVelocities, dAccelerations, numBodies);
        ////////////////////////////

        cudaDeviceSynchronize();
        cout << "Frame compute time: " << (continuousTimeNs() - computeStart) << "ns" << endl;

        // TODO 5: Download the updated positions into the hPositions array for rendering.
        cudaMemcpy(hPositions, dPositions, numBodies * sizeof(float2), cudaMemcpyDeviceToHost);
        ///////////////////////////

#ifdef GUI
        // Upload positions to OpenGL
        glBindBuffer(GL_ARRAY_BUFFER, vb);
        GL_CHECK_ERROR;
        glBufferData(GL_ARRAY_BUFFER, sizeof(float) * 2 * numBodies, hPositions, GL_STATIC_DRAW);
        GL_CHECK_ERROR;
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        GL_CHECK_ERROR;

        // Draw
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        GL_CHECK_ERROR;
        glUseProgram(sp);
        GL_CHECK_ERROR;
        glBindVertexArray(va);
        GL_CHECK_ERROR;
        glDrawArrays(GL_POINTS, 0, numBodies);
        GL_CHECK_ERROR;
        glBindVertexArray(0);
        GL_CHECK_ERROR;
        glUseProgram(0);
        GL_CHECK_ERROR;
        swapBuffers();
#endif
    }

#ifdef GUI
    cout << "Done." << endl;
    sleep(2);
#endif

    // Clean up
#ifdef GUI
    glDeleteProgram(sp);
    GL_CHECK_ERROR;
    glDeleteVertexArrays(1, &va);
    GL_CHECK_ERROR;
    glDeleteBuffers(1, &vb);
    GL_CHECK_ERROR;

    glDeleteProgram(sp);
    exitGL();
#endif

    // TODO 2: Clean up your allocated memory
    cudaFree(dPositions);
    cudaFree(dVelocities);
    cudaFree(dAccelerations);
    cudaFree(dMasses);
    //////////////////////////

    delete[] hPositions;

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
