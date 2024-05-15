
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>

#include "kernel.hh"

struct GpuConstants {
    const size_t xCount;
    const size_t yCount;
    const size_t zCount;
    const size_t voxelCount;
    const size_t rfSampleCount;
    const size_t elementCount;
    const size_t transmissionCount;
};

/*
*
*
* 
*/
__global__ void delayAndSum(const float* rfData,const float* locData, const GpuConstants* constants, const float* xRange, const float* yRange, const float *zRange, float* volume)
{
    // xyz dims 201, 201, 134 
    int tx = threadIdx.x + blockIdx.x * 8;
    int ty = threadIdx.y + blockIdx.y * 8;
    int tz = threadIdx.z + blockIdx.z * 8;

    const float Speed = 1.540f;
    const float Fs = 100000.0f; // 100 MHz
    
    if (tx >= 201 || ty >= 201 || tz >= 134)
    {
        return;
    }

    int voxelId = tz * constants->xCount * constants->yCount + ty * constants->xCount + tx;

    float voxel = volume[ voxelId ];
    float xPos = xRange[tx];
    float yPos = yRange[ty];
    float zPos = zRange[tz];

    float distance;
    int scanIndex;
    float exPos, eyPos, ezPos;
    for (int t = 0; t < constants->transmissionCount; t++)
    {
        for (int e = 0; e < constants->elementCount; e++)
        {
            exPos = locData[3 * (t * constants->elementCount + e)];
            eyPos = locData[3 * (t * constants->elementCount + e) + 1];
            ezPos = locData[3 * (t * constants->elementCount + e) + 2];

            // voxel to rx element
            distance = sqrtf(powf(xPos - exPos, 2) + powf(yPos - eyPos, 2) + powf(zPos - ezPos, 2)) + zPos;

            // tx element to voxel (only valid for plane waves under the shadow)
           // distance = distance + zPos;

            scanIndex = (int)roundf(distance * Fs/Speed);

            if (scanIndex >= (constants->rfSampleCount * constants->elementCount * constants->transmissionCount ))
            {
                continue;
            }

            voxel = voxel + rfData[t * constants->rfSampleCount * constants->elementCount + e * constants->rfSampleCount + scanIndex];
        }
    }

    volume[voxelId] = voxel;

}

__global__ void delayAndSumFast(const float* rfData, const float* locData, const GpuConstants* constants, const float* xRange, const float* yRange, const float* zRange, float* volume)
{
    __shared__ float temp[508 * 4];

    int tempSize = 508;
    
    int e = threadIdx.x;

    if (e >= 508)
    {
        return;
    }

    int x = blockIdx.x;
    int y = blockIdx.y;
    int z = blockIdx.z;


    const float Speed = 1.540f;
    const float Fs = 100000.0f; // 100 MHz

    float xPos = xRange[x];
    float yPos = yRange[y];
    float zPos = zRange[z];

    float distance;
    int scanIndex;
    float exPos, eyPos, ezPos;

    for (int t = 0; t < constants->transmissionCount; t++)
    {
        exPos = locData[3 * (t * constants->elementCount + e)];
        eyPos = locData[3 * (t * constants->elementCount + e) + 1];
        ezPos = locData[3 * (t * constants->elementCount + e) + 2];

        // voxel to rx element
        distance = sqrtf(powf(xPos - exPos, 2) + powf(yPos - eyPos, 2) + powf(zPos - ezPos, 2)) + zPos;

        // tx element to voxel (only valid for plane waves under the shadow)
        // distance = distance + zPos;

        scanIndex = (int)floorf(distance * Fs / Speed);

        temp[e] += rfData[t * constants->rfSampleCount * constants->elementCount + e * constants->rfSampleCount + scanIndex];


        //voxel = voxel + scan;
    }

    __syncthreads();
    
    for (int s = 1; s < tempSize; s *= 2)
    {
        int index = 2 * s * e;

        if (index < (tempSize - s))
        {
            temp[index] += temp[index + s];
        }
        
        __syncthreads();
    }

    if (e == 0)
    {
        volume[z * constants->xCount * constants->yCount + y * constants->xCount + x] = temp[0];
    }
    
}



void
cleanupMemory(float* floats[6], GpuConstants* constants)
{
    cudaFree(constants);

    for (int i = 0; i < 6; i++)
    {
        cudaFree(floats[i]);
    }
}

cudaError_t volumeReconstruction(Volume* volume, const md::TypedArray<float>& rfData, const md::TypedArray<float>& locData)
{
    float* dRfData = 0;
    float* dLocData = 0;
    float* dVolume = 0;
    GpuConstants* dConstants = 0;
    cudaError_t cudaStatus;
    
    float* dXPositions = 0;
    float* dYPositions = 0;
    float* dZPositions = 0;

    float* deviceData[6] = { dRfData, dLocData, dVolume, dXPositions, dYPositions, dZPositions };

    std::vector<size_t> rfDims = rfData.getDimensions();

    GpuConstants constants = {
        volume->getXCount(),
        volume->getYCount(),
        volume->getZCount(),
        volume->getCount(),
        rfDims[0],
        rfDims[1],
        rfDims[2] };

    int count;
    cudaGetDeviceCount(&count);
    std::cout << count << std::endl;

    // Transfer data to device
    {
        std::cout << "Allocating GPU memory" << std::endl;
        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to connect to GPU\n");
            cleanupMemory(deviceData, dConstants);
        }

        // Malloc arrays on GPU
        cudaStatus = cudaMalloc((void**)&dRfData, rfData.getNumberOfElements() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate rf array on device\n");
            cleanupMemory(deviceData, dConstants);
        }

        cudaStatus = cudaMalloc((void**)&dLocData, locData.getNumberOfElements() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate location array on device\n");
            cleanupMemory(deviceData, dConstants);
        }

        cudaStatus = cudaMalloc((void**)&dConstants, sizeof(GpuConstants));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate constants array on device\n");
            cleanupMemory(deviceData, dConstants);
        }

        cudaStatus = cudaMalloc((void**)&dVolume, volume->getCount() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(deviceData, dConstants);
        }

        cudaStatus = cudaMalloc((void**)&dXPositions, volume->getXCount() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(deviceData, dConstants);
        }

        cudaStatus = cudaMalloc((void**)&dYPositions, volume->getYCount() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(deviceData, dConstants);
        }

        cudaStatus = cudaMalloc((void**)&dZPositions, volume->getZCount() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(deviceData, dConstants);
        }

        std::cout << "Transferring data to GPU" << std::endl;

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dRfData, (void*)&rfData.begin()[0], rfData.getNumberOfElements() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy RF data to device\n");
            cleanupMemory(deviceData, dConstants);
        }

        cudaStatus = cudaMemcpy(dLocData, (void*)&locData.begin()[0], locData.getNumberOfElements() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy location data to device\n");
            cleanupMemory(deviceData, dConstants);
        }

        cudaStatus = cudaMemcpy(dConstants, &constants, sizeof(GpuConstants), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy constants to device\n");
            cleanupMemory(deviceData, dConstants);
        }

        cudaStatus = cudaMemcpy(dXPositions, volume->getXRange(), volume->getXCount() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy constants to device\n");
            cleanupMemory(deviceData, dConstants);
        }

        cudaStatus = cudaMemcpy(dYPositions, volume->getYRange(), volume->getYCount() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy constants to device\n");
            cleanupMemory(deviceData, dConstants);
        }

        cudaStatus = cudaMemcpy(dZPositions, volume->getZRange(), volume->getZCount() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy constants to device\n");
            cleanupMemory(deviceData, dConstants);
        }
    }

    dim3 blockDim(8, 8, 8);
    dim3 gridDim(26, 26, 17);
    std::cout << "Starting kernel" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    delayAndSum<<<gridDim,blockDim>>>(dRfData, dLocData, dConstants, dXPositions, dYPositions, dZPositions, dVolume);

    dim3 blockDim2(201, 201, 134);
    delayAndSumFast<< <blockDim2, 512 >> > (dRfData, dLocData, dConstants, dXPositions, dYPositions, dZPositions, dVolume);
    {
        // Transfer Data back
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            cleanupMemory(deviceData, dConstants);
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cudaStatus = cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // Print the elapsed time
        std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after calling kernel\n", cudaStatus);
            cleanupMemory(deviceData, dConstants);
        }


        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(volume->getData(), dVolume, volume->getCount() * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy volume data out of device\n");
            cleanupMemory(deviceData, dConstants);
        }
    }


    cleanupMemory(deviceData, dConstants);

    return cudaStatus;
}


