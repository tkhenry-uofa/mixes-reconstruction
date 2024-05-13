


#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>

#include "kernel.hh"

struct GpuConstants {
    const int xCount;
    const int yCount;
    const int zCount;
    const int voxelCount;
    const int transmissionCount;
    const int elementCount;
    const int rfSampleCount;
};


__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}


__global__ void delayAndSum(const float* rfData,const float* locData, const GpuConstants* constants, const float* xRange, const float* yRange, const float *zRange, float* volume)
{
    // xyz dims 201, 201, 134 
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;

    const float Speed = 1540;
    const float Fs = 100000000; // 100 MHz
    
    if (tx != 101 || ty >= 201 || tz >= 134)
    {
        return;
    }

    int voxelId = (tx * constants->yCount * constants->zCount) + (ty * constants->zCount) + tz;

    float* voxel = &volume[ voxelId ];
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
            distance = sqrtf(powf(xPos - exPos, 2) + powf(yPos - eyPos, 2) + powf(zPos - ezPos, 2));

            // tx element to voxel (only valid for plane waves under the shadow)
            distance = distance + zPos;

            scanIndex = roundf(distance / (Speed * Fs));

            if (scanIndex >= constants->rfSampleCount)
            {
                continue;
            }

            *voxel = *voxel + rfData[t * constants->rfSampleCount + scanIndex];
        }
    }

}

cudaError_t volumeReconstruction(Volume* volume, const CellDataArray& rfData, const CellDataArray& locData)
{
    float* dRfData = 0;
    float* dLocData = 0;
    float* dVolume = 0;
    GpuConstants* dConstants = 0;
    cudaError_t cudaStatus;
    
    float* dXPositions = 0;
    float* dYPositions = 0;
    float* dZPositions = 0;


    GpuConstants constants = {
        volume->getXCount(),
        volume->getYCount(),
        volume->getZCount(),
        volume->getCount(),
        rfData.getCellCount(),
        rfData.getColumnCount(),
        rfData.getRowCount() };

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to connect GPU\n");
        goto Error;
    }

    // Malloc arrays on GPU
    cudaStatus = cudaMalloc((void**)&dRfData, rfData.getCount() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate rf array on device\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dLocData, locData.getCount() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate location array on device\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dConstants, sizeof(GpuConstants));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate constants array on device\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dVolume, volume->getCount() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate volume on device\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dXPositions, volume->getXCount() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate volume on device\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dYPositions, volume->getYCount() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate volume on device\n");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dZPositions, volume->getZCount() * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to allocate volume on device\n");
        goto Error;
    }



    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dRfData, rfData.getData(), rfData.getCount() * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy RF data to device\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dLocData, locData.getData(), locData.getCount() * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy location data to device\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dConstants, &constants, sizeof(GpuConstants), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy constants to device\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dXPositions, volume->_xRange.data(), 201 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy constants to device\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dYPositions, volume->_yRange.data(), 201 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy constants to device\n");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dZPositions, volume->_xRange.data(), 134 * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy constants to device\n");
        goto Error;
    }

    // KERNEL


    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after calling kernel\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(volume->getData(), dVolume, volume->getCount() * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Failed to copy volume data out of device\n");
        goto Error;
    }


Error:
    cudaFree(dRfData);
    cudaFree(dLocData);
    cudaFree(dConstants);
    cudaFree(dVolume);
    cudaFree(dXPositions);
    cudaFree(dYPositions);
    cudaFree(dZPositions);

    return cudaStatus;
}