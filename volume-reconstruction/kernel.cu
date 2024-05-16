
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>

#include "kernel.hh"

/*
*
*
* 
//*/
//__global__ void delayAndSum(const float* rfData,const float* locData, const GpuConstants* constants, const float* xRange, const float* yRange, const float *zRange, float* volume)
//{
//    // xyz dims 201, 201, 134 
//    int tx = threadIdx.x + blockIdx.x * 8;
//    int ty = threadIdx.y + blockIdx.y * 8;
//    int tz = threadIdx.z + blockIdx.z * 8;
//
//    const float Speed = 1.540f;
//    const float Fs = 100000.0f; // 100 MHz
//    
//    if (tx >= 201 || ty >= 201 || tz >= 134)
//    {
//        return;
//    }
//
//    int voxelId = tz * constants->xCount * constants->yCount + ty * constants->xCount + tx;
//
//    float voxel = volume[ voxelId ];
//    float xPos = xRange[tx];
//    float yPos = yRange[ty];
//    float zPos = zRange[tz];
//
//    float distance;
//    int scanIndex;
//    float exPos, eyPos, ezPos;
//    for (int t = 0; t < constants->transmissionCount; t++)
//    {
//        for (int e = 0; e < constants->elementCount; e++)
//        {
//            exPos = locData[3 * (t * constants->elementCount + e)];
//            eyPos = locData[3 * (t * constants->elementCount + e) + 1];
//            ezPos = locData[3 * (t * constants->elementCount + e) + 2];
//
//            // voxel to rx element
//            distance = sqrtf(powf(xPos - exPos, 2) + powf(yPos - eyPos, 2) + powf(zPos - ezPos, 2)) + zPos;
//
//            // tx element to voxel (only valid for plane waves under the shadow)
//           // distance = distance + zPos;
//
//            scanIndex = (int)roundf(distance * Fs/Speed);
//
//            if (scanIndex >= (constants->rfSampleCount * constants->elementCount * constants->transmissionCount ))
//            {
//                continue;
//            }
//
//            voxel = voxel + rfData[t * constants->rfSampleCount * constants->elementCount + e * constants->rfSampleCount + scanIndex];
//        }
//    }
//
//    volume[voxelId] = voxel;
//
//}

__global__ void delayAndSumFast(const cudaPitchedPtr rfData, const cudaPitchedPtr locData, const float* xRange, const float* yRange, const float* zRange, float* volume, int sampleCount, int transmissionCount)
{
    const int elementCount = 508;
    __shared__ float temp[elementCount];
    
    int e = threadIdx.x;

    int2 test;

    if (e >= elementCount)
    {
        return;
    }

    const float samplesPerMeter = 64935.0f; // Fs/c

    const float3 voxPos = { xRange[blockIdx.x], yRange[blockIdx.y], zRange[blockIdx.z] };
    
    float distance;
    int scanIndex;
    float exPos, eyPos;
    for (int t = 0; t < transmissionCount; t++)
    {
        exPos = ((float*)locData.ptr)[2 * (t + e * transmissionCount)];
        eyPos = ((float*)locData.ptr)[2 * (t + e * transmissionCount) + 1];

        // voxel to rx element
        distance = norm3df(voxPos.x - exPos, voxPos.y - eyPos, voxPos.z) + voxPos.z;


        // tx element to voxel (only valid for plane waves under the shadow)
        // distance = distance + zPos;

        scanIndex = (int)floorf(distance * samplesPerMeter);

        temp[e] += ((float*)rfData.ptr)[t * sampleCount * elementCount + e * sampleCount + scanIndex];

    }

    __syncthreads();
    
    // Sum reduction
    int index = 0;
    for (int s = 1; s < elementCount; s *= 2)
    {
        index = 2 * s * e;

        if (index < (elementCount - s))
        {
            temp[index] += temp[index + s];
        }
        
        __syncthreads();
    }

    if (e == 0)
    {
        volume[blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x] = temp[0];
    }
    
}



void
cleanupMemory(std::vector<void*> arrays)
{
    for (void* value : arrays)
    {
        cudaFree(value);
    }
}

cudaError_t volumeReconstruction(Volume* volume, const md::TypedArray<float>& rfData, const md::TypedArray<float>& locData)
{
    cudaPitchedPtr dRfData;
    cudaPitchedPtr dLocData;
    float* dVolume = nullptr;
    
    float* dXPositions = nullptr;
    float* dYPositions = nullptr;
    float* dZPositions = nullptr;

    cudaError_t cudaStatus;

    std::vector<void*> deviceData = { dVolume, dXPositions, dYPositions, dZPositions };

    std::vector<size_t> rfDims = rfData.getDimensions();

    unsigned int sampleCount = rfDims[0];
    unsigned int elementCount = rfDims[1];
    unsigned int transmissionCount = rfDims[2];



    // Transfer data to device
    {
        std::cout << "Allocating GPU memory" << std::endl;
        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to connect to GPU\n");
        }

        cudaExtent rfExtent = { rfDims[1] * sizeof(float), rfDims[0], rfDims[2] };
        cudaStatus = cudaMalloc3D(&dRfData, rfExtent);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate rf array on device\n");
            cleanupMemory(deviceData);
        }

        deviceData.push_back(dRfData.ptr);

        std::vector<size_t> locDims = locData.getDimensions();
        cudaExtent locExtent = { locDims[0] * sizeof(float), locDims[1], locDims[2] };
        cudaStatus = cudaMalloc3D(&dLocData, locExtent);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate rf array on device\n");
            cleanupMemory(deviceData);
        }

        deviceData.push_back(dLocData.ptr);

        cudaStatus = cudaMalloc((void**)&dVolume, volume->getCount() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(deviceData);
        }

        cudaStatus = cudaMalloc((void**)&dXPositions, volume->getXCount() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(deviceData);
        }

        cudaStatus = cudaMalloc((void**)&dYPositions, volume->getYCount() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(deviceData);
        }

        cudaStatus = cudaMalloc((void**)&dZPositions, volume->getZCount() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(deviceData);
        }

        std::cout << "Transferring data to GPU" << std::endl;


        cudaStatus = cudaMemcpy(dXPositions, volume->getXRange(), volume->getXCount() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy constants to device\n");
            cleanupMemory(deviceData);
        }

        cudaStatus = cudaMemcpy(dYPositions, volume->getYRange(), volume->getYCount() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy constants to device\n");
            cleanupMemory(deviceData);
        }

        cudaStatus = cudaMemcpy(dZPositions, volume->getZRange(), volume->getZCount() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy constants to device\n");
            cleanupMemory(deviceData);
        }
    }

    dim3 blockDim(8, 8, 8);
    dim3 gridDim(26, 26, 17);
    std::cout << "Starting kernel" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //delayAndSum<<<gridDim,blockDim>>>(dRfData, dLocData, dConstants, dXPositions, dYPositions, dZPositions, dVolume);

    dim3 gridDim2(201, 201, 134);
    delayAndSumFast<< <gridDim2, 512 >> > (dRfData, dLocData, dXPositions, dYPositions, dZPositions, dVolume, sampleCount, transmissionCount);
    {
        // Transfer Data back
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            cleanupMemory(deviceData);
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
            cleanupMemory(deviceData);
        }


        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(volume->getData(), dVolume, volume->getCount() * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy volume data out of device\n");
            cleanupMemory(deviceData);
        }
    }


    cleanupMemory(deviceData);

    return cudaStatus;
}


