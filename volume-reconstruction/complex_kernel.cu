
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <cuda/std/complex>

#include "kernel.hh"


__global__ void complexDelayAndSum(const cuda::std::complex<float>* rfData, const float* locData, const float* xRange, const float* yRange, const float* zRange, float* volume, int sampleCount, int transmissionCount)
{
    const int elementCount = 508;
    __shared__ cuda::std::complex<float> temp[elementCount];
    
    int e = threadIdx.x;

    int2 test;

    if (e >= elementCount)
    {
        return;
    }
    temp[e] = 0.0f;

    // const float samplesPerMeter = 64935.0f; // Fs/c 100 MHz, 1540 m/s
    const float samplesPerMeter = 32467.5f; // 50 MHz

    const float3 voxPos = { xRange[blockIdx.x], yRange[blockIdx.y], zRange[blockIdx.z] };
    
    float distance;
    int scanIndex;
    float exPos, eyPos;
    cuda::std::complex<float> value;
    for (int t = 0; t < transmissionCount; t++)
    {
        exPos = locData[2 * (t + e * transmissionCount)];
        eyPos = locData[2 * (t + e * transmissionCount) + 1];

        // voxel to rx element
        distance = norm3df(voxPos.x - exPos, voxPos.y - eyPos, voxPos.z) + voxPos.z;

        // tx element to voxel (only valid for plane waves under the shadow)
        // distance = distance + zPos;

        scanIndex = (int)floorf(distance * samplesPerMeter);

        value = rfData[t * sampleCount * elementCount + e * sampleCount + scanIndex];

        temp[e] += value;

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

        float zero = 0.0f;
        float value = norm3df(temp[0].real(), temp[0].imag(),zero);
        volume[blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x] = value;
    }
    
}



void
cleanupMemory(float* floats[5], cuda::std::complex<float>* rfData)
{
    for (int i = 0; i < 6; i++)
    {
        cudaFree(floats[i]);
    }
    cudaFree(rfData);
}

cudaError_t complexVolumeReconstruction(Volume* volume, const md::TypedArray<std::complex<float>>& rfData, const md::TypedArray<float>& locData)
{
    cuda::std::complex<float>* dRfData = 0;
    float* dLocData = 0;
    float* dVolume = 0;
    
    float* dXPositions = 0;
    float* dYPositions = 0;
    float* dZPositions = 0;

    cudaError_t cudaStatus;

    float* deviceData[5] = {dVolume, dXPositions, dYPositions, dZPositions, dLocData };

    std::vector<size_t> rfDims = rfData.getDimensions();

    int sampleCount = rfDims[0];
    int transmissionCount = rfDims[2];

    // Transfer data to device
    {
        std::cout << "Allocating GPU memory" << std::endl;
        // Choose which GPU to run on, change this on a multi-GPU system.
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to connect to GPU\n");
            cleanupMemory(deviceData,dRfData);
            return cudaStatus;
        }

        // Malloc arrays on GPU
        cudaStatus = cudaMalloc((void**)&dRfData, rfData.getNumberOfElements() * sizeof(cuda::std::complex<float>));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate rf array on device\n");
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
        }

        cudaStatus = cudaMalloc((void**)&dLocData, locData.getNumberOfElements() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate location array on device\n");
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
        }

        cudaStatus = cudaMalloc((void**)&dVolume, volume->getCount() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
        }

        cudaStatus = cudaMalloc((void**)&dXPositions, volume->getXCount() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
        }

        cudaStatus = cudaMalloc((void**)&dYPositions, volume->getYCount() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
        }

        cudaStatus = cudaMalloc((void**)&dZPositions, volume->getZCount() * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
        }

        std::cout << "Transferring data to GPU" << std::endl;

        // Copy input vectors from host memory to GPU buffers.
        cudaStatus = cudaMemcpy(dRfData, (void*)&rfData.begin()[0], rfData.getNumberOfElements() * sizeof(cuda::std::complex<float>), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy RF data to device\n");
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
        }

        cudaStatus = cudaMemcpy(dLocData, (void*)&locData.begin()[0], locData.getNumberOfElements() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy location data to device\n");
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
        }

        cudaStatus = cudaMemcpy(dXPositions, volume->getXRange(), volume->getXCount() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy constants to device\n");
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
        }

        cudaStatus = cudaMemcpy(dYPositions, volume->getYRange(), volume->getYCount() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy constants to device\n");
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
        }

        cudaStatus = cudaMemcpy(dZPositions, volume->getZRange(), volume->getZCount() * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy constants to device\n");
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
        }
    }

    dim3 blockDim(8, 8, 8);
    dim3 gridDim(26, 26, 17);
    std::cout << "Starting kernel" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //delayAndSum<<<gridDim,blockDim>>>(dRfData, dLocData, dConstants, dXPositions, dYPositions, dZPositions, dVolume);

    dim3 gridDim2(201, 201, 134);
    complexDelayAndSum<< <gridDim2, 512 >> > (dRfData, dLocData, dXPositions, dYPositions, dZPositions, dVolume, sampleCount, transmissionCount);
    {
        // Transfer Data back
        // Check for any errors launching the kernel
        cudaStatus = cudaGetLastError();
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
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
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
        }


        // Copy output vector from GPU buffer to host memory.
        cudaStatus = cudaMemcpy(volume->getData(), dVolume, volume->getCount() * sizeof(float), cudaMemcpyDeviceToHost);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "Failed to copy volume data out of device\n");
            cleanupMemory(deviceData, dRfData);
            return cudaStatus;
        }
    }


    cleanupMemory(deviceData, dRfData);

    return cudaStatus;
}


