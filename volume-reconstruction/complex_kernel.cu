
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <cuda/std/complex>
#include <math_constants.h>

#include "kernel.hh"

// Anything more than half the diagonal diameter of the array doesn't contribute
#define MAX_LATERAL_RANGE 0.039116F 
#define INV_MAX_LATERAL_RANGE 25.5650F 

/**
* Calculates a hann aprodization based on the lateral distance to the element
* Anything further than the distance from the middle of the array to the corner gets zeroed
*/
__device__ float calculateAprodization(float3 voxPosition, float3 elePosition)
{

    float x_dist = abs(voxPosition.x - elePosition.x);
    float y_dist = abs(voxPosition.y - elePosition.y);

    // Get the lateral distance between the voxel and the element
    float lateral_dist = sqrtf(powf(x_dist, 2) + powf(y_dist, 2));

    // Normalize and shift to map 0 to the peak of the window and 1 to the left end
    lateral_dist = lateral_dist * INV_MAX_LATERAL_RANGE;

    // Everything >= 1 gets set to zero
    lateral_dist = (lateral_dist > 1.0f) ? 1.0f : lateral_dist;

    // Compress to 0-0.5
    lateral_dist = lateral_dist / 2;

    float apro = powf(cosf(CUDART_PI_F * lateral_dist), 2);

    return apro;
}

// Blocks = voxels
// Threads = rx elements
__global__ void complexDelayAndSum(const cuda::std::complex<float>* rfData, const float* locData, const float* xRange, const float* yRange, const float* zRange, float* volume, int sampleCount, int transmissionCount)
{
    const int elementCount = 508;
    __shared__ cuda::std::complex<float> temp[elementCount];
    
    const int pulse_delay = 31; // Samples to delay 

    int e = threadIdx.x;


    if (e >= elementCount)
    {
        return;
    }
    temp[e] = 0.0f;

   // const float samplesPerMeter = 64935.0f; // Fs/c 100 MHz, 1540 m/s
    const float samplesPerMeter = 32467.5f; // 50 MHz

    const float3 voxPos = { xRange[blockIdx.x], yRange[blockIdx.y], zRange[blockIdx.z] };
    
    const float z_src = -0.006f;
    float distance;
    int scanIndex;
    float exPos, eyPos;
    cuda::std::complex<float> value;
    
    // Beamform this voxel per element 
    for (int t = 0; t < transmissionCount; t++)
    {
        exPos = locData[2 * (t + e * transmissionCount)];
        eyPos = locData[2 * (t + e * transmissionCount) + 1];

        float apro = calculateAprodization(voxPos, { exPos, eyPos, 0.0f });

        // voxel to rx element
        distance = norm3df(voxPos.x - exPos, voxPos.y - eyPos, voxPos.z);

        // Plane wave
        //distance = distance + voxPos.z;

        distance += sqrt(powf(z_src - voxPos.z, 2) + powf(voxPos.x, 2)) + z_src;

        scanIndex = (int)floorf(distance * samplesPerMeter + pulse_delay);

        value = rfData[t * sampleCount * elementCount + e * sampleCount + scanIndex];

        temp[e] += value*apro;

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
        float value = norm3df(temp[0].real(), temp[0].imag(),0.0f);
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

    std::cout << "Starting kernel" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    //delayAndSum<<<gridDim,blockDim>>>(dRfData, dLocData, dConstants, dXPositions, dYPositions, dZPositions, dVolume);

    dim3 gridDim(volume->getXCount(), volume->getYCount(), volume->getZCount());
    complexDelayAndSum<< <gridDim, 512 >> > (dRfData, dLocData, dXPositions, dYPositions, dZPositions, dVolume, sampleCount, transmissionCount);
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


