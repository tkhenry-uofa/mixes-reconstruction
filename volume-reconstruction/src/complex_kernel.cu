
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <cuda/std/complex>
#include <math_constants.h>

#include "kernel.hh"

// Half array
/*#define MAX_LATERAL_RANGE 0.039116F 
#define INV_MAX_LATERAL_RANGE 25.5650F*/ 

// Full array
#define MAX_LATERAL_RANGE 0.039116F 
#define INV_MAX_LATERAL_RANGE 12.782493097453727F 


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
__global__ void complexDelayAndSum(const cuda::std::complex<float>* rfData, const float* locData, const float* xRange, const float* yRange, const float* zRange, float* volume, int sampleCount, int transmissionCount, float max_dist)
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
    float rx_distance;
    int scanIndex;
    float exPos, eyPos;
    float tx_distance;
    cuda::std::complex<float> value;
    
    // Beamform this voxel per element 
    for (int t = 0; t < transmissionCount; t++)
    {
        exPos = locData[2 * (t + e * transmissionCount)];
        eyPos = locData[2 * (t + e * transmissionCount) + 1];

        float apro = calculateAprodization(voxPos, { exPos, eyPos, 0.0f });

        // voxel to rx element
        rx_distance = norm3df(voxPos.x - exPos, voxPos.y - eyPos, voxPos.z);

        // Plane wave
        //distance = distance + voxPos.z;

        tx_distance = sqrt(powf(z_src - voxPos.z, 2) + powf(voxPos.x, 2)) + z_src;

        scanIndex = lroundf((rx_distance + tx_distance) * samplesPerMeter + pulse_delay);

        value = rfData[(t * sampleCount * elementCount) + (e * sampleCount) + scanIndex-1];
       // value = value * tx_distance / max_dist;
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

cudaError_t complexVolumeReconstruction(Volume* volume, const std::vector<std::complex<float>>& rf_data, const std::vector<float>& loc_data, const Defs::DataDims& data_dims)
{
    cuda::std::complex<float>* d_rf_data = 0;
    float* d_loc_data = 0;
    float* d_volume = 0;
    
    float* d_x_positions = 0;
    float* d_y_positions = 0;
    float* d_z_positions = 0;

    cudaError_t cuda_status;

    float* device_data[5] = {d_volume, d_x_positions, d_y_positions, d_z_positions, d_loc_data };

    // Transfer data to device
    {
        std::cout << "Allocating GPU memory" << std::endl;
        // Choose which GPU to run on, change this on a multi-GPU system.
        cuda_status = cudaSetDevice(0);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to connect to GPU\n");
            cleanupMemory(device_data,d_rf_data);
            return cuda_status;
        }

        // Malloc arrays on GPU
        size_t size = rf_data.size();
        cuda_status = cudaMalloc((void**)&d_rf_data, size * sizeof(cuda::std::complex<float>));
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to allocate rf array on device\n");
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }

        cuda_status = cudaMalloc((void**)&d_loc_data, loc_data.size() * sizeof(float));
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to allocate location array on device\n");
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }

        cuda_status = cudaMalloc((void**)&d_volume, volume->getCount() * sizeof(float));
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }

        cuda_status = cudaMalloc((void**)&d_x_positions, volume->getXCount() * sizeof(float));
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }

        cuda_status = cudaMalloc((void**)&d_y_positions, volume->getYCount() * sizeof(float));
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }

        cuda_status = cudaMalloc((void**)&d_z_positions, volume->getZCount() * sizeof(float));
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to allocate volume on device\n");
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }

        std::cout << "Transferring data to GPU" << std::endl;

        // Copy input vectors from host memory to GPU buffers.
        cuda_status = cudaMemcpy(d_rf_data, (void*)&rf_data.begin()[0], rf_data.size() * sizeof(cuda::std::complex<float>), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to copy RF data to device\n");
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }

        cuda_status = cudaMemcpy(d_loc_data, (void*)&loc_data.begin()[0], loc_data.size() * sizeof(float), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to copy location data to device\n");
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }

        cuda_status = cudaMemcpy(d_x_positions, volume->getXRange(), volume->getXCount() * sizeof(float), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to copy constants to device\n");
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }

        cuda_status = cudaMemcpy(d_y_positions, volume->getYRange(), volume->getYCount() * sizeof(float), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to copy constants to device\n");
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }

        cuda_status = cudaMemcpy(d_z_positions, volume->getZRange(), volume->getZCount() * sizeof(float), cudaMemcpyHostToDevice);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to copy constants to device\n");
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }
    }

    std::cout << "Starting kernel" << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    dim3 gridDim((unsigned int)volume->getXCount(), (unsigned int)volume->getYCount(), (unsigned int)volume->getZCount());

    float max_dist = volume->get_max_xz_dist();
    complexDelayAndSum<< <gridDim, 512 >> > (d_rf_data, d_loc_data, d_x_positions, d_y_positions, d_z_positions, d_volume, data_dims.sample_count, data_dims.transmission_count, max_dist);
    {
        // Transfer Data back
        // Check for any errors launching the kernel
        cuda_status = cudaGetLastError();
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cuda_status));
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }

        // cudaDeviceSynchronize waits for the kernel to finish, and returns
        // any errors encountered during the launch.
        cuda_status = cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;

        // Print the elapsed time
        std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "cudaDeviceSynchronize returned error code %d after calling kernel\n", cuda_status);
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }


        // Copy output vector from GPU buffer to host memory.
        cuda_status = cudaMemcpy(volume->getData(), d_volume, volume->getCount() * sizeof(float), cudaMemcpyDeviceToHost);
        if (cuda_status != cudaSuccess) {
            fprintf(stderr, "Failed to copy volume data out of device\n");
            cleanupMemory(device_data, d_rf_data);
            return cuda_status;
        }
    }


    cleanupMemory(device_data, d_rf_data);

    return cuda_status;
}


