
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <cuda/std/complex>
#include <cuda_runtime.h>
#include <math_constants.h>


#include "kernel.hh"


#define RETURN_IF_ERROR(STATUS, MESSAGE)\
if (STATUS != cudaSuccess) {            \
    std::cerr << MESSAGE << std::endl;  \
    cleanupMemory(device_data);         \
    return STATUS;                      \
}   

// Half array
#define INV_MAX_LATERAL_RANGE 25.5650F

//#define INV_MAX_LATERAL_RANGE 12.782493097453727F // Full array diagonal width
#define PULSE_DELAY 31
#define SAMPLES_PER_METER 32467.5F // 50 MHz, 1540 m/s

#define THREADS_PER_BLOCK 512


__constant__ defs::KernelConstants Constants;

static cudaError_t
load_constants(const defs::TxConfig& tx_config, const defs::RfDataDims& rf_dims)
{
    defs::KernelConstants const_struct =
    {   
        rf_dims.element_count,
        rf_dims.sample_count,
        tx_config.src_location,
        rf_dims.tx_count,
        tx_config.transmit_type
    };
    cudaError_t error = cudaMemcpyToSymbol(Constants, &const_struct, sizeof(defs::KernelConstants));
    return error;
}

/**
* Calculates a hann aprodization based on the lateral distance to the element
* Anything further than the distance from the middle of the array to the corner gets zeroed
*/
__device__ float 
calculateAprodization(float3 voxPosition, float3 elePosition)
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
    lateral_dist = lateral_dist * 0.5f;

    float apro = powf(cosf(CUDART_PI_F * lateral_dist), 2);

    return apro;
}

// Blocks = voxels
// Threads = rx elements
__global__ void 
complexDelayAndSum(const cuda::std::complex<float>* rfData, const float* locData, const float* xRange, const float* yRange, const float* zRange, float* volume)
{

    __shared__ cuda::std::complex<float> temp[THREADS_PER_BLOCK];
    
    int e = threadIdx.x;


    if (e >= Constants.element_count)
    {
        return;
    }
    temp[e] = 0.0f;

    const float3 voxPos = { xRange[blockIdx.x], yRange[blockIdx.y], zRange[blockIdx.z] };
    
    float rx_distance;
    int scanIndex;
    float exPos, eyPos;
    float tx_distance;
    cuda::std::complex<float> value;
    // Beamform this voxel per element 
    for (int t = 0; t < Constants.tx_count; t++)
    {
        exPos = locData[2 * (t + e * Constants.tx_count)];
        eyPos = locData[2 * (t + e * Constants.tx_count) + 1];

        float apro = calculateAprodization(voxPos, { exPos, eyPos, 0.0f });

        // voxel to rx element
        rx_distance = norm3df(voxPos.x - exPos, voxPos.y - eyPos, voxPos.z);

        // Plane wave
        
        switch (Constants.tx_type)
        {
            case defs::TX_PLANE:
                tx_distance = voxPos.z;
                break;

            case defs::TX_X_LINE:
                tx_distance = sqrt(powf(Constants.src_pos.z - voxPos.z, 2) + powf(Constants.src_pos.y - voxPos.y, 2)) + Constants.src_pos.z;
                break;

            case defs::TX_Y_LINE:
                tx_distance = sqrt(powf(Constants.src_pos.z - voxPos.z, 2) + powf(Constants.src_pos.x - voxPos.x, 2)) + Constants.src_pos.z;
                break;

        }

        scanIndex = lroundf((rx_distance + tx_distance) * SAMPLES_PER_METER + PULSE_DELAY);

        value = rfData[(t * Constants.sample_count * Constants.element_count) + (e * Constants.sample_count) + scanIndex-1];
        temp[e] += value;

    }

    __syncthreads();
    
    // Sum reduction
    int index = 0;
    for (int s = 1; s < Constants.element_count; s *= 2)
    {
        index = 2 * s * e;

        if (index < (Constants.element_count - s))
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
cleanupMemory(void* ptrs[6])
{
    for (int i = 0; i < 6; i++)
    {
        cudaFree(ptrs[i]);
    }
}

cudaError_t 
volumeReconstruction(Volume* volume, const std::vector<std::complex<float>>* rf_data, const defs::RfDataDims& rf_dims, const std::vector<float>* loc_data, const defs::TxConfig& tx_config)
{
    cuda::std::complex<float>* d_rf_data = 0;
    float* d_loc_data = 0;
    float* d_volume = 0;
    
    float* d_x_positions = 0;
    float* d_y_positions = 0;
    float* d_z_positions = 0;

    cudaError_t cuda_status;

    void* device_data[6] = {d_rf_data, d_volume, d_x_positions, d_y_positions, d_z_positions, d_loc_data };


    std::cout << "Allocating GPU Memory" << std::endl;
    cuda_status = cudaSetDevice(0);
    RETURN_IF_ERROR(cuda_status, "Failed to connect to GPU");

    size_t size = rf_data->size();
    cuda_status = cudaMalloc((void**)&d_rf_data, size * sizeof(cuda::std::complex<float>));
    RETURN_IF_ERROR(cuda_status, "Failed to allocate rf array on device.");

    cuda_status = cudaMalloc((void**)&d_loc_data, loc_data->size() * sizeof(float));
    RETURN_IF_ERROR(cuda_status, "Failed to allocate location array on device.");

    cuda_status = cudaMalloc((void**)&d_volume, volume->get_element_count() * sizeof(float));
    RETURN_IF_ERROR(cuda_status, "Failed to allocate volume on device.");

    cuda_status = cudaMalloc((void**)&d_x_positions, volume->get_x_count() * sizeof(float));
    RETURN_IF_ERROR(cuda_status, "Failed to allocate x data on device.");

    cuda_status = cudaMalloc((void**)&d_y_positions, volume->get_y_count() * sizeof(float));
    RETURN_IF_ERROR(cuda_status, "Failed to allocate y data on device.");

    cuda_status = cudaMalloc((void**)&d_z_positions, volume->get_z_count() * sizeof(float));
    RETURN_IF_ERROR(cuda_status, "Failed to allocate z data on device.");

    std::cout << "Transferring data to GPU" << std::endl;

    cuda_status = cudaMemcpy(d_rf_data, (void*)&rf_data->begin()[0], rf_data->size() * sizeof(cuda::std::complex<float>), cudaMemcpyHostToDevice);
    RETURN_IF_ERROR(cuda_status, "Failed to copy rf data to device.");

    cuda_status = cudaMemcpy(d_loc_data, (void*)&loc_data->begin()[0], loc_data->size() * sizeof(float), cudaMemcpyHostToDevice);
    RETURN_IF_ERROR(cuda_status, "Failed to copy location data to device.");

    cuda_status = cudaMemcpy(d_x_positions, volume->get_x_range(), volume->get_x_count() * sizeof(float), cudaMemcpyHostToDevice);
    RETURN_IF_ERROR(cuda_status, "Failed to copy x data to device.");

    cuda_status = cudaMemcpy(d_y_positions, volume->get_y_range(), volume->get_y_count() * sizeof(float), cudaMemcpyHostToDevice);
    RETURN_IF_ERROR(cuda_status, "Failed to copy y data to device.");

    cuda_status = cudaMemcpy(d_z_positions, volume->get_z_range(), volume->get_z_count() * sizeof(float), cudaMemcpyHostToDevice);
    RETURN_IF_ERROR(cuda_status, "Failed to copy z data to device.");

    cuda_status = load_constants(tx_config, rf_dims);
    RETURN_IF_ERROR(cuda_status, "Failed to send constant data to device.");


    std::cout << "Starting kernel" << std::endl;

    dim3 gridDim((unsigned int)volume->get_x_count(), (unsigned int)volume->get_y_count(), (unsigned int)volume->get_z_count());
    auto start = std::chrono::high_resolution_clock::now();
    complexDelayAndSum<<<gridDim, THREADS_PER_BLOCK >>>(d_rf_data, d_loc_data, d_x_positions, d_y_positions, d_z_positions, d_volume);

   
    cuda_status = cudaGetLastError();
    RETURN_IF_ERROR(cuda_status, "Kernel failed with error: " << cudaGetErrorString(cuda_status));

    cuda_status = cudaDeviceSynchronize();
    RETURN_IF_ERROR(cuda_status, "Cuda error code returned after sync: " << cuda_status);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;
    
    cuda_status = cudaMemcpy(volume->get_data(), d_volume, volume->get_element_count() * sizeof(float), cudaMemcpyDeviceToHost);
    RETURN_IF_ERROR(cuda_status, "Failed to copy volume from device.");
    
    cleanupMemory(device_data);

    return cuda_status;
}


