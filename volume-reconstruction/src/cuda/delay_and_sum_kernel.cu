#include "delay_and_sum_kernel.cuh"

#define PULSE_DELAY 31
#define SAMPLES_PER_METER 32467.5F // 50 MHz, 1540 m/s


cudaError_t helpers::copy_constants(defs::KernelConstants consts)
{
    return cudaMemcpyToSymbol(Constants, &consts, sizeof(defs::KernelConstants));
}

__global__ void
kernels::complexDelayAndSum(const cuda::std::complex<float>* rfData, const float* locData, float* volume, const defs::PositionTextures textures)
{
    __shared__ cuda::std::complex<float> temp[THREADS_PER_BLOCK];

    int e = threadIdx.x;

    defs::KernelConstants consts = Constants;


    if (e >= consts.element_count)
    {
        return;
    }
    temp[e] = 0.0f;

    const float3 voxPos = { 
        tex1D<float>(textures.x, blockIdx.x),
        tex1D<float>(textures.y, blockIdx.y),
        tex1D<float>(textures.z, blockIdx.z) };

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

        float apro = 1.0f;

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
            // apro = yLineAprodization(voxPos, { exPos, eyPos, 0.0f });
            break;

        }

        scanIndex = lroundf((rx_distance + tx_distance) * SAMPLES_PER_METER + PULSE_DELAY);

        value = rfData[(t * Constants.sample_count * Constants.element_count) + (e * Constants.sample_count) + scanIndex - 1];
        temp[e] += value * apro;

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
        float value = norm3df(temp[0].real(), temp[0].imag(), 0.0f);
        volume[blockIdx.z * gridDim.x * gridDim.y + blockIdx.y * gridDim.x + blockIdx.x] = value;

    }
}