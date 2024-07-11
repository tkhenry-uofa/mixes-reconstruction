#ifndef DELAY_AND_SUM_KERNEL_CUH
#define DELAY_AND_SUM_KERNEL_CUH


#include <cuda_runtime.h>
#include <cuda/std/complex>
#include <device_launch_parameters.h>
#include "../defs.hh"

#define THREADS_PER_BLOCK 512

__constant__ defs::KernelConstants Constants;

namespace helpers
{
	cudaError_t copy_constants(defs::KernelConstants consts);
}

namespace kernels
{
	__global__ void
	complexDelayAndSum(const cuda::std::complex<float>* rfData, const float* locData, float* volume, const defs::PositionTextures textures);
}




#endif // !DELAY_AND_SUM_KERNEL_CUH
