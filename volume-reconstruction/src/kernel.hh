#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <complex>

#include "volume.hh"
#include "defs.hh"


cudaError_t volumeReconstruction(Volume* volume, const std::vector<std::complex<float>>* rf_data, const std::vector<float>* loc_data, defs::TransmitType tx_type, float3 src_pos, const defs::DataDims& dims);