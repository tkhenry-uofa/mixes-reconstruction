#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <complex>

#include "volume.hh"
#include "defs.hh"


cudaError_t volumeReconstruction(Volume* volume, const std::vector<std::complex<float>>* rf_data, const defs::RfDataDims& rf_dims, const std::vector<float>* loc_data, const defs::TxConfig& tx_config);