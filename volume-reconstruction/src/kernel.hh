#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <complex>

#include "volume.hh"
#include "defs.hh"

cudaError_t complexVolumeReconstruction(Volume* volume, const std::vector<std::complex<float>>& rf_data, const std::vector<float>& loc_data, const Defs::DataDims& dims);