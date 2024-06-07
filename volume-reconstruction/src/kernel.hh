#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <complex>

#include "volume.hh"

struct DataDims {
    float element_count;
    float sample_count;
    float transmission_count;
};

cudaError_t complexVolumeReconstruction(Volume* volume, const std::vector<std::complex<float>>& rf_data, const std::vector<float>& loc_data, const DataDims& dims);