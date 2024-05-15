#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "volume.hh"



cudaError_t volumeReconstruction(Volume* volume, const md::TypedArray<float>& rfData, const md::TypedArray<float>& locData);