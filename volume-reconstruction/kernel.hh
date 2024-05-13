#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "cellDataArray.hh"
#include "volume.hh"



cudaError_t volumeReconstruction(Volume* volume, const CellDataArray& rfData, const CellDataArray& locData);