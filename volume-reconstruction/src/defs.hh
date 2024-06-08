#ifndef DEFS_H
#define DEFS_H

#include <string>
#include <cuda_runtime.h>

namespace defs
{
	enum TransmitType
	{
		TX_PLANE = 0,
		TX_X_LINE = 1,
		TX_Y_LINE = 2
	};

	struct KernelConstants
	{
		size_t element_count;
		float max_voxel_distance;
		size_t sample_count;
		float3 src_pos;
		size_t transmission_count;
		TransmitType tx_type;
	};



	struct DataDims {
		size_t element_count;
		size_t sample_count;
		size_t transmission_count;
	};


	struct VolumeDims {
		const float xMin;
		const float xMax;
		const float yMin;
		const float yMax;
		const float zMin;
		const float zMax;
		const float resolution;
	};

	static const std::string loc_data_name = "allLocs";
	static const std::string rf_data_name = "allScans";
}




#endif // DEFS_H
