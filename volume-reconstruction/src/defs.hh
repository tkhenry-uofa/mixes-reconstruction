#ifndef DEFS_H
#define DEFS_H

#include <string>

namespace Defs
{
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
