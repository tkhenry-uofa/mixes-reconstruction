
#include <string>
#include <vector>

#include <windows.h>

#include "defs.hh"

namespace dma {
	int open_mapped_file(size_t size, std::string filepath, HANDLE* handle, LPVOID* buffer);

	int load_location_data(std::vector<float>** location_array);

	int load_rf_data(ComplexVectorF** rx_data);
}




