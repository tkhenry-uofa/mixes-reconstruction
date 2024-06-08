#ifndef MAT_PARSER_H
#define MAT_PARSER_H

#include <mat.h>

#include <matrix.h>
#include <vector>
#include <complex>
#include <string>

#include "defs.hh"
#include "volume.hh"

namespace mat_parser
{
	int
	get_array_names(std::string file_path, int* array_count, std::vector<std::string>* names);

	int
	get_data_arrays(std::string file_path, std::vector<std::complex<float>>** rf_data, std::vector<float>** loc_data, Defs::DataDims* data_dims);

	int
	save_volume_data(Volume* vol, std::string file_path, std::string variable_name);
}

#endif // !MAT_PARSER_H

