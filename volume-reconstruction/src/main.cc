
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <complex>

#include "defs.hh"
#include "mat_parser.hh"
#include "volume.hh"
#include "kernel.hh"


static const float XMin = -5.0f / 1000;
static const float XMax = 75.0f / 1000;

static const float YMin = -5.0f / 1000;
static const float YMax = 5.0f / 1000;

static const float ZMin = 15.0f / 1000;
static const float ZMax = 85.0f / 1000;

static const float Resolution = 0.0003f;

static const Defs::VolumeDims vDims = { XMin, XMax, YMin, YMax, ZMin, ZMax, Resolution };

int 
matlab_c_api(Volume* volume, std::string data_dir, std::string data_file)
{
    int result = 0;
    std::string full_path = data_dir + data_file;

    int array_count;
    std::vector<std::string> array_names;
    result = mat_parser::get_array_names(full_path, &array_count, &array_names);

    std::vector<std::complex<float>>* rf_data = nullptr;
    std::vector<float>* loc_data = nullptr;
    Defs::DataDims data_dims;
    result = mat_parser::get_data_arrays(full_path, &rf_data, &loc_data, &data_dims);
    if (result != 0)
    {
        std::cerr << "Failed to get data arrays" << std::endl;
        goto Cleanup;
    }

    result = complexVolumeReconstruction(volume, *rf_data, *loc_data, data_dims);



Cleanup:
    delete rf_data;
    delete loc_data;


    return result;
}


int main()
{
    int result = 0;
    Volume* vol = new Volume(vDims);

    std::string data_dir = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\cuda_data\)";
    std::string data_file = "div_side.mat";

    result = matlab_c_api(vol, data_dir, data_file);

    std::string volume_path = data_dir + "beamformed_" + data_file;
    std::string variable_name = "volume";

    result = mat_parser::save_volume_data(vol, volume_path, variable_name);
    

    delete vol;
    return result;
}


