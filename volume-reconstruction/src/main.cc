
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <complex>

#include "kernel.hh"
#include "defs.hh"
#include "mat_parser.hh"
#include "volume.hh"



static const float XMin = -15.0f / 1000;
static const float XMax = 15.0f / 1000;

static const float YMin = -15.0f / 1000;
static const float YMax = 15.0f / 1000;

static const float ZMin = 40.0f / 1000;
static const float ZMax = 50.0f / 1000;

static const float Resolution = 0.0003f;

static const defs::VolumeDims vDims = { XMin, XMax, YMin, YMax, ZMin, ZMax, Resolution };

int 
matlab_c_api(Volume* volume, std::string data_dir, std::string data_file)
{
    int result = 0;
    std::string full_path = data_dir + data_file;
    float3 src_pos = { 0.0f, 0.0f, -0.006f };

    int array_count;
    std::vector<std::string> array_names;
    result = mat_parser::get_array_names(full_path, &array_count, &array_names);

    std::vector<std::complex<float>>* rf_data = nullptr;
    std::vector<float>* loc_data = nullptr;
    defs::DataDims data_dims;
    result = mat_parser::get_data_arrays(full_path, &rf_data, &loc_data, &data_dims);
    if (result != 0)
    {
        std::cerr << "Failed to get data arrays" << std::endl;
        return 1;
    }

    
    result = volumeReconstruction(volume, *rf_data, *loc_data, defs::TX_Y_LINE, src_pos, data_dims);

    delete rf_data;
    delete loc_data;


    return result;
}


int main()
{
    int result = 0;
    Volume* vol = new Volume(vDims);

    std::string data_dir = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\cuda_data\)";
    std::string data_file = "troubleshooting.mat";

    result = matlab_c_api(vol, data_dir, data_file);

    std::string volume_path = data_dir + "beamformed_" + data_file;
    std::string variable_name = "volume";

    result = mat_parser::save_volume_data(vol, volume_path, variable_name);
    

    delete vol;
    return result;
}


