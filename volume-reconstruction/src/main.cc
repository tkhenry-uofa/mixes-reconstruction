
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <complex>

#include "kernel.hh"
#include "defs.hh"
#include "mat_parser.hh"
#include "volume.hh"

static const float XMin = -20.0f / 1000;
static const float XMax = 20.0f / 1000;

static const float YMin = -20.0f / 1000;
static const float YMax = 20.0f / 1000;

static const float ZMin = 80.0f / 1000;
static const float ZMax = 100.0f / 1000;

static const float Resolution = 0.00015f;

static const defs::VolumeDims Volume_Dimensions = { XMin, XMax, YMin, YMax, ZMin, ZMax, Resolution };

int main()
{

    Volume* volume = new Volume(Volume_Dimensions);

    std::string data_dir = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\cuda_data\)";
    std::string data_file = "psf_90";
    std::string extension = ".mat";
    std::string full_path = data_dir + data_file + extension;

    MatParser* parser = new MatParser();

    std::vector<std::string> names;
    
    if(!parser->openFile(full_path))
    {
        return 1;
    }

    int result = volumeReconstruction(volume, parser->getRfData(), parser->getRfDims(), parser->getLocationData(), parser->getTxConfig());
    
    std::string volume_path = data_dir + data_file + "_beamformed" + extension;
    std::string variable_name = "volume";

    if (result == 0)
    {
        result = parser->SaveFloatArray(volume->get_data(), volume->get_counts().data(), volume_path, variable_name);
    }

    delete volume;
    delete parser;
   
    return result;
}


