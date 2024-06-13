
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <complex>

#include "kernel.hh"
#include "defs.hh"
#include "mat_parser.hh"
#include "volume.hh"



static const float XMin = 50.0f / 1000;
static const float XMax = 100.0f / 1000;

static const float YMin = -15.0f / 1000;
static const float YMax = 15.0f / 1000;

static const float ZMin = 60.0f / 1000;
static const float ZMax = 120.0f / 1000;

static const float Resolution = 0.0003f;

static const defs::VolumeDims Volume_Dimensions = { XMin, XMax, YMin, YMax, ZMin, ZMax, Resolution };

int main()
{

    Volume* volume = new Volume(Volume_Dimensions);

    std::string data_dir = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\cuda_data\)";
    std::string data_file = "4x4_cyst_side.mat";
    std::string full_path = data_dir + data_file;

    MatParser* parser = new MatParser();

    std::vector<std::string> names;
    
    if(!parser->openFile(full_path))
    {
        return 1;
    }

    float3 src_pos = { 0.0f, 0.0f, -0.004f };

    int result = volumeReconstruction(volume, parser->getRfData(), parser->getLocationData(), defs::TX_Y_LINE, src_pos, parser->getRfDims());
    std::string volume_path = data_dir + "beamformed_" + data_file;
    std::string variable_name = "volume";

    if (result == 0)
    {
        result = parser->SaveFloatArray(volume->get_data(), volume->get_counts().data(), volume_path, variable_name);
    }

    delete volume;
    delete parser;
   
    return result;
}


