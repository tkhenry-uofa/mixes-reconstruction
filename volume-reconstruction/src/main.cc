
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <complex>

#include <windows.h>
#include <stdio.h>

#include "kernel.hh"
#include "defs.hh"
#include "mat_parser.hh"
#include "volume.hh"
#include "dma_mapping.hh"

static const float XMin = -20.0f / 1000;
static const float XMax = 20.0f / 1000;

static const float YMin = -20.0f / 1000;
static const float YMax = 20.0f / 1000;

static const float ZMin = 30.0f / 1000;
static const float ZMax = 50.0f / 1000;

static const float Resolution = 0.00015f;

static const defs::VolumeDims Volume_Dimensions = { XMin, XMax, YMin, YMax, ZMin, ZMax, Resolution };


int beamform_from_file(std::string filepath, std::string extension)
{
    Volume* volume = new Volume(Volume_Dimensions);
    MatParser* parser = new MatParser();

    if (!parser->openFile(filepath + extension) && !parser->loadAllData())
    {
        delete volume;
        delete parser;
        return 1;
    }

    int result = volumeReconstruction(volume, parser->getRfData(), parser->getRfDims(), parser->getLocationData(), parser->getTxConfig());

    std::string volume_path = filepath+ "_beamformed" + extension;
    std::string variable_name = "volume";

    if (result == 0)
    {
        result = parser->SaveFloatArray(volume->get_data(), volume->get_counts().data(), volume_path, variable_name);
    }

    delete volume;
    delete parser;

    return result;
}

int beamform_from_mapped_page(std::string filepath, std::string extension)
{
    int result = 0;

    Volume* volume = new Volume(Volume_Dimensions);
    MatParser* parser = new MatParser();

    if (!parser->openFile(filepath + extension) && !parser->loadTxConfig())
    {
        delete volume;
        delete parser;
        return 1;
    }

    std::vector<float>* location_array = nullptr;
    ComplexVectorF* rf_data = nullptr;

    result = dma::load_location_data(&location_array);
    result = dma::load_rf_data(&rf_data);

    defs::RfDataDims dims = { 512,2825,3 };

    result = volumeReconstruction(volume, rf_data, dims, location_array, parser->getTxConfig());

    std::string volume_path = filepath + "_beamformed" + extension;
    std::string variable_name = "volume";

    if (result == 0)
    {
        result = parser->SaveFloatArray(volume->get_data(), volume->get_counts().data(), volume_path, variable_name);
    }

    delete volume;
    delete parser;
    delete rf_data;
    delete location_array;

    return result;
}



int main()
{
    int result = 0;

    std::string data_dir = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\oct\)";
    std::string data_file = "psf_40_full_sample";

    result = beamform_from_mapped_page(data_dir + data_file, ".mat");

    return result;

}


