
#include <string>
#include <vector>
#include <iostream>
#include <complex>

#include <windows.h>

#include "defs.hh"

#include "cuda/kernel.hh"

#include "data_io/dma_mapping.hh"
#include "data_io/mat_parser.hh"
#include "data_io/volume.hh"


static const float XMin = -20.0f / 1000;
static const float XMax = 20.0f / 1000;

static const float YMin = -20.0f / 1000;
static const float YMax = 20.0f / 1000;

static const float ZMin = 30.0f / 1000;
static const float ZMax = 70.0f / 1000;

static const float Resolution = 0.0003f;

static const defs::VolumeDims Volume_Dimensions = { XMin, XMax, YMin, YMax, ZMin, ZMax, Resolution };


int beamform_from_file(std::string filepath, std::string extension)
{
    Volume* volume = new Volume(Volume_Dimensions);
    MatParser* parser = new MatParser();

    if (!parser->openFile(filepath + extension) || !parser->loadAllData())
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
    std::string data_file = "fc_psf_50";

    //result = beamform_from_mapped_page(data_dir + data_file, ".mat");

    result = beamform_from_file(data_dir + data_file, ".mat");

    return result;

}


