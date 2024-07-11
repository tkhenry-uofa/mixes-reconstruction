
#include <string>
#include <vector>
#include <iostream>
#include <complex>

#include <windows.h>

#include "defs.hh"

#include "cuda/cuda_manager.cuh"
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
        result = parser->SaveFloatArray(volume->getData(), volume->getCounts().data(), volume_path, variable_name);
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
        result = parser->SaveFloatArray(volume->getData(), volume->getCounts().data(), volume_path, variable_name);
    }

    delete volume;
    delete parser;
    delete rf_data;
    delete location_array;

    return result;
}

int beamform_with_class(std::string filepath, std::string extension)
{
    MatParser* parser = new MatParser();

    if (!parser->openFile(filepath + extension) || !parser->loadAllData())
    {
        delete parser;
        return 1;
    }

    std::vector<float>* volume = nullptr;

    CudaManager* beamformer = new CudaManager(parser->getTxConfig());

    bool result = beamformer->transferLocData(*parser->getLocationData()) &&
        beamformer->configureVolume(Volume_Dimensions) &&
        beamformer->transferRfData(*parser->getRfData(), parser->getRfDims());

    if (!result)
    {
        return 1;
    }

    result = beamformer->beamform(volume);
    std::string volume_path = filepath + "_beamformed" + extension;
    std::string variable_name = "volume";

    if (result == true)
    {
        ulonglong4 vol_vec = beamformer->getVolumeDims();
        size_t vol_dims[3] = { vol_vec.x, vol_vec.y, vol_vec.z };
        result = parser->SaveFloatArray(volume->data(),vol_dims , volume_path, variable_name);
    }

    delete volume;
    delete parser;

    return result == false;
}

int main()
{
    int result = 0;

    std::string data_dir = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\oct\)";
    std::string data_file = "fc_psf_50";

    //result = beamform_from_mapped_page(data_dir + data_file, ".mat");

   // result = beamform_from_file(data_dir + data_file, ".mat");

    result = beamform_with_class(data_dir + data_file, ".mat");

    return result;

}


