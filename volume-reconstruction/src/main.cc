
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

static const float XMin = -20.0f / 1000;
static const float XMax = 20.0f / 1000;

static const float YMin = -20.0f / 1000;
static const float YMax = 20.0f / 1000;

static const float ZMin = 30.0f / 1000;
static const float ZMax = 50.0f / 1000;

static const float Resolution = 0.00015f;

static const defs::VolumeDims Volume_Dimensions = { XMin, XMax, YMin, YMax, ZMin, ZMax, Resolution };


int open_mapped_file(size_t size, std::string filepath, HANDLE* handle, LPVOID* buffer)
{


    *handle = OpenFileMapping(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        filepath.c_str());

    if (*handle == NULL) {
        std::cerr << "Could not open file mapping object: " << GetLastError() << std::endl;
        return 1;
    }

    *buffer = MapViewOfFile(
        *handle,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        size);

    if (*buffer == NULL) {
        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
        CloseHandle(*handle);
        return 1;
    }

    return 0;
}

int load_location_data(std::vector<float>** location_array)
{
    std::string filename = "Local\\LocationData";
    int result = 0;
    const int length = 3072; // Number of floats
    const size_t size = length * sizeof(float);

    HANDLE handle = nullptr;
    LPVOID buffer = nullptr;

    result = open_mapped_file(size, filename, &handle, &buffer);
    if (result != 0)
    {
        return result;
    }

    float* loc_data_p = static_cast<float*>(buffer);

    *location_array = new std::vector<float>(loc_data_p, &(loc_data_p[length]));

    UnmapViewOfFile(buffer);
    CloseHandle(handle);

    // Clean up
    return 0;
}

int load_rf_data(ComplexVectorF** rx_data)
{
    int result = 0;
    std::string filename = "Local\\RfData";

    const int length = 4339200; // Number of values
    const size_t size = length * sizeof(std::complex<float>);

    HANDLE handle = nullptr;
    LPVOID buffer = nullptr;

    result = open_mapped_file(size, filename, &handle, &buffer);

    if (result != 0)
    {
        return result;
    }

    std::complex<float>* rf_data_p = reinterpret_cast<std::complex<float>*>(buffer);
    *rx_data = new ComplexVectorF(rf_data_p, &(rf_data_p[length]));

    UnmapViewOfFile(buffer);
    CloseHandle(handle);


    return result;
}




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

    result = load_location_data(&location_array);
    result = load_rf_data(&rf_data);

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


