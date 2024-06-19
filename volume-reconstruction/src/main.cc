
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


int beamform_from_file()
{
    Volume* volume = new Volume(Volume_Dimensions);

    std::string data_dir = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\oct\)";
    std::string data_file = "psf_40_full_sample";
    std::string extension = ".mat";
    std::string full_path = data_dir + data_file + extension;

    MatParser* parser = new MatParser();

    std::vector<std::string> names;

    if (!parser->openFile(full_path))
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

int load_from_page()
{
    const char* sharedMemoryName = "Local\\LocationData";
    const size_t size = 16384;
    const int length = 3072;

    HANDLE hMapFile = OpenFileMapping(
        FILE_MAP_ALL_ACCESS,
        FALSE,
        sharedMemoryName);

    if (hMapFile == NULL) {
        std::cerr << "Could not open file mapping object: " << GetLastError() << std::endl;
        return 1;
    }

    LPVOID pBuf = MapViewOfFile(
        hMapFile,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        size);

    if (pBuf == NULL) {
        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        return 1;
    }

    float* loc_data_p = static_cast<float*>(pBuf);

    std::vector<float>loc_data(loc_data_p, &(loc_data_p[length - 1]));

    // Clean up
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);

    return 0;
}


int main()
{
    int result = 0;
    //result = beamform_from_file();
    result = load_from_page();

    return result;

}


