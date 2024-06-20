#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>
#include <complex>

#include <windows.h>
#include <stdio.h>

#include "defs.hh"

#include "dma_mapping.hh"



int dma::open_mapped_file(size_t size, std::string filepath, HANDLE* handle, LPVOID* buffer)
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

int dma::load_location_data(std::vector<float>** location_array)
{
    std::string filename = "Local\\LocationData";
    int result = 0;
    const int length = 3072; // Number of floats
    const size_t size = length * sizeof(float);

    HANDLE handle = nullptr;
    LPVOID buffer = nullptr;

    result = dma::open_mapped_file(size, filename, &handle, &buffer);
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

int dma::load_rf_data(ComplexVectorF** rx_data)
{
    int result = 0;
    std::string filename = "Local\\RfData";

    const int length = 4339200; // Number of values
    const size_t size = length * sizeof(std::complex<float>);

    HANDLE handle = nullptr;
    LPVOID buffer = nullptr;

    result = dma::open_mapped_file(size, filename, &handle, &buffer);

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

