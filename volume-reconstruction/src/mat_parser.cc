#include <iostream>
#include <stdio.h>

#include "mat_parser.hh"


int
mat_parser::get_array_names(std::string file_path, int* array_count, std::vector<std::string>* names)
{
    MATFile* file_p = matOpen(file_path.c_str(), "r");

    if (file_p == NULL)
    {
        std::cerr << "Failed to open file: " << file_path << std::endl;
        return 1;
    }

    const char** dir = (const char**)matGetDir(file_p, array_count);
    if (dir == NULL) {
        printf("Error reading directory of file %s\n", file_path.c_str());
        return(1);
    }
    else
    {
        for (int i = 0; i < *array_count; i++)
        {
            names->push_back(dir[i]);
        }
    }

    mxFree(dir);
    matClose(file_p);


    return 0;
};

int
mat_parser::get_data_arrays(std::string file_path, std::vector<std::complex<float>>** rf_data, std::vector<float>** loc_data, defs::DataDims* data_dims)
{
    int result = 0;


    mxArray* rf_array = nullptr;
    mxArray* loc_array = nullptr;

    std::vector<std::complex<float>>* rf_vec = nullptr;
    std::vector<float>* loc_vec = nullptr;

    MATFile* file_p = matOpen(file_path.c_str(), "r");
    if (file_p == NULL) {
        printf("Error opening file %s\n", file_path.c_str());
        result = 1;
        goto Cleanup;
    }

    // Get RF Data
    rf_array = matGetVariable(file_p, defs::rf_data_name.c_str());
    if (rf_array == NULL) {
        printf("Error reading in rf_data\n");
        result = 1;
        goto Cleanup;
    }

    if (!mxIsComplex(rf_array))
    {
        std::cerr << "RF Data array is not complex." << std::endl;
        result = 1;
        goto Cleanup;
    }

    const mwSize* rf_size = mxGetDimensions(rf_array);

    data_dims->sample_count = rf_size[0];
    data_dims->element_count = rf_size[1];
    data_dims->transmission_count = rf_size[2];

    // Get loc data
    loc_array = matGetVariable(file_p, defs::loc_data_name.c_str());
    if (rf_array == NULL) {
        printf("Error reading in rf_data\n");
        result = 1;
        goto Cleanup;
    }

    // mxComplexSingle and std::complex<float> are both structs of two floats so we can cast directly
    const size_t rf_total_count = mxGetNumberOfElements(rf_array);
    std::complex<float>* rf_data_p = reinterpret_cast<std::complex<float>*>(mxGetComplexSingles(rf_array));
    *rf_data = new std::vector<std::complex<float>>(rf_data_p, &(rf_data_p[rf_total_count - 1]));

    const size_t loc_total_count = mxGetNumberOfElements(loc_array);
    mxSingle* loc_data_p = mxGetSingles(loc_array);
    *loc_data = new std::vector<float>(loc_data_p, &(loc_data_p[loc_total_count - 1]));

Cleanup:

    if (matClose(file_p) != 0) {
        printf("Error closing file %s\n", file_path.c_str());
        result = 1;
    }

    mxDestroyArray(rf_array);
    mxDestroyArray(loc_array);

    return result;
};

int
mat_parser::save_volume_data(Volume* vol, std::string file_path, std::string variable_name)
{
    int result = 0;

    mwSize dims[3] = { vol->getXCount(), vol->getYCount(), vol->getZCount() };

    mxArray* volume_array = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);

    mxSingle* start = (mxSingle*)vol->getData();

    mxSetSingles(volume_array, start);


    MATFile* file_p = matOpen(file_path.c_str(), "w");
    if (!file_p)
    {
        std::cerr << "Failed to open file for volume: " << file_path << std::endl;
        result = 1;
        goto Cleanup;
    }

    result = matPutVariable(file_p, variable_name.c_str(), volume_array);
    if (result != 0)
    {
        matError mat_error = matGetErrno(file_p);
        perror("Error");
        std::cerr << "Failed load volume into file." << std::endl;
        result = 1;
        goto Cleanup;
    }

Cleanup:
    if (matClose(file_p) != 0) {
        std::cerr << "Failed to close file: " << file_path << std::endl;
        result = 1;
    }

    return result;
}

