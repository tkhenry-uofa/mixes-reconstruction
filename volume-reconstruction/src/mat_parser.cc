#include <iostream>
#include <stdio.h>

#include "mat_parser.hh"

static const char* Rf_data_name = "rx_scans";
static const char* Loc_data_name = "rx_locs";
static const char* Tx_config_name = "tx_config";

static const char* F0_name = "f0";
static const char* Fs_name = "fs";

static const char* Column_count_name = "cols";
static const char* Row_count_name = "rows";
static const char* Width_name = "width";
static const char* Pitch_name = "pitch";

static const char* X_min_name = "x_min";
static const char* x_max_name = "x_max";
static const char* Y_min_name = "y_min";
static const char* Y_max_name = "y_max";

static const char* Tx_count_name = "no_transmits";
static const char* Src_location_name = "src";
static const char* Transmit_type_name = "transmit";
static const char* Pulse_delay_name = "pulse_delay";


MatParser::~MatParser()
{
    if (_file != NULL)
    {
        matClose(_file);
    }
}

bool
MatParser::openFile(std::string file)
{
    bool success = true;

    _file = matOpen(file.c_str(), "r");

    if (_file == NULL)
    {
        std::cerr << "Failed to open file: " << file << std::endl;
        return false;
    }

    success = success && _loadRfDataArray() && _loadLocationData() && _loadTxConfig();
    return success;
}

bool
MatParser::_loadRfDataArray()
{
    bool success = false;
    mxArray* mx_array = nullptr;

    // Get RF Data
    mx_array = matGetVariable(_file, Rf_data_name);
    if (mx_array == NULL) {
        std::cerr << "Error reading rf data array." << std::endl;
        return success;
    }

    if (mxIsComplex(mx_array))
    {
        const mwSize* rf_size = mxGetDimensions(mx_array);

        _rf_data_dims.sample_count = rf_size[0];
        _rf_data_dims.element_count = rf_size[1];
        _rf_data_dims.tx_count = rf_size[2];

        // mxComplexSingle and std::complex<float> are both structs of two floats so we can cast directly
        const size_t rf_total_count = mxGetNumberOfElements(mx_array);
        std::complex<float>* rf_data_p = reinterpret_cast<std::complex<float>*>(mxGetComplexSingles(mx_array));
        _rf_data.reset(new std::vector<std::complex<float>>(rf_data_p, &(rf_data_p[rf_total_count - 1])));
        success = true;
    }
    else
    {
        std::cerr << "RF Data array is not complex." << std::endl;
    }
    mxDestroyArray(mx_array);
    return success;
}

bool
MatParser::_loadLocationData()
{
    bool success = false;

    mxArray* mx_array = nullptr;

    // Get RF Data
    mx_array = matGetVariable(_file, Loc_data_name);
    if (mx_array == NULL) {
        std::cerr << "Error reading location data array." << std::endl;
        return success;
    }

    success = true;
    const size_t loc_total_count = mxGetNumberOfElements(mx_array);
    mxSingle* loc_data_p = mxGetSingles(mx_array);
    _location_data.reset(new std::vector<float>(loc_data_p, &(loc_data_p[loc_total_count - 1])));


    mxDestroyArray(mx_array);
    return success;
}

bool
MatParser::_loadTxConfig()
{
    bool success = false;

    mxArray* struct_array = nullptr;

    // Get RF Data
    struct_array = matGetVariable(_file, Tx_config_name);
    if (struct_array == NULL) {
        std::cerr << "Error reading tx configuration struct." << std::endl;
        return success;
    }


    // TODO: Catch log and throw null returns
    mxArray* field_p = mxGetField(struct_array, 0, F0_name);
    _tx_config.f0 = (int)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, Fs_name);
    _tx_config.fs = (int)*mxGetDoubles(field_p);


    field_p = mxGetField(struct_array, 0, Column_count_name);
    _tx_config.column_count = (int)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, Row_count_name);
    _tx_config.row_count = (int)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, Width_name);
    _tx_config.width = (float)*mxGetDoubles(field_p);
    /*field_p = mxGetField(struct_array, 0, Pitch_name);
    _tx_config.pitch = (float)*mxGetDoubles(field_p);*/

    field_p = mxGetField(struct_array, 0, X_min_name);
    _tx_config.x_min = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, x_max_name);
    _tx_config.x_max = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, Y_min_name);
    _tx_config.y_min = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, Y_max_name);
    _tx_config.y_max = (float)*mxGetDoubles(field_p);

    field_p = mxGetField(struct_array, 0, Tx_count_name);
    _tx_config.tx_count = (int)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, Pulse_delay_name);
    _tx_config.pulse_delay = (float)*mxGetDoubles(field_p);

    // TODO src_location and transmit type


    return success;
}

bool
MatParser::SaveFloatArray(float* ptr, size_t dims[3], std::string file_path, std::string variable_name)
{
    bool success = false;

    mxArray* volume_array = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
    mxSetSingles(volume_array, (mxSingle*)ptr);

    MATFile* file_p = matOpen(file_path.c_str(), "w");
    if (!file_p)
    {
        std::cerr << "Failed to open file for volume: " << file_path << std::endl;
        mxDestroyArray(volume_array);
        return success;
    }

    success = matPutVariable(file_p, variable_name.c_str(), volume_array);
    if (!success)
    {
        std::cerr << "Failed to save array to file." << std::endl;
    }

    matClose(file_p);

    return success;
}