#include <iostream>
#include <stdio.h>

#include "mat_parser.hh"


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
    mx_array = matGetVariable(_file, defs::Rf_data_name);
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
    mx_array = matGetVariable(_file, defs::Loc_data_name);
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
    struct_array = matGetVariable(_file, defs::Tx_config_name);
    if (struct_array == NULL) {
        std::cerr << "Error reading tx configuration struct." << std::endl;
        return success;
    }

    // TODO: Catch log and throw null returns
    mxArray* field_p = mxGetField(struct_array, 0, defs::F0_name);
    _tx_config.f0 = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Fs_name);
    _tx_config.fs = (float)*mxGetDoubles(field_p);


    field_p = mxGetField(struct_array, 0, defs::Column_count_name);
    _tx_config.column_count = (int)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Row_count_name);
    _tx_config.row_count = (int)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Width_name);
    _tx_config.width = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Pitch_name);
    _tx_config.pitch = (float)*mxGetDoubles(field_p);

    field_p = mxGetField(struct_array, 0, defs::X_min_name);
    _tx_config.x_min = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::x_max_name);
    _tx_config.x_max = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Y_min_name);
    _tx_config.y_min = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Y_max_name);
    _tx_config.y_max = (float)*mxGetDoubles(field_p);
     
    field_p = mxGetField(struct_array, 0, defs::Tx_count_name);
    _tx_config.tx_count = (int)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Pulse_delay_name);
    _tx_config.pulse_delay = (float)*mxGetDoubles(field_p);

    field_p = mxGetField(struct_array, 0, defs::Src_location_name);
    float* src_locs = (float*)mxGetDoubles(field_p);
    _tx_config.src_location = { src_locs[0], src_locs[1], src_locs[2] };

    field_p = mxGetField(struct_array, 0, defs::Transmit_type_name);
    std::u16string tx_name =  (const char16_t*)(mxGetChars(field_p));

    if (tx_name == defs::Plane_tx_name)
    {
        _tx_config.transmit_type = defs::TX_PLANE;
    }
    else if (tx_name == defs::X_line_tx_name)
    {
        _tx_config.transmit_type = defs::TX_X_LINE;
    }
    else if (tx_name == defs::Y_line_tx_name)
    {
        _tx_config.transmit_type = defs::TX_Y_LINE;
    }
    else
    {
        std::cerr << "Invalid transmit type (read the file)." << std::endl;
    }

    success = true;
    return success;
}

bool
MatParser::SaveFloatArray(float* ptr, size_t dims[3], std::string file_path, std::string variable_name)
{

    mxArray* volume_array = mxCreateNumericArray(3, dims, mxSINGLE_CLASS, mxREAL);
    mxSetSingles(volume_array, (mxSingle*)ptr);

    MATFile* file_p = matOpen(file_path.c_str(), "w");
    if (!file_p)
    {
        std::cerr << "Failed to open file for volume: " << file_path << std::endl;
        mxDestroyArray(volume_array);
        return 1;
    }

    int error = matPutVariable(file_p, variable_name.c_str(), volume_array);
    if (error)
    {
        std::cerr << "Failed to save array to file." << std::endl;
        return error;
    }

    matClose(file_p);

    return 0;
}