#ifndef MAT_PARSER_H
#define MAT_PARSER_H


#include <vector>
#include <complex>
#include <string>

#include "defs.hh"
#include "volume.hh"

extern "C" {
	#include <mat.h>
	#include <matrix.h>
}

class MatParser
{
public:

	static bool
	SaveFloatArray(float* ptr, size_t dims[3], std::string file_path, std::string variable_name);

	MatParser():
		_array_count(0), 
		_rf_data_dims({0,0,0}), _tx_config({})
		{};

	~MatParser();

	bool 
	isOpen() { return _file != NULL; };

	// Opens the file and returns the array names
	bool 
	openFile(std::string file);

	std::vector<std::complex<float>>*
	getRfData() { return _rf_data.get(); };

	defs::RfDataDims
	getRfDims() { return _rf_data_dims; };

	std::vector<float>*
	getLocationData() { return _location_data.get(); };

	defs::TxConfig
	getTxConfig() { return _tx_config; };

	bool
	loadAllData();

	bool
	MatParser::loadTxConfig();


private:

	bool
	_loadRfDataArray();

	bool
	_loadLocationData();

	bool
	_loadTxConfig();

	defs::TxConfig _tx_config;
	
	std::vector<std::string> _array_names;
	defs::RfDataDims _rf_data_dims;

	int _array_count;

	MATFile* _file = NULL;

	std::unique_ptr<std::vector<std::complex<float>>> _rf_data;
	std::unique_ptr<std::vector<float>> _location_data;

};

#endif // !MAT_PARSER_H

