#ifndef VOLUME_HH
#define VOLUME_HH

#include <thrust/host_vector.h>

#include <vector>

#include "defs.hh"

class Volume
{
public:

	

	Volume( const defs::VolumeDims& dims);

	~Volume();


	float* getData() const { return _data; }
	const float* end() const { return _end; }
	
	
	defs::VolumeDims getDims() const { return _dims; }

	std::vector<size_t> getCounts() const { return { _xCount, _yCount, _zCount }; }
	size_t getCount() const { return _elementCount; }
	size_t getXCount() const { return _xCount; }
	size_t getYCount() const { return _yCount; }
	size_t getZCount() const { return _zCount; }

	const float* getXRange() { return _xRange.data(); }
	const float* getYRange() { return _yRange.data(); }
	const float* getZRange() { return _zRange.data(); }
	
	float get_max_xz_dist() 
	{ 
		return sqrt(powf(_dims.xMax, 2) + powf(_dims.yMax, 2) + powf(_dims.zMax, 2)); 
	}

private:

	// _data owns the memory, _dataArray and _matlabArray will point at the same location
	float* _data;
	float* _end;

	defs::VolumeDims _dims;
	
	size_t _xCount;
	size_t _yCount;
	size_t _zCount;
	size_t _elementCount;

	std::vector<float> _xRange;
	std::vector<float> _yRange;
	std::vector<float> _zRange;

	
};


#endif // VOLUME_HH

