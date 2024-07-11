#ifndef VOLUME_HH
#define VOLUME_HH

#include <thrust/host_vector.h>

#include <vector>

#include "../defs.hh"

class Volume
{
public:

	

	Volume( const defs::VolumeDims& dims);

	~Volume();


	float* getData() const { return _data; }
	const float* end() const { return _end; }
	
	
	defs::VolumeDims getDims() const { return _dims; }

	std::vector<size_t> getCounts() const { return { _x_count, _y_count, _z_count }; }
	size_t getElementCount() const { return _element_count; }
	size_t getXCount() const { return _x_count; }
	size_t getYCount() const { return _y_count; }
	size_t getZCount() const { return _z_count; }

	const float* getXRange() { return _x_range.data(); }
	const float* getYRange() { return _y_range.data(); }
	const float* getZRange() { return _z_range.data(); }
	
	float getMaxXZDist() 
	{ 
		return sqrt(powf(_dims.x_max, 2) + powf(_dims.y_max, 2) + powf(_dims.z_max, 2)); 
	}

private:

	// _data owns the memory, _dataArray and _matlabArray will point at the same location
	float* _data;
	float* _end;

	defs::VolumeDims _dims;
	
	size_t _x_count;
	size_t _y_count;
	size_t _z_count;
	size_t _element_count;

	std::vector<float> _x_range;
	std::vector<float> _y_range;
	std::vector<float> _z_range;

	
};


#endif // VOLUME_HH

