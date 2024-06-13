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


	float* get_data() const { return _data; }
	const float* end() const { return _end; }
	
	
	defs::VolumeDims get_dims() const { return _dims; }

	std::vector<size_t> get_counts() const { return { _x_count, _y_count, _z_count }; }
	size_t get_element_count() const { return _element_count; }
	size_t get_x_count() const { return _x_count; }
	size_t get_y_count() const { return _y_count; }
	size_t get_z_count() const { return _z_count; }

	const float* get_x_range() { return _x_range.data(); }
	const float* get_y_range() { return _y_range.data(); }
	const float* get_z_range() { return _z_range.data(); }
	
	float get_max_xz_dist() 
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

