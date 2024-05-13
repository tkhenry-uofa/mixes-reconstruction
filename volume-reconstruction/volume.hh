#ifndef VOLUME_HH
#define VOLUME_HH

#include <MatlabDataArray.hpp>
#include <MatlabEngine.hpp>

#include <thrust/host_vector.h>

#include <vector>

namespace me = matlab::engine;
namespace md = matlab::data;


class Volume
{
public:

	struct VolumeDims {
		const float xMin;
		const float xMax;
		const float yMin;
		const float yMax;
		const float zMin;
		const float zMax;
		const float resolution;
	};

	Volume( me::MATLABEngine* engine, const VolumeDims& dims);

	~Volume();

	size_t getCount() const
	{
		return (_xCount * _yCount * _zCount);
	}

	float* getData() const
	{
		return _data;
	}

	VolumeDims getDims() const
	{
		return _dims;
	}

	size_t getXCount() const
	{
		return _xCount;
	}

	size_t getYCount() const
	{
		return _yCount;
	}

	size_t getZCount() const
	{
		return _zCount;
	}
	
	// TODO: Move these back to private
	std::vector<float> _xRange;
	std::vector<float> _yRange;
	std::vector<float> _zRange;


private:

	// _data owns the memory, _dataArray and _matlabArray will point at the same location
	float* _data;
	float*** _dataArray;
	matlab::data::TypedArray<float>* _matlabArray;

	VolumeDims _dims;
	
	me::MATLABEngine const* _engine;

	size_t _xCount;
	size_t _yCount;
	size_t _zCount;

	
};


#endif // VOLUME_HH

