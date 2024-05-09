#pragma once

#include <MatlabDataArray.hpp>
#include <MatlabEngine.hpp>

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

	inline float*** get_data()
	{
		return _data;
	}

	inline int get_x_count()
	{
		return _xCount;
	}

	inline int get_y_count()
	{
		return _yCount;
	}

	inline int get_z_count()
	{
		return _zCount;
	}


private:

	float*** _data;

	std::vector<std::vector<std::vector<float>>> _dataVector;

	VolumeDims _dims;

	size_t _xCount;
	size_t _yCount;
	size_t _zCount;

	std::vector<float> _xRange;
	std::vector<float> _yRange;
	std::vector<float> _zRange;

	me::MATLABEngine const* _engine;
};

