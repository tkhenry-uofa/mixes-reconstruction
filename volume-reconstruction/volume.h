#pragma once

#include <MatlabDataArray.hpp>
#include <MatlabEngine.hpp>

#include <vector>

namespace me = matlab::engine;
namespace md = matlab::data;

class volume
{
public:

	volume( me::MATLABEngine* engine, float x_min, float x_max,
		float y_min, float y_max,
		float z_min, float z_max,
		float resolution);

	~volume();

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

	float _xMin;
	float _xMax;
	float _yMin;
	float _yMax;
	float _zMin;
	float _zMax;
	float _resolution;

	size_t _xCount;
	size_t _yCount;
	size_t _zCount;

	std::vector<float> _xRange;
	std::vector<float> _yRange;
	std::vector<float> _zRange;

	me::MATLABEngine const * _engine;
};

