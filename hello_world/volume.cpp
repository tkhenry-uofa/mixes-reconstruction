#include "volume.h"

volume::volume(float x_min, float x_max,
	float y_min, float y_max,
	float z_min, float z_max,
	float resolution):
    _data(nullptr),
	_x_min(x_min), _x_max(x_max),
	_y_min(y_min), _y_max(y_max),
	_z_min(z_min), _z_max(z_max),
	_resolution(resolution),
	_x_count(0), _y_count(0), _z_count(0)
{

    for (float x = _x_min; x <= _x_max; x += _resolution) 
    {
        _x_range.push_back(x);
    }

    for (float y = _y_min; y <= _y_max; y += _resolution)
    {
        _y_range.push_back(y);
    }

    for (float z = _z_min; z <= _z_max; z += _resolution)
    {
        _z_range.push_back(z);
    }

    _x_count = _x_range.size();
    _y_count = _y_range.size();
    _z_count = _z_range.size();

    _data = new float** [_x_count];
    for (int i = 0; i < _x_count; i++)
    {
        _data[i] = new float* [_y_count];
        for (int j = 0; j < _y_count; j++)
        {
            _data[i][j] = new float[_z_count] {0};
        }
    }
}

volume::~volume()
{
    for (int i = 0; i < _x_count; i++) {
        for (int j = 0; j < _y_count; j++) {
            delete[] _data[i][j];
        }
        delete[] _data[i];
    }
    delete[] _data;

}