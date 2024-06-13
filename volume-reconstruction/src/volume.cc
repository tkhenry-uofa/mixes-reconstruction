#include <cstring>

#include "volume.hh"

Volume::Volume(const defs::VolumeDims& dims): 
                _data(nullptr), _dims(dims), _end(nullptr)
{
    for (float x = _dims.x_min; x <= _dims.x_max; x += _dims.resolution) {
        _x_range.push_back(x);
    }
    for (float y = _dims.y_min; y <= _dims.y_max; y += _dims.resolution) {
        _y_range.push_back(y);
    }
    for (float z = _dims.z_min; z <= _dims.z_max; z += _dims.resolution) {
        _z_range.push_back(z);
    }

    _x_count = _x_range.size();
    _y_count = _y_range.size();
    _z_count = _z_range.size();
    _element_count = _x_count * _y_count * _z_count;

    _data = new float[_element_count];
    memset(_data, 0, _element_count * sizeof(float));

    _end = &_data[_element_count];

}

Volume::~Volume()
{
    delete _data;
}

