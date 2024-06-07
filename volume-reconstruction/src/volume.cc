#include <cstring>

#include "volume.hh"

Volume::Volume(const VolumeDims& dims): 
                _data(nullptr), _dims(dims), _end(nullptr)
{
    for (float x = _dims.xMin; x <= _dims.xMax; x += _dims.resolution) {
        _xRange.push_back(x);
    }
    for (float y = _dims.yMin; y <= _dims.yMax; y += _dims.resolution) {
        _yRange.push_back(y);
    }
    for (float z = _dims.zMin; z <= _dims.zMax; z += _dims.resolution) {
        _zRange.push_back(z);
    }

    _xCount = _xRange.size();
    _yCount = _yRange.size();
    _zCount = _zRange.size();
    _elementCount = _xCount * _yCount * _zCount;

    _data = new float[_elementCount];
    memset(_data, 0, _elementCount * sizeof(float));

    _end = &_data[_elementCount - 1];

}

Volume::~Volume()
{
    delete _data;
}

