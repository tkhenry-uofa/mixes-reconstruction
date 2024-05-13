#include "volume.hh"

Volume::Volume(me::MATLABEngine* engine, const VolumeDims& dims): 
                _data(nullptr), _dataArray(nullptr), 
                _dims(dims), _engine(engine), 
                _matlabArray(nullptr) 
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

    _data = new float[_xCount * _yCount * _zCount] {0};

}

Volume::~Volume()
{
    delete _data;
    delete _matlabArray;

}

