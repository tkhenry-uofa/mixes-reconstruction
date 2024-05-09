#include "volume.h"

Volume::Volume(me::MATLABEngine* engine, const VolumeDims& dims): 
                _engine(engine), _dims(dims) {
    
    
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

    _data = new float** [_xCount];
    for (int i = 0; i < _xCount; i++)
    {
        _data[i] = new float* [_yCount];
        for (int j = 0; j < _yCount; j++)
        {
            _data[i][j] = new float[_zCount] {0};
        }
    }

    // TODO: Change to thrust 
    _dataVector.resize(_xCount, std::vector<std::vector<float>>(_yCount, std::vector<float>(_zCount, 0.0f)));
}

Volume::~Volume()
{
    for (int i = 0; i < _xCount; i++) {
        for (int j = 0; j < _yCount; j++) {
            delete[] _data[i][j];
        }
        delete[] _data[i];
    }
    delete[] _data;

}
