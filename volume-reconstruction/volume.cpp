#include "volume.h"

volume::volume( me::MATLABEngine *engine, float x_min, float x_max,
    float y_min, float y_max,
    float z_min, float z_max,
    float resolution) :
    _data(nullptr), _engine(engine),
    _xMin(x_min), _xMax(x_max),
    _yMin(y_min), _yMax(y_max),
    _zMin(z_min), _zMax(z_max),
    _resolution(resolution),
    _xCount(0), _yCount(0), _zCount(0)
{

    for (float x = _xMin; x <= _xMax; x += _resolution)
    {
        _xRange.push_back(x);
    }

    for (float y = _yMin; y <= _yMax; y += _resolution)
    {
        _yRange.push_back(y);
    }

    for (float z = _zMin; z <= _zMax; z += _resolution)
    {
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
}

volume::~volume()
{
    for (int i = 0; i < _xCount; i++) {
        for (int j = 0; j < _yCount; j++) {
            delete[] _data[i][j];
        }
        delete[] _data[i];
    }
    delete[] _data;

}
//
//mxArray*
//volume::to_mxArray()
//{
//
//    const mwSize size[3] = { _xCount, _yCount, _zCount };
//
//    mxArray* mData = mxCreateNumericArray(3, &size[0], mxSINGLE_CLASS, mxREAL);
//
//    for (int i = 0; i < _xCount; i++)
//    {
//        for (int j = 0; j < _yCount; j++)
//        {
//            for (int k = 0; k < _zCount; k++)
//            {
//                &(mData[i][j][k]) = _data[i][j][k];
//            }
//        }
//    }
//}