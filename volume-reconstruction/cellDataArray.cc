#include <algorithm>

#include "cellDataArray.hh"

CellDataArray::CellDataArray(matlab::data::CellArray& cellArray):
    _data(nullptr), _cellCount(0), _rowCount(0), _columnCount(0), _totalCount(0)
{
    // Determine the size of the cell array
    _cellCount = cellArray.getNumberOfElements();

    

    // Assuming each cell contains a 2D float array, create a 3D pointer array
    _rowCount = cellArray[0].getDimensions()[0];
    _columnCount = cellArray[0].getDimensions()[1];

    _totalCount = _cellCount * _rowCount * _columnCount;
  
    _data = new float[_totalCount];

    //// Loop through each cell
    //matlab::data::TypedArray<float> const matrix;
    //for (size_t i = 0; i < _cellCount; ++i) {

    //    matrix = cellArray[i][0];

    //    std::copy(&matrix.begin(), &matrix.end(), _data[i * _rowCount * _columnCount]);
    //}

    cellArray.release();
}

CellDataArray::~CellDataArray()
{
    // The pointers in the array are still owned by the matlab variable.
    delete[] _data;
}