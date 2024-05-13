#include "cellDataArray.hh"

CellDataArray::CellDataArray(matlab::data::CellArray& cellArray):
    _data(nullptr), _cellCount(0), _rowCount(0), _columnCount(0)
{
    // Determine the size of the cell array
    _cellCount = cellArray.getNumberOfElements();

    // Assuming each cell contains a 2D float array, create a 3D pointer array
    _data = new float* [_cellCount];
    _rowCount = cellArray[0].getDimensions()[0];
    _columnCount = cellArray[0].getDimensions()[1];
    
    // Loop through each cell
    for (size_t i = 0; i < _cellCount; ++i) {
        // Access each cell as a 2D array
        matlab::data::TypedArray<float> matrix = cellArray[i];

        _data[i] = &matrix.begin()[0];
    }
}

CellDataArray::~CellDataArray()
{
    // The pointers in the array are still owned by the matlab variable.
    delete[] _data;
}