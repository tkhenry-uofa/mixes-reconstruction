#ifndef CELL_DATA_ARRAY_HH
#define CELL_DATA_ARRAY_HH

#include <MatlabDataArray.hpp>
#include <MatlabEngine.hpp>

class CellDataArray
{
public:

	CellDataArray(matlab::data::CellArray& cellArray);

	~CellDataArray();

	size_t getCount() const
	{
		return _cellCount * _rowCount * _columnCount;
	}

	float* getData() const { return &_data[0][0]; }

	size_t getCellCount() const { return _cellCount; }

	size_t getRowCount() const { return _rowCount; }

	size_t getColumnCount() const { return _columnCount; }


private:

	float** _data;

	size_t _cellCount;
	size_t _rowCount;
	size_t _columnCount;

};

#endif // CELL_DATA_ARRAY_HH
