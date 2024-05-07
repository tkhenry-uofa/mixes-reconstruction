#pragma once

#include <vector>

class volume
{
public:

	volume(float x_min, float x_max, 
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
		return _x_count;
	}

	inline int get_y_count()
	{
		return _y_count;
	}

	inline int get_z_count()
	{
		return _z_count;
	}

private:

	float*** _data;

	float _x_min;
	float _x_max;
	float _y_min; 
	float _y_max;
	float _z_min; 
	float _z_max;
	float _resolution;

	size_t _x_count;
	size_t _y_count;
	size_t _z_count;

	std::vector<float> _x_range;
	std::vector<float> _y_range;
	std::vector<float> _z_range;

};

