#ifndef DEFS_H
#define DEFS_H

#include <string>
#include <cuda_runtime.h>

namespace defs
{

	enum TransmitType
	{
		TX_PLANE = 0,
		TX_X_LINE = 1,
		TX_Y_LINE = 2
	};

	struct KernelConstants
	{
		size_t element_count;
		size_t sample_count;
		float3 src_pos;
		size_t tx_count;
		TransmitType tx_type;
	};


	struct DataDims {
		size_t element_count;
		size_t sample_count;
		size_t tx_count;
	};

	struct TxConfig {
		float f0; // Transducer frequency (Hz)
		float fs; // Data sample rate (Hz)

		int column_count; // Array column count
		int row_count; // Array row count;
		float width; // Element width (m)
		float pitch; // Element pitch (m)

		float x_min; // Transducer left elements (m)
		float x_max; // Transudcer right elements (m)
		float y_min; // Transducer bottom elements (m)
		float y_max; // Transducer top elements (m)

		int tx_count; // Number of transmittions
		float3 src_location; // Location of tx source (m)
		TransmitType transmit_type; // Transmit type
		float pulse_delay; // Delay to center of pulse (seconds)
	};


	struct VolumeDims {
		const float x_min;
		const float x_max;
		const float y_min;
		const float y_max;
		const float z_min;
		const float z_max;
		const float resolution;
	};

}




#endif // DEFS_H
