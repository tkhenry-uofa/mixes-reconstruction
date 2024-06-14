#ifndef DEFS_H
#define DEFS_H

#include <string>
#include <cuda_runtime.h>

namespace defs
{
	// Matlab strings are u16, even though variable names aren't
	static const std::u16string Plane_tx_name = u"plane";
	static const std::u16string X_line_tx_name = u"xLine";
	static const std::u16string Y_line_tx_name = u"yLine";
	
	static const char* Rf_data_name = "rx_scans";
	static const char* Loc_data_name = "rx_locs";
	static const char* Tx_config_name = "tx_config";

	static const char* F0_name = "f0";
	static const char* Fs_name = "fs";

	static const char* Column_count_name = "cols";
	static const char* Row_count_name = "rows";
	static const char* Width_name = "width";
	static const char* Pitch_name = "pitch";

	static const char* X_min_name = "x_min";
	static const char* x_max_name = "x_max";
	static const char* Y_min_name = "y_min";
	static const char* Y_max_name = "y_max";

	static const char* Tx_count_name = "no_transmits";
	static const char* Src_location_name = "src";
	static const char* Transmit_type_name = "transmit";
	static const char* Pulse_delay_name = "pulse_delay";

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
