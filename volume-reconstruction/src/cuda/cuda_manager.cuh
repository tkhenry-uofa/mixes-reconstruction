#ifndef CUDA_MANAGER_CUH
#define CUDA_MANAGER_CUH

#include <cuda_runtime.h>
#include <cuda/std/complex>

#include <vector>

#include "../defs.hh"
#include "../data_io/volume.hh"




class CudaManager
{

public:

	CudaManager(defs::TxConfig config);
	~CudaManager();

	bool transferLocData(const std::vector<float>& loc_data);

	bool transferRfData(const std::vector<std::complex<float>>& rf_data, const defs::RfDataDims& rf_dims);

	bool configureVolume(const defs::VolumeDims& dims);

	bool beamform(std::vector<float>* volume );

	ulonglong4 getVolumeDims()
	{
		return _vox_counts;
	}
	

private:

	void cleanupMemory();

	ulonglong4 _vox_counts;

	defs::TxConfig _tx_config;
	
	cuda::std::complex<float>* _d_rf_data = nullptr;

	float* _d_loc_data = nullptr;

	float* _d_volume = nullptr;

	cudaArray_t _position_arrays[3];

	defs::PositionTextures _textures;

	defs::RfDataDims _rf_dims;

};

#endif // !CUDA_MANAGER_CUH