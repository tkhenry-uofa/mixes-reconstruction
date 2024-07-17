#include <iostream>
#include <stdexcept>
#include <chrono>

#include "delay_and_sum_kernel.cuh"
#include "cuda_manager.cuh"

CudaManager::CudaManager( defs::TxConfig config) : _tx_config(config), _textures({0,0,0}), _position_arrays(), _vox_counts({0,0,0,0}), _rf_dims({0,0,0}), _d_volume(nullptr)
{
	std::cout << "Allocating GPU Memory" << std::endl;
	cudaError_t cuda_status = cudaSetDevice(0);

	_position_arrays[0] = nullptr;
	_position_arrays[1] = nullptr;
	_position_arrays[2] = nullptr;

	if (cuda_status != cudaSuccess)
	{
		std::cerr << "Failed set cuda device" << std::endl;
		throw std::invalid_argument("Invalid cuda device");
	}
}

CudaManager::~CudaManager()
{
	cleanupMemory();
}

void
CudaManager::cleanupMemory()
{
	cudaFree(_d_rf_data);
	cudaFree(_d_loc_data);
	cudaFree(_d_volume);

	cudaFreeArray(_position_arrays[0]);
	cudaFreeArray(_position_arrays[1]);
	cudaFreeArray(_position_arrays[2]);
}

bool
CudaManager::configureVolume(const defs::VolumeDims& dims)
{
	std::vector<float> x_range;
	std::vector<float> y_range;
	std::vector<float> z_range;

	// Destroy old data
	cudaDestroyTextureObject(_textures.x);
	cudaDestroyTextureObject(_textures.y);
	cudaDestroyTextureObject(_textures.z);

	cudaFreeArray(_position_arrays[0]);
	cudaFreeArray(_position_arrays[1]);
	cudaFreeArray(_position_arrays[2]);

	cudaFree(_d_volume);

	for (float x = dims.x_min; x <= dims.x_max; x += dims.resolution) {
		x_range.push_back(x);
	}
	for (float y = dims.y_min; y <= dims.y_max; y += dims.resolution) {
		y_range.push_back(y);
	}
	for (float z = dims.z_min; z <= dims.z_max; z += dims.resolution) {
		z_range.push_back(z);
	}

	_vox_counts = { z_range.size(), y_range.size(), z_range.size(), x_range.size() * y_range.size() * z_range.size() };


	cudaError_t cuda_status = cudaMalloc((void**)&_d_volume, _vox_counts.w * sizeof(float));
	RETURN_IF_ERROR(cuda_status, "Failed to malloc volume on device.")

	// TEXTURE SETUP
	// 32 bits in the channel 
	cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(sizeof(float) * 8, 0, 0, 0, cudaChannelFormatKindFloat);

	cudaTextureDesc tex_desc;
	memset(&tex_desc, 0, sizeof(cudaTextureDesc));
	tex_desc.addressMode[0] = cudaAddressModeClamp;
	tex_desc.filterMode = cudaFilterModePoint;
	tex_desc.readMode = cudaReadModeElementType;
	tex_desc.normalizedCoords = false;

	cudaResourceDesc tex_res_desc;
	memset(&tex_res_desc, 0, sizeof(cudaResourceDesc));
	tex_res_desc.resType = cudaResourceTypeArray;
	
	cudaError_t malloc_status = cudaMallocArray(&_position_arrays[0], &channel_desc, _vox_counts.x);
	cudaError_t memcpy_status = cudaMemcpyToArray(_position_arrays[0], 0, 0, x_range.data(), _vox_counts.x * sizeof(float), cudaMemcpyHostToDevice);
	tex_res_desc.res.array.array = _position_arrays[0];
	cudaError_t bind_status = cudaCreateTextureObject(&_textures.x, &tex_res_desc, &tex_desc, NULL);

	if (malloc_status != cudaSuccess || memcpy_status != cudaSuccess || bind_status != cudaSuccess)
	{
		std::cerr << "Failed to create x texture." << std::endl;
		return false;
	}
	
	malloc_status = cudaMallocArray(&_position_arrays[1], &channel_desc, _vox_counts.y);
	memcpy_status = cudaMemcpyToArray(_position_arrays[1], 0, 0, y_range.data(), _vox_counts.y * sizeof(float), cudaMemcpyHostToDevice);
	tex_res_desc.res.array.array = _position_arrays[1];
	bind_status = cudaCreateTextureObject(&_textures.y, &tex_res_desc, &tex_desc, NULL);

	if (malloc_status != cudaSuccess || memcpy_status != cudaSuccess || bind_status != cudaSuccess)
	{
		std::cerr << "Failed to create y texture." << std::endl;
		return false;
	}

	
	malloc_status = cudaMallocArray(&_position_arrays[2], &channel_desc, _vox_counts.z);
	memcpy_status = cudaMemcpyToArray(_position_arrays[2], 0, 0, z_range.data(), _vox_counts.z * sizeof(float), cudaMemcpyHostToDevice);
	tex_res_desc.res.array.array = _position_arrays[2];
	bind_status = cudaCreateTextureObject(&_textures.z, &tex_res_desc, &tex_desc, NULL);

	if (malloc_status != cudaSuccess || memcpy_status != cudaSuccess || bind_status != cudaSuccess)
	{
		std::cerr << "Failed to create z texture." << std::endl;
		return false;
	}
	
	return true;
}

bool
CudaManager::transferLocData(const std::vector<float>& loc_data)
{

	if (_d_loc_data != nullptr)
	{
		cudaFree(_d_loc_data);
	}

	cudaError_t cuda_status = cudaMalloc((void**)&_d_loc_data, loc_data.size() * sizeof(float));
	RETURN_IF_ERROR(cuda_status, "Failed to malloc location array on device.")

	cuda_status = cudaMemcpy(_d_loc_data, (void*)loc_data.data(), loc_data.size() * sizeof(float), cudaMemcpyHostToDevice);
	RETURN_IF_ERROR(cuda_status, "Failed to copy location array to device.")

	return cuda_status == cudaSuccess;
}

bool
CudaManager::transferRfData(const std::vector<std::complex<float>>& rf_data, const defs::RfDataDims& rf_dims)
{
	if (_d_rf_data != nullptr)
	{
		cudaFree(_d_rf_data);
	}

	_rf_dims = rf_dims;
	size_t rf_data_size= rf_data.size();
	cudaError_t cuda_status = cudaMalloc((void**)&_d_rf_data, rf_data_size * sizeof(std::complex<float>));
	RETURN_IF_ERROR(cuda_status, "Failed to malloc rf data on device.")

	cuda_status = cudaMemcpy(_d_rf_data, (void*)rf_data.data(), rf_data.size() * sizeof(std::complex<float>), cudaMemcpyHostToDevice);
	RETURN_IF_ERROR(cuda_status, "Failed to copy rf data to device.")

	return cuda_status == cudaSuccess;
}

bool
CudaManager::beamform(std::vector<float>** volume)
{
	*volume = nullptr;
	defs::KernelConstants const_struct =
	{
		_rf_dims.element_count,
		_rf_dims.sample_count,
		_tx_config.src_location,
		_rf_dims.tx_count,
		_tx_config.transmit_type,
		_vox_counts
	};

	cudaError_t cuda_status = helpers::copy_constants(const_struct);
	RETURN_IF_ERROR(cuda_status, "Failed to copy constants to device.")


	dim3 gridDim((unsigned int)_vox_counts.x, (unsigned int)_vox_counts.y, (unsigned int)_vox_counts.z);
	auto start = std::chrono::high_resolution_clock::now();
	kernels::complexDelayAndSum <<<gridDim, THREADS_PER_BLOCK >>> (_d_rf_data, _d_loc_data, _d_volume, _textures);

	cuda_status = cudaGetLastError();
	RETURN_IF_ERROR(cuda_status, "Kernel failed.")
	cuda_status = cudaDeviceSynchronize();
	RETURN_IF_ERROR(cuda_status, "Sync failed.")

	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double> elapsed = end - start;
	std::cout << "Kernel duration: " << elapsed.count() << " seconds" << std::endl;

	*volume = new std::vector<float>(_vox_counts.w);
	cuda_status = cudaMemcpy((*volume)->data(), _d_volume, _vox_counts.w * sizeof(float), cudaMemcpyDeviceToHost);
	RETURN_IF_ERROR(cuda_status, "Copying volume to CPU failed.")

	return cuda_status == cudaSuccess;
	
}