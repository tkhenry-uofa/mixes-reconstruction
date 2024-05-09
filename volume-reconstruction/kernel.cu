
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>

#include <MatlabDataArray.hpp>
#include <MatlabEngine.hpp>

#include "volume.h"

static const float XMin = -15.0 / 1000;
static const float XMax = 15.0 / 1000;

static const float YMin = -15.0 / 1000;
static const float YMax = 15.0 / 1000;

static const float ZMin = 40.0 / 1000;
static const float ZMax = 60.0 / 1000;

static const float Resolution = 0.00015;

cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size);

namespace me = matlab::engine;
namespace md = matlab::data;

__global__ void addKernel(int* c, const int* a, const int* b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

void callSQRT() {

    using namespace matlab::engine;

    // Start MATLAB engine synchronously
    std::unique_ptr<MATLABEngine> matlabPtr = startMATLAB();

    //Create MATLAB data array factory
    matlab::data::ArrayFactory factory;

    // Define a four-element typed array
    matlab::data::TypedArray<double> const argArray =
        factory.createArray({ 1,4 }, { -2.0, 2.0, 6.0, 8.0 });

    // Call MATLAB sqrt function on the data array
    matlab::data::Array const results = matlabPtr->feval(u"sqrt", argArray);

    // Display results
    for (int i = 0; i < results.getNumberOfElements(); i++) {
        double a = argArray[i];
        std::complex<double> v = results[i];
        double realPart = v.real();
        double imgPart = v.imag();
        std::cout << "Square root of " << a << " is " <<
            realPart << " + " << imgPart << "i" << std::endl;
    }
}



int main()
{

    std::string path = R"(C:\Users\tkhen\source\repos\cuda\hello_world\data\psf_0050_16_scans.mat)";

    md::ArrayFactory factory;

    printf("Initializing matlab.\n");
    std::unique_ptr<me::MATLABEngine> engine = me::startMATLAB();

    printf("Loading matlab data.\n");

    std::unique_ptr<md::StructArray> fileContents;

    try {
        // Call MATLAB 'load' function

        fileContents.reset( new md::StructArray(engine->feval(u"load", factory.createCharArray(path))));

        // Displaying loaded data (Optional)
        /*engine->feval(u"disp", { result });*/

    }
    catch (const std::exception& e) {
        std::cerr << "Error loading file: " << e.what() << std::endl;
        return -1;
    }

    size_t fieldCount = fileContents->getNumberOfFields();

    if (fieldCount != 2)
    {
        std::cerr << "Expected 2 fields in file, instead found " << fieldCount << std::endl;
        return -1;
    }

    md::Range<md::ForwardIterator, md::MATLABFieldIdentifier const> fileRange = fileContents->getFieldNames();

    md::ForwardIterator<md::MATLABFieldIdentifier const> currentValue = fileRange.begin();

    std::vector<std::string> fieldNames;
    for (; currentValue != fileRange.end(); currentValue++)
    {
        fieldNames.push_back(*currentValue);
    }

    md::CellArray allRfData = (*fileContents)[0][fieldNames[0]];
    md::CellArray allLocData = (*fileContents)[0][fieldNames[1]];


    volume* vol = new volume(engine.get(), XMin, XMax, YMin, YMax, ZMin, ZMax, Resolution);

    delete vol;

    return 0;
}





// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int* c, const int* a, const int* b, unsigned int size)
{
    int* dev_a = 0;
    int* dev_b = 0;
    int* dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel <<<1, size >>> (dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);

    return cudaStatus;
}

