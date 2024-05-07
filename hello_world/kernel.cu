
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>

#include <mat.h>
#include <matrix.h>

#include "volume.h"

const float X_min = -15.0 / 1000;
const float X_max = 15.0 / 1000;

const float Y_min = -15.0 / 1000;
const float Y_max = 15.0 / 1000;

const float Z_min = 40.0 / 1000;
const float Z_max = 60.0 / 1000;

const float Resolution = 0.00015;

cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size);

__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}



int main()
{
    /*const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };*/

    std::string path = R"(C:\Users\tkhen\source\repos\cuda\hello_world\data\psf_0050_16_scans.mat)";

    MATFile *file = matOpen(path.c_str(), "r");

    std::vector<std::string> varNames;
    std::vector<mxArray*> mat_vars;
    std::string name;
    mxArray* var = NULL;
    int size = 0;
    if (!file)
    {
        fprintf(stderr, "Failed to open .mat file.");
        return 1;
    }
    else
    {
        char** rawNames = matGetDir(file, &size);

        if (size != 2)
        {
            fprintf(stderr, "Expected two matlab variables, instead read %d.", size);
            return 1;
        }

        printf("Loading matlab data, variable names:\n");


        for (int i = 0; i < size; i++)
        {
            name = std::string(rawNames[i]);
            varNames.push_back(name);
            std::cout << name << std::endl;

            var = matGetVariable(file, name.c_str());

            if (!var)
            {
                printf("Failed to load var '%s'.", name.c_str());
                return 1;
            }

            mat_vars.push_back(var);
        } 
    }

   
    const int acquisition_count = mxGetDimensions(mat_vars[0])[0];


    std::vector<const mxArray*> rf_data_array;
    std::vector<const mxArray*> el_location_array;

    for (int i = 0; i < acquisition_count; i++)
    {
        rf_data_array.push_back(mxGetCell(mat_vars[0],i));
        el_location_array.push_back(mxGetCell(mat_vars[1], i));
    }
    
    const mxArray* test = rf_data_array[0];

    const int sample_count = mxGetDimensions(test)[0];
    const int element_count = mxGetDimensions(test)[1];


    volume* vol = new volume(X_min, X_max, Y_min, Y_max, Z_min, Z_max, Resolution);

    delete vol;



    //// Add vectors in parallel.
    //cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "addWithCuda failed!");
    //    return 1;
    //}

    //printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
    //    c[0], c[1], c[2], c[3], c[4]);

    //// cudaDeviceReset must be called before exiting in order for profiling and
    //// tracing tools such as Nsight and Visual Profiler to show complete traces.
    //cudaStatus = cudaDeviceReset();
    //if (cudaStatus != cudaSuccess) {
    //    fprintf(stderr, "cudaDeviceReset failed!");
    //    return 1;
    //}


    return 0;
}





// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
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
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

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

