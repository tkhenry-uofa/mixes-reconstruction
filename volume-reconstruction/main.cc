
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>

#include <MatlabDataArray.hpp>
#include <MatlabEngine.hpp>

#include "cellDataArray.hh"
#include "volume.hh"
#include "kernel.hh"

static const float XMin = -15.0f / 1000;
static const float XMax = 15.0f / 1000;

static const float YMin = -15.0f / 1000;
static const float YMax = 15.0f / 1000;

static const float ZMin = 40.0f / 1000;
static const float ZMax = 60.0f / 1000;

static const float Resolution = 0.00015f;

static const Volume::VolumeDims VolumeDims = { XMin, XMax, YMin, YMax, ZMin, ZMax, Resolution };


namespace me = matlab::engine;
namespace md = matlab::data;

int main()
{

    std::string dataRoot = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\cuda_data\)";

    std::string dataPath = dataRoot + "psf_0050_16_scans.mat";

    md::ArrayFactory factory;

    printf("Initializing matlab\n");
    std::unique_ptr<me::MATLABEngine> engine = me::startMATLAB();

    printf("Loading matlab data\n");

    std::unique_ptr<md::StructArray> fileContents;

    try {

        fileContents.reset(new md::StructArray(engine->feval(u"load", factory.createCharArray(dataPath))));

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

    md::TypedArray<float> matRfData = (*fileContents)[0][fieldNames[1]];
    md::TypedArray<float> matLocData = (*fileContents)[0][fieldNames[0]];

    
    Volume* vol = new Volume(engine.get(), VolumeDims);

    cudaError_t error = volumeReconstruction(vol, matRfData, matLocData);
    
    std::cout << "Saving data" << std::endl;
    matlab::data::TypedArray<float> myTypedArray = factory.createArray(vol->getCounts(), vol->getData(), vol->end());

    std::u16string name = u"newTest";
 
    std::u16string filePath = uR"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\cuda_data\testVolume.mat)";
    engine->setVariable(name, myTypedArray);

    engine->eval(u"save('" + filePath + u"');");


    std::cout << "Press any key to exit." << std::endl;
    std::cin;

    delete vol;

    return 0;
}



