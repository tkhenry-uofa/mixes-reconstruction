
#include <stdio.h>
#include <string>
#include <vector>
#include <iostream>

#include <MatlabDataArray.hpp>
#include <MatlabEngine.hpp>

#include "volume.hh"
#include "kernel.hh"

static const std::string loc_data_name = "allLocs";
static const std::string rf_data_name = "allScans";

static const float XMin = -25.0f / 1000;
static const float XMax = 25.0f / 1000;

static const float YMin = -5.0f / 1000;
static const float YMax = 5.0f / 1000;

static const float ZMin = 30.0f / 1000;
static const float ZMax = 60.0f / 1000;

static const float Resolution = 0.00015f;

static const Volume::VolumeDims VolumeDims = { XMin, XMax, YMin, YMax, ZMin, ZMax, Resolution };


namespace me = matlab::engine;
namespace md = matlab::data;

int main()
{

    std::string dataRoot = R"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\cuda_data\)";

    std::string dataPath = dataRoot + "cyst_100k.mat";

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

    if (fieldCount < 2)
    {
        std::cerr << "Expected at least 2 fields in file, instead found " << fieldCount << std::endl;
        return -1;
    }

    md::Range<md::ForwardIterator, md::MATLABFieldIdentifier const> fileRange = fileContents->getFieldNames();

    md::ForwardIterator<md::MATLABFieldIdentifier const> currentValue = fileRange.begin();

    int i = 0;
    std::string fieldName;
    bool have_rf_data = false;
    bool have_loc_data = false;
    std::cout << "Found matlab variables:" << std::endl;
    for (; currentValue != fileRange.end(); currentValue++)
    {
        fieldName = *currentValue;
        std::cout << "    " << fieldName << std::endl;

        if (fieldName == rf_data_name)
        {
            have_rf_data = true;
        }
        else if (fieldName == loc_data_name)
        {
            have_loc_data = true;
        }
    }

    if (!have_rf_data )
    {
        std::cerr << "Cannot find rf data '" << rf_data_name << "'" << std::endl;
        return -1;
    }
    else if (!have_loc_data)
    {
        std::cerr << "Cannot find element location data '" << loc_data_name << "'" << std::endl;
        return -1;
    }

    md::TypedArray<std::complex<float>> matRfData = (*fileContents)[0][rf_data_name];
    md::TypedArray<float> matLocData = (*fileContents)[0][loc_data_name];

    Volume* vol = new Volume(engine.get(), VolumeDims);

    cudaError_t error = complexVolumeReconstruction(vol, matRfData, matLocData);
    
    std::cout << "Saving data" << std::endl;
    matlab::data::TypedArray<float> myTypedArray = factory.createArray(vol->getCounts(), vol->getData(), vol->end());

    std::u16string name = u"volume";
 
    std::u16string filePath = uR"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\cuda_data\cyst_100k_beamformed.mat)";
    engine->setVariable(name, myTypedArray);

    engine->eval(u"save('" + filePath + u"');");

    delete vol;

    return 0;
}



