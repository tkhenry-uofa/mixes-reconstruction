//int matlab_cpp_engine(Volume* volume, std::string data_dir, std::string data_file)
//{
//
//    std::string data_path = data_dir + data_file;
//
//    md::ArrayFactory factory;
//
//    printf("Initializing matlab\n");
//    std::unique_ptr<me::MATLABEngine> engine = me::startMATLAB();
//
//    printf("Loading matlab data\n");
//
//    std::unique_ptr<md::StructArray> file_contents;
//
//    try {
//
//        file_contents.reset(new md::StructArray(engine->feval(u"load", factory.createCharArray(data_path))));
//
//    }
//    catch (const std::exception& e) {
//        std::cerr << "Error loading file: " << e.what() << std::endl;
//        return -1;
//    }
//
//    size_t field_count = file_contents->getNumberOfFields();
//
//    if (field_count < 2)
//    {
//        std::cerr << "Expected at least 2 fields in file, instead found " << field_count << std::endl;
//        return -1;
//    }
//
//    md::Range<md::ForwardIterator, md::MATLABFieldIdentifier const> file_range = file_contents->getFieldNames();
//
//    md::ForwardIterator<md::MATLABFieldIdentifier const> current_value = file_range.begin();
//
//    int i = 0;
//    std::string field_name;
//    bool have_rf_data = false;
//    bool have_loc_data = false;
//    std::cout << "Found matlab variables:" << std::endl;
//    for (; current_value != file_range.end(); current_value++)
//    {
//        field_name = *current_value;
//        std::cout << "    " << field_name << std::endl;
//
//        if (field_name == rf_data_name)
//        {
//            have_rf_data = true;
//        }
//        else if (field_name == loc_data_name)
//        {
//            have_loc_data = true;
//        }
//    }
//
//    if (!have_rf_data)
//    {
//        std::cerr << "Cannot find rf data '" << rf_data_name << "'" << std::endl;
//        return -1;
//    }
//    else if (!have_loc_data)
//    {
//        std::cerr << "Cannot find element location data '" << loc_data_name << "'" << std::endl;
//        return -1;
//    }
//
//    md::TypedArray<std::complex<float>> mat_rf_data = (*file_contents)[0][rf_data_name];
//    md::TypedArray<float> mat_loc_data = (*file_contents)[0][loc_data_name];
//
//    std::vector<std::complex<float>> rf_data_vector(mat_rf_data.begin(), mat_rf_data.end());
//    std::vector<float> loc_data_vector(mat_loc_data.begin(), mat_loc_data.end());
//    
//    cudaError_t error = complexVolumeReconstruction(volume, rf_data_vector, loc_data_vector, mat_rf_data.getDimensions());
//
//    std::cout << "Saving data" << std::endl;
//    matlab::data::TypedArray<float> myTypedArray = factory.createArray(volume->getCounts(), volume->getData(), volume->end());
//
//    std::u16string name = u"volume";
//
//    std::u16string filePath = uR"(C:\Users\tkhen\OneDrive\Documents\MATLAB\lab\mixes\data\cuda_data\div_side_beamformed.mat)";
//    engine->setVariable(name, myTypedArray);
//
//    engine->eval(u"save('" + filePath + u"');");
//
//    return 0;
//}