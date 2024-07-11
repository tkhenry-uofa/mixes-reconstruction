#include <iostream>

#include "shared_memory_handler.hh"

const std::string SharedMemoryHandler::File_prefix = R"(Local\)";
const std::string SharedMemoryHandler::Pipe_prefix = R"(\\.\pipe\)";

SharedMemoryHandler::SharedMemoryHandler(std::string filename, size_t size): 
	_filename(filename), _size(size), _handle(nullptr), _file_ptr(nullptr)
{

}

SharedMemoryHandler::~SharedMemoryHandler()
{
	if (_file_ptr)
	{
		UnmapViewOfFile(_file_ptr);
	}

	if (_handle)
	{
		CloseHandle(_handle);
	}
}

bool
SharedMemoryHandler::create_file()
{
    // Save location data
    _handle = CreateFileMapping(
        INVALID_HANDLE_VALUE,   // Use paging file
        NULL,                   // Default security
        PAGE_READWRITE,         // Read/write access
        0,                      // Maximum object size (high-order DWORD)
        _size,             // Maximum object size (low-order DWORD)
        _filename.c_str());      // Name of mapping object

    if (_handle == NULL) {
        std::cout << "Could not create file mapping object: " << GetLastError() << std::endl;
        return false;
    }

    _file_ptr = MapViewOfFile(
        _handle,
        FILE_MAP_ALL_ACCESS,
        0,
        0,
        _size);

    if (_file_ptr == NULL) {
        std::cout << "Could not map view of file: " << GetLastError() << std::endl;
        CloseHandle(_handle);
        return false;
    }

    return true;
}

void*
SharedMemoryHandler::wait_for_data()
{
    void* data = nullptr;



    return data;
}