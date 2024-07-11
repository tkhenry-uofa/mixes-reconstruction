#ifndef SHARED_MEMORY_HANDLER_HH
#define SHARED_MEMORY_HANDLER_HH

#include <string>
#include <complex>
#include <Windows.h>

class SharedMemoryHandler
{
public:

	static const std::string File_prefix;
	static const std::string Pipe_prefix;

	SharedMemoryHandler(std::string filename, size_t size);
	~SharedMemoryHandler();

	inline void* const get_file_ptr() { return _file_ptr; }

	bool create_file();

	void* wait_for_data();



private:

	HANDLE _handle; // Handle to the shared file
	LPVOID _file_ptr; // Pointer to the start of the file buffer

	std::string _filename;

	size_t _size;
};


#endif // SHARED_MEMORY_HANDLER_HH

