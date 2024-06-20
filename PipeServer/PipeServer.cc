#include <windows.h>
#include <iostream>
#include "defs.hh"

int pipe_test()
{
    // Define the pipe name
    const char* pipeName = R"(\\.\pipe\DataPipe)";

    // Create a named pipe
    HANDLE hPipe = CreateNamedPipeA(
        pipeName,
        PIPE_ACCESS_DUPLEX,
        PIPE_TYPE_BYTE | PIPE_READMODE_BYTE | PIPE_WAIT,
        1,                  // Number of instances
        1024,               // Out buffer size
        1024,               // In buffer size
        0,                  // Default timeout
        NULL                // Default security attributes
    );

    if (hPipe == INVALID_HANDLE_VALUE) {
        std::cerr << "CreateNamedPipe failed, error: " << GetLastError() << std::endl;
        return 1;
    }

    std::cout << "Waiting for client to connect to the pipe..." << std::endl;

    // Wait for the client to connect
    BOOL result = ConnectNamedPipe(hPipe, NULL) ? TRUE : (GetLastError() == ERROR_PIPE_CONNECTED);

    if (!result) {
        std::cerr << "ConnectNamedPipe failed, error: " << GetLastError() << std::endl;
        CloseHandle(hPipe);
        return 1;
    }

    std::cout << "Client connected, sending message..." << std::endl;

    // Write a message to the pipe
    const char* message = "Hello from the server!";
    DWORD bytesWritten;
    result = WriteFile(
        hPipe,
        message,
        (DWORD)strlen(message) + 1,
        &bytesWritten,
        NULL
    );

    if (!result) {
        std::cerr << "WriteFile failed, error: " << GetLastError() << std::endl;
        CloseHandle(hPipe);
        return 1;
    }

    std::cout << "Message sent, waiting for response..." << std::endl;

    // Read the response from the pipe
    char buffer[1024];
    DWORD bytesRead;
    result = ReadFile(
        hPipe,
        buffer,
        sizeof(buffer) - 1,
        &bytesRead,
        NULL
    );

    if (!result) {
        std::cerr << "ReadFile failed, error: " << GetLastError() << std::endl;
        CloseHandle(hPipe);
        return 1;
    }

    buffer[bytesRead] = '\0'; // Null-terminate the string

    std::cout << "Received response: " << buffer << std::endl;

    // Close the pipe
    CloseHandle(hPipe);

    std::cout << "Server exiting." << std::endl;
    return 0;
}



int tx_array_test()
{
    // TODO: Catch log and throw null returns
    mxArray* field_p = mxGetField(struct_array, 0, defs::F0_name);
    _tx_config.f0 = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Fs_name);
    _tx_config.fs = (float)*mxGetDoubles(field_p);


    field_p = mxGetField(struct_array, 0, defs::Column_count_name);
    _tx_config.column_count = (int)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Row_count_name);
    _tx_config.row_count = (int)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Width_name);
    _tx_config.width = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Pitch_name);
    _tx_config.pitch = (float)*mxGetDoubles(field_p);

    field_p = mxGetField(struct_array, 0, defs::X_min_name);
    _tx_config.x_min = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::x_max_name);
    _tx_config.x_max = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Y_min_name);
    _tx_config.y_min = (float)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Y_max_name);
    _tx_config.y_max = (float)*mxGetDoubles(field_p);

    field_p = mxGetField(struct_array, 0, defs::Tx_count_name);
    _tx_config.tx_count = (int)*mxGetDoubles(field_p);
    field_p = mxGetField(struct_array, 0, defs::Pulse_delay_name);
    _tx_config.pulse_delay = (float)*mxGetDoubles(field_p);

    field_p = mxGetField(struct_array, 0, defs::Src_location_name);
    float* src_locs = (float*)mxGetDoubles(field_p);
    _tx_config.src_location = { src_locs[0], src_locs[1], src_locs[2] };

    field_p = mxGetField(struct_array, 0, defs::Transmit_type_name);
    std::u16string tx_name = (const char16_t*)(mxGetChars(field_p));

    if (tx_name == defs::Plane_tx_name)
    {
        _tx_config.transmit_type = defs::TX_PLANE;
    }
    else if (tx_name == defs::X_line_tx_name)
    {
        _tx_config.transmit_type = defs::TX_X_LINE;
    }
    else if (tx_name == defs::Y_line_tx_name)
    {
        _tx_config.transmit_type = defs::TX_Y_LINE;
    }
    else
    {
        std::cerr << "Invalid transmit type (read the file)." << std::endl;
    }
}



int page_map()
{
    const char* sharedMemoryName = "Local\\LocationData";
    const size_t size = 16384; // size in bytes

    // Create a memory-mapped file
    HANDLE hMapFile = CreateFileMapping(
        INVALID_HANDLE_VALUE,   // Use paging file
        NULL,                   // Default security
        PAGE_READWRITE,         // Read/write access
        0,                      // Maximum object size (high-order DWORD)
        size,                   // Maximum object size (low-order DWORD)
        sharedMemoryName);      // Name of mapping object

    if (hMapFile == NULL) {
        std::cerr << "Could not create file mapping object: " << GetLastError() << std::endl;
        return 1;
    }

    // Map the memory to the process's address space
    LPVOID pBuf = MapViewOfFile(
        hMapFile,               // Handle to map object
        FILE_MAP_ALL_ACCESS,    // Read/write permission
        0,
        0,
        size);

    if (pBuf == NULL) {
        std::cerr << "Could not map view of file: " << GetLastError() << std::endl;
        CloseHandle(hMapFile);
        return 1;
    }

    // Use the shared memory
    memcpy(pBuf, "Hello from the test process!", 29);

    // Clean up
    UnmapViewOfFile(pBuf);
    CloseHandle(hMapFile);

    return 0;
}

int main() 
{
    return page_map();
}
