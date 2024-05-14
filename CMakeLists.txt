cmake_minimum_required(VERSION 3.26)
project(parallel_project CUDA)

set(CMAKE_CUDA_STANDARD 17)

# Add source files to the project
add_executable(parallel_project
        main.cu
        cpu_processing.cu
        filters.cu
        gpu_processing.cu
        job_loader.cu
        png_tools.cu
)

# Set properties for CUDA separable compilation
set_target_properties(parallel_project PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}  # Output binary in the project root
)

# Find the PNG library
find_package(PNG REQUIRED)

# Include directories for PNG
include_directories(${PNG_INCLUDE_DIRS})

# Link with any necessary libraries (e.g., CUDA runtime, libpng)
target_link_libraries(parallel_project ${CUDA_LIBRARIES} ${PNG_LIBRARIES})
