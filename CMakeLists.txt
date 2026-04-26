cmake_minimum_required(VERSION 3.14)
project(DataFrameLib)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find Apache Arrow
find_package(Arrow REQUIRED)
find_package(ArrowDataset REQUIRED)

# Tell CMake where to find your .h files
include_directories(${CMAKE_SOURCE_DIR}/include)

# Gather all your .cpp files from the src directory
file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

# Create the executable
add_executable(DataFrameLib ${SOURCES})

# Link Apache Arrow libraries
target_link_libraries(DataFrameLib PRIVATE Arrow::arrow_shared ArrowDataset::arrow_dataset_shared)

target_link_libraries(DataFrameLib arrow_shared parquet_shared)