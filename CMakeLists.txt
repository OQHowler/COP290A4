cmake_minimum_required(VERSION 3.16)
project(DataFrameLib CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Arrow REQUIRED)
find_package(Parquet REQUIRED)

# Gather all library sources from src/ (excluding main.cpp since it's just for local testing)
file(GLOB LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(REMOVE_ITEM LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp")

# Build the main library target that the autograder expects
add_library(dataframelib ${LIB_SOURCES})
target_link_libraries(dataframelib PUBLIC arrow_shared parquet_shared)
target_include_directories(dataframelib PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_features(dataframelib PUBLIC cxx_std_20)

# Local test driver (commented out for final submission so it doesn't conflict with the grader)
#add_executable(DataFrameLibTest ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cpp)
#target_link_libraries(DataFrameLibTest PRIVATE dataframelib)