cmake_minimum_required(VERSION 3.0)
project(BIEXPDELTABARANKINORDERJ)

# Locate source files
file(GLOB SOURCES "src/*.c")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Build the executable
add_executable(BIEXPDELTABARANKINORDERJ ${SOURCES})

# link to libm if it exists 
find_library(MATH_LIBRARY m)
if(MATH_LIBRARY)
    message(STATUS "linking against libm")
    target_link_libraries(BIEXPDELTABARANKINORDERJ PRIVATE ${MATH_LIBRARY})
endif()

