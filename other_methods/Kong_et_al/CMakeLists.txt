#Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
#Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
#Authors: Patrick Wieschollek, Orazio Gallo, Jinwei Gu, and Jan Kautz

cmake_minimum_required(VERSION 2.8)
project( Align )

SET (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set (CMAKE_CXX_STANDARD 11)
set (CMAKE_CXX_FLAGS "--std=c++11")

find_package( OpenCV REQUIRED )
find_package( Eigen3 REQUIRED )
find_package(CUDA REQUIRED)

include(FindCUDA)
set (CUDA_PROPAGATE_HOST_FLAGS ON)

include_directories("/usr/include/eigen3/" SYSTEM)


find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message("NO OMP")
endif()


cuda_add_executable( physicalreflectionseparation physicalreflectionseparation.cu )
target_link_libraries( physicalreflectionseparation ${OpenCV_LIBS})


add_subdirectory(gtest)
enable_testing()
include_directories(${gtest_SOURCE_DIR}/include ${gtest_SOURCE_DIR})


cuda_add_executable( verify verify.cu )
target_link_libraries( verify ${OpenCV_LIBS}  gtest gtest_main)

