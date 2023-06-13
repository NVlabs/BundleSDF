# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


cmake_minimum_required(VERSION 3.15)
project(BundleTrack LANGUAGES CUDA CXX)

message("CMAKE_CXX_COMPILER_VERSION: ${CMAKE_CXX_COMPILER_VERSION}")

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -g")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -gencode arch=compute_30,code=sm_30 -gencode arch=compute_75,code=sm_75 -O0 -G -g -std=c++14 -lineinfo -use_fast_math --default-stream per-thread")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35 -gencode arch=compute_75,code=sm_75 -O3 -std=c++14 -use_fast_math --default-stream per-thread")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -dc -gencode=arch=compute_86,code=sm_86 -gencode=arch=compute_52,code=sm_52  -gencode=arch=compute_60,code=sm_60   -gencode=arch=compute_61,code=sm_61   -gencode=arch=compute_70,code=sm_70   -gencode=arch=compute_75,code=sm_75   -gencode=arch=compute_80,code=sm_80  -O3 -std=c++14 -use_fast_math --default-stream per-thread")
# set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -Xcompiler -fPIC -gencode arch=compute_86,code=sm_86  -O0 -G -g -std=c++14 -use_fast_math --default-stream per-thread")


# if (${CMAKE_BUILD_TYPE} STREQUAL "Release")
#   set(COMPILE_OPTIONS "-fopenmp -march=native -msse -msse2 -msse3 -msse4 -msse4.2 -mavx2 -ftree-vectorize -O3")
#   set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${COMPILE_OPTIONS}")
#   message("Compile with " ${COMPILE_OPTIONS})
# endif (${CMAKE_BUILD_TYPE} STREQUAL "Release")

# set(GUROBI 1)
add_definitions(-DG2O=0)
add_definitions(-DTIMER=0)
add_definitions(-DPRINT_RESIDUALS_DENSE=0)
add_definitions(-DPRINT_RESIDUALS_SPARSE=0)
# add_definitions(-DPRINT_ITER_POSES=1)
add_definitions(-DCUDA_RANSAC=1)
add_definitions(-DCUDA_MATCHING=1)
# add_definitions(-DDEBUG=1)

find_package(Boost REQUIRED COMPONENTS system program_options serialization)
find_package(PCL REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(OpenMP REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(BLAS REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
find_package(GLUT REQUIRED)
find_package(LAPACK)

find_package(pybind11 REQUIRED)
find_package(Python COMPONENTS Interpreter Development REQUIRED)
set(PYBIND11_CPP_STANDARD -std=c++14)
# set(PYBIND11_FINDPYTHON TRUE)

include_directories(
  src
  /usr/include/
  /usr/local/include/
  ${Boost_INCLUDE_DIRS}
  ${PYTHON_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
  ${GLUT_INCLUDE_DIRS}
  ${CUDA_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
  ${OPENGL_INCLUDE_DIR}
  ${CSPARSE_INCLUDE_DIR}
  ${PROJECT_SOURCE_DIR}/src/cuda/
  ${PROJECT_SOURCE_DIR}/src/cuda/Solver/
  ${PROJECT_SOURCE_DIR}/src/Thirdparty
  ${PROJECT_SOURCE_DIR}/src/Thirdparty/g2o
)

if (${GUROBI})
  message("Using Gurobi")
  add_definitions(-DGUROBI=1)
  include_directories("${PROJECT_SOURCE_DIR}/gurobi903/linux64/include")
  set(GUROBI_LIBS "${PROJECT_SOURCE_DIR}/gurobi903/linux64/lib/libgurobi_g++5.2.a" "${PROJECT_SOURCE_DIR}/gurobi903/linux64/lib/libgurobi90.so")
endif (${GUROBI})

file(GLOB MY_SRC ${PROJECT_SOURCE_DIR}/src/*.cpp ${PROJECT_SOURCE_DIR}/src/cuda/*.cpp ${PROJECT_SOURCE_DIR}/src/cuda/Solver/*.cpp)
file(GLOB G2O_LIBS ${PROJECT_SOURCE_DIR}/src/Thirdparty/g2o/lib/libg2o*)
file(GLOB Boost_LIBRARIES "/usr/lib/x86_64-linux-gnu/libboost*.so")

message("G2O_LIBS " ${G2O_LIBS})
message("Boost_LIBRARIES " ${Boost_LIBRARIES})

set(YAML_CPP_LIBRARIES "/usr/local/lib/libyaml-cpp.so")


list(REMOVE_ITEM PCL_LIBRARIES pcl_simulation)
message("PCL_LIBRARIES " ${PCL_LIBRARIES})

#============================ CUDA =======================================
file(GLOB CUDA_FILES
  "${PROJECT_SOURCE_DIR}/src/*.cu"
  "${PROJECT_SOURCE_DIR}/src/cuda/*.cu"
  "${PROJECT_SOURCE_DIR}/src/cuda/Solver/*.cu"
)
message("CUDA_FILES " ${CUDA_FILES})


cuda_add_library(MY_CUDA_LIB SHARED ${CUDA_FILES})
target_link_libraries(MY_CUDA_LIB ${CUDA_LIBRARIES})
set_target_properties(MY_CUDA_LIB
   PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

add_library(${PROJECT_NAME} SHARED ${MY_SRC})
target_link_libraries(${PROJECT_NAME} ${Boost_LIBRARIES} ${OpenCV_LIBRARIES} ${PCL_LIBRARIES} ${OpenMP_CXX_FLAGS} ${OPENGL_LIBRARY}  ${YAML_CPP_LIBRARIES} ${CUDA_LIBRARIES} ${OPENGL_LIBRARIES} ${GLUT_LIBRARY}  zmq ${GUROBI_LIBS} ${PYTHON_LIBRARIES} ${G2O_LIBS} MY_CUDA_LIB
)

############# Pybind
# pybind11_add_module(my_cpp pybind_interface/pybind_api.cpp ${MY_SRC})
pybind11_add_module(my_cpp pybind_interface/pybind_api.cpp ${CMAKE_SOURCE_DIR}/src/Frame.cpp)
target_link_libraries(my_cpp PRIVATE ${PROJECT_NAME} MY_CUDA_LIB ${YAML_CPP_LIBRARIES} ${PYTHON_LIBRARIES} ${CUDA_LIBRARIES})
