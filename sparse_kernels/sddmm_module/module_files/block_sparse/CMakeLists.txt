cmake_minimum_required(VERSION 3.13)

#cmake_policy(SET CMP0104 OLD)

project(SPATHA CUDA CXX C)

##
### Handle Dependencies.
##
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules)
include(cmake/Dependencies.cmake)

##
### C++ and CUDA flags.
##

# Requires C++11.
set(CMAKE_CXX_STANDARD 17)

#cmake_policy(SET CMP0104 OLD)

create_cuda_gencode_flags(CUDA_GENCODE_FLAGS ${CUDA_ARCHS})
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} ${CUDA_GENCODE_FLAGS}")

##
### Project build.
##
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

##################################

set(SPMM_SPATHA_BENCHS_SOURCES
    benchmark/benchmark.spmm_nm.cu)

set_source_files_properties(${SPMM_SPATHA_BENCHS_SOURCES} PROPERTIES LANGUAGE CUDA)

set(SPATHA_SRCS)
add_subdirectory(util)
add_subdirectory(spmm)
add_subdirectory(sddmm)

add_definitions(-DGPU_CC=${CUDA_ARCHS})
add_executable(benchmark.spmm_nm ${SPMM_SPATHA_BENCHS_SOURCES})
set_source_files_properties(${benchmark.spmm_nm} PROPERTIES LANGUAGE CUDA)
target_link_libraries(benchmark.spmm_nm
spatha_util spatha spatha_sddmm ${SPUTNIK_BENCHMARK_LIBS} ${CMAKE_TORCH_LIBRARIES})
target_include_directories(benchmark.spmm_nm
	PUBLIC ${TORCH_INCLUDE_DIRS})

##################################

set(SDDMM_OPENSPARSELT_EXAMPLE_SOURCES
    example/sddmm.blockwise.cu)

set_source_files_properties(${SDDMM_OPENSPARSELT_EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)

add_executable(sddmm.blockwise ${SDDMM_OPENSPARSELT_EXAMPLE_SOURCES})
set_source_files_properties(${sddmm.blockwise} PROPERTIES LANGUAGE CUDA)
target_link_libraries(sddmm.blockwise
spatha_util spatha_sddmm ${SPUTNIK_BENCHMARK_LIBS})

