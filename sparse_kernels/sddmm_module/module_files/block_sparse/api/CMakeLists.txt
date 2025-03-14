set(SDDMM_OPENSPARSELT_BENCH_SOURCES
    ./argparse_util.h
    ./timing_util.h
    ./benchmark.spmm_nm.cu
)

set_source_files_properties(${SDDMM_OPENSPARSELT_BENCH_SOURCES} PROPERTIES LANGUAGE CUDA)

#add_library(spatha_sddmm SHARED
#	${SDDMM_SOURCES})

add_definitions(-DGPU_CC=${CUDA_ARCHS})
add_library(openSparseLt_bench STATIC
	${SDDMM_OPENSPARSELT_BENCH_SOURCES})

target_link_libraries(openSparseLt_bench
    PUBLIC cudart nvToolsExt
)

target_include_directories(openSparseLt_bench
	PUBLIC .)