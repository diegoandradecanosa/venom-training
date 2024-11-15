set(SPMM_OPENSPARSELT_EXAMPLE_SOURCES
    #./spmm.blockwise.cu
    ./sddmm.blockwise.cu
)

set_source_files_properties(${SPMM_OPENSPARSELT_EXAMPLE_SOURCES} PROPERTIES LANGUAGE CUDA)

#add_library(spmm SHARED
#	${SPMM_SOURCES})

add_definitions(-DGPU_CC=${CUDA_ARCHS})
add_library(openSparseLt_example STATIC
	${SPMM_OPENSPARSELT_EXAMPLE_SOURCES})

target_link_libraries(openSparseLt_example
    PUBLIC cudart nvToolsExt
)

target_include_directories(openSparseLt_example
	PUBLIC .)