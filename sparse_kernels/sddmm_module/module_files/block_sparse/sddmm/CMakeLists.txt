set(SDDMM_SPATHA_SOURCES
    ./blockwise_library.cu
)

#cmake_policy(SET CMP0104 OLD)

set_source_files_properties(${SDDMM_SPATHA_SOURCES} PROPERTIES LANGUAGE CUDA)

include(GNUInstallDirs)

add_definitions(-DGPU_CC=${CUDA_ARCHS})

add_library(sddmm
	STATIC ${SDDMM_SPATHA_SOURCES})