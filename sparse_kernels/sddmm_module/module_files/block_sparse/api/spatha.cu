/*
 * Copyright (C) 2023 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "../sddmm/sddmm_op.h"
#include "../sddmm/sddmm_library_decl.h"
#include "../sddmm/blockwise_library.cu"

#include <torch/extension.h>

using namespace spatha_sddmm;

/**************** SDDMM ****************/


torch::Tensor sddmm(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}


torch::Tensor sddmm_32x32x32_32x32x32_16x16x16_2(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_32x32x32_32x32x32_16x16x16_2(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}


torch::Tensor sddmm_32x32x32_32x32x32_16x16x16_4(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_32x32x32_32x32x32_16x16x16_4(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}


torch::Tensor sddmm_32x64x32_32x32x32_16x16x16_2(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_32x64x32_32x32x32_16x16x16_2(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}


torch::Tensor sddmm_32x64x32_32x32x32_16x16x16_3(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_32x64x32_32x32x32_16x16x16_3(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}


torch::Tensor sddmm_64x32x32_32x32x32_16x16x16_2(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_64x32x32_32x32x32_16x16x16_2(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}


torch::Tensor sddmm_64x32x32_32x32x32_16x16x16_3(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_64x32x32_32x32x32_16x16x16_3(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}


torch::Tensor sddmm_64x64x32_32x32x32_16x16x16_2(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_64x64x32_32x32x32_16x16x16_2(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}


torch::Tensor sddmm_64x64x32_32x32x32_16x16x16_3(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_64x64x32_32x32x32_16x16x16_3(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}

/******************************* */


torch::Tensor sddmm_32x32x32_32x32x32_16x16x16_3(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_32x32x32_32x32x32_16x16x16_3(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}

//
torch::Tensor sddmm_64x64x32_64x32x32_16x16x16_2(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_64x64x32_64x32x32_16x16x16_2(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}

//
torch::Tensor sddmm_64x32x32_32x32x32_16x16x16_4(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_64x32x32_32x32x32_16x16x16_4(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}

//
torch::Tensor sddmm_64x32x32_64x32x32_16x16x16_2(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_64x32x32_64x32x32_16x16x16_2(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}

// 32,64,32-32,32,32-16,16,16-4
torch::Tensor sddmm_32x64x32_32x32x32_16x16x16_4(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_32x64x32_32x32x32_16x16x16_4(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}

//32,64,32-32,64,32-16,16,16 2
torch::Tensor sddmm_32x64x32_32x64x32_16x16x16_2(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_32x64x32_32x64x32_16x16x16_2(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}


//32,64,32-32,64,32-16,16,16 3
torch::Tensor sddmm_32x64x32_32x64x32_16x16x16_3(torch::Tensor A_matrix,
                    torch::Tensor B_matrix,
                    torch::Tensor C_metadata,
                    torch::Tensor C_indices,
                    int C_num_rows,
                    int C_num_cols,
                    int A_num_cols,
                    int n,
                    int m,
                    int nnz,
                    int seed,
                    int mbrow,
                    int brow)
{

    return sddmm_cuda_32x64x32_32x64x32_16x16x16_3(A_matrix,
                      B_matrix,
                      C_metadata,
                      C_indices,
                      C_num_rows,
                      C_num_cols,
                      A_num_cols,
                      n,
                      m,
                      nnz,
                      seed,
                      mbrow,
                      brow);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("sddmm",     &sddmm,      "Custom SDDMM kernel with default kernels");
    m.def("sddmm_32x32x32_32x32x32_16x16x16_2",     &sddmm_32x32x32_32x32x32_16x16x16_2,      "Custom SDDMM kernel with 32x32x32 32x32x32 16x16x16 2 parameters");
    m.def("sddmm_32x32x32_32x32x32_16x16x16_4",     &sddmm_32x32x32_32x32x32_16x16x16_4,      "Custom SDDMM kernel with 32x32x32 32x32x32 16x16x16 4 parameters");
    m.def("sddmm_32x64x32_32x32x32_16x16x16_2",     &sddmm_32x64x32_32x32x32_16x16x16_2,      "Custom SDDMM kernel with 32x64x32 32x32x32 16x16x16 2 parameters");
    m.def("sddmm_32x64x32_32x32x32_16x16x16_3",     &sddmm_32x64x32_32x32x32_16x16x16_3,      "Custom SDDMM kernel with 32x64x32 32x32x32 16x16x16 3 parameters");
    m.def("sddmm_64x32x32_32x32x32_16x16x16_2",     &sddmm_64x32x32_32x32x32_16x16x16_2,      "Custom SDDMM kernel with 64x32x32 32x32x32 16x16x16 2 parameters");
    m.def("sddmm_64x32x32_32x32x32_16x16x16_3",     &sddmm_64x32x32_32x32x32_16x16x16_3,      "Custom SDDMM kernel with 64x32x32 32x32x32 16x16x16 3 parameters");
    m.def("sddmm_64x64x32_32x32x32_16x16x16_2",     &sddmm_64x64x32_32x32x32_16x16x16_2,      "Custom SDDMM kernel with 64x64x32 32x32x32 16x16x16 2 parameters");
    m.def("sddmm_64x64x32_32x32x32_16x16x16_3",     &sddmm_64x64x32_32x32x32_16x16x16_3,      "Custom SDDMM kernel with 64x64x32 32x32x32 16x16x16 3 parameters");
    m.def("sddmm_32x32x32_32x32x32_16x16x16_3",     &sddmm_32x32x32_32x32x32_16x16x16_3,      "Custom SDDMM kernel with 32x32x32 32x32x32 16x16x16 3 parameters");
    m.def("sddmm_64x64x32_64x32x32_16x16x16_2",     &sddmm_64x64x32_64x32x32_16x16x16_2,      "Custom SDDMM kernel with 64x64x32 64x32x32 16x16x16 2 parameters");
    m.def("sddmm_64x32x32_32x32x32_16x16x16_4",     &sddmm_64x32x32_32x32x32_16x16x16_4,      "Custom SDDMM kernel with 64x32x32 32x32x32 16x16x16 4 parameters");
    m.def("sddmm_64x32x32_64x32x32_16x16x16_2",     &sddmm_64x32x32_64x32x32_16x16x16_2,      "Custom SDDMM kernel with 64x32x32 64x32x32 16x16x16 2 parameters");
    m.def("sddmm_32x64x32_32x32x32_16x16x16_4",     &sddmm_32x64x32_32x32x32_16x16x16_4,      "Custom SDDMM kernel with 32x64x32 32x32x32 16x16x16 4 parameters");
    m.def("sddmm_32x64x32_32x64x32_16x16x16_2",     &sddmm_32x64x32_32x64x32_16x16x16_2,      "Custom SDDMM kernel with 32x64x32 32x64x32 16x16x16 2 parameters");
    m.def("sddmm_32x64x32_32x64x32_16x16x16_3",     &sddmm_32x64x32_32x64x32_16x16x16_3,      "Custom SDDMM kernel with 32x64x32 32x64x32 16x16x16 3 parameters");
}
