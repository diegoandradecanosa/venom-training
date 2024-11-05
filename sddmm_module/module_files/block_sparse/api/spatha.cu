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

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
    m.def("sddmm",     &sddmm,      "Custom SDDMM kernel");
}
