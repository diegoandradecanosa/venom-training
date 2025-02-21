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

#include "./blockwise_op.h"
#include "./blockwise_format.h"
#include "../common/library_util.h"

//#include <ATen/ATen.h>
//#include <ATen/core/Tensor.h>
//#include <ATen/Utils.h>

#include "cuda_fp16.h"
//#include <c10/cuda/CUDAException.h>
//#include <torch/library.h>
//#include <torch/extension.h>

//using namespace at;
namespace spatha{

#define INSTANTIATE_FUNC(type, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
type##InitFn_t NAME_FUNC(type, Init, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
    = type##InitFn<ShapeBase<BM, BLOCK_N, BLOCK_K>, ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE>; \
type##ExecFn_t NAME_FUNC(type, Exec, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
    = type##ExecFn<ShapeBase<BM, BLOCK_N, BLOCK_K>, ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE>;

INSTANTIATE_FUNC(SddmmNM, 32, 32, 32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 32, 32, 32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 32, 32, 32, 32, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 32, 64, 32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 32, 64, 32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 32, 64, 32, 32, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 32, 64, 32, 32, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 32, 64, 32, 32, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 32, 64, 32, 32, 64, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 64, 32, 32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 64, 32, 32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 64, 32, 32, 32, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 32, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 32, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 32, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 32, 64, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 64, 32, 32, 64, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 64, 32, 32, 64, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 64, 32, 32, 64, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 64, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 64, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 64, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 64, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 64, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 64, 64, 32, 16, 16, 16, 4);

// **
INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 32, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 64, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 64, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 64, 32, 32, 16, 16, 16, 4);

// **
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 32, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 32, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 32, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 32, 64, 32, 16, 16, 16, 4);

// **
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 64, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 64, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 64, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 64, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 64, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 64, 64, 32, 16, 16, 16, 4);

// **
INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 128, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 128, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 128, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 128, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 128, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 128, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 128, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 128, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 128, 64, 32, 16, 16, 16, 4);

// **
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 32, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 32, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 32, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 32, 64, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 64, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 128, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 128, 64, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 128, 64, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 128, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 128, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 128, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 128, 128, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 128, 128, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 128, 128, 32, 16, 16, 16, 4);



torch::Tensor sddmm_cuda(torch::Tensor lhs_matrix,
                         torch::Tensor rhs_matrix,
                         torch::Tensor A_metadata,
                         torch::Tensor A_indices,
                         int A_num_rows,
                         int A_num_cols,
                         int B_num_cols,
                         int vec_length,
                         int n,
                         int m,
                         int nnz,
                         int seed,
                         int mbrow,
                         int brow)
{

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(rhs_matrix.device());

    auto output_matrix = torch::empty({A_num_rows, B_num_cols/m*n}, options);

    SddmmBlockwiseOp<ShapeBase<128, 32, 32>,   // block tile
                    ShapeBase<64, 32, 32>,   // warp tile
                    ShapeBase<16, 16, 16>,   // mma shape
                    2>                       // number of pipeline stage
                    op;

    op.initialize(A_num_rows, B_num_cols,
        reinterpret_cast<uint *>(A_metadata.data_ptr<int>()),
        reinterpret_cast<uint *>(A_indices.data_ptr<int>()),
        A_num_cols,
        reinterpret_cast<half *>(lhs_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(rhs_matrix.data_ptr<torch::Half>()),
        m, n, brow, mbrow,
        reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>())
        );

    op();

    cudaDeviceSynchronize();

    return output_matrix;
}


}
