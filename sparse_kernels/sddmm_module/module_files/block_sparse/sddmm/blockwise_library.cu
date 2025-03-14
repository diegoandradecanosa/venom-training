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

#include <ATen/ATen.h>
#include <ATen/core/Tensor.h>
#include <ATen/Utils.h>

#include "ATen/cuda/CUDAContext.h"

#include "cuda_fp16.h"
#include <c10/cuda/CUDAException.h>
#include <torch/library.h>
#include <torch/extension.h>

#include <chrono>

#include <cublas_v2.h>
#include <cusparse_v2.h>
//#include <cusparseLt.h>

using namespace at;
namespace spatha_sddmm{

#define INSTANTIATE_FUNC(type, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
type##InitFn_t NAME_FUNC(type, Init, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
    = type##InitFn<ShapeBase<BM, BLOCK_N, BLOCK_K>, ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE>; \
type##ExecFn_t NAME_FUNC(type, Exec, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
    = type##ExecFn<ShapeBase<BM, BLOCK_N, BLOCK_K>, ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE>;


INSTANTIATE_FUNC(SddmmNM, 32, 32, 32, 32, 32, 32, 16, 16, 16, 2); // Default
INSTANTIATE_FUNC(SddmmNM, 32, 32, 32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 32, 32, 32, 32, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 32, 64, 32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 32, 64, 32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 32, 64, 32, 32, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 32, 64, 32, 32, 64, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 32, 64, 32, 32, 64, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 32, 64, 32, 32, 64, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 64, 32, 32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 64, 32, 32, 32, 32, 32, 16, 16, 16, 3);
INSTANTIATE_FUNC(SddmmNM, 64, 32, 32, 32, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 32, 32, 32, 16, 16, 16, 2);
INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 32, 32, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 32, 32, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 32, 64, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 32, 64, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 32, 64, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 64, 32, 32, 64, 32, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 64, 32, 32, 64, 32, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 64, 32, 32, 64, 32, 32, 16, 16, 16, 4);

INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 64, 32, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 64, 32, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 64, 32, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 64, 64, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 64, 64, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 64, 64, 32, 64, 64, 32, 16, 16, 16, 4);


//INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 32, 32, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 32, 32, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 32, 32, 32, 16, 16, 16, 4);


INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 64, 32, 32, 16, 16, 16, 2);

//INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 64, 32, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 64, 32, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 32, 32, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 32, 32, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 32, 32, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 32, 64, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 32, 64, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 32, 64, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 64, 32, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 64, 32, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 64, 32, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 64, 64, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 64, 64, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 64, 64, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 128, 32, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 128, 32, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 32, 32, 128, 32, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 128, 32, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 128, 32, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 128, 32, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 128, 64, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 128, 64, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 64, 32, 128, 64, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 32, 32, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 32, 32, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 32, 32, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 32, 64, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 32, 64, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 32, 64, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 32, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 32, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 32, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 64, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 64, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 64, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 128, 64, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 128, 64, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 128, 64, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 128, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 128, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 64, 128, 32, 16, 16, 16, 4);

//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 128, 128, 32, 16, 16, 16, 2);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 128, 128, 32, 16, 16, 16, 3);
//INSTANTIATE_FUNC(SddmmNM, 128, 128, 32, 128, 128, 32, 16, 16, 16, 4);




/**
 * Default kernel parameters. 32,32,32 32,32,32 16,16,16 2
 */
torch::Tensor sddmm_cuda(torch::Tensor A_matrix,
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

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    /* SddmmBlockwiseOp<ShapeBase<32, 32, 32>,
                    ShapeBase<32, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    2>
                    op; */

    //./src/benchmark_sddmm --sparsity-type n-to-m --sddmm spatha --gemm cuBlas --precision half --meta-block-size 32 --block-size 4 --nn_row 2 --mm_row 8 --m 128 --k 64 --n 64 --d 0.5 --bm 128 --bn 32 --bk 32 --wm 64 --wn 32 --wk 32 --mm 16 --mn 16 --mk 16 --nstage 2 --check --random
#ifdef V_64
    SddmmBlockwiseOp<ShapeBase<64, 32, 32>,
                    ShapeBase<64, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    2>
                    op;
#else
    SddmmBlockwiseOp<ShapeBase<128, 32, 32>,
                ShapeBase<32, 32, 32>,
                ShapeBase<16, 16, 16>,
                2>
                op;
#endif

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);

    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
}


/**
 * Common part for all sddmm_cuda calls, receives a SddmmBlockwiseOp with the tiling parameters.
 */
/*
torch::Tensor sddmm_cuda_run(SddmmBlockwiseOp* op_pointer,
    torch::Tensor A_matrix,
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

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op_pointer->initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op_pointer();

    return output_matrix;

}
*/

/*
 * 32, 32, 32, 32, 32, 32, 16, 16, 16, 2
 * Default configuration.
 */
torch::Tensor sddmm_cuda_32x32x32_32x32x32_16x16x16_2(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<32, 32, 32>,
                    ShapeBase<32, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    2>
                    op;

    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
    //return sddmm_cuda_run(&op, A_matrix, B_matrix, C_metadata, C_indices, C_num_rows, C_num_cols, A_num_cols, n, m, nnz, seed, mbrow, brow);
}



/*
 * 32, 32, 32, 32, 32, 32, 16, 16, 16, 4
 */
torch::Tensor sddmm_cuda_32x32x32_32x32x32_16x16x16_4(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<32, 32, 32>,
                    ShapeBase<32, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    4>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
    //return sddmm_cuda_run(op, A_matrix, B_matrix, C_metadata, C_indices, C_num_rows, C_num_cols, A_num_cols, n, m, nnz, seed, mbrow, brow);
}



/*
 * 32, 64, 32, 32, 32, 32, 16, 16, 16, 2
 */
torch::Tensor sddmm_cuda_32x64x32_32x32x32_16x16x16_2(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<32, 64, 32>,
                    ShapeBase<32, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    2>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
    //return sddmm_cuda_run(op, A_matrix, B_matrix, C_metadata, C_indices, C_num_rows, C_num_cols, A_num_cols, n, m, nnz, seed, mbrow, brow);
}



/*
 * 32, 64, 32, 32, 32, 32, 16, 16, 16, 3
 */
torch::Tensor sddmm_cuda_32x64x32_32x32x32_16x16x16_3(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<32, 64, 32>,
                    ShapeBase<32, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    3>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
    //return sddmm_cuda_run(op, A_matrix, B_matrix, C_metadata, C_indices, C_num_rows, C_num_cols, A_num_cols, n, m, nnz, seed, mbrow, brow);
}



/*
 * 64, 32, 32, 32, 32, 32, 16, 16, 16, 2
 */
torch::Tensor sddmm_cuda_64x32x32_32x32x32_16x16x16_2(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<64, 32, 32>,
                    ShapeBase<32, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    2>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
    //return sddmm_cuda_run(op, A_matrix, B_matrix, C_metadata, C_indices, C_num_rows, C_num_cols, A_num_cols, n, m, nnz, seed, mbrow, brow);
}



/*
 * 64, 32, 32, 32, 32, 32, 16, 16, 16, 3
 */
torch::Tensor sddmm_cuda_64x32x32_32x32x32_16x16x16_3(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<64, 32, 32>,
                    ShapeBase<32, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    3>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
    //return sddmm_cuda_run(op, A_matrix, B_matrix, C_metadata, C_indices, C_num_rows, C_num_cols, A_num_cols, n, m, nnz, seed, mbrow, brow);
}



/*
 * 64, 64, 32, 32, 32, 32, 16, 16, 16, 2
 */
torch::Tensor sddmm_cuda_64x64x32_32x32x32_16x16x16_2(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<64, 64, 32>,
                    ShapeBase<32, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    2>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
    //return sddmm_cuda_run(op, A_matrix, B_matrix, C_metadata, C_indices, C_num_rows, C_num_cols, A_num_cols, n, m, nnz, seed, mbrow, brow);
}



/*
 * 64, 64, 32, 32, 32, 32, 16, 16, 16, 3
 */
torch::Tensor sddmm_cuda_64x64x32_32x32x32_16x16x16_3(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<64, 64, 32>,
                    ShapeBase<32, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    3>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
}




/*
 * 32, 32, 32, 32, 32, 32, 16, 16, 16, 3
 */
torch::Tensor sddmm_cuda_32x32x32_32x32x32_16x16x16_3(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<32, 32, 32>,
                    ShapeBase<32, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    3>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
}


/*
 * 64, 64, 32, 64, 32, 32, 16, 16, 16, 2
 */
torch::Tensor sddmm_cuda_64x64x32_64x32x32_16x16x16_2(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<64, 64, 32>,
                    ShapeBase<64, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    2>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
}

/*
 * 64, 32, 32, 32, 32, 32, 16, 16, 16, 4
 */
torch::Tensor sddmm_cuda_64x32x32_32x32x32_16x16x16_4(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<64, 32, 32>,
                    ShapeBase<32, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    4>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
}


/*
 * 64, 32, 32, 64, 32, 32, 16, 16, 16, 2
 */
torch::Tensor sddmm_cuda_64x32x32_64x32x32_16x16x16_2(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<64, 32, 32>,
                    ShapeBase<64, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    2>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
}


/*
 * 32, 64, 32, 32, 32, 32, 16, 16, 16, 4
 */
torch::Tensor sddmm_cuda_32x64x32_32x32x32_16x16x16_4(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<32, 64, 32>,
                    ShapeBase<32, 32, 32>,
                    ShapeBase<16, 16, 16>,
                    4>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
}


/*
 * 32, 64, 32, 32, 64, 32, 16, 16, 16, 2
 */
torch::Tensor sddmm_cuda_32x64x32_32x64x32_16x16x16_2(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<32, 64, 32>,
                    ShapeBase<32, 64, 32>,
                    ShapeBase<16, 16, 16>,
                    2>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
}




/*
 * 32, 64, 32, 32, 64, 32, 16, 16, 16, 3
 */
torch::Tensor sddmm_cuda_32x64x32_32x64x32_16x16x16_3(torch::Tensor A_matrix,
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
    SddmmBlockwiseOp<ShapeBase<32, 64, 32>,
                    ShapeBase<32, 64, 32>,
                    ShapeBase<16, 16, 16>,
                    3>
                    op;
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(B_matrix.device());

    auto output_matrix = torch::empty({C_num_rows, C_num_cols/m*n}, options);

    BlockwiseSpTensor<half> spmat;
    spmat.init_sparse_device(C_num_rows,
                             C_num_cols,
                             brow,
                             n,
                             m,
                             nnz,
                             reinterpret_cast<int *>(C_metadata.data_ptr<int>()),
                             reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                             reinterpret_cast<int *>(C_indices.data_ptr<int>()),
                             seed,
                             mbrow,
                             128);


    op.initialize(
        spmat,
        A_num_cols,
        reinterpret_cast<half *>(A_matrix.data_ptr<torch::Half>()),
        reinterpret_cast<half *>(B_matrix.data_ptr<torch::Half>())
    );

    op();

    return output_matrix;
}



}

