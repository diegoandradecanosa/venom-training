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
#include <cusparseLt.h>

using namespace at;
namespace spatha{

#define INSTANTIATE_FUNC(type, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
type##InitFn_t NAME_FUNC(type, Init, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
    = type##InitFn<ShapeBase<BM, BLOCK_N, BLOCK_K>, ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE>; \
type##ExecFn_t NAME_FUNC(type, Exec, BM, BLOCK_N, BLOCK_K, WARP_M, WARP_N, WARP_K, MMA_M, MMA_N, MMA_K, NSTAGE) \
    = type##ExecFn<ShapeBase<BM, BLOCK_N, BLOCK_K>, ShapeBase<WARP_M, WARP_N, WARP_K>, ShapeBase<MMA_M, MMA_N, MMA_K>, NSTAGE>;

#define CHECK_CUSPARSELT(func)                                                                         \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
    }                                                                          \
}


// block_sz = 32
INSTANTIATE_FUNC(SpmmNM, 32, 16, 16, 32, 16, 16, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 16, 32, 64, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 16, 32, 64, 16, 16, 8, 16, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 16, 32, 64, 16, 16, 8, 16, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 128, 32, 64, 128, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 128, 16, 32, 128, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 16, 32, 64, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 16, 32, 32, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 16, 32, 16, 16, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 16, 32, 16, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 16, 32, 16, 16, 16, 8, 16, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 16, 32, 16, 16, 16, 8, 16, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 32, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 32, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 32, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 32, 32, 32, 32, 32, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 32, 32, 32, 32, 32, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 32, 32, 32, 32, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 32, 32, 64, 32, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 32, 64, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 32, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 32, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 32, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 16, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 16, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 16, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 64, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 64, 32, 64, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 32, 32, 64, 32, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 128, 32, 64, 128, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 128, 128, 32, 64, 128, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 32, 64, 32, 32, 16, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 32, 32, 16, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 64, 32, 32, 16, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 64, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 32, 128, 64, 32, 64, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 32, 128, 128, 32, 128, 128, 16, 8, 32, 2);

// block_sz = 64
INSTANTIATE_FUNC(SpmmNM, 64, 32, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64, 32, 32, 64, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64, 32, 32, 64, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64, 64, 16, 64, 32, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 16, 64, 32, 16, 16, 8, 16, 3);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 16, 64, 32, 16, 16, 8, 16, 4);

INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 32, 4);


INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 32, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 32, 64, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 32, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64, 64, 64, 32, 64, 64, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 16, 3);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 64, 32, 32, 16, 8, 16, 4);

INSTANTIATE_FUNC(SpmmNM, 64, 32, 128, 64, 32, 128, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 64, 32, 16, 32, 32, 16, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 64, 16, 16, 64, 16, 16, 16, 8, 16, 2);

// block_sz = 128
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 4);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 5);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 32, 32, 16, 8, 32, 6);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 32, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 32, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 32, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 128, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 128, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 128, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 128, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 128, 64, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 128, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 64, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 128, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 128, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 128, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 64, 128, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 64, 128, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 64, 128, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 64, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 64, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 128, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 128, 64, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 128, 64, 64, 16, 8, 32, 4);


INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 64, 32, 16, 8, 32, 4);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 64, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 16, 128, 32, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 16, 128, 32, 16, 16, 8, 16, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 16, 16, 64, 16, 16, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 128, 16, 16, 128, 16, 16, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 128, 32, 32, 128, 32, 32, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 128, 32, 32, 32, 16, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 32, 32, 32, 16, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 32, 32, 32, 16, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 128, 64, 32, 16, 8, 32, 2);

INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 128, 128, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 128, 128, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 32, 128, 128, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 32, 32, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 32, 32, 64, 16, 8, 32, 3);

INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 64, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 128, 64, 64, 64, 64, 16, 8, 32, 3);

// block_sz = 256
INSTANTIATE_FUNC(SpmmNM, 256, 64, 16, 64, 32, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 32, 16, 64, 32, 16, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 32, 32, 64, 32, 32, 16, 8, 16, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 16, 64, 64, 16, 16, 8, 16, 2);

INSTANTIATE_FUNC(SpmmNM, 256, 32, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 32, 32, 64, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 256, 32, 32, 64, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 256, 64, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 32, 64, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 32, 64, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 256, 128, 32, 64, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 128, 32, 64, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 256, 128, 32, 64, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 256, 128, 32, 32, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 128, 32, 32, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 256, 128, 32, 32, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 256, 64, 64, 64, 32, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 64, 64, 32, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 64, 64, 32, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 256, 64, 64, 64, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 64, 64, 64, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 256, 64, 64, 64, 64, 64, 16, 8, 32, 4);

///
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 32, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 32, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64, 64, 32, 32, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 32, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 32, 64, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 32, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 32, 64, 64, 32, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 32, 64, 64, 32, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 32, 64, 64, 32, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 16, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 16, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 64, 64, 16, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 16, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 16, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 64, 16, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 32, 16, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 32, 16, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128, 64, 32, 32, 16, 32, 16, 8, 32, 4);

//
INSTANTIATE_FUNC(SpmmNM, 128,32, 32, 32, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128,32, 32, 32, 32, 32, 16, 8, 32, 3);

INSTANTIATE_FUNC(SpmmNM, 128,64, 32, 32, 32, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128,64, 32, 32, 32, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 128,64, 32, 32, 32, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 128,128,32, 32, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 128,128,32, 32, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64,128,32, 64, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 64, 64, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 64, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64,128,64, 64, 64, 64, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64,128,64, 64, 64, 64, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64,128,64, 64, 64, 64, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64,128,32, 32, 64, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 32, 64, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 32, 64, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64,128,32, 32, 128, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 32, 128, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 32, 128, 32, 16, 8, 32, 4);

INSTANTIATE_FUNC(SpmmNM, 64,128,32, 64, 128, 32, 16, 8, 32, 2);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 64, 128, 32, 16, 8, 32, 3);
INSTANTIATE_FUNC(SpmmNM, 64,128,32, 64, 128, 32, 16, 8, 32, 4);

/*
cusparseLtHandle_t handle;
bool handle_initialized = false;

torch::Tensor _cslt_compress(const torch::Tensor sparse_input)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(sparse_input.device());

    if (!handle_initialized){
        TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
        handle_initialized = true;
    }

    cusparseLtMatDescriptor_t matA;
    auto          type  = CUDA_R_16F;

    TORCH_CUDASPARSE_CHECK( cusparseLtStructuredDescriptorInit(
                                        &handle,
                                        &matA,
                                        sparse_input.size(0),
                                        sparse_input.size(1),
                                        sparse_input.size(1),
                                        16,
                                        type,
                                        CUSPARSE_ORDER_ROW,
                                        CUSPARSELT_SPARSITY_50_PERCENT) );

    //--------------------------------------------------------------------------
    // Compress the A matrix
    size_t compressed_size;
    TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompressedSize2(
        &handle,
        &matA,
        &compressed_size));

    auto compressed_tensor = torch::empty(compressed_size/2, options);
    half *dA_compressed = reinterpret_cast<half *>(compressed_tensor.data_ptr<torch::Half>());

    TORCH_CUDASPARSE_CHECK(cusparseLtSpMMACompress2(
        &handle,
        &matA,
        true,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        sparse_input.data_ptr(),
        dA_compressed,
        nullptr));

    TORCH_CUDASPARSE_CHECK(
        cusparseLtMatDescriptorDestroy(&matA));

    return compressed_tensor;
}

torch::Tensor spmm_cuda(
                        torch::Tensor compressed_tensor,
                        torch::Tensor rhs_matrix,
                        torch::Tensor bias_opt,
                        int A_num_rows,
                        int A_num_cols,
                        int B_num_cols,
                        int vec_length,
                        int n_row,
                        int m_row,
                        int nnz,
                        int seed,
                        int mbrow,
                        int brow)
{
    auto options = torch::TensorOptions().dtype(torch::kFloat16).device(rhs_matrix.device());

    auto output_matrix = torch::empty({B_num_cols,A_num_rows}, options);

    cusparseLtMatDescriptor_t matA, matB, matC;
    cusparseLtMatmulPlan_t    plan;

    cusparseLtMatmulDescriptor_t   matmul;
    cusparseLtMatmulAlgSelection_t alg_sel;

    if (!handle_initialized){
        TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));
        handle_initialized = true;
    }
    TORCH_CUDASPARSE_CHECK(cusparseLtInit(&handle));

    TORCH_CUDASPARSE_CHECK( cusparseLtStructuredDescriptorInit(
                                        &handle,
                                        &matA,
                                        A_num_rows,
                                        A_num_cols,
                                        A_num_cols,
                                        16,
                                        CUDA_R_16F,
                                        CUSPARSE_ORDER_ROW,
                                        CUSPARSELT_SPARSITY_50_PERCENT) );
    TORCH_CUDASPARSE_CHECK( cusparseLtDenseDescriptorInit(
                                        &handle,
                                        &matB,
                                        A_num_cols,
                                        B_num_cols,
                                        B_num_cols,
                                        16,
                                        CUDA_R_16F,
                                        CUSPARSE_ORDER_ROW) );
    TORCH_CUDASPARSE_CHECK( cusparseLtDenseDescriptorInit(
                                        &handle,
                                        &matC,
                                        A_num_rows,
                                        B_num_cols,
                                        A_num_rows,
                                        16,
                                        CUDA_R_16F,
                                        CUSPARSE_ORDER_COL) );
    //--------------------------------------------------------------------------
    // matmul, algorithm selection, and plan initialization
    TORCH_CUDASPARSE_CHECK( cusparseLtMatmulDescriptorInit(
                                            &handle,
                                            &matmul,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            CUSPARSE_OPERATION_NON_TRANSPOSE,
                                            &matA,
                                            &matB,
                                            &matC,
                                            &matC,
                                            CUSPARSE_COMPUTE_16F) );
    // set bias pointer for matmut, need to assign to get location
    void* dBias = bias_opt.data_ptr();
    TORCH_CUDASPARSE_CHECK(cusparseLtMatmulDescSetAttribute(
                                        &handle,
                                        &matmul, CUSPARSELT_MATMUL_BIAS_POINTER,
                                        &dBias,
                                        sizeof(dBias)));
    TORCH_CUDASPARSE_CHECK( cusparseLtMatmulAlgSelectionInit(
                                            &handle,
                                            &alg_sel,
                                            &matmul,
                                            CUSPARSELT_MATMUL_ALG_DEFAULT) );
    size_t workspace_size=0;
    TORCH_CUDASPARSE_CHECK( cusparseLtMatmulPlanInit(&handle,
                                                    &plan,
                                                    &matmul,
                                                    &alg_sel,
                                                    workspace_size) );

    TORCH_CUDASPARSE_CHECK( cusparseLtMatmulGetWorkspace(&handle,
                                                        &plan,
                                                        &workspace_size) );

    //--------------------------------------------------------------------------
    // execute SpMM cuSparseLt kernel
    //auto compressed_tensor = _cslt_compress(A_values);

    float alpha = 1.0f;
    float beta = 0.0f;
    TORCH_C


    TORCH_CUDASPARSE_CHECK( cusparseLtMatmul(
                &handle,
                &plan,
                &alpha,
                reinterpret_cast<half *>(compressed_tensor.data_ptr<torch::Half>()),
                reinterpret_cast<half *>(rhs_matrix.data_ptr<torch::Half>()),
                &beta,
                reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                nullptr,
                nullptr,
                0));

    TORCH_CUDASPARSE_CHECK(
        cusparseLtMatDescriptorDestroy(&matA));
    TORCH_CUDASPARSE_CHECK(
        cusparseLtMatDescriptorDestroy(&matB));
    TORCH_CUDASPARSE_CHECK(
        cusparseLtMatDescriptorDestroy(&matC));
    TORCH_CUDASPARSE_CHECK(
        cusparseLtMatmulPlanDestroy(&plan));

    return output_matrix;
}
*/

torch::Tensor spmm_cuda_64x64x32_32x64x32_16x8x32_2(torch::Tensor A_metadata,
                        torch::Tensor A_indices,
                        torch::Tensor A_values,
                        torch::Tensor rhs_matrix,
                        torch::Tensor bias,
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

    auto output_matrix = torch::empty({B_num_cols,A_num_rows}, options);


    SpmmBlockwiseOp<ShapeBase<64, 64, 32>,  // block tile
                ShapeBase<32, 64, 32>,       // warp tile
                ShapeBase<16, 8, 32>,        // mma shape
                2>                           // number of pipeline stage
                op;

    op.initialize(A_num_rows, A_num_cols,
                  reinterpret_cast<half *>(A_values.data_ptr<torch::Half>()),
                  reinterpret_cast<uint *>(A_metadata.data_ptr<int>()), reinterpret_cast<uint *>(A_indices.data_ptr<int>()),
                  B_num_cols,
                  reinterpret_cast<half *>(rhs_matrix.data_ptr<torch::Half>()),
                  1.0f,
                  reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                  0.0f, m, n, brow, mbrow,
                  reinterpret_cast<half *>(bias.data_ptr<torch::Half>()));

    op();

    return output_matrix;
}

torch::Tensor spmm_cuda_64x64x32_32x64x32_16x8x32_3(torch::Tensor A_metadata,
                        torch::Tensor A_indices,
                        torch::Tensor A_values,
                        torch::Tensor rhs_matrix,
                        torch::Tensor bias,
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

    auto output_matrix = torch::empty({B_num_cols,A_num_rows}, options);

    SpmmBlockwiseOp<ShapeBase<64, 64, 32>,  // block tile
                ShapeBase<32, 64, 32>,       // warp tile
                ShapeBase<16, 8, 32>,        // mma shape
                3>                           // number of pipeline stage
                op;

    op.initialize(A_num_rows, A_num_cols,
                  reinterpret_cast<half *>(A_values.data_ptr<torch::Half>()),
                  reinterpret_cast<uint *>(A_metadata.data_ptr<int>()), reinterpret_cast<uint *>(A_indices.data_ptr<int>()),
                  B_num_cols,
                  reinterpret_cast<half *>(rhs_matrix.data_ptr<torch::Half>()),
                  1.0f,
                  reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                  0.0f, m, n, brow, mbrow,
                  reinterpret_cast<half *>(bias.data_ptr<torch::Half>()));

    op();

    return output_matrix;
}

torch::Tensor spmm_cuda_128x64x32_32x64x32_16x8x32_2(torch::Tensor A_metadata,
                        torch::Tensor A_indices,
                        torch::Tensor A_values,
                        torch::Tensor rhs_matrix,
                        torch::Tensor bias,
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

    auto output_matrix = torch::empty({B_num_cols,A_num_rows}, options);

    //printf("spmm_cuda_128x64x32_32x64x32_16x8x32_2: [%d:%d:%d] %d x %d x %d \n", 128, n, m, A_num_rows, A_num_cols, B_num_cols);


    SpmmBlockwiseOp<ShapeBase<128, 64, 32>,  // block tile
                ShapeBase<32, 64, 32>,       // warp tile
                ShapeBase<16, 8, 32>,        // mma shape
                2>                           // number of pipeline stage
                op;

    op.initialize(A_num_rows, A_num_cols,
                  reinterpret_cast<half *>(A_values.data_ptr<torch::Half>()),
                  reinterpret_cast<uint *>(A_metadata.data_ptr<int>()), reinterpret_cast<uint *>(A_indices.data_ptr<int>()),
                  B_num_cols,
                  reinterpret_cast<half *>(rhs_matrix.data_ptr<torch::Half>()),
                  1.0f,
                  reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                  0.0f, m, n, brow, mbrow,
                  reinterpret_cast<half *>(bias.data_ptr<torch::Half>()));

    op();

    return output_matrix;
}

torch::Tensor spmm_cuda_128x64x32_32x64x32_16x8x32_3(torch::Tensor A_metadata,
                        torch::Tensor A_indices,
                        torch::Tensor A_values,
                        torch::Tensor rhs_matrix,
                        torch::Tensor bias,
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

    auto output_matrix = torch::empty({B_num_cols,A_num_rows}, options);

    SpmmBlockwiseOp<ShapeBase<128, 64, 32>,  // block tile
                ShapeBase<32, 64, 32>,       // warp tile
                ShapeBase<16, 8, 32>,        // mma shape
                3>                           // number of pipeline stage
                op;

    op.initialize(A_num_rows, A_num_cols,
                  reinterpret_cast<half *>(A_values.data_ptr<torch::Half>()),
                  reinterpret_cast<uint *>(A_metadata.data_ptr<int>()), reinterpret_cast<uint *>(A_indices.data_ptr<int>()),
                  B_num_cols,
                  reinterpret_cast<half *>(rhs_matrix.data_ptr<torch::Half>()),
                  1.0f,
                  reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                  0.0f, m, n, brow, mbrow,
                  reinterpret_cast<half *>(bias.data_ptr<torch::Half>()));

    op();

    return output_matrix;
}

torch::Tensor spmm_cuda(torch::Tensor A_metadata,
                        torch::Tensor A_indices,
                        torch::Tensor A_values,
                        torch::Tensor rhs_matrix,
                        torch::Tensor bias,
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

    auto output_matrix = torch::empty({B_num_cols,A_num_rows}, options);

#ifdef V_64
    SpmmBlockwiseOp<ShapeBase<64, 64, 32>,  // block tile
                ShapeBase<32, 64, 32>,      // warp tile
                ShapeBase<16, 8, 32>,       // mma shape
                2>                          // number of pipeline stage
                op;
#else
    SpmmBlockwiseOp<ShapeBase<128, 64, 32>,  // block tile
                ShapeBase<32, 64, 32>,       // warp tile
                ShapeBase<16, 8, 32>,        // mma shape
                2>                           // number of pipeline stage
                op;
#endif

    op.initialize(A_num_rows, A_num_cols,
                  reinterpret_cast<half *>(A_values.data_ptr<torch::Half>()),
                  reinterpret_cast<uint *>(A_metadata.data_ptr<int>()), reinterpret_cast<uint *>(A_indices.data_ptr<int>()),
                  B_num_cols,
                  reinterpret_cast<half *>(rhs_matrix.data_ptr<torch::Half>()),
                  1.0f,
                  reinterpret_cast<half *>(output_matrix.data_ptr<torch::Half>()),
                  0.0f, m, n, brow, mbrow,
                  reinterpret_cast<half *>(bias.data_ptr<torch::Half>()));

    op();

    return output_matrix;
}

}
