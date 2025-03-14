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


#pragma once
#include "blockwise_format.h"
#include "blockwise_kernel.h"

namespace spatha_sddmm {

struct SddmmBlockwiseOpState {
    size_t shared_mem_size;
    dim3 gridDim;
    dim3 blockDim;
    bool initSuccess = false;
    struct Argument_t {
        int M, N, K;
        int nn, mm, brow, mbrow;
        half *A, *B, *C_values, *D;
        //int  *A_bsr_indptr, *A_bsr_indices, *A_row_indices;
        uint  *C_metadata;
        uint  *C_indices;
        float alpha, beta;
    } args;
};

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape,
    typename MmaShape,
    // pipeline config
    int NStage
>
class SddmmBlockwiseOp {
    static constexpr int BM = ThreadBlockShape::M;
    static constexpr int BK = 1;
    using AccumulatorType = float;
    using ASwizzle = Swizzle16BWiseXor;//Swizzle8BWiseXor;
    using BSwizzle = Swizzle16BWiseXor; //Swizzle16BWiseXor;//Swizzle8BWiseXor;
    using CSwizzle = Swizzle8BWiseXor;
    // launch state
public:
    SddmmBlockwiseOpState _state;
    using KernelImpl = SddmmBlockwiseKernel<BM, BK, ThreadBlockShape,
                        WarpShape, MmaShape, NStage, AccumulatorType,
                        ASwizzle, BSwizzle, CSwizzle>;


    void initialize(BlockwiseSpTensor<half> &spmat, int N, half *A, half *B);

    void initialize(BlockwiseSpTensor<half> &spmat, int N, half *A, half *B, half *D);

    void initialize(BlockwiseSpTensor<half> &spmat, int N, half *A,
    float alpha, half *B, float beta, half *D);

    void operator()(cudaStream_t stream = NULL);

};


// *** device kernel ***
template<typename KernelImpl> __global__
void _sddmmBlockwiseKernel(typename SddmmBlockwiseOpState::Argument_t args)
{
    extern __shared__ half shared_mem_workspace[];
    KernelImpl k;

    k.mainLoop(args.M, args.N, args.K, args.nn, args.mm, args.brow, args.mbrow, args.A, args.B, args.C_indices, args.C_values, args.C_metadata, shared_mem_workspace);
}

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape,
    typename MmaShape,
    // pipeline config
    int NStage
>
void SddmmBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage>::initialize(
    BlockwiseSpTensor<half> &spmat, int K, half *A, float alpha, half *B,
    float beta, half *D)
{
    //assert(spmat.brow == BM && "sparse matrix pattern and kernel mismatch\n");
    //assert(spmat.bcol == BK && "sparse matrix pattern and kernel mismatch\n");
    assert(spmat.device_synced && "sparse matrix is not synced to device\n");

    // set argument
    BlockwiseSpTensor<half>::DeviceRef &C_ref = spmat.device_ref;

    this->_state.args = SddmmBlockwiseOpState::Argument_t({spmat.nrow,
        spmat.ncol, K,
        spmat.n, spmat.m, spmat.brow, spmat.mbrow,
        A, B, C_ref.values, D,
        C_ref.metadata,
        C_ref.csb_indices,
        1.0, beta});

    // compute shared memory buffer size
    size_t input_buffer_size_dyn = 0;
    size_t input_buffer_size = input_buffer_size_dyn +
                               KernelImpl::input_buffer_size_static;
    size_t output_buffer_size_dyn = 0;
    //if (spmat.row_permute)
        // with row swizzling, need buffer for row indices
        //output_buffer_size_dyn = sizeof(int) * BM;
    size_t output_buffer_size = output_buffer_size_dyn +
                                KernelImpl::output_buffer_size_static;

    this->_state.shared_mem_size = max(input_buffer_size, output_buffer_size);
    //printf("Shared size: %d \n", this->_state.shared_mem_size);
    if (this->_state.shared_mem_size >= 32*1024) {
        // set kernel attribute
        if (cudaSuccess != cudaFuncSetAttribute(
            _sddmmBlockwiseKernel<KernelImpl>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, this->_state.shared_mem_size)
        ||  cudaSuccess != cudaFuncSetAttribute(
            _sddmmBlockwiseKernel<KernelImpl>,
            cudaFuncAttributePreferredSharedMemoryCarveout, 100)) {
            cudaError_t err = cudaGetLastError();
            std::cerr << "Set kernel attribute failed: " << cudaGetErrorString(err);
            this->_state.initSuccess = false;
        }
    }

    // calculate launch configuration
    int gdimX = KernelImpl::GridMappingXYToMN ?
                (spmat.nrow / KernelImpl::block_M) : (CEIL(spmat.ncol/spmat.m*4, KernelImpl::block_N));
    int gdimY = KernelImpl::GridMappingXYToMN ?
                (CEIL(spmat.ncol/spmat.m*4, KernelImpl::block_N)) : (spmat.nrow / KernelImpl::block_M);
    this->_state.gridDim = dim3(gdimX, gdimY, 1);
    this->_state.blockDim = dim3(KernelImpl::blockDim, 1, 1);

    this->_state.initSuccess = true;
}


template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape,
    typename MmaShape,
    // pipeline config
    int NStage
>
void SddmmBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage>::initialize(
    BlockwiseSpTensor<half> &spmat, int K, half *A, half *B, half *D) {
    this->initialize(spmat, K, A, 1.0f, B, 1.0f, D);
}

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape,
    typename MmaShape,
    // pipeline config
    int NStage
>
void SddmmBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage>::initialize(
    BlockwiseSpTensor<half> &spmat, int K, half *A, half *B) {
    this->initialize(spmat, K, A, 1.0f, B, 0.0f, nullptr);
}

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape,
    typename MmaShape,
    // pipeline config
    int NStage
>
void SddmmBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage>::operator()(
    cudaStream_t stream)
{
    _sddmmBlockwiseKernel<KernelImpl><<<this->_state.gridDim, this->_state.blockDim, this->_state.shared_mem_size, stream>>>(this->_state.args);
}


// pure-function version of the original c++-object Op
// function handle easy for benchmarking, testing

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape,
    typename MmaShape,
    // pipeline config
    int NStage
>
SddmmBlockwiseOpState SddmmNMInitFn(BlockwiseSpTensor<half> &spmat, int K, half *A, half *B)
{
    SddmmBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage> op;
    op.initialize(spmat, K, A, B);
    return op._state;
}

template<
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape,
    typename MmaShape,
    // pipeline config
    int NStage
>
void SddmmNMExecFn(SddmmBlockwiseOpState &state, cudaStream_t stream = NULL)
{
    using KernelImpl = typename SddmmBlockwiseOp<ThreadBlockShape, WarpShape, MmaShape, NStage>::KernelImpl;
    _sddmmBlockwiseKernel<KernelImpl><<<state.gridDim, state.blockDim,
        state.shared_mem_size, stream>>>(state.args);
}

// signature of blockSpmmInit(...)
typedef SddmmBlockwiseOpState (*SddmmNMInitFn_t) (BlockwiseSpTensor<half>&, int, half*, half*);

// signature of blockSpmmRun(...)
typedef void (*SddmmNMExecFn_t) (SddmmBlockwiseOpState&, cudaStream_t);

}