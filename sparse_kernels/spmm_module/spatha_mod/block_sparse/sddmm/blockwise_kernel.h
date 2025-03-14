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

#include "../common/base.h"
#include "../common/mma.h"
#include "../common/memcpy.h"
#include "../common/swizzle.h"
#include "../common/epilogue.h"

namespace spatha {

template<
    // block-sparse pattern
    int BM_, int BK_,
    // tiling shapes
    typename ThreadBlockShape,
    typename WarpShape,
    typename MmaShape,
    // threadblock level pipeline stage
    int      kThreadBlockStage,
    // type of accumulator
    typename AccumulatorType,
    // type of shared memory swizzling
    typename ASwizzle,
    typename BSwizzle,
    typename CSwizzle,
    // pipeline configuration
    bool     UseRegisterDoubleBuffer = false,
    //bool     UseRegisterDoubleBuffer = true,
    bool     UseMinimumSync = true,
    bool     GridMappingXYToMN_ = true
>
struct SddmmBlockwiseKernel {
    static constexpr int BM      = BM_;
    static constexpr int BK      = BK_;
    static constexpr int block_M = ThreadBlockShape::M;
    static constexpr int block_N = ThreadBlockShape::N;
    static constexpr int block_K = ThreadBlockShape::K;
    static constexpr int warp_M  = WarpShape::M;
    static constexpr int warp_N  = WarpShape::N;
    static constexpr int mma_M   = MmaShape::M;
    static constexpr int mma_N   = MmaShape::N;
    static constexpr int mma_K   = MmaShape::K;
    /* static constexpr int brow    = 4;
    static constexpr int mbrow   = 16; */
    static_assert(BM == block_M,
    "Only support threadblock shape M == BM in block-sparse shape.\n");

    static_assert(WarpShape::K == ThreadBlockShape::K,
    "K-dim of warp and threadblock tiling must be the same. "
    "Split-K not supported.\n");

    static_assert( block_M % warp_M == 0);
    static_assert( block_N % warp_N == 0);
    static_assert( warp_M  % mma_M  == 0);
    static_assert( warp_N  % mma_N  == 0);
    static_assert( block_K % mma_K  == 0);
    static_assert( block_K % BK     == 0);
    //static_assert( kThreadBlockStage > 1);

    /* static_assert( warp_N % 16 == 0,
    "Only support warp shape M>=16 for performance.\n"); */

    static constexpr int metaPrefetchBlock = 256; //4 warps*4
    static_assert(metaPrefetchBlock / (block_K) >= kThreadBlockStage);

    // precompute constants
    static constexpr bool GridMappingXYToMN = GridMappingXYToMN_;
    static constexpr int blockDim = 32 * (block_M/warp_M) * (block_N/warp_N);

    // use multi-stage shared-mem buffer (needed by async copy)
    /* static constexpr size_t input_buffer_size_static =
        (block_M*block_K + block_N*block_K) * kThreadBlockStage * sizeof(half) + metaPrefetchBlock * 2 * sizeof(uint); //FIXME: block_K*(block_N/mm_row)*4 */
    static constexpr size_t input_buffer_size_static =
        (block_M*block_K + block_M/16*block_K + block_N*block_K) * kThreadBlockStage * sizeof(half) + metaPrefetchBlock * 2 * sizeof(uint); //FIXME: block_K*(block_N/mm_row)*4

    static constexpr size_t output_buffer_size_static =
        (block_M * block_N) * sizeof(half); //FIXME: block_M*(block_N/mm_row)*4
    /* static constexpr size_t output_buffer_size_static =
        (block_M/(mma_M*2) * block_N/mma_N * (32+5) * 8) * sizeof(half); */
    /* static constexpr size_t output_buffer_size_static =
        (block_M/(mma_M*2)/(mma_M*2) * block_N/mma_N * (32+5) * 8/mma_N * (32+5) * 8) * sizeof(half); */

    // mainloop interface
    __device__ __forceinline__
    void mainLoop(const int M, const int N, const int K,
        const int n, const int m, const int brow, const int mbrow,
        const half *A,
        const half *B,
        const uint* C_indices, half* C_values, const uint* C_metadata,
        half *shared_mem_workspace
    );

    __device__ __forceinline__
    void epilogue(const int M, const int N, half *D, half *shared_mem_workspace, float alpha, const half *C, const uint* C_metadata, float beta, const int nn, const int mm, const int brow, const int mbrow
    );

};

template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape,
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__
void SddmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle,
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::mainLoop
(const int M, const int N, const int K,
        const int nn, const int mm, const int brow, const int mbrow,
        const half *A,
        const half *B,
        const uint* C_indices, half* C_values, const uint* C_metadata,
        half *shared_mem_workspace)
{
    const int warp_id = (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    const int kAccess = 8;

    const int N_sp_p = ROUND_UP((ROUND_UP(N, mm)/mm)*nn, 16);
    const int col_n  = N_sp_p*2;
    const int mcol_n = N_sp_p/8;

    int idx_block_M = GridMappingXYToMN ? blockIdx.x : blockIdx.y;
    int idx_block_N = GridMappingXYToMN ? blockIdx.y : blockIdx.x;
    int idx_warp_M  = warp_id % (block_M / warp_M);
    int idx_warp_N  = warp_id / (block_M / warp_M);

    const int n_offset = idx_block_N*(block_N*mm/4);

    const half* A_panel = &A[idx_block_M*block_M*K];
    const half* B_panel = &B[n_offset*K];//
    const uint* C_index = &C_indices[idx_block_M*col_n+idx_block_N*block_N];
    const uint* C_meta  = &C_metadata[
        idx_block_M*(block_M>>1)*mcol_n +
        idx_warp_M*(warp_M>>1)*mcol_n +
        idx_block_N*16*(block_N/16) +
        idx_warp_N*16*(warp_N/16) +
        (lane>>2)*2];

    // compute global to shared copy constants
    const int iter_copy_A = CEIL(block_M * block_K / kAccess, blockDim);
    const int iter_copy_B = CEIL(block_N * block_K / kAccess, blockDim);

    // compute shared memory buffer addresses
    const int NStage = kThreadBlockStage;
    const int size_of_tile_A = block_M * block_K + block_M/16*block_K;
    const int size_of_tile_B = block_N * block_K;
    half *shared_A = shared_mem_workspace;
    half *shared_B = shared_A + size_of_tile_A * NStage;
    uint *shared_I = (uint*)(shared_B + size_of_tile_B * NStage);

    // compute shared memory offsets
    int A_warp_panel_offset = idx_warp_M * (warp_M * block_K + (warp_M/16)*block_K);
    const int smem_lda = block_K;
    int B_warp_panel_offset = idx_warp_N * warp_N * block_K;
    const int smem_ldb = block_K;

    ASwizzle aSwizzle;
    BSwizzle bSwizzle;

    // define mma buffers
    typedef typename my_mma::fragment_a_rowmajor<MmaShape> FragmentA;
    typedef typename my_mma::fragment_b_colmajor<MmaShape> FragmentB;
    typedef typename my_mma::fragment_c<MmaShape, AccumulatorType> FragmentC;
    typedef typename my_mma::fragment_meta_sparse<MmaShape> FragmentMeta;

    const int iter_mma_M = warp_M / mma_M;
    const int iter_mma_N = warp_N / mma_N;
    const int load_mma_M = iter_mma_M/2;
    const int kWarpStage = (UseRegisterDoubleBuffer ? 2 : 1);
    FragmentA afrag[kWarpStage][iter_mma_M];
    FragmentB bfrag[kWarpStage][iter_mma_N];
    FragmentC cfrag[iter_mma_M][iter_mma_N];
    FragmentMeta metafrag[load_mma_M][iter_mma_N];

    #pragma unroll
    for(int m=0; m<load_mma_M; m++){
        for(int n=0; n<iter_mma_N; n++){
            my_mma::load_meta_sync(metafrag[m][n].x, C_meta, m*16*mcol_n + n*16);
        }
    }

    // load indices first
    const int num_tile = CEIL(K, block_K);
    int num_block = block_N;

    my_pipeline::Pipeline<NStage, true> pipe;
    int fetch = 0, compute = 0;

    //int pos = lane%4;
    uint mindex[load_mma_M*iter_mma_N*8];

    my_pipeline::copy_and_sync((uint*)shared_I, (const uint*)C_index,
                                min(num_block, metaPrefetchBlock));

    for(; compute < num_tile; compute++) {

        for (; fetch < compute + NStage; fetch++) {
            pipe.acquire_writer();

            // fetch data
            if (fetch < num_tile) {
                half *shared_tile_A = shared_A + (fetch % NStage) * size_of_tile_A;
                half *shared_tile_B = shared_B + (fetch % NStage) * size_of_tile_B;

                #pragma unroll
                for (int i = 0; i < iter_copy_A; i++) {
                    int idx = (threadIdx.x + blockDim*i) * kAccess;
                    int nz_block_idx = (idx / block_K);

                    const half *src = A_panel + nz_block_idx*K + fetch*block_K + (idx % block_K);
                          half *dst = shared_tile_A + (idx/512)*block_K + (idx/512)*512 + aSwizzle(idx%512); //512=8*8*8=8*64

                    my_pipeline::cp_async_pred_zfill<16>(dst, src);
                }

                #pragma unroll
                for (int i = 0; i < iter_copy_B; i++) {
                    int idx = (threadIdx.x + blockDim*i) * kAccess;
                    int nz_block_idx = (idx / block_K);

                    int n_base = shared_I[ nz_block_idx ];
                    int n = (nz_block_idx/4)*mm + n_base;
                    //bool zfill = (n>N);
                    bool zfill = ((n+n_offset)>=N);

                    const half *src = B_panel + n*K + fetch*block_K + (idx % block_K);
                          half *dst = shared_tile_B + (idx/512)*512 + bSwizzle(idx%512);

                    my_pipeline::cp_async_pred_zfill<16>(dst, src, true, zfill);
                }

            }
            pipe.commit_stage();
        }
        pipe.acquire_reader();

        half *shared_tile_A = shared_A + (compute % NStage) * size_of_tile_A;
        half *shared_tile_B = shared_B + (compute % NStage) * size_of_tile_B;

        for (int k = 0; k < block_K / mma_K; k++) {

            #pragma unroll
            for (int m = 0; m < load_mma_M; m+=1) {
                int offset = k*mma_K;
                int disp = 2*mma_M*block_K;
                disp = m*(disp+(disp/512)*block_K);
                my_mma::load_matrix_sync2<ASwizzle>(afrag[0][m*2], shared_tile_A + disp, offset, smem_lda, A_warp_panel_offset);
                my_mma::load_matrix_sync2<ASwizzle>(afrag[0][m*2+1], shared_tile_A + disp, offset+2*block_K, smem_lda, A_warp_panel_offset);
            }

            #pragma unroll
            for (int n = 0; n < iter_mma_N; n++) {
                int offset = B_warp_panel_offset + n*mma_N*block_K + k*mma_K;
                my_mma::load_matrix_sync2<BSwizzle>(bfrag[0][n], shared_tile_B, offset, smem_ldb);
            }

            #pragma unroll
            for (int m = 0; m < iter_mma_M; m++) {
                #pragma unroll
                for (int n = 0; n < iter_mma_N; n++) {
                    my_mma::mma_sync(cfrag[m][n], afrag[0][m], bfrag[0][n], cfrag[m][n]);
                }
            }
        }

        pipe.release_reader();
    }

    int pos = lane%4;
    const half* C_panel = &C_values[idx_block_M*block_M*N_sp_p +
        idx_warp_M*warp_M*N_sp_p +
        idx_block_N*16*block_N/2 +
        idx_warp_N*16*warp_N/2 +
        (lane/16)*16*N_sp_p +
        (lane%16)*8];

    /* int n_col =
        idx_block_N*16*block_N/2 +
        idx_warp_N*16*warp_N/2 +
        (lane/16)*16*N_sp_p +
        (lane%16)*8; */

    #pragma unroll
    for(int m=0; m<load_mma_M; m++){
        #pragma unroll
        for(int n=0; n<iter_mma_N; n++){
            //float result[8];
            half result[8];

            result[0] = cfrag[m*2  ][n].x[(metafrag[m][n].x[0] >> (pos*4) & 0x3)];
            result[1] = cfrag[m*2  ][n].x[(metafrag[m][n].x[0] >> (pos*4+2) & 0x3)];
            result[2] = cfrag[m*2  ][n].x[(metafrag[m][n].x[0] >> (pos*4+16) & 0x3)+4];
            result[3] = cfrag[m*2  ][n].x[(metafrag[m][n].x[0] >> (pos*4+18) & 0x3)+4];
            result[4] = cfrag[m*2+1][n].x[(metafrag[m][n].x[1] >> (pos*4) & 0x3)];
            result[5] = cfrag[m*2+1][n].x[(metafrag[m][n].x[1] >> (pos*4+2) & 0x3)];
            result[6] = cfrag[m*2+1][n].x[(metafrag[m][n].x[1] >> (pos*4+16) & 0x3)+4];
            result[7] = cfrag[m*2+1][n].x[(metafrag[m][n].x[1] >> (pos*4+18) & 0x3)+4];

            //if((n_col+n*16*mma_N/2)<N){
                *(float4*)(C_panel + m*32*N_sp_p + n*16*mma_N/2) = *(float4*)(result);
            //}
        }
    }
}


template<
    int BM, int BK, typename ThreadBlockShape, typename WarpShape,
    typename MmaShape, int kThreadBlockStage, typename AccumulatorType,
    typename ASwizzle, typename BSwizzle, typename CSwizzle,
    bool UseRegisterDoubleBuffer, bool UseMinimumSync, bool GridMappingXYToMN
>
__device__ __forceinline__
void SddmmBlockwiseKernel<BM, BK, ThreadBlockShape, WarpShape, MmaShape,
    kThreadBlockStage, AccumulatorType, ASwizzle, BSwizzle, CSwizzle,
    UseRegisterDoubleBuffer, UseMinimumSync, GridMappingXYToMN>::epilogue
(const int M, const int N, half *D, half *shared_mem_workspace, float alpha, const half *C_values, const uint* C_metadata, float beta, const int nn, const int mm, const int brow, const int mbrow)
{
    epilogue_impl<BM, block_N, blockDim, warp_M, warp_N, mma_M, mma_N, MmaShape, CSwizzle, GridMappingXYToMN>(M, N, D, shared_mem_workspace, alpha, C_values, C_metadata, beta, nn, mm, brow, mbrow);
}

}