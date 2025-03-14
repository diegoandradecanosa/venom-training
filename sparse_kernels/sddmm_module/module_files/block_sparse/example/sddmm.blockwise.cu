
// benchmark for block-sparse spmm

#include "../sddmm/sddmm_op.h"
#include "../cuda_array.h"
#include <stdio.h>

using namespace spatha_sddmm;
using namespace std;

int roundUp(int numToRound, int multiple)
{
    if (multiple == 0)
        return numToRound;

    int remainder = numToRound % multiple;
    if (remainder == 0)
        return numToRound;

    return numToRound + multiple - remainder;
}

int main(int argc, const char** argv)
{
    int m = 32, n = 32, k = 64;
    float density = 0.2;
    unsigned seed = 2023;
    bool row_permute = false;

    int bm = 32;
    int meta_block_sz = 32;
    int block_sz = 4;
    int mm_row = 4;
    int nn_row = 2;

    // define a blockwise sparse matrix
    BlockwiseSpTensor<half> spmat;
    spmat.init_random(m, n, block_sz, nn_row, mm_row, 0.5, seed, meta_block_sz, bm);
    spmat.transform_and_sync_device();

    CudaRandomArray<half> A;
    CudaRandomArray<half> B;

    A.initialize(k*m);
    B.initialize(k*n);

    A.sync_device();
    B.sync_device();

    /* for(int i=0; i<k; i++){
        for(int j=0; j<m; j++){
            printf("%.2f ", __half2float(A.host_array[i*m+j]));
        }
        printf("\n");
    }
    printf("\n"); */
    for(int i=0; i<m; i++){
        for(int j=0; j<k; j++){
            if(j!=0 && j%8==0){
                cout << "  ";
            }
            printf("%.2f ", __half2float(A.host_array[i*k+j]));
        }
        printf("\n");
    }
    printf("\n");

    for(int i=0; i<n; i++){
        for(int j=0; j<k; j++){
            if(j!=0 && j%8==0){
                cout << "  ";
            }
            printf("%.2f ", __half2float(B.host_array[i*k+j]));
        }
        printf("\n");
    }
    printf("\n");

    //*************************************
    int m_fixed = 4; // !IMPORTANT! value fixed because of NVIDIA architecture (2:4)
    int bits_elem_meta=2;
    int mrow_m = 2;
    int bits_elem_cols=8;
    int brow_fixed = 16;

    // general variables N:M format
    int A_num_cols_sp = (n/mm_row)*nn_row;
    int A_num_cols_sp_pad = roundUp((roundUp(n, mm_row)/mm_row)*nn_row, 16);
    int bm_m = m/bm;
    int mbrow_m = bm/meta_block_sz;
    int mbrow_m2 = meta_block_sz/brow_fixed;
    int brow_m = brow_fixed/block_sz;
    int nelems=(sizeof(uint)*8)/bits_elem_meta;
    int mcol_nn = nelems/mrow_m/nn_row; //8/n
    int mcol_n = A_num_cols_sp_pad/nn_row/mcol_nn; // n/(4*4)
    int col_n = A_num_cols_sp_pad/nn_row;

    cout << "metadata: " << endl;
    for(int bm_i=0; bm_i<1; bm_i++){
        for(int ii=0; ii<meta_block_sz/2; ii++){
            for(int jj=0; jj<mcol_n; jj++){
                cout << "row: " << ii+jj*meta_block_sz/2 << ", bits = [ ";
                for(int mbrow_iii=0; mbrow_iii<mrow_m; mbrow_iii++){
                    for(int mcol_ii=0; mcol_ii<mcol_nn; mcol_ii++){
                        for (int n_i=0; n_i<nn_row; n_i++) {
                            cout << ((spmat.metadata[bm_i*meta_block_sz/2*mcol_n + ii+jj*meta_block_sz/2] >> (mbrow_iii*(nelems/mrow_m)*bits_elem_meta+mcol_ii*nn_row*bits_elem_meta+n_i*bits_elem_meta)) & 0x3);
                        }
                        cout << " ";
                    }
                }cout << " ], =" << spmat.metadata[ii+jj*meta_block_sz/2] << " ";
            } cout << endl;
        }
    }

    cout << "indices: " << col_n << " " << endl;
    for(int ii=0; ii<2; ii++){
        for(int jj=0; jj<col_n; jj++){
            cout << "row: " << ii << ", col: " << jj << ", bits = [ ";
            for (int w = 0; w < m_fixed; w++) {
                cout << spmat.indices[ii*col_n*m_fixed + jj*m_fixed + w] << " ";
            }
            cout << "], ";
        } cout << endl << endl;
    }

    SddmmBlockwiseOp<ShapeBase<32, 32, 32>,   // block tile
                     ShapeBase<32, 32, 32>,   // warp tile
                     ShapeBase<16, 16, 16>,   // mma shape
                     2>                       // number of pipeline stage
                     op;

    op.initialize(spmat, k, A.device_ptr, B.device_ptr);

    op();

    cudaDeviceSynchronize();
    spmat.sync_host();

    std::vector<float> C_ref(m * n);
    //get_host_reference<half>(spmat, n, B.host_array, 1.0f, C.host_array, 1.0f, C_ref);

    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            C_ref[i*n+j] = 0.0f;

            for (int k1 = 0; k1 < k; k1++) {
                C_ref[i*n+j] += __half2float(A.host_array[i*k+k1]) * __half2float(B.host_array[k1+j*k]);
            }
        }
    }

    for(int i=0; i<m; i++){
        for(int j=0; j<n; j++){
            printf("%.2f ", C_ref[i*n+j]);
        }
        printf("\n");
    }
    printf("\n");

    std::cout << "verify correctness" << std::endl;

    for(int i=0; i<2; i++){
        cout << endl;
        for(int j=0; j<16*A_num_cols_sp_pad; j++){
            if(j!=0 && j%8==0){
                cout << "  ";
                cout << j/8 << ": ";
            }
            //if(cnt==3){
            //    cout << endl;
            //    cnt=0;
            //}
            //cnt++;
            cout << __half2float(spmat.values[i*16*A_num_cols_sp_pad+j]) << " ";
        }
        cout << endl;
    }
    printf("\n");

    /* bool passed = true;
    for (int i = 0; i < m; i++) {
        for (int j = 0; j < n; j++) {
            int idx = i*n + j;
            float d = static_cast<float>(C.host_array[idx]);
            float d_ref = C_ref[idx];
            if (fabs(d_ref - d) > 1e-3 * fabs(d_ref)) {
                printf("i = %d, j = %d, result %f != %f\n", i, j, d, d_ref);
                passed = false;
            }
        }
    }

    if (passed) std::cout << "Passed\n";
    else        std::cout << "Failed\n";

    return passed ? EXIT_SUCCESS: EXIT_FAILURE; */
}