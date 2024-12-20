

"""
Declarations and code for exeuting sparse kernels from Venom using V:N:M masks.

"""

import torch
import sten

import ctypes
import math
import functools
try:
    cache = functools.cache
except AttributeError:
    cache = functools.lru_cache(maxsize=None)

from .native_scripting import compile

# Import Venom kernels.
import spatha
import spatha_sddmm
#from grouped_nmv_tensor import VenomTensor, venom_mask_sparsify

__all__ = [
    "FixedMaskTensor",
    "round_up",
    "venom2dense",
    "dense2venom",
    "SparseVNMTensor",
    "VENOMSparsifier",
    "NMVectorSparsifier",
    "VenomSparseLinearFunction",
    "VenomSparseLinear",
    "linear_to_venom",
    "linear_to_venom_with_masks",
    "linear_to_venom_detecting_mask",
    "find_and_replace_with_venom",
]



class FixedMaskTensor:
    def __init__(self, val, mask, n, m, v):
        assert torch.all(
            torch.isclose(mask, torch.zeros_like(mask))
            | torch.isclose(mask, torch.ones_like(mask))
        )
        self.val = val
        self.mask = mask
        self.n = n
        self.m = m
        self.v = v

    @staticmethod
    def from_dense(tensor, n, m, v):
        mask = torch.where(
            tensor.abs() < 1e-6,
            torch.zeros_like(tensor, dtype=torch.bool),
            torch.ones_like(tensor, dtype=torch.bool),
        )
        return FixedMaskTensor(tensor * mask, mask, n, m, v)

    def to_dense(self):
        return copy.deepcopy(self.val)

    def numel(self):
        return self.val.numel()

    def to(self, device=None, dtype=None, non_blocking=False, copy=False):
        return FixedMaskTensor(
            self.val.to(device=device, dtype=dtype, copy=True),
            self.mask.to(device=device, dtype=dtype, copy=True),
            self.n,
            self.m,
            self.v,
        )

    @property
    def shape(self):
        return self.val.shape

    @property
    def device(self):
        return self.val.device

    @property
    def dtype(self):
        return self.val.dtype


def round_up(x,y):
    return math.ceil(x/y)*y

@cache
def venom2dense(dense_shape, dense_dtype, n, m, tileM):
    nrows = dense_shape[0]
    ncols = dense_shape[1]

    A_size = nrows*ncols
    density = n/m

    brow = 4 #this->brow = brow_;
    mbrow = 32 #this->mbrow = mbrow_;

    bm   = tileM
    # !IMPORTANT! constants because of architecture constraints
    m_fixed = 4
    bits_elem_meta=2
    mrow_m = 2
    bits_elem_cols=8
    brow_fixed = 16
    nelems=32//bits_elem_meta #(sizeof(uint)*8)=32
    nelems_col = nelems//mrow_m

    A_num_cols_sp = (ncols/m)*n
    A_num_cols_sp_pad_nm = (round_up(ncols, m)/m)*n
    A_num_cols_sp_pad = round_up((round_up(ncols, m)/m)*n, 16)
    A_nnz = nrows*A_num_cols_sp_pad

    assert dense_dtype in (torch.float32, torch.float64)
    dtype = "float" if dense_dtype == torch.float32 else "double"
    lib = compile(
        f"""
        #include <iostream>
        #include <algorithm>
        #include <utility>
        #include <cstdlib>
        #include <cstdio>
        #include <cmath>
        #include <functional>
        #include <tuple>
        #include <vector>
        #include <numeric>
        #include <chrono>

        using namespace std;


        extern "C" void func3({dtype}* hA_dense, {dtype}* hA_values, int *hA_columns, int *hA_metadata){{
            //this->hA_dense.resize(this->A_size, 0);

            // general variables N:M format
            int bm_m = {nrows}/{bm};
            int mbrow_m = {bm}/{mbrow};
            int mbrow_m2 = {mbrow}/{brow_fixed};
            int brow_m = {brow_fixed}/{brow};
            // metadata
            int mcol_kk = {nelems}/{mrow_m}/{n};
            int mcol_k = {A_num_cols_sp_pad}/{n}/mcol_kk;
            // indices
            int col_kk = mcol_kk;
            int col_k = {A_num_cols_sp_pad}/{n}/col_kk;

            uint indexes[{nelems}];
            uint columns[col_kk*{m_fixed}];

            for(int bm_i=0; bm_i<bm_m; bm_i++){{
                for(int mbrow_i=0; mbrow_i<mbrow_m; mbrow_i++){{
                    for(int mbrow_i2=0; mbrow_i2<mbrow_m2; mbrow_i2++){{
                        for(int brow_i=0; brow_i<brow_m; brow_i++){{
                            for(int mcol_i=0; mcol_i<mcol_k; mcol_i++){{
                                //read columns indexes
                                for(int col_i=0; col_i<col_kk; col_i++){{
                                    for(int col_ii=0; col_ii<{m_fixed}; col_ii++){{
                                        columns[col_i*{m_fixed} + col_ii] =
                                        hA_columns[bm_i*col_k*col_kk*{m_fixed} + mcol_i*col_kk*{m_fixed} + col_i*{m_fixed} + col_ii];
                                    }}
                                }}
                                // read metadata
                                for(int mbrow_ii=0; mbrow_ii<({brow}/{mrow_m}); mbrow_ii++){{
                                    for(int mbrow_iii=0; mbrow_iii<{mrow_m}; mbrow_iii++){{
                                        for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){{
                                            for (int n_i=0; n_i<{n}; n_i++) {{
                                                indexes[
                                                    mbrow_iii*{n} +
                                                    mcol_ii*{mrow_m}*{n} +
                                                    n_i] =
                                                (((hA_metadata[
                                                    bm_i*mcol_k*{bm}/{mrow_m} +
                                                    mbrow_i*mcol_k*{mbrow}/{mrow_m} +
                                                    mbrow_i2*{brow_fixed}/{mrow_m} +
                                                    brow_i*{brow}/{mrow_m}  +
                                                    mcol_i*{mbrow}/{mrow_m} +
                                                    mbrow_ii]) >> (mbrow_iii*({nelems}/{mrow_m})*{bits_elem_meta}+mcol_ii*{n}*{bits_elem_meta}+n_i*{bits_elem_meta})) & 0x3);
                                            }}
                                        }}
                                    }}

                                    for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){{
                                        for(int mbrow_iii=0; mbrow_iii<{mrow_m}; mbrow_iii++){{
                                            for(int n_i=0; n_i<{n}; n_i++){{
                                                unsigned int index = columns[mcol_ii*{m_fixed} + indexes[mcol_ii*{mrow_m}*{n}+mbrow_iii*{n}+n_i]];

                                                if((mcol_i*{m}*mcol_kk + mcol_ii*{m} + index) < {ncols}){{
                                                    hA_dense[
                                                        bm_i*{bm}*{ncols} +
                                                        mbrow_i*{mbrow}*{ncols} +
                                                        mbrow_i2*{brow_fixed}*{ncols} +
                                                        brow_i*{brow}*{ncols} +
                                                        mcol_i*{m}*mcol_kk +
                                                        mbrow_ii*{mrow_m}*{ncols} +
                                                        mcol_ii*{m} +
                                                        mbrow_iii*{ncols} +
                                                        index] =
                                                    hA_values[
                                                        bm_i*{bm}*{A_num_cols_sp_pad} +
                                                        mbrow_i*{mbrow}*{A_num_cols_sp_pad}+
                                                        mbrow_i2*{brow_fixed}*{A_num_cols_sp_pad}+
                                                        brow_i*{brow}*{nelems}/{mrow_m}+
                                                        mcol_i*{brow_fixed}*{nelems}/{mrow_m} +
                                                        mbrow_ii*{mrow_m}*{n} +
                                                        mcol_ii*{n}*{brow} +
                                                        mbrow_iii*{n} +
                                                        n_i];
                                                }}
                                            }}
                                        }}
                                    }}
                                }}
                            }}
                        }}
                    }}
                }}
            }}
        }}
        """,
    )
    lib.func3.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return lib.func3


@cache
def dense2venom(dense_shape, dense_dtype, n, m, tileM):
    nrows = dense_shape[0]
    ncols = dense_shape[1]

    brow = 4 #this->brow = brow_;
    mbrow = 32 #this->mbrow = mbrow_;

    bm   = tileM
    # !IMPORTANT! constants because of architecture constraints
    m_fixed = 4
    bits_elem_meta=2
    mrow_m = 2
    bits_elem_cols=8
    brow_fixed = 16
    nelems=32//bits_elem_meta #(sizeof(uint)*8)=32
    nelems_col = nelems//mrow_m

    A_num_cols_sp = (ncols//m)*n
    A_num_cols_sp_pad_nm = (round_up(ncols, m)/m)*n
    A_num_cols_sp_pad = round_up((round_up(ncols, m)/m)*n, 16)
    A_nnz = nrows*A_num_cols_sp_pad

    assert dense_dtype in (torch.float32, torch.float64)
    dtype = "float" if dense_dtype == torch.float32 else "double"
    lib = compile(
        f"""
        #include <iostream>
        #include <algorithm>
        #include <utility>
        #include <cstdlib>
        #include <cstdio>
        #include <cmath>
        #include <functional>
        #include <tuple>
        #include <vector>
        #include <numeric>
        #include <chrono>

        using namespace std;


        extern "C" void func2({dtype}* sparse, int* masks, {dtype}* hA_values, int *hA_columns, int *hA_metadata){{

            int bm_m = {nrows}/{bm};
            int mbrow_m = {bm}/{mbrow};
            int mbrow_m2 = {mbrow}/{brow_fixed};
            int brow_m = {brow_fixed}/{brow};
            // metadata
            int mcol_kk = {nelems}/{mrow_m}/{n};
            int mcol_k = {A_num_cols_sp_pad}/{n}/mcol_kk;
            // indices
            int col_kk = mcol_kk;
            int col_k = {A_num_cols_sp_pad}/{n}/col_kk;

            {dtype} values[{nelems}];
            uint indexes[{nelems}];
            uint columns[col_kk*{m_fixed}];

            int max_idx = 0;

            for(int bm_i=0; bm_i<bm_m; bm_i++){{
                for(int mbrow_i=0; mbrow_i<mbrow_m; mbrow_i++){{
                    for(int mbrow_i2=0; mbrow_i2<mbrow_m2; mbrow_i2++){{
                        for(int brow_i=0; brow_i<brow_m; brow_i++){{
                            for(int mcol_i=0; mcol_i<mcol_k; mcol_i++){{
                                for(int col_i=0; col_i<col_kk; col_i++){{
                                    for(int col_ii=0; col_ii<{m_fixed}; col_ii++){{
                                        columns[col_i*{m_fixed} + col_ii] =
                                        hA_columns[bm_i*col_k*col_kk*{m_fixed} + mcol_i*col_kk*{m_fixed} + col_i*{m_fixed} + col_ii];
                                    }}
                                }}
                                for(int mbrow_ii=0; mbrow_ii<({brow}/{mrow_m}); mbrow_ii++){{
                                    for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){{
                                        for(int mbrow_iii=0; mbrow_iii<{mrow_m}; mbrow_iii++){{
                                            int pos=0;
                                            for(int n_i=0; n_i<{m_fixed}; n_i++){{
                                                unsigned int index = columns[mcol_ii*{m_fixed} + n_i];

                                                if((mcol_i*{m}*mcol_kk + mcol_ii*{m} + index) < {ncols}){{
                                                    int nnz = masks[
                                                            bm_i*{bm}*{ncols} +
                                                            mbrow_i*{mbrow}*{ncols} +
                                                            mbrow_i2*{brow_fixed}*{ncols} +
                                                            brow_i*{brow}*{ncols} +
                                                            mcol_i*{m}*mcol_kk +
                                                            mbrow_ii*{mrow_m}*{ncols} +
                                                            mcol_ii*{m} +
                                                            mbrow_iii*{ncols} +
                                                            index];

                                                    if(nnz != 0){{
                                                        indexes[
                                                            mbrow_iii*{n} +
                                                            mcol_ii*{mrow_m}*{n} +
                                                            pos] = n_i;

                                                        values[
                                                            mcol_ii*{mrow_m}*{n} +
                                                            mbrow_iii*{n} +
                                                            pos] =
                                                        sparse[
                                                            bm_i*{bm}*{ncols} +
                                                            mbrow_i*{mbrow}*{ncols} +
                                                            mbrow_i2*{brow_fixed}*{ncols} +
                                                            brow_i*{brow}*{ncols} +
                                                            mcol_i*{m}*mcol_kk +
                                                            mbrow_ii*{mrow_m}*{ncols} +
                                                            mcol_ii*{m} +
                                                            mbrow_iii*{ncols} +
                                                            index];

                                                        pos+=1;
                                                    }}
                                                }} else {{
                                                    if(n_i<2){{
                                                        indexes[
                                                            mbrow_iii*{n} +
                                                            mcol_ii*{mrow_m}*{n} +
                                                            pos] = 0;

                                                        values[
                                                            mcol_ii*{mrow_m}*{n} +
                                                            mbrow_iii*{n} +
                                                            pos] = 0;

                                                        pos+=1;
                                                    }}
                                                }}
                                            }}
                                        }}
                                    }}
                                    // write metadata
                                    unsigned int meta=0;
                                    for(int mbrow_iii=0; mbrow_iii<{mrow_m}; mbrow_iii++){{
                                        for(int mcol_ii=0; mcol_ii<mcol_kk; mcol_ii++){{
                                            for (int n_i=0; n_i<{n}; n_i++) {{

                                                int idx = bm_i*{bm}*{A_num_cols_sp_pad} +
                                                        mbrow_i*{mbrow}*{A_num_cols_sp_pad}+
                                                        mbrow_i2*{brow_fixed}*{A_num_cols_sp_pad}+
                                                        brow_i*{brow}*{nelems}/{mrow_m}+
                                                        mcol_i*{brow_fixed}*{nelems}/{mrow_m} +
                                                        mbrow_ii*{mrow_m}*{n} +
                                                        mcol_ii*{n}*{brow} +
                                                        mbrow_iii*{n} +
                                                        n_i;

                                                max_idx = (idx>max_idx)?(idx):(max_idx);

                                                hA_values[
                                                        idx] =
                                                values[
                                                    mcol_ii*{mrow_m}*{n} +
                                                    mbrow_iii*{n} +
                                                    n_i];

                                                unsigned int tmp = indexes[
                                                            mbrow_iii*{n} +
                                                            mcol_ii*{mrow_m}*{n} +
                                                            n_i];
                                                meta |= (tmp << (mbrow_iii*({nelems}/{mrow_m})*{bits_elem_meta}+mcol_ii*{n}*{bits_elem_meta}+n_i*{bits_elem_meta}));
                                            }}
                                        }}
                                    }}
                                    hA_metadata[bm_i*mcol_k*{bm}/{mrow_m} +
                                                mbrow_i*mcol_k*{mbrow}/{mrow_m} +
                                                mbrow_i2*{brow_fixed}/{mrow_m} +
                                                brow_i*{brow}/{mrow_m}  +
                                                mcol_i*{mbrow}/{mrow_m} +
                                                mbrow_ii] = meta;
                                }}
                            }}
                        }}
                    }}
                }}
            }}
            //cout << "max_idx: " << max_idx << endl;
        }}
        """,
    )
    lib.func2.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
    ]
    return lib.func2


class SparseVNMTensor:
    def __init__(self, n_, m_, v_, device, dense = None, mask = None, values = None, metadata = None, columns=None):
        """
            either  [dense, mask, columns]
            or      [mask, values, columns, metadata]
            have to be given
        """
        
        self.n = n_
        self.m = m_
        self.v = v_
        self.nnz = 0
        

        if dense is not None:
            #print("SparseVNMTensor from dense tensor")
            self.dense = dense.cpu().to(dtype=torch.float32)
            self.nrows, self.ncols = dense.shape
        else:
            #print("SparseVNMTensor from compressed components")
            self.dense = None
            self.nrows, self.ncols = mask.shape
            
        self.device=device
        self.columns = columns.contiguous() if columns is not None else None
        self.values = values.contiguous() if values is not None else None
        self.metadata = metadata.contiguous() if metadata is not None else None

        self.mask = mask

        if self.values is None and self.metadata is None:
            # Compress dense tensor according to mask.
            #print("Compressing tensor to venom...")
            self.to_venom(dense.cpu().to(dtype=torch.float32), mask.cpu())
        else:
            # Complete remaining attributes.
            #print("storing nnz...")
            A_num_cols_sp_pad = round_up((round_up(self.ncols, self.m)/self.m)*self.n, 16)
            self.nnz = self.nrows*A_num_cols_sp_pad

    def to_venom(self, dense_, mask_):
        impl_builder = (
            dense2venom
            )
        func = impl_builder(
                dense_.shape,
                dense_.dtype,
                self.n,
                self.m,
                self.v
            )

        self.nrows, self.ncols = dense_.shape
        A_num_cols_sp_pad = round_up((round_up(self.ncols, self.m)/self.m)*self.n, 16)
        self.nnz = self.nrows*A_num_cols_sp_pad
        m_fixed = 4
        mrow_m = 2
        bits_elem_meta=2

        nelems = 32//bits_elem_meta
        nelems_col = nelems//mrow_m

        self.values = torch.zeros(self.nrows * A_num_cols_sp_pad, dtype=torch.float32, device="cpu")
        self.metadata = torch.zeros(self.nrows//mrow_m * A_num_cols_sp_pad//nelems_col, dtype=torch.int32, device="cpu")

        func(dense_.data_ptr(), mask_.data_ptr(), self.values.data_ptr(), self.columns.data_ptr(), self.metadata.data_ptr())
        
        # Test: Convert values to half. Model encounters unexpected floats. Myabe it is the values tensor.
        self.values.half()
        

    def to_dense(self):
        local_values = self.values.clone().float().cpu()
        local_columns = self.columns.cpu()
        local_metadata = self.metadata.cpu()
        
        impl_builder = (
            venom2dense
            )
        func = impl_builder(
                (self.nrows, self.ncols),
                torch.float32,
                self.n,
                self.m,
                self.v
            )
        # initialize with ones
        dense = torch.zeros((self.nrows, self.ncols), dtype=self.values.dtype, device="cpu", requires_grad=True)

        #print("values device:", local_values.device, "columns device:", local_columns.device, "metadata device:", local_metadata.device)
        func(dense.data_ptr(), local_values.data_ptr(), local_columns.data_ptr(), local_metadata.data_ptr())

        return dense.to(device="cuda:0").half()







#def sddmm_dispatch(lhs_matrix, rhs_matrix, A_metadata, A_values, A_indices, A_num_rows, A_num_cols, B_num_cols, v, n, m, nnz):
#    """
#        Performs matrix multiplication of two matrix for backpropagation, lhs and rhs, 
#        resulting in a matriz with a sparse distribution equal to another matrix, A.
#        
#        The values of lhs and rhs are given as tensors, and the sparsity distribution of A is given with the indices and metadata information.
#        
#    """
#
#
#    output = spatha.sddmm(
#                          lhs_matrix.to(device='cuda:0'),   # lhs operand
#                          rhs_matrix.to(device='cuda:0'),   # rhs operand
#                          A_metadata.to(device='cuda:0'),    # metada for output sparsity distribution
#                          A_values.to(dtype=torch.half).to(device='cuda:0'),    # Values for output sparsity distribution
#                          A_indices.to(device='cuda:0'),           # indices for output sparsity distribution
#                          A_num_rows,
#                          A_num_cols,         
#                          B_num_cols,         
#                          v,          
#                          n,                # N
#                          m,                # M
#                          nnz,              # nnz
#                          0,                # seed
#                          32,               # mbrow
#                          4                 # brow
#                          )
#
#    return output

#def sparse_dense_mul_dispatch(sparse_values, sparse_indices, sparse_metadata, dense, nrows_sp, ncols_sp, ncols_d, m, n, v, nnz, bias):
#
#    dense_ = dense.contiguous()
#
#    output = spatha.spmm(
#                          sparse_metadata,  # metadata
#                          sparse_indices,   # indices
#                          sparse_values,    # values
#                          dense_,           # rhs_matrix
#                          bias,             # bias
#                          nrows_sp,         # A_num_rows
#                          ncols_sp,         # A_num_cols
#                          ncols_d,          # B_num_cols
#                          v,                # V
#                          n,                # N
#                          m,                # M
#                          nnz,              # nnz
#                          0,                # seed
#                          32,               # mbrow
#                          4                 # brow
#                          )
#
#    return output



class VENOMSparsifier:
    """
    Sparsifier for venom tensors. Creates a VENOM compressed tensor (SparseVNMTensor) built from the given tensor.
    
    Can receive the mask to apply if available, or use a random one
    """
    def __init__(self, v, n, m):
        self.v = v
        self.n = n
        self.m = m
        
        
    def __call__(self, tensor, grad_fmt=None, mask=None):
        nrows, ncols = tensor.shape
        columns = torch.zeros(nrows//self.v, ncols//self.m*4, dtype=torch.int32)
        columns = columns.reshape((-1,4)) + torch.tensor([0,1,2,3], dtype=torch.int32)
        columns = columns.reshape((nrows//self.v, ncols//self.m*4))

        if mask is None:
            mask = NMVectorSparsifier.get_random_mask(tensor, self.m, self.v)

        sparse_mtx = sten.SparseTensorWrapper.wrapped_from_dense(
            SparseVNMTensor(self.n, self.m, self.v, tensor.device, dense=tensor, mask=mask, columns=columns ),
            tensor,
            grad_fmt,
        )

        return sparse_mtx


class NMVectorSparsifier:
    def __init__(self, n, m, v):
        self.n = n
        self.m = m
        self.v = v

    @staticmethod
    def get_random_mask(tensor, m, v):
        mask = torch.zeros(tensor.shape, dtype=tensor.dtype)
        m_tmp = torch.cat( (torch.tensor([1,0,1,0]), torch.zeros(m-4)), 0 )
        mask = mask.reshape(-1, v, m) + m_tmp
        mask = mask.reshape(tensor.shape)

        return mask

    def __call__(self, tensor, grad_fmt=None):
        nrows, ncols = tensor.shape
        columns = torch.zeros(nrows//self.v, ncols//self.m*4, dtype=torch.int32)
        columns = columns.reshape((-1,4)) + torch.tensor([0,1,2,3], dtype=torch.int32)
        columns = columns.reshape((nrows//self.v, ncols//self.m*4))

        mask = NMVectorSparsifier.get_random_mask(tensor, self.m, self.v)

        sparse_mtx = sten.SparseTensorWrapper.wrapped_from_dense(
            SparseVNMTensor(self.n, self.m, self.v, tensor.device, dense=tensor, mask=mask, columns=columns ),
            tensor,
            grad_fmt,
        )

        return sparse_mtx
    
    
# Autograd function to perform forward and backwards operations using VENOM tensors and sparse computation from spatha
class VenomSparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, 
                input, 
                sparse_weights_values, 
                bias, 
                sparse_weights_columns, 
                sparse_weights_metadata, 
                sparse_weights_dense,
                venom_v, venom_n, venom_m, nnz):
        ctx.save_for_backward(input, bias, sparse_weights_columns, sparse_weights_metadata, sparse_weights_dense)
        
        nrows, ncols = sparse_weights_dense.shape
        ctx.sparse_weights_num_rows = nrows
        ctx.sparse_weights_num_cols = ncols
        ctx.v = venom_v
        ctx.n = venom_n
        ctx.m = venom_m
        ctx.nnz = nnz
        
        spmm_input = torch.flatten(input, start_dim=0, end_dim=-2).half().T.contiguous()
        #invalid_type_tensor = torch.ones(sparse_weights_metadata.shape, dtype=int)
        #print("\nTypes: metadata:",  type(sparse_weights_metadata), sparse_weights_metadata.dtype, 
        #        "\ncolumns:", type(sparse_weights_columns), sparse_weights_columns.dtype ,
        #        "\nvalues:", type(sparse_weights_values), sparse_weights_values.dtype ,
        #        "\ninput:", type(spmm_input), spmm_input.dtype ,
        #        "\nbias:", type(bias), bias.dtype,
        #        "\nnrows", type(nrows),
        #        "\nncols", type(ncols),
        #        "\nspmm_input.shape[1]", type(spmm_input.shape[1]),
        #        "\nvenom_v", type(venom_v),
        #        "\nvenom_n", type(venom_n),
        #        "\nvenom_m", type(venom_m),
        #        "\nnnz", type(nnz),
        #        )
        #print("\nSnippets: metadata:", sparse_weights_metadata[:8], 
        #        "\ncolumns:",sparse_weights_columns[:8, :8] ,
        #        "\nvalues:", sparse_weights_values[:8] ,
        #        "\ninput:", spmm_input[:8, :8] ,
        #        "\nbias:", bias[:8],
        #        "\nnrows", nrows,
        #        "\nncols", ncols,
        #        "\nspmm_input.shape[1]", spmm_input.shape[1],
        #        "\nvenom_v", venom_v,
        #        "\nvenom_n", venom_n,
        #        "\nvenom_m", venom_m,
        #        "\nnnz", nnz,
        #        )
        #print("\nInvalid tensor: type:", type(invalid_type_tensor), 
        #      "\ndtype:", invalid_type_tensor.dtype, 
        #      "\nSnippet:", invalid_type_tensor[:8])
        
        spmm_output = spatha.spmm(sparse_weights_metadata,               # metadata
                          sparse_weights_columns,                        # indices
                          sparse_weights_values,                         # values
                          spmm_input,                                    # rhs_matrix
                          bias,                                          # Bias
                          nrows,                                         # A_num_rows
                          ncols,                                         # A_num_cols
                          spmm_input.shape[1],                           # B_num_cols
                          venom_v,                                       # vec_length
                          venom_n,                                       # n
                          venom_m,                                       # m
                          nnz,                                           # nnz
                          0,                                             # seed
                          32,                                            # mbrow
                          4                                              # brow
                          )
        #print("SPMM completed")
        spmm_output = spmm_output.reshape((*input.shape[0:-1], -1))[..., :nrows]
        #print("SPMM completed. Output dtype:", spmm_output.dtype)
        return spmm_output
            

    @staticmethod
    def backward(ctx, grad_output):
        input, bias, weights_columns, weights_metadata, weights_dense  = ctx.saved_tensors
        
        global global_grad_output  
        global_grad_output = grad_output 

        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weights_dense
        
        if ctx.needs_input_grad[1]: 
            #grad_weight = grad_output.t() @ input.to("cuda:0")
            
            sddmm_grad_output = torch.flatten(grad_output, start_dim=0, end_dim=-2)
            sddmm_grad_output = sddmm_grad_output.T.contiguous()
            sddmm_input = torch.flatten(input, start_dim=0, end_dim=-2).half()
            sddmm_input = sddmm_input.T.contiguous()# Transpose both since sddmm does A@B.T, but we want A.T@B.
            
            #print("\nTypes: sddmm_grad_output:",  type(sddmm_grad_output), sddmm_grad_output.dtype, 
            #      "\nTypes: sddmm_input:",  type(sddmm_input), sddmm_input.dtype, 
            #      "\nTypes: weights_metadata:",  type(weights_metadata), weights_metadata.dtype, 
            #      "\nTypes: weights_columns:",  type(weights_columns), weights_columns.dtype 
            #)
            
            compressed_grad_weights = spatha_sddmm.sddmm(
                            sddmm_grad_output,             # A_matrix
                            sddmm_input,                   # B_matrix
                            weights_metadata,              # C_metadata
                            weights_columns,               # C_indices
                            ctx.sparse_weights_num_rows,   # C_num_rows
                            ctx.sparse_weights_num_cols,   # C_num_cols    
                            sddmm_input.shape[1], 
                            ctx.n,              # N
                            ctx.m,              # M
                            ctx.nnz,            # nnz
                            0,                             # seed
                            32,                            # mbrow
                            4                              # brow
                            )
            grad_weight = compressed_grad_weights.flatten()
            #print("SDDMM completed: output dtype:", compressed_grad_weights.dtype)
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        # Needs one return value for each forward argument.
        #      input,      sparse_weights_values, bias,      sparse_weights_columns, sparse_weights_metadata, sparse_weights_dense, venom_v, venom_n, venom_m, nnz
        #print("Backwards output dtypes: grad_input:", grad_input.dtype, "grad_weight:", grad_weight.dtype, "grad_bias:", grad_bias.dtype)
        return grad_input, grad_weight,           grad_bias, None,                   None,                    None,                 None,    None,    None,    None
    


# Torch.nn.Module subclass that sparsifies an existing torch.nn.lineal layer and uses dense algebra with a masked weights tensor.
class VenomSparseLinear(torch.nn.Module):
    def __init__(self, v, n, m, original: torch.nn.Linear = None, weights=None, bias=None, mask=None):
        super(VenomSparseLinear, self).__init__()        

        if original is not None and mask is None:
            # Built compressing original torch.nn.Linear module's weights
            #print("Compressing original weights tensor with random mask...")
            self.compressed_weights = NMVectorSparsifier(n, m, v)(original.weight).wrapped_tensor
        elif original is not None and mask is not None:
            # Build by compressing the original torch.nn.Linear module's weights, but using the given mask, not a random one
            #print("Compressing original weights tensor using specific mask...")
            self.compressed_weights = VENOMSparsifier(v, n, m)(original.weight, mask=mask).wrapped_tensor
        else:
            # Original is not provided, build from weights and mask tensors
            assert weights is not None
            assert mask is not None
            self.compressed_weights = VENOMSparsifier(v, n, m)(weights, mask=mask).wrapped_tensor

        self.values = torch.nn.Parameter(self.compressed_weights.values.to(device="cuda:0").half())
        self.values.requires_grad = True
        self.columns = self.compressed_weights.columns.to(device="cuda:0")
        self.metadata = self.compressed_weights.metadata.to(device="cuda:0")

        # Copy bias from original if provided
        if original is not None:
            self.bias = original.bias.to(device="cuda:0").half()
        else:
            assert bias is not None
            self.bias = bias.to(device="cuda:0")

        self.dense = torch.nn.Parameter(self.compressed_weights.to_dense().to(device="cuda:0").half())
        self.mask = self.compressed_weights.mask

        self.nrows_sp = self.compressed_weights.nrows
        self.ncols_sp = self.compressed_weights.ncols
        self.nnz      = self.compressed_weights.nnz
        
        self.v = v
        self.n = n
        self.m = m

    def forward(self, input):

        return VenomSparseLinearFunction.apply(input, self.values, self.bias, self.columns, self.metadata, self.dense, self.v, self.n, self.m, self.nnz)

    def clear_grads(self):
        #self.dense.grad = None
        self.bias.grad = None


"""
class VenomSpmm(torch.nn.Module):
    def __init__(self, original: torch.nn.Linear, v:int, n:int, m:int):
        super().__init__()
        self._v = v
        self._n = n
        self._m = m
        self.bias = original.bias
        #self.bias = torch.zeros(original.bias.shape, dtype=original.bias.dtype, device=original.bias.device)

        # Convert weights from original module to SrNM
        w = VenomSparsifier(n, m, v)(original.weight).wrapped_tensor

        self.values = torch.nn.Parameter(w.values)
        self.columns = w.columns
        self.metadata = w.metadata

        self.nrows_sp = w.nrows
        self.ncols_sp = w.ncols
        self.nnz      = w.nnz

    def forward(self, input):

        flattened_input = torch.flatten(input, start_dim=0, end_dim=-2)

        ncols_d  = flattened_input.T.shape[1]
        DM, _    = flattened_input.shape
        
        bias2 = torch.zeros(self.bias.shape, dtype=self.bias.dtype, device=self.bias.device)

        output = sparse_dense_mul_dispatch( self.values, 
                                            self.columns, 
                                            self.metadata, 
                                            flattened_input.T, 
                                            self.nrows_sp, 
                                            self.ncols_sp,
                                            ncols_d, 
                                            self._m, 
                                            self._n, 
                                            self._v, 
                                            self.nnz, 
                                            self.bias)
        #print(output.shape)
        #print("bias", self.bias.shape, self.bias.dtype)
        #print(DM)
        
        output = output.reshape((*input.shape[0:-1], -1))[..., :DM]
        #output = output.reshape((32,512,1024))
        
        return output
"""

def check_VNM(v:int, n:int, m:int, m_fixed:int, tensor: torch.Tensor, verbose:bool=False):
    shape = tensor.shape

    blocks = (shape[0]/v)*(shape[1]/m)
    ok_blocks = 0
    invalid_blocks = 0

    for row_block in range(0, shape[0], v):
        for column_block in range(0, shape[1], m):
            #print(f'Block {row_block}:{row_block+v}, {column_block}:{column_block+m}: ', end="")
            any_row_invalid = False
            non_zero_columns = []
            invalid_rows = 0
            for row in range(row_block, row_block+v):
                non_zero_position = 0
                empty_positions = 0
                for column in range(column_block, column_block+m):
                    if row < shape[0] and column < shape[1]:
                        if tensor[row][column] == 0:
                            empty_positions+=1
                        else:
                            non_zero_position += 1
                            if column not in non_zero_columns:
                                non_zero_columns.append(column)
                if non_zero_position > n:
                    any_row_invalid = True
                    invalid_rows += 1
            if invalid_rows > 0:
                print(f'\t{invalid_rows} rows with  more than {n} non zero positions,')

            #print(f'{len(non_zero_columns)} columns with some non zero element')
            if len(non_zero_columns) > m_fixed:
                any_row_invalid = True
                print(f'\tBlock has {len(non_zero_columns)} columns with some non zero element')
                print(f'\tError: more than the {m_fixed} columns with some non zero elements allowed.')

            if any_row_invalid:
                invalid_blocks+=1
                print(f'\tBlock {row_block}:{row_block+v}, {column_block}:{column_block+m} does not meet requirements')
            else:
                #print(f'\tBlock {row_block}:{row_block+64}, {column_block}:{column_block+10} meets requirements')
                ok_blocks+=1
            if verbose: print(f'{ok_blocks+invalid_blocks}/{blocks} processed blocks. {ok_blocks} valid ones, {invalid_blocks} invalid. Processing block {row_block}:{row_block+v}, {column_block}:{column_block+m}                                             ', end="\r")
    if not verbose: print(f'{ok_blocks+invalid_blocks}/{blocks} processed blocks. {ok_blocks} valid ones, {invalid_blocks} invalid.')

def check_sparsity(tensor: torch.Tensor):

    zero_elements = 0
    non_zero_elements = 0
    total_elements = 0

    for row in range(tensor.shape[0]):
        for column in range(tensor.shape[1]):
            total_elements+=1
            if tensor[row][column] == 0:
                zero_elements+=1
            else:
                non_zero_elements+=1

    print(f'Tensor calculated sparsity: {zero_elements/total_elements}')


def linear_to_venom(mod, v:int, n:int, m:int):
    """
    replaces linear modules for spmm sparse kernels from venom.
    """
    if isinstance(mod, torch.nn.Linear):
        return VenomSparseLinear(v, n, m, mod)

    for name, submod in mod.named_children():
        if isinstance(submod, VenomSparseLinear) or name=="classifier":
            continue
        if isinstance(submod, torch.nn.Linear):
            setattr(mod, name, VenomSparseLinear(v, n, m, submod))
        elif submod is not mod:
            linear_to_venom(submod, v, n, m)

    return mod


def linear_to_venom_with_masks(mod, v:int, n:int, m:int, tensor_to_mask_mapping):
    """
    Replaces linear modules for spmm sparse kernels from venom. 
    Expects to find linear modules with weights already masked, as it 
    will create the mask from those weights, looking for 0 values in the weights tensor.
    
    """
    
    if isinstance(mod, torch.nn.Linear):
        
        if mod.weight in tensor_to_mask_mapping:
            print("Found module weight tensor in tensor_to_mask_mapping.")
            mask = tensor_to_mask_mapping[mod.weight]
            num_zeros = torch.sum(mask == 0).item()
            num_ones = torch.sum(mask == 1).item()
            print("External if mask: ones:", num_ones, "zeros:", num_zeros, "Sparsity:", num_ones/(num_ones+num_zeros))
            return VenomSparseLinear(v, n, m, original=mod, mask=mask)
        else:
            print("Module weight tensor not present in tensor_to_mask_mapping.")
            mask = torch.where(mod.weight != 0, torch.ones_like(mod.weight), torch.zeros_like(mod.weight))
            num_zeros = torch.sum(mask == 0).item()
            num_ones = torch.sum(mask == 1).item()
            print("External if detected mask: ones:", num_ones, "zeros:", num_zeros, "Sparsity:", num_ones/(num_ones+num_zeros))
        
        #check_VNM(v=v, n=n, m=m, m_fixed=n+2, tensor=mask)
        

    for name, submod in mod.named_children():
        if isinstance(submod, VenomSparseLinear) or name=="classifier":
            continue
        if isinstance(submod, torch.nn.Linear):
            if submod.weight in tensor_to_mask_mapping:
                print("Found module weight tensor in tensor_to_mask_mapping.")
                mask = tensor_to_mask_mapping[submod.weight]
                num_zeros = torch.sum(mask == 0).item()
                num_ones = torch.sum(mask == 1).item()
                print("Internal if mask: ones:", num_ones, "zeros:", num_zeros, "Sparsity:", num_ones/(num_ones+num_zeros))
                setattr(mod, name, VenomSparseLinear(v, n, m, original=submod, mask=mask))
            else:
                print("submodule", name, "weight tensor not present in tensor_to_mask_mapping.")
                mask = torch.where(submod.weight != 0, torch.ones_like(submod.weight), torch.zeros_like(submod.weight))
                num_zeros = torch.sum(mask == 0).item()
                num_ones = torch.sum(mask == 1).item()
                print("Internal if detected mask: ones:", num_ones, "zeros:", num_zeros, "Sparsity:", num_ones/(num_ones+num_zeros))
            #check_VNM(v=v, n=n, m=m, m_fixed=n+2, tensor=mask)
            #setattr(mod, name, VenomSparseLinear(v, n, m, original=submod, mask=mask))
        elif submod is not mod:
            linear_to_venom_with_masks(submod, v, n, m, tensor_to_mask_mapping)

    return mod


def linear_to_venom_detecting_mask(mod, v:int, n:int, m:int):
    """
    Replaces linear modules for spmm sparse kernels from venom. 
    Expects to find linear modules with weights already masked, as it 
    will create the mask from those weights, looking for 0 values in the weights tensor.
    
    """
    
    if isinstance(mod, torch.nn.Linear):
        mask = torch.where(mod.weight != 0, torch.ones_like(mod.weight), torch.zeros_like(mod.weight))
        num_zeros = torch.sum(mask == 0).item()
        num_ones = torch.sum(mask == 1).item()
        #print("External if mask: ones:", num_ones, "zeros:", num_zeros, "Sparsity:", num_ones/(num_ones+num_zeros))
        #check_VNM(v=v, n=n, m=m, m_fixed=n+2, tensor=mask)
        # Only replace module if mask spartisy matches intended sparsity
        if n/m == num_ones/(num_ones+num_zeros):
            return VenomSparseLinear(v, n, m, original=mod, mask=None)
        else:
            print("Mask sparsity does not match N:M expected sparsity.")
            return mod

    for name, submod in mod.named_children():
        if isinstance(submod, VenomSparseLinear) or name=="classifier":
            continue
        if isinstance(submod, torch.nn.Linear):
            mask = torch.where(submod.weight != 0, torch.ones_like(submod.weight), torch.zeros_like(submod.weight))
            num_zeros = torch.sum(mask == 0).item()
            num_ones = torch.sum(mask == 1).item()
            #print("Internal if mask: ones:", num_ones, "zeros:", num_zeros, "Sparsity:", num_ones/(num_ones+num_zeros))
            #check_VNM(v=v, n=n, m=m, m_fixed=n+2, tensor=mask)
            if n/m == num_ones/(num_ones+num_zeros):
                setattr(mod, name, VenomSparseLinear(v, n, m, original=submod, mask=None))
            else:
                print("Mask sparsity does not match N:M expected sparsity.")
        elif submod is not mod:
            linear_to_venom_detecting_mask(submod, v, n, m)

    return mod


def find_and_replace_with_venom(model, target_tensor, tensor_mask, v:int, n:int, m:int):
    # Go over the model's children and check if it is a linear component.
    # and then if it has target_tensor as weights.
    
    if isinstance(model, torch.nn.Linear):
        #breakpoint()
        if model.weight is target_tensor:
            print("Found same tensor.")
            return VenomSparseLinear(v, n, m, weights=target_tensor, mask=tensor_mask, bias=model.bias)
        elif model.weight.shape == target_tensor.shape and  torch.allclose(model.weight, target_tensor):
            print("Found a tensor with the same data in the list.") 
            return VenomSparseLinear(v, n, m, weights=target_tensor, mask=tensor_mask, bias=model.bias)
        else: 
            print("Different tensors...")
    
    for name, submod in model.named_children():
        if isinstance(submod, VenomSparseLinear) or name=="classifier":
            continue
        if isinstance(submod, torch.nn.Linear) and submod.weight.shape == target_tensor.shape and (submod.weight is target_tensor or torch.allclose(submod.weight, target_tensor)):
            print("Replacing Linear component named", name)
            setattr(model, name, VenomSparseLinear(v, n, m, weights=target_tensor, mask=tensor_mask, bias=submod.bias))
        elif submod is not model:
            find_and_replace_with_venom(submod, target_tensor, tensor_mask, v, n, m)
    
    # Return model with replaced children
    return model

