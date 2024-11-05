import torch
import sten

from native_scripting import compile
import functools
import ctypes
import time
import math
from heapq import nlargest
import timeit
import statistics

import spatha
import spatha_sddmm

try:
    cache = functools.cache
except AttributeError:
    cache = functools.lru_cache(maxsize=None)


def report_time(name, data, number, v, n, m):
    for d in data:
        time_ms = d / number * 1000
        #print(f'n {n} m {m} format {name} time_ms {time_ms:.3f}')
    ds = [(d / number * 1000) for d in data]
    mean = statistics.mean(ds)
    median = statistics.median(ds)
    std = statistics.stdev(ds)

    if name == "n:m":
        cfg = str(n)+","+str(m)+","
    else:
        cfg = "0,0,"
    print(
        "0,"+cfg+str(v)+","+str(mean)+","+str(median)+","+str(std)+","+str(len(ds))
    )

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
            cout << "max_idx: " << max_idx << endl;
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

def round_up(x,y):
    return math.ceil(x/y)*y

class SparseVNMTensor:
    def __init__(self, n_, m_, v_, dense_, mask_, columns_, device_):
        self.n = n_
        self.m = m_
        self.v = v_
        self.nnz = 0
        self.nrows = None
        self.ncols = None
        
        self.dense = dense_.cpu().to(dtype=torch.float32)
        self.device=device_
        
        self.columns = columns_
        self.values = None
        self.metadata = None

        self.mask = mask_

        self.to_venom(dense_.cpu().to(dtype=torch.float32), mask_.cpu())

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

    def to_dense(self):
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
        
        func(dense.data_ptr(), self.values.data_ptr(), self.columns.data_ptr(), self.metadata.data_ptr())

        return dense.to(device="cuda:0").half()

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
            SparseVNMTensor(self.n, self.m, self.v, tensor, mask, columns, tensor.device),
            tensor,
            grad_fmt,
        )

        return sparse_mtx



# Autograd function to perform forward and backwards operations using VENOM tensors using dense algebra with masked tensors
class VenomMaskedLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias=None):
        ctx.save_for_backward(input, weight, bias)
        
        #print("VenomMaskedLinearFunction input shape:", input.shape)
        #print("VenomMaskedLinear input:\n", input[:8, :8] )
        #print("VenomMaskedLinear weights:\n", weight.t()[:8])

        output = torch.matmul(input, weight.t())
        if bias is not None:
            output += bias
        
        #print("VenomMaskedLinear forward output shape:", output.shape)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight, bias = ctx.saved_tensors

        global global_grad_output  
        global_grad_output = grad_output 

        #print(input.device, weight.device, grad_output.device)
        #print("grad_output shape", grad_output.shape)

        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            
            grad_input = grad_output @ weight.to("cuda:0")
            # grad_input = grad_output @ weigths.T, according to sten library.
            #grad_input = grad_output @ weight.T.to("cuda:0")
            #print("Computed grad_input, shape:", grad_input.shape, "input shape:", input.shape)
        if ctx.needs_input_grad[1]:
            #grad_weights = torch.from_numpy(input.T @ grad_outputs) # According to sten
            #grad_weight = input.T @ grad_output
            grad_weight = grad_output.T.contiguous() @ input.to("cuda:0") # Roberto example
            #print("Computed grad_weight, shape:", grad_weight.shape, "weights shape:", weight.shape)
        
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
            #print("Computed grad_bias, shape:", grad_bias.shape, "Bias shape:", bias.shape)
        
        #print(ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2])
        #print(grad_output)
        #print(input)
        #print(grad_weight)
        #print(grad_input[:4, :4])
        #print(grad_weight[:4, :4])
        
        return grad_input, grad_weight, grad_bias


# Torch.nn.Module subclass that sparsifies an existing torch.nn.lineal layer and uses dense algebra with a masked weights tensor.
class VenomMaskedLinear(torch.nn.Module):
    def __init__(self, original: torch.nn.Linear, v, n, m):
        super(VenomMaskedLinear, self).__init__()        

        self.w = NMVectorSparsifier(n, m, v)(original.weight).wrapped_tensor

        self.values = torch.nn.Parameter(self.w.values.to(device="cuda:0").half())
        self.columns = self.w.columns.to(device="cuda:0")
        self.metadata = self.w.metadata.to(device="cuda:0")

        self.bias = original.bias

        self.dense = torch.nn.Parameter(self.w.to_dense())
        self.mask = self.w.mask

        self.nrows_sp = self.w.nrows
        self.ncols_sp = self.w.ncols
        self.nnz      = self.w.nnz

    def forward(self, input):
        
        return VenomMaskedLinearFunction.apply(input, self.dense, self.bias)
    
    def clear_grads(self):
        self.dense.grad = None
        self.bias.grad = None


# Autograd function to perform forward and backwards operations using VENOM tensors and sparse computation from spatha
class VenomSparseLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, sparse_weights, bias=None):
        ctx.save_for_backward(input, bias) # sparse_weights cannot be saved as it is not a tensor
        ctx.sparse_weights = sparse_weights
        
        #ctx.A_num_rows = A_num_rows
        #ctx.A_num_cols = A_num_cols
        #ctx.B_num_cols = B_num_cols
        #ctx.v = v
        #ctx.n = n
        #ctx.m = m
        #ctx.nnz = nnz
        
        #output = torch.matmul(input, weight.t())
        #if bias is not None:
        #    output += bias
        #transposed_flattened_input = torch.flatten(input, start_dim=0, end_dim=-2).T
        #B_num_cols  = transposed_flattened_input.shape[1]
        
        #print("Transposed input shape:", transposed_flattened_input.shape)
        #print("VenomSparseLinear Transposed input:\n", transposed_flattened_input[:8, :8] )
        #print("VenomSparseLinear weights:\n", sparse_weights.values[:8])
        
        # output = torch.matmul(input, weights.T). Since we can't transpose weights (compressed)
        # output = torch.matmul(weights, input.T).T
        # spmm(weights, input.T) -> (weights*input).T
        spmm_input = input.T.contiguous()
        spmm_output = spatha.spmm(sparse_weights.metadata.cuda(),               # metadata
                          sparse_weights.columns.cuda(),                        # indices
                          sparse_weights.values.to(dtype=torch.half).cuda(),    # values
                          spmm_input,                                              # rhs_matrix
                          bias.to(dtype=torch.half).cuda(),                     # Bias
                          sparse_weights.nrows,                                 # A_num_rows
                          sparse_weights.ncols,                                 # A_num_cols
                          spmm_input.shape[1],                                     # B_num_cols
                          sparse_weights.v,                                     # vec_length
                          sparse_weights.n,                                     # n
                          sparse_weights.m,                                     # m
                          sparse_weights.nnz,                                   # nnz
                          0,                                                    # seed
                          32,                                                   # mbrow
                          4                                                     # brow
                          )
        #output = output.reshape((*input.shape[0:-1], -1))[..., :sparse_weights.nrows]
        #dense_masked_weight = torch.nn.Parameter(sparse_weights.to_dense())
        #output1 = torch.matmul(input, dense_masked_weight.t())
        #output2 = torch.matmul(dense_masked_weight, input.T).T
        #print("Dense (dense_masked_weight, input.T).T and sparse allclose?", torch.allclose(output2, spmm_output, atol=0.05))
        
        #torch.set_printoptions(linewidth=1000, threshold=10000)
        #print("*******************************\nCompressed weights values:\n", sparse_weights.values[:16])
        #print("Dense masked weights:\n", dense_masked_weight[:16, :16])
        
        #print("Transposed operations allclose?", torch.allclose(output1, output2, atol=0.001))
        #if bias is not None:
        #    print("adding bias")
        #    spmm_output += bias
        #print("VenomSparseLinear forward output shape:", spmm_output.shape)
        return spmm_output
            

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        sparse_weights = ctx.sparse_weights
        
        global global_grad_output  
        global_grad_output = grad_output 

        print(input.device, sparse_weights.device, grad_output.device)

        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ sparse_weights.dense.half().to("cuda:0")
        
        #if ctx.needs_input_grad[1]: # Do it always, due to weights being sparse, there is no 
        #grad_weight = grad_output.t() @ input.to("cuda:0")
        #grad_weight = grad_output.T.contiguous() @ input.to("cuda:0")
        print("Launching sddmm kernel")
        
        sddmm_grad_output = grad_output.T.contiguous()
        sddmm_input = input.T.contiguous()# Transpose both since sddmm does A@B.T, but we want A.T@B
        print("Types:",  type(sddmm_grad_output),    
                type(sddmm_input),                 
                type(sparse_weights.metadata),
                type(sparse_weights.columns), 
                type(sparse_weights.nrows),
                type(sparse_weights.ncols),
                type(sparse_weights.n),                     
                type(sparse_weights.m),                     
                type(sparse_weights.nnz),     
                type(0),                     
                type(32),                    
                type(4)                      
                )
        print("Values:",  sddmm_grad_output.shape, sddmm_grad_output[:8, :8],    
                sddmm_input.shape, sddmm_input[:8, :8],                 
                sparse_weights.metadata[:8],
                sparse_weights.columns[:8], 
                sparse_weights.nrows,
                sparse_weights.ncols,
                sparse_weights.n,                     
                sparse_weights.m,                     
                sparse_weights.nnz,     
                0,                     
                32,                    
                4, sep="\n"                      
                )
        
        print("sddmm_grad_output.shape:", sddmm_grad_output.shape)
        print("sddmm_input.shape:", sddmm_input.shape)
        
        compressed_grad_weights = spatha_sddmm.sddmm(
                        sddmm_grad_output,             # A_matrix
                        sddmm_input,                   # B_matrix
                        sparse_weights.metadata,       # C_metadata
                        sparse_weights.columns,        # C_indices
                        sparse_weights.nrows,          # C_num_rows
                        sparse_weights.ncols,          # C_num_cols    
                        sddmm_input.shape[1], 
                        sparse_weights.n,              # N
                        sparse_weights.m,              # M
                        sparse_weights.nnz,            # nnz
                        0,                             # seed
                        32,                            # mbrow
                        4                              # brow
                        )
        # Densify to return dense gradients. Can be optimized
        #print("Densifying sddmm output")
        #compressed_grad_weights = compressed_grad_weights.float().cpu()
        #columns = sparse_weights.columns.cpu()
        #metadata = sparse_weights.metadata.cpu()
        #impl_builder = (
        #            venom2dense
        #            )
        #func = impl_builder(
        #        (sparse_weights.nrows, sparse_weights.ncols),
        #        torch.float32, 
        #        sparse_weights.n,
        #        sparse_weights.m,
        #        sparse_weights.v
        #    )
        #grad_weight = torch.zeros((sparse_weights.nrows, sparse_weights.ncols), dtype=compressed_grad_weights.dtype, device="cpu", requires_grad=True)
        #func(grad_weight.data_ptr(), compressed_grad_weights.data_ptr(), columns.data_ptr(), metadata.data_ptr())
        #print("")
        #
        grad_weight = compressed_grad_weights
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        print(ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2])
        """ print(grad_output)
        print(input)
        print(grad_weight) """
        print(grad_input[:4, :4])
        print(grad_weight[:4, :4])
        
        return grad_input, grad_weight, grad_bias


# Torch.nn.Module subclass that sparsifies an existing torch.nn.lineal layer and uses dense algebra with a masked weights tensor.
class VenomSparseLinear(torch.nn.Module):
    def __init__(self, original: torch.nn.Linear|VenomMaskedLinear, v, n, m):
        super(VenomSparseLinear, self).__init__()        

        if isinstance(original, torch.nn.Linear):
            self.w = NMVectorSparsifier(n, m, v)(original.weight).wrapped_tensor

            self.values = torch.nn.Parameter(self.w.values.to(device="cuda:0").half())
            self.columns = self.w.columns.to(device="cuda:0")
            self.metadata = self.w.metadata.to(device="cuda:0")
        elif isinstance(original, VenomMaskedLinear):
            self.w = original.w
            self.values = original.values.to(device="cuda:0")
            self.columns = original.columns.to(device="cuda:0")
            self.metadata = original.metadata.to(device="cuda:0")

        self.bias = original.bias

        #self.dense = torch.nn.Parameter(self.w.to_dense())
        self.mask = self.w.mask

        self.nrows_sp = self.w.nrows
        self.ncols_sp = self.w.ncols
        self.nnz      = self.w.nnz

    def forward(self, input):
        
        return VenomSparseLinearFunction.apply(input, self.w, self.bias)

    def clear_grads(self):
        #self.dense.grad = None
        self.bias.grad = None



def main():

    torch.set_printoptions(precision=2)
    global_grad_output = None


    # Define input and standard dense computation of compatible size.
    input = torch.randn(256, 64, requires_grad=True, dtype=torch.half, device="cuda:0")
    dense_linear_layer = torch.nn.Linear(64, 128, bias=True, dtype=torch.half, device="cuda:0")
    

    # Set VENOM parameters
    v=32
    n=2
    m=8

    # Create a module that uses dense algebra with masked tensors.
    masked_dense_layer = VenomMaskedLinear(dense_linear_layer, v, n, m)
    print( type(masked_dense_layer) )
    # Create a new standard linear module but replace the weigths with the masked ones to have a reference result.
    torch_linear_layer_with_masked_weights = torch.nn.Linear(64, 128, bias=True, dtype=torch.half, device="cuda:0")
    # Copy masked weights from masked_dense_layer, assign them to the weights in this instance of standard linear module
    torch_linear_layer_with_masked_weights.weight = torch.nn.Parameter(masked_dense_layer.dense.detach().clone())
    torch_linear_layer_with_masked_weights.bias = torch.nn.Parameter(masked_dense_layer.bias.detach().clone())
    # Create sparse linear module, copying the weights from the masked one to ensure the same computations
    sparse_layer = VenomSparseLinear(masked_dense_layer, v, n, m)


    # Compute modules result. 
    #   Standard with masked weights, 
    #   VenomMaskedLinear that creates the masked weights and still uses dense algebra
    #   VenomSparseLinear that uses sparse matrix multiplication
    reference_result_dense_with_masked_weights = torch_linear_layer_with_masked_weights(input)
    masked_dense_result = masked_dense_layer(input)
    sparse_result = sparse_layer(input)
    
    # Compute dense result outside of pytorch
    #dense = input @ masked_dense_layer.dense.T + masked_dense_layer.bias



    # Check masked and dense computation is the same 
    linear_close_masked = torch.allclose(reference_result_dense_with_masked_weights.cuda(), masked_dense_result, atol=0.05)
    print("Torch.nn.Linear equal to VenomMaskedLinear:", linear_close_masked)
    if not linear_close_masked:
        print("Torch.nn.Linear result:\n", reference_result_dense_with_masked_weights[:8, :8] )
        print("VenomMaskedLinear:\n", masked_dense_result[:8, :8] )
    sparse_close_to_masked = torch.allclose(masked_dense_result.cuda(), sparse_result.cuda(), atol=0.05)
    print("Sparse result equal to Masked:", sparse_close_to_masked )
    if not sparse_close_to_masked:
        print("VenomMaskedLinear:\n", masked_dense_result[:8, :8] )
        print("VenomSparseLinear:\n", sparse_result[:8, :8] )
    #print("Manual result:\n", dense[:8, :8] )


    # Check backwards operations are equivalent
    # Empty grads
    input.grad = None
    torch_linear_layer_with_masked_weights.weight.grad = None
    torch_linear_layer_with_masked_weights.bias.grad = None
    masked_dense_layer.clear_grads()
    sparse_layer.clear_grads
    
    print("Testing reference backwards...")
    # Calculate loss and run backwards on reference result.
    reference_result_dense_with_masked_weights.sum().backward(retain_graph=True)
    
    reference_input_grad = input.grad.detach().clone()
    reference_weight_grad = torch_linear_layer_with_masked_weights.weight.detach().clone()
    #print("reference_input_grad shape:", reference_input_grad.shape)
    #print("reference_weight_grad shape:", reference_weight_grad.shape)
    
    print("Testing masked backwards...")
    # Clear input gradient to not accumulate the gradients of each module.
    input.grad = None
    # Calculate loss and run backwards on masked result.
        #reference_result_dense_with_masked_weights = torch_linear_layer_with_masked_weights(input)
        #reference_result_dense_with_masked_weights.sum().backward()
    masked_dense_result = masked_dense_layer(input)
    masked_dense_result.sum().backward(retain_graph=True)
    masked_dense_input_grad = input.grad.detach().clone()
    masked_dense_weight_grad = masked_dense_layer.dense.grad.detach().clone()
    #masked_dense_input_grad = input.grad.detach().clone()
    #masked_dense_weight_grad = torch_linear_layer_with_masked_weights.weight.detach().clone()
    
    # Check reference and masked gradients are close
    reference_close_to_masked_input_grad = torch.allclose(reference_input_grad, masked_dense_input_grad, atol=0.05)
    #print("Are input gradients close between reference and masked dense?:", reference_close_to_masked_input_grad)
    reference_close_to_masked_weight_grad = torch.allclose(reference_weight_grad, masked_dense_weight_grad, atol=0.05)
    print("Are gradients close between reference and masked dense? Input:", reference_close_to_masked_input_grad, 
          " weights:", reference_close_to_masked_weight_grad)
    if not reference_close_to_masked_input_grad:
        print("Reference input grad:\n", reference_input_grad[:8, :8])
        print("Masked input grad:\n", masked_dense_input_grad[:8, :8])
    if not reference_close_to_masked_weight_grad:
        print("Reference weight grad:\n", reference_weight_grad[:8, :8])
        print("Masked weight grad:\n", masked_dense_weight_grad[:8, :8])
    
    
    # Check gradients using sparse operation.
    print("Testing sparse backwards...")
    # Clear input gradient to not accumulate the gradients of each module.
    input.grad = None
    # Calculate loss and run backwards on sparse result.
    """
    sparse_result.sum().backward(retain_graph=True)
    sparse_input_grad = input.grad.detach().clone()
    sparse_weight_grad = torch_linear_layer_with_masked_weights.weight.grad.detach().clone()
    
    # Check masked and sparse gradients are close
    masked_close_to_sparse_input_grad = torch.allclose(masked_dense_input_grad, sparse_input_grad)
    #print("Are input gradients close between reference and masked dense?:", reference_close_to_masked_input_grad)
    masked_close_to_sparse_weight_grad = torch.allclose(masked_dense_weight_grad, sparse_weight_grad)
    print("Are gradients close between masked and sparse? Input:", masked_close_to_sparse_input_grad, 
          " weights:", masked_close_to_sparse_weight_grad)
    
    """
    
    
    
    # Measure performance
    num_repeats = 100
    number=1
    
    aggregated_reference_forward_time = 0
    aggregated_masked_forward_time = 0
    aggregated_sparse_forward_time = 0
    aggregated_reference_backward_time = 0
    aggregated_masked_backward_time = 0
    aggregated_sparse_backward_time = 0
    
    
    reference_result_dense_with_masked_weights = torch_linear_layer_with_masked_weights(input)
    masked_dense_result = masked_dense_layer(input)
    sparse_result = sparse_layer(input)
    
    # Dense using masked weights times
    timeit.repeat('reference_output = torch_linear_layer_with_masked_weights(input)', repeat=10, number=number, globals=locals())
    dense_times = timeit.repeat('reference_output = torch_linear_layer_with_masked_weights(input)', repeat=num_repeats, number=number, globals=locals())
    report_time('dense-forward', dense_times, number, v, n, m)
    
    # Custom layer using masked weights
    timeit.repeat('masked_output = masked_dense_layer(input)', repeat=10, number=number, globals=locals())
    dense_times = timeit.repeat('masked_output = masked_dense_layer(input)', repeat=num_repeats, number=number, globals=locals())
    report_time('masked-forward', dense_times, number, v, n, m)
    
    # Sparse layer.
    timeit.repeat('sparse_output = sparse_layer(input)', repeat=10, number=number, globals=locals())
    dense_times = timeit.repeat('sparse_output = sparse_layer(input)', repeat=num_repeats, number=number, globals=locals())
    report_time('sparse-forward', dense_times, number, v, n, m)

    # Measure backwards time
    timeit.repeat('reference_result_dense_with_masked_weights.sum().backward(retain_graph=True)', repeat=10, number=number, globals=locals())
    dense_times = timeit.repeat('reference_result_dense_with_masked_weights.sum().backward(retain_graph=True)', repeat=num_repeats, number=number, globals=locals())
    report_time('dense-backward', dense_times, number, v, n, m)
    
    timeit.repeat('masked_dense_result.sum().backward(retain_graph=True)', repeat=10, number=number, globals=locals())
    dense_times = timeit.repeat('masked_dense_result.sum().backward(retain_graph=True)', repeat=num_repeats, number=number, globals=locals())
    report_time('masked-backward', dense_times, number, v, n, m)
    
    timeit.repeat('sparse_result.sum().backward(retain_graph=True)', repeat=10, number=number, globals=locals())
    dense_times = timeit.repeat('sparse_result.sum().backward(retain_graph=True)', repeat=num_repeats, number=number, globals=locals())
    report_time('sparse-backward', dense_times, number, v, n, m)
    

if __name__ == "__main__":
    main()
