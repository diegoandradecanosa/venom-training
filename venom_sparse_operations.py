

"""
Declarations and code for exeuting sparse kernels from Venom using V:N:M masks.

"""

import torch
import sten

# Import Venom kernels.
import spatha
from grouped_nmv_tensor import VenomTensor, venom_mask_sparsify




def sparse_dense_mul_dispatch(sparse_values, sparse_indices, sparse_metadata, dense, nrows_sp, ncols_sp, ncols_d, m, n, v, nnz, bias):

    dense_ = dense.contiguous()

    output = spatha.spmm_128x64x32_32x64x32_16x8x32_2(
                          sparse_metadata.to(device='cuda:0'),  # metadata
                          sparse_indices.to(device='cuda:0'),   # indices
                          sparse_values.to(dtype=torch.half).to(device='cuda:0'),    # values
                          dense_.to(device='cuda:0'),           # rhs_matrix
                          bias.to(device='cuda:0'),             # bias
                          nrows_sp,         # A_num_rows
                          ncols_sp,         # A_num_cols
                          ncols_d,          # B_num_cols
                          v,                # V
                          n,                # N
                          m,                # M
                          nnz,              # nnz
                          0,                # seed
                          32,               # mbrow
                          4                 # brow
                          )

    return output


class VenomSparsifier:
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
        # random pruning (cuSparseLt-like approach) -> mask, columns
        nrows, ncols = tensor.shape
        columns = torch.zeros(nrows//self.v, ncols//self.m*4, dtype=torch.int32)
        columns = columns.reshape((-1,4)) + torch.tensor([0,1,2,3], dtype=torch.int32)
        columns = columns.reshape((nrows//self.v, ncols//self.m*4))

        mask = VenomSparsifier.get_random_mask(tensor, self.m, self.v)

        sparse_mtx = sten.SparseTensorWrapper.wrapped_from_dense(
            VenomTensor(self.n, self.m, self.v, tensor, mask, columns, tensor.device),
            tensor,
            grad_fmt,
        )

        return sparse_mtx

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
        
        """ if self.bias is not None:
            output += self.bias.unsqueeze(0).expand_as(output) """
        
        output = output.reshape((*input.shape[0:-1], -1))[..., :DM]
        #output = output.reshape((32,512,1024))
        
        return output


def linear_to_venom_spmm(mod, v:int, n:int, m:int):
    """
    replaces linear modules for spmm sparse kernels from venom.
    """
    if isinstance(mod, torch.nn.Linear):
        return VenomSpmm(mod, v, n, m)

    for name, submod in mod.named_children():
        if isinstance(submod, VenomSpmm):
            continue
        if isinstance(submod, torch.nn.Linear):
            setattr(mod, name, VenomSpmm(submod, v, n, m))
        elif submod is not mod:
            linear_to_venom_spmm(submod, v, n, m)

    return mod