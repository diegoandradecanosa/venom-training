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

from venom_tensor_diego import SparseVNMTensor

try:
    cache = functools.cache
except AttributeError:
    cache = functools.lru_cache(maxsize=None)



# Set VENOM parameters. Done here for testing purposes, should be set in main and passed through methods.
v=64
n=2
m=8









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


def sparse_dense_mul_dispatch(sparse_values, sparse_indices, sparse_metadata, dense, nrows_sp, ncols_sp, ncols_d, m, n, v, nnz, bias):

    dense_ = dense.contiguous()

    output = spatha.spmm(
                          sparse_metadata,  # metadata
                          sparse_indices,   # indices
                          sparse_values,    # values
                          dense_,           # rhs_matrix
                          bias,             # bias
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
    def forward(ctx, 
                input, 
                sparse_weights_values, 
                bias, 
                sparse_weights_columns, 
                sparse_weights_metadata, 
                sparse_weights_dense,
                venom_v, venom_n, venom_m, nnz):# using underscores to avoid conflict with global variables.
        #ctx.save_for_backward(input, bias) # sparse_weights cannot be saved as it is not a tensor
        #ctx.sparse_weights = sparse_weights
        ctx.save_for_backward(input, bias, sparse_weights_columns, sparse_weights_metadata, sparse_weights_dense)
        
        nrows, ncols = sparse_weights_dense.shape
        ctx.sparse_weights_num_rows = nrows
        ctx.sparse_weights_num_cols = ncols
#        ctx.B_num_cols = B_num_cols
        ctx.v = venom_v
        ctx.n = venom_n
        ctx.m = venom_m
        ctx.nnz = nnz
        
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
        
        
        #output = sparse_dense_mul_dispatch( values,
        #                                    columns,
        #                                    metadata,
        #                                    flattened_input.T,
        #                                    nrows_sp,
        #                                    ncols_sp,
        #                                    ncols_d,
        #                                    m,
        #                                    n,
        #                                    v,
        #                                    0,
        #                                    bias)
        
        spmm_input = torch.flatten(input, start_dim=0, end_dim=-2).T.contiguous()
        spmm_output = spatha.spmm(sparse_weights_metadata,               # metadata
                          sparse_weights_columns,                        # indices
                          sparse_weights_values,                         # values
                          spmm_input,                                    # rhs_matrix
                          bias,                                          # Bias
                          nrows,                          # A_num_rows
                          ncols,                          # A_num_cols
                          spmm_input.shape[1],                           # B_num_cols
                          venom_v,                              # vec_length
                          venom_n,                              # n
                          venom_m,                              # m
                          nnz,                            # nnz
                          0,                                             # seed
                          32,                                            # mbrow
                          4                                              # brow
                          )
        spmm_output = spmm_output.reshape((*input.shape[0:-1], -1))[..., :nrows]
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
        #input, bias = ctx.saved_tensors
        input, bias, weights_columns, weights_metadata, weights_dense  = ctx.saved_tensors
        #ctx.save_for_backward(input, bias, sparse_weights.columns, 
        # sparse_weights.metadata, sparse_weights.dense)
        #sparse_weights = ctx.sparse_weights
        
        global global_grad_output  
        global_grad_output = grad_output 

        #print(input.device, sparse_weights.device, grad_output.device)

        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weights_dense
        
        if ctx.needs_input_grad[1]: 
            
            #grad_weight = grad_output.t() @ input.to("cuda:0")
            #grad_weight = grad_output.T.contiguous() @ input.to("cuda:0")
            #print("Launching sddmm kernel")
            
            sddmm_grad_output = torch.flatten(grad_output, start_dim=0, end_dim=-2)
            sddmm_grad_output = sddmm_grad_output.T.contiguous()
            sddmm_input = torch.flatten(input, start_dim=0, end_dim=-2)
            sddmm_input = sddmm_input.T.contiguous()# Transpose both since sddmm does A@B.T, but we want A.T@B.
            #print("Types:",  type(sddmm_grad_output),    
            #        type(sddmm_input),                 
            #        type(weights_metadata),
            #        type(weights_columns), 
            #        type(ctx.sparse_weights_num_rows),
            #        type(ctx.sparse_weights_num_cols),
            #        type(ctx.n),                     
            #        type(ctx.m),                     
            #        type(ctx.nnz),     
            #        type(0),                     
            #        type(32),                    
            #        type(4)                      
            #        )
            #print("Values:",  sddmm_grad_output.shape, sddmm_grad_output[:8, :8],    
            #        sddmm_input.shape, sddmm_input[:8, :8],                 
            #        weights_metadata[:8],
            #        weights_columns[:8], 
            #        ctx.sparse_weights_num_rows,
            #        ctx.sparse_weights_num_cols,
            #        ctx.n,                     
            #        ctx.m,                     
            #        ctx.nnz,     
            #        0,                     
            #        32,                    
            #        4, sep="\n"                      
            #        )
            #
            #print("sddmm_grad_output.shape:", sddmm_grad_output.shape)
            #print("sddmm_input.shape:", sddmm_input.shape)
            
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
            #print("SDDMM kernel completed")
            grad_weight = compressed_grad_weights.flatten()
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        #print(ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2])
        """ print(grad_output)
        print(input)
        print(grad_weight) """
        #print(grad_input[:4, :4])
        #print(grad_weight[:4, :4])
    
        # Needs one return value for each forward argument.
        #      input,      sparse_weights_values, bias,      sparse_weights_columns, sparse_weights_metadata, sparse_weights_dense, venom_v, venom_n, venom_m, nnz
        return grad_input, grad_weight,           grad_bias, None,                   None,                    None,                 None,    None,    None,    None
    


# Torch.nn.Module subclass that sparsifies an existing torch.nn.lineal layer and uses dense algebra with a masked weights tensor.
class VenomSparseLinear(torch.nn.Module):
    def __init__(self, original: torch.nn.Linear|VenomMaskedLinear, v, n, m):
        super(VenomSparseLinear, self).__init__()        

        if isinstance(original, torch.nn.Linear):
            self.compressed_weights = NMVectorSparsifier(n, m, v)(original.weight).wrapped_tensor

            self.values = torch.nn.Parameter(self.compressed_weights.values.to(device="cuda:0").half())
            self.values.requires_grad = True
            self.columns = self.compressed_weights.columns.to(device="cuda:0")
            self.metadata = self.compressed_weights.metadata.to(device="cuda:0")
        elif isinstance(original, VenomMaskedLinear):
            self.compressed_weights = original.w
            self.values = original.values.to(device="cuda:0")
            self.columns = original.columns.to(device="cuda:0")
            self.metadata = original.metadata.to(device="cuda:0")

        self.bias = original.bias

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


############################### Clases da integración de roberto ##########################

class VenomLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, values, bias, columns, metadata, dense):
        ctx.save_for_backward(input, values, bias, columns, metadata, dense)

        flattened_input = torch.flatten(input, start_dim=0, end_dim=-2)

        ncols_d  = flattened_input.T.shape[1]
        DM, _    = flattened_input.shape

        nrows_sp, ncols_sp = dense.shape

        output = sparse_dense_mul_dispatch( values,
                                            columns,
                                            metadata,
                                            flattened_input.T,
                                            nrows_sp,
                                            ncols_sp,
                                            ncols_d,
                                            m,
                                            n,
                                            v,
                                            0,
                                            bias)

        output = output.reshape((*input.shape[0:-1], -1))[..., :nrows_sp]

        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, values, bias, columns, metadata, dense = ctx.saved_tensors

        nrows_sp, ncols_sp = dense.shape

        grad_input = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ dense

        if ctx.needs_input_grad[1]:
            flattened_grad_output = torch.flatten(grad_output, start_dim=0, end_dim=-2)
            grad_output_ = flattened_grad_output.T.contiguous()
            flattened_input = torch.flatten(input, start_dim=0, end_dim=-2)
            input_ = flattened_input.T.contiguous()

            #print(grad_output_.shape, input_.shape)

            grad_weight = spatha_sddmm.sddmm(
                                            grad_output_,
                                            input_,
                                            metadata,
                                            columns,
                                            nrows_sp,
                                            ncols_sp,
                                            input_.shape[1],
                                            n,
                                            m,
                                            0,
                                            0,
                                            32,
                                            4)

        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)

        return grad_input, grad_weight.flatten(), grad_bias, None, None, None


class SrnmSpmm(torch.nn.Module):
    def __init__(self, original: torch.nn.Linear):
        super(SrnmSpmm, self).__init__()

        w = NMVectorSparsifier(n, m, v)(original.weight).wrapped_tensor

        self.values = torch.nn.Parameter(w.values.to(device="cuda:0").half())
        self.columns = w.columns.to(device="cuda:0")
        self.metadata = w.metadata.to(device="cuda:0")

        self.bias = original.bias

        self.dense = torch.nn.Parameter(w.to_dense())
        #self.mask = self.w.mask

    def forward(self, input):
        return VenomLinearFunction.apply(
                                        input,
                                        self.values,
                                        self.bias,
                                        self.columns,
                                        self.metadata,
                                        self.dense)

############################### Clases da integración de roberto ##########################


def main():

    torch.set_printoptions(precision=2)
    global_grad_output = None

    #input_shape = [16384, 4096]
    #layer_shape = [4096, 8192]
    input_shape = [256, 64]
    layer_shape = [64, 128]

    # Define input and standard dense computation of compatible size.
    input = torch.randn(input_shape[0], input_shape[1], requires_grad=True, dtype=torch.half, device="cuda:0")
    dense_linear_layer = torch.nn.Linear(layer_shape[0], layer_shape[1], bias=True, dtype=torch.half, device="cuda:0")
    



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
    #sparse_layer = SrnmSpmm(dense_linear_layer)


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
    #sparse_layer.clear_grads
    
    print("Testing reference backwards...")
    # Calculate loss and run backwards on reference result.
    reference_result_dense_with_masked_weights.sum().backward(retain_graph=True)
    
    reference_input_grad = input.grad.detach().clone()
    reference_weight_grad = torch_linear_layer_with_masked_weights.weight.grad.detach().clone()
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
    #print("Testing sparse backwards...")
    # Clear input gradient to not accumulate the gradients of each module.
    input.grad = None
    # Calculate loss and run backwards on sparse result.
    
    sparse_result.sum().backward(retain_graph=True)
    sparse_input_grad = input.grad.detach().clone()
    #print("backwards completed")
    #sparse_weight_grad = torch_linear_layer_with_masked_weights.weight.grad.detach().clone()
    sparse_weight_grad = sparse_layer.values.grad.detach().clone()
    #print("weight grad extracted.")
    # Build a new VenomTensor to densify compressed gradients    n_, m_, v_, dense_, mask_, columns_, device_):
    #densified_weights_grad = SparseVNMTensor(n, m, v, 
    #                                         sparse_layer.compressed_weights.device,
    #                                         sparse_layer.dense.detach().clone(), 
    #                                         sparse_layer.mask.detach().clone(), 
    #                                         sparse_layer.columns.detach().clone(), 
    #                                         sparse_layer.compressed_weights.device)
    venom_compressed_weights_grad = SparseVNMTensor(n, m, v, 
                                             sparse_layer.compressed_weights.device,
                                             mask = sparse_layer.mask.detach().clone(),
                                             values = sparse_weight_grad, 
                                             columns = sparse_layer.columns.detach().clone(), 
                                             metadata = sparse_layer.metadata.detach().clone())
    #print("New sparse tensor created")
    #venom_compressed_weights_grad.values = sparse_weight_grad
    #print("Densifying")
    densified_weights_grad = venom_compressed_weights_grad.to_dense()
    #venom_compressed_weights_grad.dense = venom_compressed_weights_grad.to_dense()
    
    # Check masked and sparse gradients are close
    masked_close_to_sparse_input_grad = torch.allclose(masked_dense_input_grad, sparse_input_grad)
    #print("Are input gradients close between reference and masked dense?:", reference_close_to_masked_input_grad)
    masked_close_to_sparse_weight_grad = torch.allclose(masked_dense_weight_grad, densified_weights_grad)
    print("Are gradients close between masked and sparse? Input:", masked_close_to_sparse_input_grad, 
          " weights:", masked_close_to_sparse_weight_grad)
    if not masked_close_to_sparse_input_grad:
        print("Reference input grad:\n", masked_dense_input_grad[:8, :8])
        print("Masked input grad:\n", sparse_input_grad[:8, :8])
    if not masked_close_to_sparse_weight_grad:
        print("Reference weight grad:\n", masked_dense_weight_grad[:8, :8])
        print("Masked weight grad:\n", densified_weights_grad[:8, :8])
    
    
    
    
    # Measure performance
    num_repeats = 10
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
    print("\nTimes for forward and backwards passes: 0,cfg,v,mean,median,std,len")
    print("Dense, torch linear layer with masked weights forward times  ", end="")
    timeit.repeat('reference_output = torch_linear_layer_with_masked_weights(input)', repeat=10, number=number, globals=locals())
    dense_times_forward = timeit.repeat('reference_output = torch_linear_layer_with_masked_weights(input)', repeat=num_repeats, number=number, globals=locals())
    report_time('dense-forward', dense_times_forward, number, v, n, m)
    
    # Custom layer using masked weights
    print("Masked using dense algebra, forward times                    ", end="")
    timeit.repeat('masked_output = masked_dense_layer(input)', repeat=10, number=number, globals=locals())
    masked_times_forward = timeit.repeat('masked_output = masked_dense_layer(input)', repeat=num_repeats, number=number, globals=locals())
    report_time('masked-forward', masked_times_forward, number, v, n, m)
    
    # Sparse layer.
    print("Sparse forward times                                         ", end="")
    timeit.repeat('sparse_output = sparse_layer(input)', repeat=10, number=number, globals=locals())
    sparse_times_forward = timeit.repeat('sparse_output = sparse_layer(input)', repeat=num_repeats, number=number, globals=locals())
    report_time('sparse-forward', sparse_times_forward, number, v, n, m)

    
    # Measure backwards time
    print("Backwards pass")
    print("Dense, torch linear layer with masked weights backward times ", end="")
    timeit.repeat('reference_result_dense_with_masked_weights.sum().backward(retain_graph=True)', repeat=10, number=number, globals=locals())
    dense_times_backward = timeit.repeat('reference_result_dense_with_masked_weights.sum().backward(retain_graph=True)', repeat=num_repeats, number=number, globals=locals())
    report_time('dense-backward', dense_times_backward, number, v, n, m)
    
    print("Masked using dense algebra, backward times                   ", end="")
    timeit.repeat('masked_dense_result.sum().backward(retain_graph=True)', repeat=10, number=number, globals=locals())
    masked_times_backward = timeit.repeat('masked_dense_result.sum().backward(retain_graph=True)', repeat=num_repeats, number=number, globals=locals())
    report_time('masked-backward', masked_times_backward, number, v, n, m)
    
    print("Sparse backward times                                        ", end="")
    timeit.repeat('sparse_result.sum().backward(retain_graph=True)', repeat=10, number=number, globals=locals())
    sparse_times_backward = timeit.repeat('sparse_result.sum().backward(retain_graph=True)', repeat=num_repeats, number=number, globals=locals())
    report_time('sparse-backward', sparse_times_backward, number, v, n, m)
    

if __name__ == "__main__":
    main()
