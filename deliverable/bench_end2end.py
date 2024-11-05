#
# Copyright (C) 2024 Roberto Lopez Castro (roberto.lopez.castro@udc.es). All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import statistics
import subprocess
import ctypes

import sten

import math
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from pathlib import Path

import timeit
import argparse
import sys
import time

from venom_tensor import SparseVNMTensor

import spatha
import spatha_sddmm

parser = argparse.ArgumentParser()

parser.add_argument('-m', type=int, default=8)
parser.add_argument('-n', type=int, default=2)
parser.add_argument('-v', type=int, default=128)
parser.add_argument('-bs', type=int, default=16)

parser.add_argument('--profile', action='store_true', default=False)
parser.add_argument('--sparsetime', action='store_true', default=False)

args = parser.parse_args()

m          = args.m
n          = args.n
v          = args.v
bs         = args.bs
sparsetime = args.sparsetime


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

def report_time(name, data, number):
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

def linear_to_spmm(mod, weights_to_sparsify):
    if isinstance(mod, torch.nn.Linear):
        return SrnmSpmm(mod)

    for name, m in mod.named_children():
        if isinstance(m, SrnmSpmm) or name=="classifier":
            continue
        if isinstance(m, torch.nn.Linear):
            setattr(mod, name, SrnmSpmm(m))
        elif m is not mod:
            linear_to_spmm(m, weights_to_sparsify)

    return mod

from transformers import BertForSequenceClassification, AutoTokenizer

def transformer_encoder_layer_prototype(num_repeats, number):
    model = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2)
    #model2 = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2).to(device='cuda:0').half()

    input = torch.randint(low=0, high=100, size=(bs, 512))#, dtype=torch.half)

    weights_to_sparsify = [
        module
        for module_name, module in model.named_modules()
        if (
            isinstance(module, torch.nn.modules.linear.Linear)
            and "encoder.layer" in module_name
        )
    ]
    #print(weights_to_sparsify)
    model = model.to(device='cuda:0').half()
    input = input.to(device='cuda:0')

    sparse_model = linear_to_spmm(model, weights_to_sparsify)

    labels = torch.randint(low=0, high=2, size=(bs,)).to(device='cuda:0')  # Random labels for the batch

    torch.set_grad_enabled(True)

    sparse_model.train()
    #model2.train()
    #print("Starting execution")
    output = sparse_model(input, labels=labels)
    loss = output.loss
    loss.backward()
    #print("Warm-up executed")

    if args.profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        #with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
            with record_function("model_inference"):
                output = sparse_model(input)
                output.sum().backward()
        prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
        prof.export_chrome_trace("trace_sparse.json")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
        #exit()

        #with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        """ with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
            with record_function("model_inference"):
                output = model2(input)
                output.sum().backward()
        prof.export_stacks("/tmp/profiler_stacks_dense.txt", "self_cuda_time_total")
        prof.export_chrome_trace("trace_dense.json")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
        exit() """

    #warmup
    """ timeit.repeat('import torch; output = model2(input, labels=labels); loss=output.loss; loss.backward()', repeat=10, number=number, globals=locals())
    dense_times = timeit.repeat('import torch; output = model2(input, labels=labels); loss=output.loss; loss.backward()', repeat=num_repeats, number=number, globals=locals())
    report_time('dense', dense_times, number) """

    #warmup
    timeit.repeat('import torch; output = sparse_model(input, labels=labels); loss=output.loss; loss.backward()', repeat=10, number=number, globals=locals())
    sparse_times = timeit.repeat('import torch; output = sparse_model(input, labels=labels); loss=output.loss; loss.backward()', repeat=num_repeats, number=number, globals=locals())
    report_time('n:m', sparse_times, number)

if __name__ == "__main__":
    torch. set_grad_enabled(False)
    transformer_encoder_layer_prototype(num_repeats=30, number=1)