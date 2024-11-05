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

from transformers import BertForSequenceClassification, AutoTokenizer

def transformer_encoder_layer_prototype(num_repeats, number):
    model2 = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2).to(device='cuda:0').half()

    input = torch.randint(low=0, high=100, size=(bs, 512)).to(device='cuda:0')
    labels = torch.randint(low=0, high=2, size=(bs,)).to(device='cuda:0')  # Random labels for the batch

    torch.set_grad_enabled(True)

    model2.train()
    #model2.train()
    #print("Starting execution")
    output = model2(input, labels=labels)
    loss = output.loss
    loss.backward()
    #print("Warm-up executed")

    if args.profile:
        #with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
            with record_function("model_inference"):
                output = model2(input)
                output.sum().backward()
        prof.export_stacks("/tmp/profiler_stacks_dense.txt", "self_cuda_time_total")
        prof.export_chrome_trace("trace_dense.json")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
        exit()

    #warmup
    timeit.repeat('import torch; output = model2(input, labels=labels); loss=output.loss; loss.backward()', repeat=10, number=number, globals=locals())
    dense_times = timeit.repeat('import torch; output = model2(input, labels=labels); loss=output.loss; loss.backward()', repeat=num_repeats, number=number, globals=locals())
    report_time('dense', dense_times, number)

if __name__ == "__main__":
    torch. set_grad_enabled(False)
    transformer_encoder_layer_prototype(num_repeats=30, number=1)