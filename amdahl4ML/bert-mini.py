import statistics
import subprocess
import ctypes
import os

#import sten

import math
import numpy as np
import torch
from torch.profiler import profile, record_function, ProfilerActivity

from pathlib import Path

import timeit
import argparse
import sys
import time
import transformers


working_path = '/media/rtx3090/Disco2TB/inno4scale_shared/'
# Load the shared library
lib = ctypes.cdll.LoadLibrary(working_path+"controlled_wait_kernel.so")

# Define the function signature
lib.wait_kernel.argtypes = [
    ctypes.c_void_p,  # input pointer
    ctypes.c_void_p,  # output pointer
    ctypes.c_int,     # size of the array
    ctypes.c_ulonglong      # wait time microseconds
]





parser = argparse.ArgumentParser()

parser.add_argument('--profile', action='store_true', default=False)

parser.add_argument('--layer', action='store_true', default=False)

args = parser.parse_args()


class WaitKernelFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, wait_time_microseconds, weight):
        #input = input.contiguous()
        input_shape = input.shape()
        weight_shape = weight.shape()
        output_shape = (input_shape[0], weight_shape[1])
        #prints


        output = torch.zeros(output_shape)
        size = input.numel()


        lib.wait_kernel(
            input.data_ptr(),
            output.data_ptr(),
            size,
            wait_time_microseconds
        )

        return output


class WaitKernelModule(torch.nn.Module):
    def __init__(self, wait_time_microseconds, original):
        super(WaitKernelModule, self).__init__()
        self.weight = original.weight
        self.wait_time_microseconds = wait_time_microseconds

    def forward(self, input):
        return WaitKernelFunction.apply(input, self.wait_time_microseconds, self.weight)



def report_time(name, data, number):

    for d in data:
        time_ms = d / number * 1000
        #print(f'n {n} m {m} format {name} time_ms {time_ms:.3f}')

    ds = [(d / number * 1000) for d in data]

    mean = statistics.mean(ds)
    median = statistics.median(ds)
    std = statistics.stdev(ds)
 
    #print(
    #    str(mean)+","+str(median)+","+str(std)+","+str(len(ds))
    #)
    print('\tMean:', mean, '\n\tmedian:', median, '\n\tstandard deviation:', std, '\n\tsamples:', len(ds))




def replace_operation(module, delay: int):
    if isinstance(module, torch.nn.Linear) or isinstance(module, WaitKernelModule):
        return WaitKernelModule(delay, module)

    for name, m in module.named_children():
        #if isinstance(m, Controlled_wait):
        #    continue
        if isinstance(m, torch.nn.Linear) or isinstance(module, WaitKernelModule):
            setattr(module, name, WaitKernelModule(delay))
        elif m is not module:
            replace_operation(m, delay)

    return module



def transformer_encoder_layer_prototype(num_repeats, number):
    model = transformers.AutoModelForCausalLM.from_pretrained(
        "prajjwal1/bert-mini",
        torch_dtype='auto',
        low_cpu_mem_usage=True,
        cache_dir="/media/rtx3090/Disco2TB/inno4scale_shared/models"
    )
    model = model.to(device='cuda:0').half()
    
    submodel = model.bert.encoder
    
    
    layer = model.bert.encoder.layer[0]

    model_input = torch.randint(low=0, high=100, size=(32, 512))
    model_input = model_input.to(device='cuda:0')
    layer_input = torch.rand(size=(1024, 32, 256), dtype=torch.half)
    layer_input = layer_input.to(device='cuda:0')


    print('Pytorch version:', torch.__version__)
    #warmup
    if args.layer:
        print('Running first layer from bert-mini model...')
        timeit.repeat('output = layer(layer_input)', repeat=10, number=number, globals=locals())
        dense_times = timeit.repeat('output = layer(layer_input)', repeat=num_repeats, number=number, globals=locals())
        report_time('dense', dense_times, number)
    else:
        print('Running full bert-mini model...')
        timeit.repeat('output = model(model_input)', repeat=10, number=number, globals=locals())
        dense_times = timeit.repeat('output = model(model_input)', repeat=num_repeats, number=number, globals=locals())
        report_time('dense', dense_times, number)

    if args.profile:
        #with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
        print('Flash_forward_attention enabled?', torch.backends.cuda.flash_sdp_enabled())
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True, experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
            with record_function("model_inference"):
                if args.layer:
                    print('Profiling first layer from bert-mini model...')
                    output = layer(layer_input)
                else:
                    print('Profiling all bert-mini model layers...')
                    #next_input = layer_input
                    #for layer_index in range(len(model.bert.encoder.layer)):
                    #    next_input = model.bert.encoder.layer[layer_index](next_input)
                    output = submodel(layer_input)
                    #output = model(model_input)
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
        prof.export_chrome_trace(working_path+"/Scripts/amdahl4ML/trace_dense_standard.json")

        if not args.layer:
            weights_to_sparsify = [
                module
                for module_name, module in model.named_modules()
                if (
                    isinstance(module, torch.nn.modules.linear.Linear)
                    and "layer" in module_name
                )
            ]
            print('Capas:\n', [(module_name, type(module)) for module_name, module in model.named_modules()])
            # Run model replacing some operations with a long delay
            long_delay_model = replace_operation(submodel, 200)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True, experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
                with record_function("model_inference"):
                    print('\n\nProfiling model replacing operation with long controlled wait...\n')
                    output = long_delay_model(layer_input)
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
            prof.export_chrome_trace(working_path+"/Scripts/amdahl4ML/trace_dense_long_delay.json")

            # Run again with a short delay on the same operations
            short_delay_model = replace_operation(long_delay_model, 100)
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True, experimental_config=torch._C._profiler._ExperimentalConfig(verbose=True)) as prof:
                with record_function("model_inference"):
                    print('\n\nProfiling model replacing operation with short controlled wait...\n')
                    output = short_delay_model(layer_input)
            print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
            prof.export_chrome_trace(working_path+"/Scripts/amdahl4ML/trace_dense_short_delay.json")
                
        #prof.export_stacks(working_path+"/Scripts/amdahl4ML/profiler_stacks_dense_cuda.txt", "self_cuda_time_total")
        #prof.export_stacks(working_path+"/Scripts/amdahl4ML/profiler_stacks_dense_cpu.txt", "self_cpu_time_total")
        

        # Export to ONNX
        if args.layer:
            torch.onnx.export(layer, layer_input, working_path+"/Scripts/amdahl4ML/bert-mini.layer.onnx", verbose=True)
        else:
            torch.onnx.export(model, model_input, working_path+"/Scripts/amdahl4ML/bert-mini.onnx", verbose=True)
        
        exit()
    
    

        
        

if __name__ == "__main__":
    torch.set_grad_enabled(False)
    transformer_encoder_layer_prototype(num_repeats=100, number=1)
