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

import os
import statistics
import torch
import torch.distributed as dist
from torch.profiler import profile, record_function, ProfilerActivity
from torch.utils.data import DataLoader, DistributedSampler

import time
import timeit
import argparse
import numpy as np

import venom_sparse_operations as vsp



parser = argparse.ArgumentParser()

parser.add_argument('-m', type=int, default=4)
parser.add_argument('-n', type=int, default=2)
parser.add_argument('-v', type=int, default=64)
parser.add_argument('-bs', type=int, default=16)

parser.add_argument('--profile', action='store_true', default=False)

args = parser.parse_args()

m          = args.m
n          = args.n
v          = args.v
bs         = args.bs


def train_epoch(model, train_loader, epoch, rank, profiler):
    
    #import sys
    #if 'venom_sparse_operations' not in sys.modules:
    #import venom_sparse_operations as internal_vsp
    #internal_vsp.reset_time_accumulators()
    venom_times = vsp.transposition_reshape_timer()
    venom_times.reset_times()
    
    for batch_idx, batch in enumerate(train_loader):
            
        input_ids, attention_mask, labels = (
            torch.tensor(np.array(batch['input_ids'])).to(rank),
            torch.tensor(np.array(batch['attention_mask'])).to(rank),
            batch['label'].to(rank)
        )

        # Check tensor shapes
        if input_ids.size(0) != labels.size(0):
             #continue
             min_size = min(input_ids.size(0), labels.size(0))
             input_ids = input_ids[:min_size]
             attention_mask = attention_mask[:min_size]
             labels = labels[:min_size]


            # Utilizar precisión mixta con autocast
            
            #outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            #loss = outputs.loss.mean()


        with torch.cuda.amp.autocast():
            # Pasar los datos por el modelo
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.mean()  # Para DataParallel

        # Backpropagation con GradScaler
        #scaler.scale(loss).backward()
        loss.backward()
        # Actualizar los parámetros del modelo usando GradScaler
        #scaler.step(optimizer)

        # Actualizar el scheduleri
        #scaler.update()
        #optimizer.zero_grad()

        #lr_scheduler.step()
        #manager.optimizer_post_step(model, optimizer, epoch, steps_per_epoch=steps_per_epoch)

        if profiler is not None:
            profiler.step()
        if batch_idx % 10 == 0 and rank == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(input_ids)}/{len(train_loader)*len(input_ids)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')
            
    print("Accumulated SPMM transpose time:", venom_times.spmm_transposition_time/1000000, "milliseconds.")# Accumulator gives nanoseconds, convert to milli
    print("Accumulated SDDMM transpose time:", venom_times.sddmm_transposition_time/1000000, "milliseconds.")# Accumulator gives nanoseconds, convert to milli


def report_time(name, v, n, m, bs, data, number, time_offset_milliseconds):
    for d in data:
        time_ms = (d / number * 1000)+time_offset_milliseconds
        #print(f'n {n} m {m} format {name} time_ms {time_ms:.3f}')
    ds = [(d / number * 1000)+time_offset_milliseconds for d in data]
    mean = statistics.mean(ds)
    median = statistics.median(ds)
    std = statistics.stdev(ds)

    print(
        str(name)+","+str(v)+","+str(n)+","+str(m)+","+str(bs)+","+str(mean)+","+str(median)+","+str(std)+","+str(len(ds))
    )

def setup(rank, world_size):
    #backend='nccl',
    dist.init_process_group( rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


from transformers import BertForSequenceClassification, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_from_disk, DatasetDict


def load_distributed_model_and_train(num_repeats, number, rank, world_size):
    
    venom_times = vsp.transposition_reshape_timer()
    
    model_name = 'bert-large-uncased'
    model_downloaded_folder='./bert-large-uncased-binary'
    model = BertForSequenceClassification.from_pretrained(model_downloaded_folder, num_labels=2)
    #model2 = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2).to(device='cuda:0').half()
    model.to(local_rank)
    

    
    # Load dataset
    #dataset = load_dataset("squad")
    dataset = load_from_disk('../../../../pancho/Decaulion_env/sst2_dataset')
    
    # truncate dataset to 10 batches to perform short tests.
    #max_samples =bs*10
    #truncated_dataset = DatasetDict({
    #    split: split_dataset.take(max_samples)
    #    for split, split_dataset in dataset.items()
    #})
    #dataset = truncated_dataset
    
    tokenizer = AutoTokenizer.from_pretrained(model_downloaded_folder)
    encoded_dataset = dataset.map(lambda examples: tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128), batched=True)
    train_sampler = DistributedSampler(encoded_dataset['train'], num_replicas=world_size, rank=rank)
    train_loader = DataLoader(encoded_dataset['train'], sampler=train_sampler, batch_size=int(os.environ["BATCH_SIZE"]), num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True, drop_last=True)

    #input = torch.randint(low=0, high=100, size=(bs, 512))#, dtype=torch.half)
    
    distributed_model = DDP(model, device_ids=[rank], output_device=local_rank, find_unused_parameters=True)

    # Change linear modules to venom, using sinthetic 64:2:8 masks.
    print("Model loaded and changing to venom kernels with "+str(v)+":"+str(n)+":"+str(m)+" mask.")
    distributed_model = vsp.linear_to_venom(distributed_model, v, n, m)

    #model = model.cuda().half()
    # Move model to appropiate device
    distributed_model = distributed_model.to(rank)
    #input = input.cuda()
    #labels = torch.randint(low=0, high=2, size=(bs,)).to(device='cuda:0')  # Random labels for the batch

    torch.set_grad_enabled(True)

    #Enable training
    distributed_model.train()
    #model2.train()
    #print("Starting execution")
    #output = distributed_model(input, labels=labels)
    #loss = output.loss
    #loss.backward()
    #print("Warm-up executed")
    
    if args.profile:
        # profiling enabled, run once with profiler enabled.
        with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=17, warmup=2, active=1),
            record_shapes=True,
            profile_memory=True,
            with_stack=True) as profiler:

            train_epoch(distributed_model, train_loader, 1, rank, profiler)

        
    else:
        
        start = time.time_ns()
        train_epoch(distributed_model, train_loader, 1, rank, None)
        end = time.time_ns()
        total_time_nanoseconds = end-start
        print("name,v,n,m,bs,total_milliseconds,milliseconds_excluding_transpose")
        print('venom'+","+str(v)+","+str(n)+","+str(m)+","+str(bs)+","+str(total_time_nanoseconds/1000000)+","+str((total_time_nanoseconds-venom_times.sddmm_transposition_time)/1000000))
        
        
        # No profiling, repeat multiple times to measure statistically reliable times.
        #timeit.repeat(setup='from __main__ import train_epoch', stmt='train_epoch(distributed_model, train_loader, 1, rank, None)', repeat=10, number=number, globals=locals())
        #sparse_times = timeit.repeat(setup='from __main__ import train_epoch', stmt='train_epoch(distributed_model, train_loader, 1, rank, None)', repeat=num_repeats, number=number, globals=locals())
        #report_time('venom', v, n, m, bs, sparse_times, number)

    cleanup()
        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        #with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        #    with record_function("model_inference"):
        #        output = distributed_model(input, labels=labels)
        #        loss = output.loss
        #        loss.backward()
        #prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
        #prof.export_chrome_trace("trace_sparse.json")
        #print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    
    

if __name__ == "__main__":
    
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    
    os.environ["TOKENIZERS_PARALLELISM"] = "false" # Disable tokenizer parallelism to avoid conflicting with DDP.
    
    torch.cuda.set_device(local_rank)
    setup(rank, world_size)
    
    torch.set_grad_enabled(False)
    load_distributed_model_and_train(num_repeats=1, number=1, rank=rank, world_size=world_size)
