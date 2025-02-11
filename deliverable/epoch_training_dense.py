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

import timeit
import argparse




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

def setup(rank, world_size):
    #backend='nccl',
    dist.init_process_group( rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()


from transformers import BertForSequenceClassification, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset


def transformer_encoder_layer_prototype(num_repeats, number, rank, world_size):
    
    model_name = 'bert-large-uncased'
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=2)
    #model2 = BertForSequenceClassification.from_pretrained('bert-large-uncased', num_labels=2).to(device='cuda:0').half()
    distributed_model = DDP(model, device_ids=[rank])

    
    # Load dataset
    dataset = load_dataset("squad")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    encoded_dataset = dataset.map(lambda examples: tokenizer(examples['text'], padding='max_length', truncation=True, max_length=128), batched=True)
    train_sampler = DistributedSampler(encoded_dataset['train'], num_replicas=world_size, rank=rank)
    train_loader = DataLoader(encoded_dataset['train'], sampler=train_sampler, batch_size=int(os.environ["BATCH_SIZE"]), num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True, drop_last=True)
    


    input = torch.randint(low=0, high=100, size=(bs, 512))#, dtype=torch.half)

    #print(weights_to_sparsify)
    #model = model.cuda().half()
    distributed_model = distributed_model.cuda().half()
    input = input.cuda()
    labels = torch.randint(low=0, high=2, size=(bs,)).to(device='cuda:0')  # Random labels for the batch

    torch.set_grad_enabled(True)

    #Enable training
    distributed_model.train()
    #model2.train()
    #print("Starting execution")
    output = distributed_model(input, labels=labels)
    loss = output.loss
    loss.backward()
    #print("Warm-up executed")

    if args.profile:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
        #with profile(activities=[ProfilerActivity.CUDA], record_shapes=True, with_stack=True) as prof:
            with record_function("model_inference"):
                output = distributed_model(input, labels=labels)
                loss = output.loss
                loss.backward()
        prof.export_stacks("/tmp/profiler_stacks.txt", "self_cuda_time_total")
        prof.export_chrome_trace("trace_sparse.json")
        print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))

    #warmup
    timeit.repeat('import torch; output = distributed_model(input, labels=labels); loss=output.loss; loss.backward()', repeat=10, number=number, globals=locals())
    sparse_times = timeit.repeat('import torch; output = distributed_model(input, labels=labels); loss=output.loss; loss.backward()', repeat=num_repeats, number=number, globals=locals())
    report_time('n:m', sparse_times, number)
    
    cleanup()

if __name__ == "__main__":
    
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    
    torch.cuda.set_device(local_rank)
    setup(rank, world_size)
    
    torch. set_grad_enabled(False)
    transformer_encoder_layer_prototype(num_repeats=30, number=1)