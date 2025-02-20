import os
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import BertForSequenceClassification, BertTokenizer, get_scheduler
from datasets import load_from_disk
from torch.utils.data import DataLoader, DistributedSampler
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

import torch.profiler
import numpy as np
from sparseml.pytorch.optim import ScheduledModifierManager



def setup(rank, world_size):
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def train(model, train_loader, optimizer, scaler, manager, epoch, rank, lr_scheduler, profiler, steps_per_epoch):
    Epoch=epoch
    for batch_idx, batch in enumerate(train_loader):

        epoch=Epoch+batch_idx/len(train_loader)

        manager.optimizer_pre_step(model, optimizer, epoch, steps_per_epoch=steps_per_epoch)
        manager.update(model, optimizer, epoch, steps_per_epoch=steps_per_epoch)

        input_ids, attention_mask, labels = (
            torch.tensor(np.array(batch['input_ids'])).to(rank),
            torch.tensor(np.array(batch['attention_mask'])).to(rank),
            batch['label'].to(rank))


        if input_ids.size(0) != labels.size(0):
            min_size = min(input_ids.size(0), labels.size(0))
            input_ids = input_ids[:min_size]
            attention_mask = attention_mask[:min_size]
            labels = labels[:min_size]

        with torch.cuda.amp.autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss.mean()

        scaler.scale(loss).backward()
        scaler.step(optimizer)

        scaler.update()
        optimizer.zero_grad()

        manager.optimizer_post_step(model, optimizer, epoch, steps_per_epoch=steps_per_epoch)
        lr_scheduler.step()

        profiler.step()

        if batch_idx % 10 == 0 and rank == 0:
             print(f'Train Epoch: {epoch} [{batch_idx * len(input_ids)}/{len(train_loader.dataset)} '
                  f'({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def main():
    rank = int(os.environ['SLURM_PROCID'])
    world_size = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ['SLURM_LOCALID'])
    traza =str(os.environ['NOMBRE_TRAZA'])

    torch.cuda.set_device(local_rank)
    setup(rank, world_size)

    model = BertForSequenceClassification.from_pretrained('<model_name>', num_labels=2)
    model.to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)


    tokenizer = BertTokenizer.from_pretrained('<model_name>')
    dataset = load_from_disk('<dataset_name>')
    encoded_dataset = dataset.map(lambda examples: tokenizer(examples['sentence'], padding='max_length', truncation=True, max_length=128), batched=True)

    train_sampler = DistributedSampler(encoded_dataset['train'], num_replicas=world_size, rank=rank)
    train_loader = DataLoader(encoded_dataset['train'], sampler=train_sampler, batch_size=int(os.environ["BATCH_SIZE"]), num_workers=int(os.environ["SLURM_CPUS_PER_TASK"]), pin_memory=True, drop_last=True)
    optimizer = AdamW(ddp_model.parameters(), lr=1e-6)
    
    
    num_epochs = 50
    num_training_steps = num_epochs * len(train_loader)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)


    manager = ScheduledModifierManager.from_yaml("<recipe_name>.yaml")
    manager.initialize(model, epoch=0.0, optimizer=optimizer)
    scaler = torch.cuda.amp.GradScaler()
    
    model.train()
    
    #wait, warmup, and active frequency chosed in the scheduler
    #must be selected carefully in order to avoid overloads or errors
    with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=20, warmup=2, active=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f'../traces/{traza}'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True) as profiler:

        for epoch in range(1, num_epochs + 1):
            train(ddp_model, train_loader, optimizer, scaler, manager, epoch, local_rank, lr_scheduler, profiler, steps_per_epoch=len(train_loader))

    cleanup()

if __name__ == '__main__':
    main()
