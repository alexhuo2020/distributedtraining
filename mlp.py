## Train a two layers feedforward network with pipeline parallelism using torch.distributed.pipelining on 2 GPUs
## torchrun --nnodes=1 --nproc_per_node=2 mlp.py

import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

import torch.distributed as dist
from torch.distributed.pipelining import ScheduleGPipe, pipeline, SplitPoint, build_stage, ScheduleInterleaved1F1B, ScheduleFlexibleInterleaved1F1B, ScheduleLoopedBFS
import numpy as np 
torch.manual_seed(1234)

import argparse

def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    # destroy the process group
    dist.destroy_process_group()
    

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        x = F.relu(self.first_layer(x))
        return self.second_layer(x)
            
    
        
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train MLP with Pipeline Parallelism.")

    # Add arguments
    parser.add_argument("--schedule", type=str, required=True, default='GPipe')
    parser.add_argument("--model_dim", type=int, required=True, default=1)
    parser.add_argument("--microbatch_size", type=int, required=True, default=2)
    parser.add_argument("--num_batches", type=int, required=True, default=1)

    # Parse arguments
    args = parser.parse_args()

    schedule_type = args.schedule
    model_dim = args.model_dim
    microbatch_size = args.microbatch_size
    num_batches = args.num_batches

    chunks = 2
    setup()
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    x = torch.randn((microbatch_size,model_dim))
    model = MLP(model_dim,model_dim,model_dim)

    import copy 
    model_copy = copy.deepcopy(model)
    optimizer_copy = torch.optim.Adam(model_copy.parameters(), lr=1)

    pipe = pipeline(module = model,
                    mb_args=(x,),
                    split_spec = {
                        "first_layer": SplitPoint.END,
                    })
    print(pipe)
    optimizer = torch.optim.Adam(model.parameters(), lr=1)
    assert pipe.num_stages == world_size

    stage = pipe.build_stage(rank, torch.device(f"cuda:{local_rank}"))
    
    loss_fn=torch.nn.MSELoss(reduction='sum')#reduction="mean")
    loss_fn = lambda input, target : torch.sum((input - target)**2)

    if schedule_type == 'GPipe':
        schedule = ScheduleGPipe(stage, chunks, loss_fn = loss_fn)
    elif schedule_type == 'Interleaved1F1B':
        schedule = ScheduleInterleaved1F1B(stage, chunks, loss_fn = loss_fn)
    elif schedule_type == 'FlexibleInterleaved1F1B':
        schedule = ScheduleFlexibleInterleaved1F1B(stage, chunks, loss_fn = loss_fn)
    elif schedule_type == 'LoopedBFS':
        schedule = ScheduleLoopedBFS(stage, chunks, loss_fn = loss_fn)

    # X_train = np.array([[2], [3], [4], [5]])
    # y_train = np.array([[2], [4], [6], [8]])
    X_train = torch.randn((num_batches * microbatch_size * world_size, model_dim))
    y_train = torch.randn((num_batches * microbatch_size * world_size, model_dim))
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_train = torch.chunk(X_train, num_batches)
    y_train = torch.chunk(y_train, num_batches)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir'), record_shapes=True) as prof:
        for idx, (X_, y_) in enumerate(zip(X_train, y_train)):
            X_=X_.to(torch.device(f"cuda:{local_rank}"))
            y_=y_.to(torch.device(f"cuda:{local_rank}"))

            if rank == 0:
                schedule.step(X_)
            elif rank == world_size - 1:
                losses = []
                out = schedule.step(target = y_, losses = losses)
            else:
                out = schedule.step()
            
                
            if rank == world_size - 1:
                print(out, losses)
            dist.barrier()
            optimizer.step()

    for name, param in model.named_parameters():
        if str(param.device).startswith('cuda'):
            print(f"{rank} get param {name} : {param}, {param.device}")
            print(f"{rank} get param grad {name} : {param.grad}")

    l_f=torch.nn.MSELoss(reduction='sum')

    loss = l_f(model_copy(X_train.to('cpu')),y_train.to('cpu'))
    loss.backward()
    optimizer_copy.step()
    if rank == 0:
        for name, param in model_copy.named_parameters():
            print(f" get param  {name} : {param}")
            print(f" get param grad {name} : {param.grad}")
    
    
    cleanup()