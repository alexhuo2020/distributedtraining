## training of single linear layer (linear regression) with FSDP + TP
## run with `torchrun --nnodes=1 --nproc_per_node=4 lr_ddp.py` to get the result 
import os
import numpy as np 
import torch 
import torch.nn as nn 
import torch.distributed as dist 
from torch.profiler import profile, record_function, ProfilerActivity
import os

torch.manual_seed(1234)

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import (
   enable_wrap,
   wrap)

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import loss_parallel

from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module

import argparse 

def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    # destroy the process group
    dist.destroy_process_group()

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias = bias)
    
    def forward(self, X):
        return self.linear(X)    

    def optimizer(self, learning_rate=1e-3):
        return torch.optim.SGD(self.parameters(), lr = learning_rate)
    
    def _train(self, X, y, learning_rate=1e-3, n_iterations = 10000, ):
        optimizer = self.optimizer(learning_rate)
        losses = []
        for i in range(n_iterations):
            y_pred = self.forward(X)
            loss = torch.mean((y_pred - y)**2)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            optimizer.zero_grad()
        return losses

    def predict(self, X):
        return self.linear(X)

# FSDP Training Function
def train_fsdp(model, optimizer, rank, world_size, tp_size, X, y, n_iterations=10000):
    """Train in fsdp
    """
    # split the data across process group and move the data to GPU
    # n_samples, n_features = X.shape
    # chunk_size = n_samples // world_size 
    # start = rank * chunk_size
    # end = n_samples if rank == world_size - 1 else (rank + 1) * chunk_size
    # X_subset = X[start:end]
    # y_subset = y[start:end]
    # print(f"rank {rank}")
    
    X_subset, y_subset = split_and_assign_data(X, y, world_size, rank, tp_size)#X_subset.to(rank), y_subset.to(rank)
    

    print(f"{rank} get data X: {X_subset}, y: {y_subset}") # print the data each process get

    

    losses = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir'), record_shapes=True) as prof:

        for i in range(n_iterations):
            model.train()
            optimizer.zero_grad()
            print(f"X_subset size {X_subset}")
            y_pred = model(X_subset)
            
            loss = torch.mean((y_pred - y_subset)**2)
            print(f"{rank} get loss {loss}")

            loss.backward()
            for name, param in model.named_parameters():
                print(f"{rank} get grad {name} : {param.grad}")
            optimizer.step()
            print(f"{rank} optimizer get {optimizer.state_dict()}")

            losses.append(loss.item())
            if rank == 0:
                print(f"Rank {rank}: Iteration {i+1}/{n_iterations}, Loss: {loss.item()}")
    if rank == 0:
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    cleanup()
    return losses

def split_and_assign_data(X, y, world_size, rank, tp_size=2):
    """
    Split data for FSDP and replicate it across TP groups.
    """
    try:
        dp_size = world_size // tp_size
        
        dp_rank = rank // tp_size
        tp_rank = rank % tp_size

        # Ensure data splitting respects boundaries
        n_samples = X.size(0)
        print(dp_size)
        print(world_size)
        chunk_size = (n_samples + dp_size - 1) // dp_size  # Ceiling division
        start_idx = dp_rank * chunk_size
        end_idx = min(n_samples, (dp_rank + 1) * chunk_size)
        X_shard = X[start_idx:end_idx]
        y_shard = y[start_idx:end_idx]

        local_device = f"cuda:{rank}"
        X_shard = X_shard.to(local_device)
        y_shard = y_shard.to(local_device)

        # Validate and broadcast data within TP group
        tp_group = dist.new_group(ranks=[dp_rank * tp_size + i for i in range(tp_size)])
        dist.broadcast(X_shard, src=dp_rank * tp_size, group=tp_group)
        dist.broadcast(y_shard, src=dp_rank * tp_size, group=tp_group)

        return X_shard, y_shard

    except Exception as e:
        print(f"Error in split_and_assign_data: {e}")
        raise



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train linear regression with FSDP + TP.")

    # Add arguments
    parser.add_argument("--shardingstrategy", type=str, default='full')
    parser.add_argument("--tp_size", type=int, default=2)

    # Parse arguments
    args = parser.parse_args()
    shardingstrategy = args.shardingstrategy
    tp_size = args.tp_size

    setup()
    rank = int(os.environ['RANK'])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank)

    model = LinearLayer(2,2)#.to(rank)
        
    layer_tp_plan = {
    "linear": ColwiseParallel()
    }

    assert world_size % tp_size ==0
    dp_size = world_size // tp_size
    
    print(f"dp size {dp_size}")

    device_mesh = init_device_mesh("cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp"))
    print(f"Device Mesh created: {device_mesh=}")

    tp_mesh = device_mesh['tp']
    dp_mesh = device_mesh['dp']
    dp_rank = dp_mesh.get_local_rank()


    parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan=layer_tp_plan,
    )

    for name, param in model.named_parameters():
        print(f"{rank} get param {name} : {param}")

    if shardingstrategy == 'full':
        fsdp_model = FSDP(model, sharding_strategy=ShardingStrategy.FULL_SHARD, device_mesh = dp_mesh, use_orig_params=True) 
    elif shardingstrategy == 'no':
        fsdp_model = FSDP(model, sharding_strategy=ShardingStrategy.NO_SHARD, device_mesh = dp_mesh, use_orig_params=True) 
    elif shardingstrategy == 'gradop':
        fsdp_model = FSDP(model, sharding_strategy=ShardingStrategy.SHARD_GRAD_OP, device_mesh = dp_mesh, use_orig_params=True) 
    elif shardingstrategy == 'hybrid':
        fsdp_model = FSDP(model, sharding_strategy=ShardingStrategy.HYBRID_SHARD, device_mesh = dp_mesh, use_orig_params=True) 
    elif shardingstrategy == 'hybrid2':
        fsdp_model = FSDP(model, sharding_strategy=ShardingStrategy._HYBRID_SHARD_ZERO2, device_mesh = dp_mesh, use_orig_params=True)     


    print(f"FSDP model created")
    for name, param in model.named_parameters():
        print(f"{rank} get param {name} : {param}")

    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-3, foreach=True)
    # import matplotlib.pyplot as plt 
    X_train = torch.tensor([[1, 2], [1, 3], [1, 4], [1, 5]], dtype=torch.float32)
    y_train = torch.tensor([[2,3], [4,5], [6,7], [8,9]], dtype=torch.float32)
    print(f"dp {dp_rank}")
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    print(f" wordsize {world_size}")
    train_fsdp(fsdp_model, optimizer, rank, world_size, tp_size, X_train, y_train, n_iterations=1)
