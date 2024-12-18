## DDP training of single linear layer (linear regression)
## run with `torchrun --nnodes=1 --nproc_per_node=2 lr_ddp.py` to get the result 

import os
import numpy as np 
import torch 
import torch.nn as nn 
import torch.distributed as dist 
from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(1234)

from torch.nn.parallel import DistributedDataParallel as DDP


def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()

class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.linear = nn.Linear(input_dim, output_dim, bias = bias)
    
    def forward(self, X):
        return self.linear(X)
    
    def init_weights(self):
        # Initialize weights to zero
        nn.init.ones_(self.linear.weight)
        if self.linear.bias is not None:
            nn.init.ones_(self.linear.bias)
    

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
def train_ddp(model, optimizer, rank, world_size, input_dim, output_dim, X, y, learning_rate=1e-3, n_iterations=10000):
    """train with distributed data parallel"""
    n_samples, n_features = X.shape
    chunk_size = n_samples // world_size 
    start = rank * chunk_size
    end = n_samples if rank == world_size - 1 else (rank + 1) * chunk_size
    X_subset = X[start:end]
    y_subset = y[start:end]
    
    X_subset, y_subset = X_subset.to(rank), y_subset.to(rank)
    print(f"{rank} get data X: {X_subset}, y: {y_subset}")
    losses = []
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir'), record_shapes=True) as prof:

        for i in range(n_iterations):
            model.train()
            optimizer.zero_grad()
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



if __name__ == '__main__':
    setup()
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    model = nn.Linear(1, 1)
    for name, param in model.named_parameters():
        print(f"{name}: {param}")
    model.to(rank)
    model = DDP(model) 
    print(list(model.parameters()))
    
    X_train = np.array([[2], [3], [4], [5]])
    y_train = np.array([[2], [4], [6], [8]])

    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    learning_rate = 1e-3
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    train_ddp(model, optimizer, rank, world_size, 1, 2, X_train, y_train, n_iterations=1)