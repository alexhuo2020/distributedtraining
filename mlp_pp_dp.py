## Combine DP with PP, Requires at least 4 GPUs to run
import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F
from torch.profiler import profile, record_function, ProfilerActivity

import torch.distributed as dist
from torch.distributed.pipelining import ScheduleGPipe, pipeline, SplitPoint, build_stage
import torch.distributed as dist 
from torch.profiler import profile, record_function, ProfilerActivity

torch.manual_seed(1234)

from torch.nn.parallel import DistributedDataParallel as DDP

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


