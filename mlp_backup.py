## Train a four layers feedforward network with pipeline parallelism using torch.distributed.pipelining
import os
import torch 
import torch.nn as nn 
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed.pipelining import ScheduleGPipe, pipeline, SplitPoint, build_stage
from torch.distributed.pipelining._IR import LossWrapper 
import numpy as np 

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
        self.third_layer = nn.Linear(hidden_dim, hidden_dim)
        self.fourth_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        x = F.relu(self.first_layer(x))
        x = F.relu(self.second_layer(x))
        x = F.relu(self.third_layer(x))
        return self.fourth_layer(x)


class OutputLossWrapper(LossWrapper):
    def __init__(self, module, loss_fn):
        super().__init__(module, loss_fn)

    def forward(self, input, target):
        output = self.module(input)
        return output, self.loss_fn(output, target)

if __name__ == '__main__':
    chunks = 2
    setup()
    local_rank = int(os.environ['LOCAL_RANK'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])

    x = torch.randn((2,1))
    model = MLP(1,1,2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    pipe = pipeline(module = model,
                    mb_args=(x,),
                    split_spec = {
                        "third_layer": SplitPoint.BEGINNING,
                    })
    
    assert pipe.num_stages == world_size

    stage = pipe.build_stage(rank, torch.device(f"cuda:{local_rank}"))
    loss_fn=torch.nn.MSELoss(reduction="sum")

    schedule = ScheduleGPipe(stage, chunks, loss_fn = loss_fn)

    

    X_train = np.array([[2], [3], [4], [5]])
    y_train = np.array([[2], [4], [6], [8]])

    X_train = torch.tensor(X_train, dtype=torch.float32).to(torch.device(f"cuda:{local_rank}"))
    y_train = torch.tensor(y_train, dtype=torch.float32).to(torch.device(f"cuda:{local_rank}"))


    if rank == 0:
        schedule.step(X_train)
    elif rank == world_size - 1:
        losses = []
        out = schedule.step(target = y_train, losses = losses)
    else:
        out = schedule.step()
        
    if rank == world_size - 1:
        print(out, losses)
    
    cleanup()



