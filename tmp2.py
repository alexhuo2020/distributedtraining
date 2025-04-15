import torch 
import torch.nn as nn 
import torch.nn.functional as F
from pippy import ScheduleGPipe, pipeline, SplitPoint

class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim):
        super().__init__()
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.second_layer = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, x):
        x = F.relu(self.first_layer(x))
        return self.second_layer(x)
            
x = torch.randn((1,2))
model = MLP(2,2,2)
pipe = pipeline(module = model,
                num_chunks=2,
                example_args=(x,),
                split_spec = {
                    "first_layer": SplitPoint.END,
                })
