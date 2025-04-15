## TP training of single linear layer (linear regression)
## run with `torchrun --nnodes=1 --nproc_per_node=2 lr_tp.py` to get the result 

import os
import torch
import torch.nn as nn
import torch.distributed as dist 
from torch.profiler import profile, record_function, ProfilerActivity
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor.parallel import loss_parallel

torch.manual_seed(1234)
from torch.distributed.tensor.parallel import ColwiseParallel, RowwiseParallel, parallelize_module


def setup():
    # initialize the process group
    dist.init_process_group("nccl")


def cleanup():
    dist.destroy_process_group()


class LinearLayer(nn.Module):
    def __init__(self, input_dim, output_dim, bias=False):
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
        return torch.optim.Adam(self.parameters(), lr = learning_rate)
    
    def _train(self, X, y, learning_rate=1e-3, n_iterations = 1, ):
        optimizer = self.optimizer(learning_rate)
        losses = []
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], on_trace_ready=torch.profiler.tensorboard_trace_handler('./log_dir'), record_shapes=True) as prof:

            for i in range(n_iterations):
                y_pred = self.forward(X)
                with loss_parallel():
                    loss = torch.mean((y_pred - y)**2)
                    print(f"{rank} get loss {loss}")
                    print(f"{rank} get data {X}")
                    loss.backward()
                    for name, param in model.named_parameters():
                        print(f"{rank} get grad {name} : {param.grad}")

                losses.append(loss.item())
                
                # optimizer.step()
                # print(f"{rank} optimizer get {optimizer.state_dict()}")
    
                optimizer.zero_grad()
        if rank == 0:
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        return losses

    def predict(self, X):
        return self.linear(X)


if __name__ == '__main__':
    setup()
    rank = int(os.environ['RANK'])
    torch.cuda.set_device(rank)

    model = LinearLayer(2,2)#.to(rank)
    deep_copy_model =  LinearLayer(2,2)
    deep_copy_model.load_state_dict(model.state_dict())
    
    print(model)
    for name, param in model.named_parameters():
        print(f"{rank} get param {name} : {param}")

        
    layer_tp_plan = {
    "linear": ColwiseParallel()
    }

    tp_mesh = init_device_mesh("cuda", (2,))
    print(tp_mesh)

    parallelize_module(
        module=model,
        device_mesh=tp_mesh,
        parallelize_plan=layer_tp_plan,
    )
    
    for name, param in model.named_parameters():
        print(f"{rank} get param {name} : {param}")

    X_train = torch.tensor([[1, 2], [1, 3], [1, 4], [1, 5]], dtype=torch.float32).to(rank)
    y_train = torch.tensor([[2,3], [4,5], [6,7], [8,9]], dtype=torch.float32).to(rank)

    print(model(X_train))
    if rank == 0:
        print(deep_copy_model(X_train.to('cpu')))

    model._train(X_train, y_train)

    print(model(X_train))
    cleanup()
