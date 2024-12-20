# Understand distributed training Part 1: linear regression example

Distributed training is very important in training large machine learning models to utilize the amzing computational power of GPU clusters. To understand how this is done, we first use the linear regression as an example to demonstrate how to apply Data Parallelism (DP), Tensor Parallelism (TP) and a combination of them to train the linear regression model.

## Implement a linear regression using PyTorch
Linear regression can be done in two ways: using gradient descent or by direct matrix inverse (or pseudo inverse) computations. Let's do it with gradient descent. Suppose the input training data are $X\in \mathbb{R}^{Np}, Y\in \mathbb{R}^{Nq}$ where $N$ is the number of data points, $p$ are the number of features and $q$ are the number of prediction features. A linear regression model is a linear mapping $y = W x + b$ where $W \in \mathbb{R}^{qp}, b\in \mathbb{R}^q$. To find the optimal $W,b$, we minimize the MSE loss
$$\text{MSE} = \frac1N\sum_{i=1}^N (W X_i  + b - Y_i)^2 $$
or in the matrix form
$$\text{MSE} = \frac1N \|X W^T + b - Y\|^2$$
To train the model, what we need is 
- define the model, which is a simple linear layer `model = nn.Linear(p,q)`
- set an optimizer, `optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)`
- compute the loss, `loss = torch.mean((model(X)-y)**2)`
- do gradient descent,
    + compute the gradient $\frac{\partial \text{MSE}}{\partial (W,b)}$ using `loss.backward()`
    + update the parameters $W\gets W - \eta \frac{\partial \text{MSE}}{\partial W}, b \gets b - \eta \frac{\partial \text{MSE}}{\partial b}  $ using `optimizer.step()`

```python
def train(model, optimizer, X, y, n_iterations = 10000):
    losses = []
    for i in range(n_iterations):
        y_pred = self.forward(X)
        loss = torch.mean((y_pred - y)**2)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()
        optimizer.zero_grad()
    return losses
```
## Data Parallelism
When we need to do linear regression on huge amount of data, we need to process different chuncks of data on different devices. Although this can be done using CPU clusters, using a GPU clusters may speed up the training process. We can split the data into $M$ chunks, with each GPU process each chunk.
+ Gradients are averaged and synchronized across nodes after each batch
+ Communication overhead during gradient synchronization grows with model size and node count.
+ Gradient on node $i$ ($X_i,y_i$ denote the data on node $i$):
    $$ \text{grad}_i = \frac{\partial L(X_i,y_i) }{ \partial \beta}$$
+ communicate to gather gradients and do summation, done by `all reduce` $\text{grad} = \sum_{i=1}^m \text{grad}_i  $
+ update rule on each node:
    $$\beta \gets \beta - \eta \text{grad} $$
+ All nodes have the copy of the same model parameters

Example: 

Suppose we have data 
$$X = [[1,1], [1, 2], [1,2], [1,4]]$$
$$y = [2, 4, 6, 8]$$
We augment $X$ so that bias does not require a special treatment.
$$X = \left(\begin{array}{ccc} 1 & 1 & 2\\
1 & 2 & 3 \\
1 & 3 & 4 \\
1 & 4 & 5 \end{array}\right)$$




### Fully Shareded Data Parallel: when weights cannot fit one GPU
+ a good tutorial: https://www.edge-ai-vision.com/2024/05/fully-sharded-data-parallelism-fsdp/
+ weights are distributed acorss multiple nodes
+ Forward Pass:
    + collect weights to run the forward pass (`all gather`)
    + data are distributed across different nodes and each node compute its all predictions
    + delete abundant model parameters on each node, keep its own weights
    + compute loss locally
+ Backward Pass:
    + collect weights 
    + compute gradients locally for each data 
    + run `all reduce scatter` to distribute gradients 
    + update local weights
    + discard abundant weights
+ characteristic
    + Communicates smaller shards of gradients, potentially reducing communication overhead.
    + has forward pass all-gather requirement, i.e. all weights must fit one node. However, gradients are sharded.




- Tensor Parallelism: distributed model weights across multiple nodes
    + if the model weights are too large, we need to do model sharding 
    + split $\beta = [\beta^1,\ldots, \beta^m]$ where $\beta^i \in \mathbb{R}^{p/m}$
    + denote the partition of indices by $S_1,\ldots, S_m$
    + when compute the forward pass, each node computes
        $$y_i^k = \sum_{j \in S_k} X_{ij} \beta^k_j$$
        $$y_i = \sum_{k} y_i^k$$
    + we need `all reduce sum` to compute the final prediction
    + for gradients
        $$\frac{\partial L}{\partial \beta_j} =  \frac{2}{N} \sum_{i=1}^N  \left(y_i - \sum_{k=1}^p X_{ik} \beta_k\right) X_{ij}, j \in S_j $$


    + we need to gather all the gradients
        $$[\frac{\partial L}{\partial \beta^1},\ldots, \frac{\partial L}{\partial \beta^m} ]$$
    + this is done by `all gather`
    + for backward processes, we have 
        $$\frac{\partial L}{\partial \beta^k} = \frac{\partial L}{\partial y^k} \frac{\partial y^k}{\partial \beta^k}$$
        and we also need to run `all gather` to gather all the gradients





