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
 