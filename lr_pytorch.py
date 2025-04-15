import torch 
import torch.nn as nn 
torch.manual_seed(1234)

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
    
    def train(self, X, y, learning_rate=1e-3, n_iterations = 100):
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
    




if __name__ == '__main__':
    from sklearn.datasets import make_regression
    from sklearn.model_selection import train_test_split
    import matplotlib.pyplot as plt 

    n_samples = 10000
    n_features = 2
    X, y = make_regression(n_samples=n_samples, n_features=n_features, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42)

    device = 'cpu'
    # X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    # y_train= torch.tensor(y_train, dtype=torch.float32)[:, None].to(device)
    # X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
    # y_test = torch.tensor(y_test, dtype=torch.float32)[:, None].to(device)
    X_train = torch.tensor([[1, 2], [1, 3], [1, 4], [1, 5]], dtype=torch.float32).to(device)
    y_train = torch.tensor([[2,3], [4,5], [6,7], [8,9]], dtype=torch.float32).to(device)
    X_test = X_train
    y_test = y_train
    # Train the model
    model = LinearLayer(n_features, 1).to(device)
    from time import time 
    t = time()
    losses = model.train(X_train, y_train)
    print(time()-t)
    plt.plot(losses)
    plt.show()

    # Predict
    y_pred = model.predict(X_test)

    plt.scatter(X_test[:,1].cpu(), y_test[:,0].cpu(), c='blue', marker='^')
    plt.scatter(X_test[:,1].cpu(), y_pred[:,0].detach().cpu(), c='red')
    plt.show()
