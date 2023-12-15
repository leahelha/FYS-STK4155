import numpy as np
import torch 
import torch.optim as op
import torch.nn as nn
import matplotlib.pyplot as plt


class PINN(nn.Module):

    def __init__(self, num_hidden_nodes, num_layers, activation_function):
        """
        Creating model and layers
        """
        super(PINN, self).__init__()
        
        layers = [nn.Linear(2, num_hidden_nodes), activation_function]

        for i in range(num_layers-2):
            layers += [nn.Linear(num_hidden_nodes, num_hidden_nodes), activation_function]
        
        layers += [nn.Linear(num_hidden_nodes, 1)]

        self.model = nn.Sequential(*layers)
        device=torch.device("mps")
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_normal_(layer.weight) 
                torch.nn.init.zeros_(layer.bias)
    
    def forward(self, X):
        """
        Forward propagation
        """
        return self.model(X)
    
    def trial(self, x, t):
        """
        Trial function to be used in cost function
        """
        g = (1-t)*torch.sin(torch.pi*x) + x*(1-x)*t*self.model(torch.hstack((x,t)))
        return g
    
    def Cost(self, X):
        """
        Cost function for our 1D heat model
        """
        x, t = X[:, 0:1], X[:, 1:2]

        x.requires_grad_(True)
        t.requires_grad_(True)

        g = self.trial(x, t)

        u_dt = torch.autograd.grad(g, t, grad_outputs=torch.ones_like(g), create_graph=True)[0]
        u_dx = torch.autograd.grad(g, x, grad_outputs=torch.ones_like(g), create_graph=True)[0]
        u_dxx = torch.autograd.grad(u_dx, x, grad_outputs=torch.ones_like(u_dx), create_graph=True)[0]

        cost = u_dt - u_dxx 

        return cost
    

def train(X, optimizer, model):
    """
    
    """

    optimizer.zero_grad()
    loss = torch.mean(torch.square(model.Cost(X)))
    loss.backward()

    optimizer.step()

    return loss

    

def main():
    """
    """
    device= torch.device("mps")
    N = 100
    M = 100

    x = np.linspace(0, 1, N)
    t = np.linspace(0, 1, M)

    x_mesh, t_mesh = np.meshgrid(x, t)

    x_ = x_mesh.reshape(-1 ,1)
    t_ = t_mesh.reshape(-1, 1)

    X = np.hstack((x_,t_))

    X = torch.tensor(X,dtype=torch.float32,device=device)

    activation_function = nn.Tanh()

    
    model = PINN(5, 5, activation_function)
    optimizer = op.Adam(model.parameters(), lr=0.1)

    
    epochs = 100
    for epoch in range(epochs):
        loss = train(X, optimizer, model)
        if epoch in [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]:
            print(f'loss = {loss.item()}')


    # Example: Plotting a PyTorch tensor
    
    #X_plot = torch.tensor(np.hstack((x_.reshape(-1, 1), t_.reshape(-1, 1))), dtype=torch.float32)
   
    u_pred = model.trial(X[:, 0:1], X[:,1:2])

    u_pred = u_pred.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy array

    # Reshape for 2D plotting
    u_pred = u_pred.reshape((M, N))

    
    
    
    anal = np.sin(np.pi*x_mesh)*np.exp(-np.pi**2*t_mesh)
    abs_error = abs(anal-u_pred)

    # Specify more levels
    levels = np.linspace(u_pred.min(), u_pred.max(), 80)
    levels2 = np.linspace(abs_error.min(), abs_error.max(), 80)

    fig, ax = plt.subplots(3, 3)
    im1 = ax[0].contourf(x_mesh, t_mesh, u_pred, levels = levels, cmap='viridis')
    ax[0].set_title('Predicted')
    
    im2 = ax[1].contourf(x_mesh, t_mesh, anal, levels = levels, cmap='viridis')
    ax[1].set_title('Analytical')
   
    im3 = ax[2].contourf(x_mesh, t_mesh, abs_error, levels = levels2, cmap='viridis')
    ax[2].set_title('Absolutue error')

    plt.ylabel('time t')
    plt.xlabel('position x')
    fig.colorbar(im1, ax=ax[0])
    fig.colorbar(im2, ax=ax[1])
    fig.colorbar(im3, ax=ax[2])
    plt.show()
    
    print(np.shape(u_pred))

    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_, t_, u_pred, cmap='viridis')
    ax.set_xlabel('X')
    ax.set_ylabel('T')
    ax.set_zlabel('U')
    plt.show()
    
    """






if __name__=="__main__":
    main()