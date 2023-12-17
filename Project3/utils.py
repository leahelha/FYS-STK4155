import numpy as np
import torch 
import torch.optim as op
import torch.nn as nn
import matplotlib.pyplot as plt

from NN import PINN




def initilize_data(N, M, D_list):
    """
    Initialize the training data
    """
    device= torch.device("cpu")

    x = np.linspace(0, 1, N)
    t = np.linspace(0, 1, M)

    x_mesh, t_mesh = np.meshgrid(x, t)

    x_ = x_mesh.reshape(-1 ,1)
    t_ = t_mesh.reshape(-1, 1)

    #D_list = [0.5, 1, 5, 10, 20]  # list of diffusion coefficients for training

    X_data = np.zeros((len(x_), 3, len(D_list)))

    for i in range(len(D_list)):
        X = np.hstack((x_,t_, np.ones_like(x_)*D_list[i]))

        X_data[:, :, i] = X

    X_data = torch.tensor(X_data,dtype=torch.float32,device=device)

    return X_data

def initilize_model(nodes, layers, activation_function, optimizer):
    """
    Initialize the model
    """
    activation_function = nn.Tanh() 
    model = PINN(nodes, layers, activation_function)  
    optimizer = op.Adam(model.parameters(), lr=0.01)

    return model, optimizer



def train(epochs, X_data, model, optimizer, label=None):
    """
    Train the PINN, and save the model 
    """

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = torch.mean(torch.square(model.Cost(X_data)))
        loss.backward()

        optimizer.step()
        if epoch %100==0:
            print(f'epoch {epoch}: loss = {loss.item()}')
    
    trained_model = torch.save(model.state_dict(), f'model{label}.pt')





def predict(X, model):
    """
    Use the saved model and make predictions for different D
    """
    '''Prediction, error handling, and plotting'''
    #X_plot = torch.tensor(np.hstack((x_.reshape(-1, 1), t_.reshape(-1, 1))), dtype=torch.float32)
    u_pred = model.trial(X[:, 0:1], X[:,1:2], X[:,2:3])

    u_pred = u_pred.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy array

    


def plot_colormaps(x_mesh, t_mesh, u_pred, label=None):
    """
    Plot the a color map predicted model alongside the analytical. 
    Plot the absolute error in colormap (difference)
    Plot MSE and R2 scores
    """
    anal = np.sin(np.pi*x_mesh)*np.exp(-np.pi**2*t_mesh)

    # Reshape for 2D plotting
    u_pred = u_pred.reshape((len(x_mesh), len(t_mesh)))
    
    
    abs_error = abs(anal-u_pred)

    # Specify more levels
    levels = np.linspace(u_pred.min(), u_pred.max(), 80)
    levels2 = np.linspace(abs_error.min(), abs_error.max(), 10)
    
    fig, ax = plt.subplots(1, 2, sharey=True)
    im1 = ax[0].contourf(x_mesh, t_mesh, u_pred, levels = levels, cmap='viridis')
    ax[0].set_title('Predicted')
    
    im2 = ax[1].contourf(x_mesh, t_mesh, anal, levels = levels, cmap='viridis')
    ax[1].set_title('Analytical')
   
    ax[0].set_ylabel('Time t')
    ax[0].set_xlabel('Position x')
    ax[1].set_xlabel('Position x')
    
    fig.colorbar(im1, ax=ax[0])
    fig.colorbar(im2, ax=ax[1])

    plt.savefig(f'Predict_vs_analytical_{label}.pdf')

    fig, ax = plt.subplot(1)
    err = ax[0].contourf(x_mesh, t_mesh, abs_error, levels = levels2, cmap='viridis')
    ax[0].set_title('Absolutue error')
    ax[0].set_xlabel('Position x')
    ax[0].set_ylabel('Time t')
    fig.colorbar(err, ax=ax[0])

    plt.savefig(f'Absolute error_{label}.pdf')



def plot_MSE():
    """
    Plot R2 scores
    """
    ...

def plot_R2():
    """
    Plot R2 scores
    """
    ...



def main(D_list):
    X_data = initilize_data(100, 100, D_list=D_list)
    model, optimizer = initilize_model(9, 5, )


if __name__=="__main__":
    D_list = [1]
    main(D_list)



"""
def main():
   
    
    '''Training'''
    device= torch.device("cpu")
    N = 100
    M = 100

    x = np.linspace(0, 1, N)
    t = np.linspace(0, 1, M)

    x_mesh, t_mesh = np.meshgrid(x, t)

    x_ = x_mesh.reshape(-1 ,1)
    t_ = t_mesh.reshape(-1, 1)

    D_list = [0.5, 1, 5, 10, 20]  # list of diffusion coefficients for training

    X_data = np.zeros((len(x_), 3, len(D_list)))

    for i in range(len(D_list)):
        X = np.hstack((x_,t_, np.ones_like(x_)*D_list[i]))

        X_data[:,:,i] = X

    X_data = torch.tensor(X_data,dtype=torch.float32,device=device)

    '''Editables'''
    activation_function = nn.Tanh() 
    model = PINN(9, 5, activation_function)  
    optimizer = op.Adam(model.parameters(), lr=0.01)

    
    epochs = 1000
    for epoch in range(epochs):
        loss = train(X_data, optimizer, model)
        if epoch %100==0:
            print(f'epoch {epoch}: loss = {loss.item()}')


    '''Prediction, error handling, and plotting'''
    #X_plot = torch.tensor(np.hstack((x_.reshape(-1, 1), t_.reshape(-1, 1))), dtype=torch.float32)
   
    u_pred = model.trial(X[:, 0:1], X[:,1:2])

    u_pred = u_pred.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy array

    # Reshape for 2D plotting
    u_pred = u_pred.reshape((M, N))

    
    
    
    anal = np.sin(np.pi*x_mesh)*np.exp(-np.pi**2*t_mesh)
    abs_error = abs(anal-u_pred)

    # Specify more levels
    levels = np.linspace(u_pred.min(), u_pred.max(), 80)
    levels2 = np.linspace(abs_error.min(), abs_error.max(), 10)
    

    fig, ax = plt.subplots(1, 3, sharey=True)
    im1 = ax[0].contourf(x_mesh, t_mesh, u_pred, levels = levels, cmap='viridis')
    ax[0].set_title('Predicted')
    
    im2 = ax[1].contourf(x_mesh, t_mesh, anal, levels = levels, cmap='viridis')
    ax[1].set_title('Analytical')
   
    im3 = ax[2].contourf(x_mesh, t_mesh, abs_error, levels = levels2, cmap='viridis')
    ax[2].set_title('Absolutue error')

    ax[0].set_ylabel('time t')
    ax[0].set_xlabel('position x')
    ax[1].set_xlabel('position x')
    ax[2].set_xlabel('position x')

    fig.colorbar(im1, ax=ax[0])
    fig.colorbar(im2, ax=ax[1])
    fig.colorbar(im3, ax=ax[2])
    plt.show()
    
    print(np.shape(u_pred))



"""