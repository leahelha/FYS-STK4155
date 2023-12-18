import numpy as np
import torch 
import torch.optim as op
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from pathlib import Path

from NN import PINN

device= torch.device("cpu")



def initialize_data(N, M):
    """
    Initialize the training data
    """
    

    x = np.linspace(0, 1, N)
    t = np.linspace(0, 1, M)

    x_mesh, t_mesh = np.meshgrid(x, t)

    x_ = x_mesh.reshape(-1 ,1)
    t_ = t_mesh.reshape(-1, 1)

    X = np.hstack((x_,t_))
   

    X_data = torch.tensor(X,dtype=torch.float32,device=device)

    return X_data

def initialize_model(nodes, layers, activation_function):
    """
    Initialize the model with ADAM optimizer
    """
   
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
    
    trained_model = torch.save(model.state_dict(), f'model_{label}.pt')





def predict(X, model):
    """
    Use the saved model and make predictions 
    """
    '''Prediction, error handling, and plotting'''
    #X_plot = torch.tensor(np.hstack((x_.reshape(-1, 1), t_.reshape(-1, 1))), dtype=torch.float32)
    u_pred = model.trial(X[:, 0:1], X[:,1:2]) #, X[:,2:3])

    u_pred = u_pred.detach().cpu().numpy()  # Convert PyTorch tensor to NumPy array

    return u_pred

    


def plot_colormaps(x_mesh, t_mesh, u_pred, label=None):
    """
    Plot the a color map predicted model alongside the analytical. 
    Plot the absolute error in colormap (difference)
    Plot MSE and R2 scores
    """
    here = Path(__file__).parent.absolute()

    directory = Path(f"{here}/Plots/{label}")

    if not directory.exists():
        os.makedirs(directory, exist_ok=True)

    
    anal = analytical(x_mesh, t_mesh, D=1)

    # Reshape for 2D plotting
    u_pred = u_pred.reshape((len(x_mesh), len(t_mesh)))
    
    
    abs_error = abs(anal-u_pred)

    # Specify more levels
    levels = np.linspace(u_pred.min(), u_pred.max(), 80)
    levels2 = np.linspace(abs_error.min(), abs_error.max(), 10)
    
    fig, ax = plt.subplots(1, 2, sharey=True)
    im1 = ax[0].contourf(x_mesh, t_mesh, u_pred, levels = levels, cmap='viridis')
    ax[0].set_title('Predicted NN')
    
    im2 = ax[1].contourf(x_mesh, t_mesh, anal, levels = levels, cmap='viridis')
    ax[1].set_title('Analytical')
   
    ax[0].set_ylabel('Time t')
    ax[0].set_xlabel('Position x')
    ax[1].set_xlabel('Position x')
    
    fig.colorbar(im1, ax=ax[0])
    fig.colorbar(im2, ax=ax[1])

    plt.savefig(f'{here}/Plots/{label}/Predict_vs_analytical_{label}.pdf')

    plt.figure()
    err = plt.contourf(x_mesh, t_mesh, abs_error, levels = levels2, cmap='viridis')
    plt.title('Absolutue error')
    plt.xlabel('Position x')
    plt.ylabel('Time t')
    plt.colorbar(err)

    plt.savefig(f'{here}/Plots/{label}/Absolute error_{label}.pdf')



def MSE(y, y_pred):
    """
    Calculate MSE
    """
    assert y.shape == y_pred.shape, "Input arrays must have the same shape"

    mse = np.mean((y-y_pred)**2)

    return mse


def analytical(x, t, D):
    a = np.sin(np.pi*x)*np.exp(-D*np.pi**2*t)
    return a

def explicit_scheme(u, dx, dt, N, M):
    """
    Function which performs the explicit scheme.
    
    """
    alpha = dt / (dx ** 2)
    if alpha > 0.5:  # Stability condition
        raise ValueError(f"Stability condition not met. alpha = {alpha}, for dt={dt} and dx^2 ={dx**2}")
   
    for j in range(0, M-1):    # time 
        for i in range(1, N-1):    # space 
            u[i, j+1] = alpha*u[i-1, j] + (1-2*alpha)*u[i,j] + alpha*u[i+1, j]

    return u


def plot_explicit(t_index, dx, color, a_color):
    T = 1 #total time
    L = 1
    dt = 0.5*dx**2

    N = int(L/dx)+1
    M = int(T/dt)+1

    x = np.linspace(0, L, N)
    u = np.zeros((N, M))

    t_vals=np.linspace(0,1,M)
    t = t_vals[t_index]
    

    # BC
    u[0, :] = 0
    u[-1, :] = 0
    # IC
    u[:, 0] = np.sin(np.pi*x)

    u = explicit_scheme(u, dx, dt, N, M)
    anal = analytical(x, t, 1)
    
    rmse = np.sqrt(MSE(u[:, t_index], anal))
    

    print(f'dx = {dx}')
    print(f'N = {N}')
    print(f'M = {M}')
    print(f'dt = {dt}, t = {t}, total time = {T}')
    print(f'RMSE = {rmse:.2e}')
    print('\n')
    plt.plot(x, u[:, t_index], color, label=f"t = {t :.2f}")
    plt.plot(x, anal, a_color, label=f"Analytical t = {t :.2f}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u(x, t)")
    #plt.savefig(f'Explicit_scheme_{dx}.pdf')


def plot_heatmap():
    """
    Heatmap plot for gridsearching
    """
    ...

def plot_NN_line(t_index, dx, model, color='blue', a_color='m3'):
    """
    NN equivalent to plot explicit
    """
    T = 1 #total time
    L = 1
    dt = 0.5*dx**2

    N = int(L/dx)+1
    M = int(T/dt)+1

    x = np.linspace(0, L, N)
    t_vals = np.linspace(0, T, M)
    t_point = t_vals[t_index]
    
    t = np.ones_like(x)*t_point

    X = np.hstack((x.reshape(-1,1), t.reshape(-1,1)))
    X = torch.tensor(X, dtype=torch.float32, device=device)

    
    u_pred = predict(X, model)

    analy = analytical(x, t_point, 1)

    print('NN')
    print(np.shape(u_pred[:, 0]))
    print(np.shape(analy))
    mse = MSE(u_pred[:, 0], analy)

    
    print(f'dx = {dx}')
    print(f'N = {N}')
    print(f'M = {M}')
    print(f'dt = {dt}, t = {t}, total time = {T}')
    print(f'MSE = {mse}')
    print('\n')

    plt.plot(x, u_pred, color, label=f"NN t = {t_point :.2f}")
    #plt.plot(x, analy, a_color, label=f"Analytical t = {t_point :.2f}")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("u(x, t)")


def main(D_list):
    N = 100
    M = 100

    x = np.linspace(0, 1, N)
    t = np.linspace(0, 1, M)

    x_mesh, t_mesh = np.meshgrid(x, t)

    activation = [nn.Tanh(), nn.ReLU(), nn.Sigmoid()]

    X_data = initialize_data(N=N, M=M)
    model, optimizer = initialize_model(nodes=9, layers=5, activation_function=activation[0])

    print('Initializing done')
    # train(epochs=1000, X_data=X_data, model=model, optimizer=optimizer, label='9x5')

    model.load_state_dict(torch.load('model_9x5.pt'))

    u_pred = predict(X=X_data, model=model)


    plot_colormaps(x_mesh, t_mesh, u_pred, label='9x5')


# if __name__=="__main__":
#     D_list = [1]
#     main(D_list)



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