import numpy as np
import torch
import torch.optim as op
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from pathlib import Path
import seaborn as sns
import json
from NN import PINN

torch.manual_seed(43)
np.random.seed(43)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run_from_config(config):
    # load config.json

    N = config['N']
    M = config['M']

    x = np.linspace(0, 1, N)
    t = np.linspace(0, 1, M)

    x_, t_ = np.meshgrid(x, t)

    x_ = x_.reshape(-1, 1)
    t_ = t_.reshape(-1, 1)

    X = np.hstack((x_, t_))

    X = torch.tensor(X, dtype=torch.float32, device=device)

    activation_function = config['activation_function']
    activation_name = config['activation_name']
    num_hidden_nodes = config['num_hidden_nodes']
    num_layers = config['num_layers']
    lr = config['lr']

    model = PINN(num_hidden_nodes, num_layers, activation_function)
    optimizer = op.Adam(model.parameters(), lr=lr)
    scheduler = op.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)

    epochs = config['epochs']
    loss_list = []
    epochs_list = []
    for epoch in range(epochs):

        loss = train_step(X, optimizer, model)
        if epoch % 100 == 0:
            loss_list.append(loss.item())
            epochs_list.append(epoch)
            print(f'{epoch} loss = {loss.item()}')
            scheduler.step()

    # Save model
    torch.save(model.state_dict(
    ), f'model{num_hidden_nodes}x{num_layers}_{activation_name}.pt')

    # Print RMSE
    u_pred = predict(X, model)

    u_pred = u_pred.reshape((M, N))

    X, T = np.meshgrid(x, t)

    anal = np.sin(np.pi*X)*np.exp(-np.pi**2*T)

    RMSE = np.sqrt(np.mean((anal-u_pred)**2))

    print(f'RMSE = {RMSE}')

    # Plot loss
    plt.figure()
    plt.plot(epochs_list, loss_list)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.title(
        f'Loss for {num_hidden_nodes}x{num_layers} {activation_name} activation function')
    plt.savefig(
        f'Loss_{num_hidden_nodes}x{num_layers}_{activation_name}_N{N}_M{M}.pdf')


def initialize_data(N, M):
    """
    Initialize the training data
    """

    x = np.linspace(0, 1, N)
    t = np.linspace(0, 1, M)

    x_mesh, t_mesh = np.meshgrid(x, t)

    x_ = x_mesh.reshape(-1, 1)
    t_ = t_mesh.reshape(-1, 1)

    X = np.hstack((x_, t_))

    X_data = torch.tensor(X, dtype=torch.float32, device=device)

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
        if epoch % 100 == 0:
            print(f'epoch {epoch}: loss = {loss.item()}')

    trained_model = torch.save(model.state_dict(), f'model_{label}.pt')


def train_step(X, optimizer, model):
    optimizer.zero_grad()
    loss = torch.mean(torch.square(model.Cost(X)))
    loss.backward()

    optimizer.step()

    return loss


def run(num_hidden, num_layers, activation_function, lr, save=False):
    # empty cache
    torch.cuda.empty_cache()
    N = 10
    M = 10

    x = np.linspace(0, 1, N)
    t = np.linspace(0, 1, M)

    x_, t_ = np.meshgrid(x, t)

    x_ = x_.reshape(-1, 1)
    t_ = t_.reshape(-1, 1)

    X = np.hstack((x_, t_))

    X = torch.tensor(X, dtype=torch.float32, device=device)
    model = PINN(num_hidden, num_layers, activation_function)
    optimizer = op.Adam(model.parameters(), lr=lr)

    epochs = 5000
    loss_list = []
    epochs_list = []
    for epoch in range(epochs):

        loss = train_step(X, optimizer, model)
        if epoch % 100 == 0:
            loss_list.append(loss.item())
            epochs_list.append(epoch)

            print(f'{epoch} loss = {loss.item()}')
    # Save model
    if save == True:
        torch.save(model.state_dict(), f'model{num_hidden}x{num_layers}.pt')

    # Example: Plotting a PyTorch tensor

    # X_plot = torch.tensor(np.hstack((x_.reshape(-1, 1), t_.reshape(-1, 1))), dtype=torch.float32)

    u_pred = model.trial(
        X[:, 0:1], X[:, 1:2])

    u_pred = u_pred.detach().cpu().numpy()

    u_pred = u_pred.reshape((M, N))

    X, T = np.meshgrid(x, t)

    anal = np.sin(np.pi*X)*np.exp(-np.pi**2*T)

    RMSE = np.sqrt(np.mean((anal-u_pred)**2))

    return RMSE


def grid_search():
    # Grid search
    num_hidden_nodes_list = [20, 30, 40, 50]
    num_layers_list = [1, 2, 3, 4]
    activation_list = [nn.Sigmoid(), nn.ReLU(), nn.Tanh(), nn.SiLU()]
    activation_names = ['Sigmoid', 'ReLU', 'Tanh']
    # Dictionary to store best configuration
    Best = {'RMSE': 1e10, 'num_hidden_nodes': 0, 'num_layers': 0,
            'activation_function': None, 'lr': 0}
    Best_rmse = 1e10
    for a, activation_function in enumerate(activation_list):
        for lr in [0.0001, 0.001, 0.01, 0.1]:
            print(
                f'################Running {activation_names[a]} activation function, lr={lr}##################')
            RMSE_list = np.zeros(
                (len(num_hidden_nodes_list), len(num_layers_list)))
            for i, num_hidden_nodes in enumerate(num_hidden_nodes_list):
                for j, num_layers in enumerate(num_layers_list):
                    print(f'Running {num_hidden_nodes}x{num_layers}')
                    RMSE_list[i, j] = run(
                        num_hidden_nodes, num_layers, activation_function, lr)
                    print(
                        f'RMSE for {num_hidden_nodes}x{num_layers} = {RMSE_list[i, j]}')
            # Find minimum value
            min_val_idx = np.unravel_index(
                np.argmin(RMSE_list), RMSE_list.shape)
            # Update best configuration
            if RMSE_list[min_val_idx] < Best_rmse:
                Best_rmse = RMSE_list[min_val_idx]
                Best['RMSE'] = RMSE_list[min_val_idx]
                Best['num_hidden_nodes'] = num_hidden_nodes_list[min_val_idx[0]]
                Best['num_layers'] = num_layers_list[min_val_idx[1]]
                Best['activation_function'] = activation_function
                Best['lr'] = lr

            print(f'Minimum RMSE index = {min_val_idx}')
            print(f'Minimum RMSE = {RMSE_list[min_val_idx]}')
            # New figure
            plt.figure()
            # Create the heatmap
            ax = sns.heatmap(RMSE_list, annot=True, cmap='viridis', xticklabels=num_layers_list,
                             yticklabels=num_hidden_nodes_list)
            ax.set(xlabel='Layers', ylabel='Hidden nodes per layer')
            ax.collections[0].colorbar.set_label("RMSE")
            # Title
            ax.set_title(
                f'RMSE for {activation_names[a]} activation function, lr={lr}')

            # Highlight the minimum value with a rectangle
            ax.add_patch(plt.Rectangle(
                (min_val_idx[1], min_val_idx[0]), 1, 1, fill=False, edgecolor='red', lw=3))

            # Save and show the plot
            plt.savefig(f'heatmap_{activation_names[a]}_{lr}.pdf')
    print('Best = ', Best)


def predict(X, model):
    """
    Use the saved model and make predictions 
    """
    '''Prediction, error handling, and plotting'''
    # X_plot = torch.tensor(np.hstack((x_.reshape(-1, 1), t_.reshape(-1, 1))), dtype=torch.float32)
    u_pred = model.trial(X[:, 0:1], X[:, 1:2])  # , X[:,2:3])

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
    im1 = ax[0].contourf(x_mesh, t_mesh, u_pred, levels=levels, cmap='viridis')
    ax[0].set_title('Predicted NN')

    im2 = ax[1].contourf(x_mesh, t_mesh, anal, levels=levels, cmap='viridis')
    ax[1].set_title('Analytical')

    ax[0].set_ylabel('Time t')
    ax[0].set_xlabel('Position x')
    ax[1].set_xlabel('Position x')

    fig.colorbar(im1, ax=ax[0])
    fig.colorbar(im2, ax=ax[1])

    plt.savefig(f'{here}/Plots/{label}/Predict_vs_analytical_{label}.pdf')

    plt.figure()
    err = plt.contourf(x_mesh, t_mesh, abs_error,
                       levels=levels2, cmap='viridis')
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
        raise ValueError(
            f"Stability condition not met. alpha = {alpha}, for dt={dt} and dx^2 ={dx**2}")

    for j in range(0, M-1):    # time
        for i in range(1, N-1):    # space
            u[i, j+1] = alpha*u[i-1, j] + (1-2*alpha)*u[i, j] + alpha*u[i+1, j]

    return u


def plot_explicit(t_index, dx, color, a_color):
    T = 1  # total time
    L = 1
    dt = 0.5*dx**2

    N = int(L/dx)+1
    M = int(T/dt)+1

    x = np.linspace(0, L, N)
    u = np.zeros((N, M))

    t_vals = np.linspace(0, 1, M)
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
    # plt.savefig(f'Explicit_scheme_{dx}.pdf')





def plot_NN_line(t_index, dx, model, color='blue', a_color='m3'):
    """
    NN equivalent to plot explicit
    """
    T = 1  # total time
    L = 1
    dt = 0.5*dx**2

    N = int(L/dx)+1
    M = int(T/dt)+1

    x = np.linspace(0, L, N)
    t_vals = np.linspace(0, T, M)
    t_point = t_vals[t_index]

    t = np.ones_like(x)*t_point

    X = np.hstack((x.reshape(-1, 1), t.reshape(-1, 1)))
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
    # plt.plot(x, analy, a_color, label=f"Analytical t = {t_point :.2f}")
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
    model, optimizer = initialize_model(
        nodes=9, layers=5, activation_function=activation[0])

    print('Initializing done')
    # train(epochs=1000, X_data=X_data, model=model, optimizer=optimizer, label='9x5')

    model.load_state_dict(torch.load('model_9x5.pt'))

    u_pred = predict(X=X_data, model=model)

    plot_colormaps(x_mesh, t_mesh, u_pred, label='9x5')
