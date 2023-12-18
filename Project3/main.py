import numpy as np
import torch
import torch.optim as op
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from pathlib import Path


from NN import PINN
from utils import predict, train, plot_colormaps, MSE, initialize_data, initialize_model, plot_explicit, plot_NN_line, grid_search, run_from_config


torch.manual_seed(43)
np.random.seed(43)

here = Path(__file__).parent.absolute()


def main(activation, label, training=False, colormap=False, compare_lineplot=False, explicit=False):
    x = np.linspace(0, 1, N)
    t = np.linspace(0, 1, M)

    x_mesh, t_mesh = np.meshgrid(x, t)

    X_data = initialize_data(N=N, M=M)
    model, optimizer = initialize_model(
        nodes=nodes, layers=layers, activation_function=activation)

    if training == True:
        train(epochs=1000, X_data=X_data, model=model,
              optimizer=optimizer, label=f'{label}')

    elif training == False:
        model.load_state_dict(torch.load(f'model_{label}.pt'))

    u_pred = predict(X=X_data, model=model)

    if colormap == True:
        plt.figure()
        plot_colormaps(x_mesh, t_mesh, u_pred, label=f'{label}')

    if compare_lineplot == True:
        """Comparing NN and FTCS/Explicit Euler"""
        plt.figure()
        plot_explicit(t_index=100, dx=0.01, color='blue', a_color='c3')
        plot_explicit(t_index=2500, dx=0.01, color='green', a_color='k3')
        plot_NN_line(t_index=100, dx=0.01, model=model,
                     color='orange', a_color='g3')
        plot_NN_line(t_index=2500, dx=0.01, model=model,
                     color='red', a_color='m3')
        plt.title(
            f"Comparison of NN prediction for ∆x = {0.01} with analytical")
        plt.show()

    if explicit == True:
        """ Explicit model plot"""

        plt.figure()
        plot_explicit(t_index=1, dx=0.1, color='blue', a_color='c3')
        plot_explicit(t_index=25, dx=0.1, color='green', a_color='k3')
        plt.title(f"∆x = {0.1}")
        plt.savefig(f'{here}/Plots/Explicit/Explicit_scheme_1_10.pdf')

        plt.figure()
        plot_explicit(t_index=100, dx=0.01, color='blue', a_color='c3')
        plot_explicit(t_index=2500, dx=0.01, color='green', a_color='k3')
        plt.title(f"∆x = {0.01}")
        plt.savefig(f'{here}/Plots/Explicit/Explicit_scheme_1_100.pdf')


if __name__ == "__main__":

    """


    N = 100
    M = 100

    nodes = 9
    layers = 5
    activation = [nn.Tanh(), nn.ReLU(), nn.Sigmoid()]

    label = f'{nodes}x{layers}_{activation[0]}'

    main(activation[0], label, training=True)  # Comment out when done training
    main(activation[0], label, training=False,
         colormap=True, compare_lineplot=True, explicit=True)
    """
    # To train from config, uncomment the following:

    config = {'N': 10, 'M': 10,
              'num_hidden_nodes': 40, 'num_layers': 3,
              'activation_function': nn.SiLU(),
              'activation_name': 'SiLU',
              'lr': 0.01,
              'epochs': 40000
              }
    run_from_config(config)

    # To run grid search, uncomment the following:
    # grid_search()
