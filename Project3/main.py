import numpy as np
import torch 
import torch.optim as op
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from pathlib import Path


from NN import PINN
from utils import predict, train, plot_colormaps, MSE, initialize_data, initialize_model, plot_explicit

here = Path(__file__).parent.absolute()

def main(activation, label, training=False, colormap=False):
    x = np.linspace(0, 1, N)
    t = np.linspace(0, 1, M)

    x_mesh, t_mesh = np.meshgrid(x, t)

    X_data = initialize_data(N=N, M=M)
    model, optimizer = initialize_model(nodes=nodes, layers=layers, activation_function=activation)


    if training==True:
        train(epochs=1000, X_data=X_data, model=model, optimizer=optimizer, label=f'{label}')

    elif training == False:
        model.load_state_dict(torch.load(f'model_{label}.pt'))

    u_pred = predict(X=X_data, model=model)
     
    if colormap == True:
        plot_colormaps(x_mesh, t_mesh, u_pred, label=f'{label}')
    
    
if __name__=="__main__":

    """ Explicit model plot"""
    plt.figure()
    plot_explicit(t_index=1, dx=0.1, color='blue', pred_color='c3')
    plot_explicit(t_index=25, dx=0.1, color='green', pred_color='k3')
    plt.savefig(f'{here}/Plots/Explicit/Explicit_scheme_1_10.pdf')
    

    plt.figure()
    plot_explicit(t_index=100, dx=0.01, color='blue', pred_color='c3')
    plot_explicit(t_index=2500, dx=0.01, color='green', pred_color='k3')
    plt.savefig(f'{here}/Plots/Explicit/Explicit_scheme_1_100.pdf')

    

    """Neural network training and plots"""
    N = 100  
    M = 100

    nodes = 9
    layers = 5
    activation = [nn.Tanh(), nn.ReLU(), nn.Sigmoid()]

    label = f'{nodes}x{layers}_{activation[0]}'  

    
    # main(activation[0], label, training=True, colormap=False)
    # main(activation[0], label, training=False, colormap=True)

