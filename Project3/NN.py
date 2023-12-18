import numpy as np
import torch
import torch.optim as op
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class PINN(nn.Module):

    def __init__(self, num_hidden_nodes, num_layers, activation_function):
        """
        Creating model and layers
        """
        super(PINN, self).__init__()

        layers = [nn.Linear(2, num_hidden_nodes), activation_function]

        for i in range(num_layers):
            layers += [nn.Linear(num_hidden_nodes,
                                 num_hidden_nodes), activation_function]

        layers += [nn.Linear(num_hidden_nodes, 1)]

        self.model = nn.Sequential(*layers)
        self.model.to(device)
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
        g = (1-t)*torch.sin(torch.pi*x) + x*(1-x) * \
            t*self.model(torch.hstack((x, t)))
        return g

    def Cost(self, X):
        """
        Cost function for our 1D heat model
        """
        x, t = X[:, 0:1], X[:, 1:2]  # , X[:, 2:3]

        x.requires_grad_(True)
        t.requires_grad_(True)

        g = self.trial(x, t)

        u_dt = torch.autograd.grad(
            g, t, grad_outputs=torch.ones_like(g), create_graph=True)[0]
        u_dx = torch.autograd.grad(
            g, x, grad_outputs=torch.ones_like(g), create_graph=True)[0]
        u_dxx = torch.autograd.grad(
            u_dx, x, grad_outputs=torch.ones_like(u_dx), create_graph=True)[0]

        cost = u_dt - u_dxx  # D*u_dxx

        return cost
