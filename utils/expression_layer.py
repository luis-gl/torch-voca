import numpy as np
import torch
from torch import nn
from utils.ops import FCLayer

class ExpressionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.expression_basis_fname = config['expression_basis_fname']
        self.init_expression = config['init_expression']

        self.num_vertices = config['num_vertices']
        self.expression_dim = config['expression_dim']

        init_exp_basis = np.zeros((3*self.num_vertices, self.expression_dim))

        if self.init_expression:
            init_exp_basis[:, :min(self.expression_dim, 100)] = np.load(self.expression_basis_fname)[:, :min(self.expression_dim, 100)]

        init_exp_basis = torch.from_numpy(init_exp_basis).type(torch.float32)
        
        self.decoder = FCLayer(in_units=self.expression_dim, out_units=3*self.num_vertices, init_weights=init_exp_basis)

    def forward(self, features):
        exp_offset = self.decoder(features)
        exp_offset = torch.reshape(exp_offset, (-1, self.num_vertices, 3, 1))

        return exp_offset