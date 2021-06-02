import torch
from torch import nn

class ExpressionLayer(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.expression_basis_fname = config['expression_basis_fname']
        self.init_expression = config['init_expression']

        self.num_vertices = config['num_vertices']
        self.expression_dim = config['expression_dim']

        self.decoder = nn.Linear(self.expression_dim, 3*self.num_vertices)

    def forward(self, features):
        exp_offset = self.decoder(features)
        exp_offset = torch.reshape(exp_offset, (-1, self.num_vertices, 3, 1))

        return exp_offset