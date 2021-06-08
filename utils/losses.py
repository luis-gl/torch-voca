import torch
from torch import nn

class VelocityLoss(nn.Module):
    def __init__(self, config, rec_loss):
        super().__init__()
        self.config = config
        self.reconstruction_loss = rec_loss

    def forward(self, predicted, target):
        if self.config['velocity_weight'] > 0:
            assert (self.config['num_consecutive_frames'] >= 2)
            verts_predicted = torch.reshape(predicted, (-1, self.config['num_consecutive_frames'], self.config['num_vertices'], 3))
            x1_pred = torch.reshape(verts_predicted[:, -1, :], (-1, self.config['num_vertices'], 3, 1))
            x2_pred = torch.reshape(verts_predicted[:, -2, :], (-1, self.config['num_vertices'], 3, 1))
            velocity_pred = x1_pred - x2_pred

            verts_target = torch.reshape(target, (-1, self.config['num_consecutive_frames'], self.config['num_vertices'], 3))
            x1_target = torch.reshape(verts_target[:, -1, :], (-1, self.config['num_vertices'], 3, 1))
            x2_target = torch.reshape(verts_target[:, -2, :], (-1, self.config['num_vertices'], 3, 1))
            velocity_real = x1_target - x2_target

            velocity_loss = self.config['velocity_weight']*self.reconstruction_loss(velocity_pred, velocity_real)
            return velocity_loss
        else:
            return 0.0

class AccelerationLoss(nn.Module):
    def __init__(self, config, rec_loss):
        super().__init__()
        self.config = config
        self.reconstruction_loss = rec_loss

    def forward(self, predicted, target):
        if self.config['acceleration_weight'] > 0.0:
            assert (self.config['num_consecutive_frames'] >= 3)
            verts_predicted = torch.reshape(predicted, (-1, self.config['num_consecutive_frames'], self.config['num_vertices'], 3))
            x1_pred = torch.reshape(verts_predicted[:, -1, :], (-1, self.config['num_vertices'], 3, 1))
            x2_pred = torch.reshape(verts_predicted[:, -2, :], (-1, self.config['num_vertices'], 3, 1))
            x3_pred = torch.reshape(verts_predicted[:, -3, :], (-1, self.config['num_vertices'], 3, 1))
            acc_pred = x1_pred - 2 * x2_pred + x3_pred

            verts_target = torch.reshape(target, (-1, self.config['num_consecutive_frames'], self.config['num_vertices'], 3))
            x1_target = torch.reshape(verts_target[:, -1, :], (-1, self.config['num_vertices'], 3, 1))
            x2_target = torch.reshape(verts_target[:, -2, :], (-1, self.config['num_vertices'], 3, 1))
            x3_target = torch.reshape(verts_target[:, -3, :], (-1, self.config['num_vertices'], 3, 1))
            acc_target = x1_target - 2 * x2_target + x3_target

            acceleration_loss = self.config['acceleration_weight']*self.reconstruction_loss(acc_pred, acc_target)
            return acceleration_loss
        else:
            return 0.0

class VertsRegularizerLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, expression_offset):
        if self.config['verts_regularizer_weight'] > 0.0:
            verts_regularizer_loss = self.config['verts_regularizer_weight']*torch.mean(torch.sum(torch.abs(expression_offset), dim=2))
            return verts_regularizer_loss
        else:
            return 0.0