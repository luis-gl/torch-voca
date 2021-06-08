import torch
from torch import nn
from utils.speech_encoder import SpeechEncoder
from utils.expression_layer import ExpressionLayer
from utils.losses import *

class VOCAModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.config = config

        self.speech_encoder = SpeechEncoder(config)
        self.expression_layer = ExpressionLayer(config)

        model_parameters = list(self.speech_encoder.parameters()) + list(self.expression_layer.parameters())
        self.optimizer = torch.optim.Adam(model_parameters, lr=config['learning_rate'], betas=(config['adam_beta1_value'], 0.999))

        self.reconstruction_loss = nn.L1Loss()
        self.velocity_loss = VelocityLoss(config=config)
        self.acceleration_loss = AccelerationLoss(config=config)
        self.verts_reg_loss = VertsRegularizerLoss(config=config)