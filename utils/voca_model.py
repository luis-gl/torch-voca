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
        self.encoder_optimizer = torch.optim.Adam(self.speech_encoder.parameters(), lr=config['learning_rate'], betas=(config['adam_beta1_value'], 0.999))

        self.expression_layer = ExpressionLayer(config)
        self.decoder_optimizer = torch.optim.Adam(self.expression_layer.parameters(), lr=config['learning_rate'], betas=(config['adam_beta1_value'], 0.999))

        self.reconstruction_loss = nn.L1Loss()
        self.velocity_loss = VelocityLoss(config=config)
        self.acceleration_loss = AccelerationLoss(config=config)
        self.verts_reg_loss = VertsRegularizerLoss(config=config)


    def forward(self, processed_audio, condition, face_templates):
        features = self.speech_encoder(processed_audio, condition)
        exp_offset = self.expression_layer(features)
        predicted = exp_offset + face_templates

        return predicted