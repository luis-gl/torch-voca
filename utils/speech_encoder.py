import torch
from torch import nn

from utils.ops import *

class SpeechEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._speech_encoding_dim = config['expression_dim']
        self._condition_speech_features = config['condition_speech_features']
        self._speech_encoder_size_factor = config['speech_encoder_size_factor']

        # batch normalization para el input de las características obtenidas por DeepSpeech
        #self.batch_norm = nn.BatchNorm2d(num_features=29, eps=1e-5, momentum=0.9)
        # En tensorflow, el batchnorm usa el formato [N, H, W, C] y los autores ingresan el tensor de [N, 16, 29, 1] tal cual
        # en tf, se usa decay en vez de momentum, momentum = 1 - decay segun el foro
        # https://discuss.pytorch.org/t/convering-a-batch-normalization-layer-from-tf-to-pytorch/20407
        self.batch_norm = nn.BatchNorm2d(num_features=1, eps=1e-5, momentum=0.1)

        # en el paper dicen utilizar convoluciones de tiempo pero el código en tensorflow usa conv2d
        self.time_convs = nn.Sequential(
            CustomConv2d(in_ch=37, out_ch=32, k_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(), # [128, 32, 8, 1]
            CustomConv2d(in_ch=32, out_ch=32, k_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(), # [128, 32, 4, 1]
            CustomConv2d(in_ch=32, out_ch=64, k_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU(), # [128, 64, 2, 1]
            CustomConv2d(in_ch=64, out_ch=64, k_size=(3,1), stride=(2,1), padding=(1,0)),
            nn.ReLU() # [128, 64, 1, 1]
        )

        self.flatten = nn.Flatten()

        self.fc_layers = nn.Sequential(
            FCLayer(72, 128),
            nn.Tanh(),
            FCLayer(128, self._speech_encoding_dim)
        )

    def forward(self, speech_features, condition):
        #speech_features = speech_features.permute(0,2,1,3)
        speech_features = speech_features.permute(0,3,1,2)
        features_norm = self.batch_norm(speech_features)
        # Regresar a la forma original el dato
        features_norm = features_norm.permute(0, 2, 3, 1)

        speech_features_reshaped = torch.reshape(features_norm, (-1, features_norm.shape[1], 1, features_norm.shape[2]))
        condition_reshaped = torch.reshape(condition, (-1, condition.shape[1], 1, 1)).permute(0, 2, 3, 1)
        #condition_reshaped = torch.reshape(condition, (-1, condition.shape[1], 1, 1))

        # función equivalente en pytorch a tf.transpose en tensores de n-dimensiones
        speech_feature_condition = torch.tile(condition_reshaped, (1, features_norm.shape[1], 1, 1))
        speech_features_reshaped = torch.cat((speech_features_reshaped, speech_feature_condition), dim=-1)
        # transformar el tensor a la forma de pytorch [N, C, H, W]
        speech_features_reshaped = speech_features_reshaped.permute(0, 3, 1, 2)
        #speech_feature_condition = torch.tile(condition_reshaped, (1, 1, features_norm.shape[2], 1))
        #speech_features_reshaped = torch.cat((features_norm, speech_feature_condition), dim=1)

        features = self.time_convs(speech_features_reshaped)

        features_flat = self.flatten(features)
        concatenated = torch.cat((features_flat, condition), dim=1)
        fc_result = self.fc_layers(concatenated)

        return fc_result