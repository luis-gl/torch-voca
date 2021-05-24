import torch
from torch import nn

class SpeechEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self._speech_encoding_dim = config['expression_dim']
        self._condition_speech_features = config['condition_speech_features']
        self._speech_encoder_size_factor = config['speech_encoder_size_factor']

        # batch normalization para el input de las características obtenidas por DeepSpeech
        self.batch_norm = nn.BatchNorm1d(num_features=29, eps=1e-5, momentum=0.9)

        # en el paper dicen utilizar convoluciones de tiempo pero el código en tensorflow usa conv2d
        self.time_convs = nn.Sequential(
            nn.Conv2d(in_channels=37, out_channels=32, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(1,3), stride=(1,2), padding=(0,1)),
            nn.ReLU(),
        )

        self.flatten = nn.Flatten()

        self.fc_layers = nn.Sequential(
            nn.Linear(72, 128),
            nn.Tanh(),
            nn.Linear(128, self._speech_encoding_dim)
        )

    def forward(self, speech_features, condition):
        speech_features = speech_features.permute(0, 2, 1)
        features_norm = self.batch_norm(speech_features)
        speech_features_reshaped = torch.reshape(features_norm, (-1, features_norm.shape[1], 1, features_norm.shape[2]))

        condition_reshaped = torch.reshape(condition, (-1, condition.shape[1], 1, 1))

        # función equivalente en pytorch a tf.transpose en tensores de n-dimensiones
        speech_feature_condition = condition_reshaped.permute(0, 1, 3, 2)
        speech_feature_condition = torch.tile(speech_feature_condition, (1, 1, 1, features_norm.shape[2]))
        speech_features_reshaped = torch.cat((speech_features_reshaped, speech_feature_condition), dim=1)

        features = self.time_convs(speech_features_reshaped)

        features_flat = self.flatten(features)
        concatenated = torch.cat((features_flat, condition), dim=1)
        fc_result = self.fc_layers(concatenated)

        return fc_result