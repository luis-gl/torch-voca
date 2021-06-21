from torch import nn
from utils.speech_encoder import SpeechEncoder
from utils.expression_layer import ExpressionLayer

class VOCAModel(nn.Module):
    def __init__(self, config, batcher):
        super().__init__()
        
        self.config = config
        self.batcher = batcher
        self.num_render_sequences = config['num_render_sequences']

        self.speech_encoder = SpeechEncoder(config)
        self.expression_layer = ExpressionLayer(config)
    
    def forward(self, processed_audio, condition, face_template):
        encoded = self.speech_encoder(processed_audio, condition)
        exp_offset = self.expression_layer(encoded)
        predicted = exp_offset + face_template
        return predicted, exp_offset