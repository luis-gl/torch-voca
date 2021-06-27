'''
Basado en el código de entrenamiento del modelo original en tensorflow de VOCA,
adaptado a funcionalidades en Pytorch, la mayoría de este código pertenece al autor del repositorio original.
Repositorio original: https://github.com/TimoBolkart/voca
'''

from torch import nn
from utils.init_trunc_norm import truncated_normal_

class FCLayer(nn.Module):
    def __init__(self, in_units, out_units, init_weights=None, weightini=0.1, bias=0.0):
        super().__init__()
        self.layer = nn.Linear(in_units, out_units)

        # inicialización de pesos
        if init_weights is not None:
            self.layer.weight.data = init_weights
        elif weightini == 0.0:
            nn.init.constant_(self.layer.weight, weightini)
        else:
            #nn.init.normal_(self.layer.weight, std=weightini)
            self.layer.weight.data = truncated_normal_(self.layer.weight.data, std=weightini)
        
        # inicialización de bias
        nn.init.constant_(self.layer.bias, bias)
    
    def forward(self, x):
        return self.layer(x)

class CustomConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, k_size, stride=(0,0), padding=(0,0), std_dev=0.02):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=k_size, stride=stride, padding=padding)

        # inicialización de pesos y bias
        #nn.init.normal_(self.conv_layer.weight, std=std_dev)
        self.conv_layer.weight.data = truncated_normal_(self.conv_layer.weight.data, std=std_dev)
        #nn.init.normal_(self.conv_layer.bias, std=std_dev)
        self.conv_layer.bias.data = truncated_normal_(self.conv_layer.bias.data, std=std_dev)
    
    def forward(self, x):
        return self.conv_layer(x)