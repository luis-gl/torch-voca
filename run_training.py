'''
Basado en el código de entrenamiento del modelo original en tensorflow de VOCA,
adaptado a funcionalidades en Pytorch, la mayoría de este código pertenece al autor del repositorio original.
Repositorio original: https://github.com/TimoBolkart/voca
'''

import os
import shutil
import configparser

import torch

from config_parser import read_config, create_default_config
from utils.data_handler import DataHandler
from utils.batcher import Batcher
from utils.voca_model import VOCAModel

def get_train_elements():
    # Prior to training, please adapt the hyper parameters in the config_parser.py and run the script to generate
    # the training config file use to train your own VOCA model.

    pkg_path, _ = os.path.split(os.path.realpath(__file__))
    init_config_fname = os.path.join(pkg_path, 'training_config.cfg')
    if not os.path.exists(init_config_fname):
        print('Config not found %s' % init_config_fname)
        create_default_config(init_config_fname)

    config = configparser.ConfigParser()
    config.read(init_config_fname)

    # Path to cache the processed audio
    config.set('Input Output', 'processed_audio_path', './training_data/processed_audio_%s.pkl' % config.get('Audio Parameters', 'audio_feature_type'))

    checkpoint_dir = config.get('Input Output', 'checkpoint_dir')
    if os.path.exists(checkpoint_dir):
        print('Checkpoint dir already exists %s' % checkpoint_dir)
        key = input('Press "q" to quit, "x" to erase existing folder, and any other key to continue training: ')
        if key.lower() == 'q':
            return
        elif key.lower() == 'x':
            try:
                shutil.rmtree(checkpoint_dir, ignore_errors=True)
            except:
                print('Failed deleting checkpoint directory')

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    config_fname = os.path.join(checkpoint_dir, 'config.pkl')
    if os.path.exists(config_fname):
        print('Use existing config %s' % config_fname)
    else:
        with open(config_fname, 'w') as fp:
            config.write(fp)
            fp.close()

    config = read_config(config_fname)
    data_handler = DataHandler(config)
    batcher = Batcher(data_handler)

    return config, data_handler, batcher

def train_model():
    pass

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Usando {}'.format(device))

    config, data_handler, batcher = get_train_elements()
    
    model = VOCAModel(config=config)
       

if __name__ == '__main__':
    main()