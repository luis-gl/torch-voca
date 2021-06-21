'''
Basado en el código de entrenamiento del modelo original en tensorflow de VOCA,
adaptado a funcionalidades en Pytorch, la mayoría de este código pertenece al autor del repositorio original.
Repositorio original: https://github.com/TimoBolkart/voca
'''

import os
import shutil
import configparser

import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
import torch
from torch import nn

from config_parser import read_config, create_default_config
from utils.data_handler import DataHandler
from utils.batcher import Batcher
from utils.voca_model import VOCAModel
from utils.model_render import ModelRender
from utils.losses import *

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

    return config, batcher

def train_step(config, batcher, model, optimizer, loss_fn_dict, device):
    
    model.train()

    optimizer.zero_grad()

    # Obtener datos para evaluarlos
    processed_audio, face_vertices, face_templates, subject_idx = batcher.get_training_batch(config['batch_size'])
    processed_audio = np.expand_dims(processed_audio, -1)
    face_vertices = np.expand_dims(face_vertices, -1)
    face_templates = np.expand_dims(face_templates, -1)

    # Convertimos los datos a tensores y los movemos al dispositivo
    processed_audio = torch.from_numpy(processed_audio).type(torch.float32).to(device)
    face_vertices = torch.from_numpy(face_vertices).type(torch.float32).to(device)
    face_templates = torch.from_numpy(face_templates).type(torch.float32).to(device)
    subject_idx = torch.from_numpy(subject_idx)

    condition = nn.functional.one_hot(subject_idx, batcher.get_num_training_subjects()).to(device)

    # Procesar los datos en la red para obtener los movimientos de vértices 3D
    predicted, exp_offset = model(processed_audio, condition, face_templates)

    # Calculamos la pérdida
    rec_loss = loss_fn_dict['rec'](predicted, face_vertices)
    vel_loss = loss_fn_dict['vel'](predicted, face_vertices)
    # Según el paper, usan las dos funciones de pérdida anteriores,
    # sin embargo también calculan dos más que tienen peso 0 (modificable en el archivo de configuración)
    acc_loss = loss_fn_dict['accel'](predicted, face_vertices)
    verts_reg_loss = loss_fn_dict['verts'](exp_offset)
    # Los pesos para cada función de pérdida se encuentran en el archivo de configuración
    # Para el reconstruction_loss, se mantiene el peso en 1 siempre
    loss = rec_loss + vel_loss + acc_loss + verts_reg_loss
    
    # Backpropagation y optimización
    loss.backward()
    optimizer.step()

    return loss

def validation_step(config, batcher, model, loss_fn_dict, device):
    
    model.eval()

    num_training_subjects = batcher.get_num_training_subjects()

    with torch.no_grad():

        # Obtener datos para evaluarlos
        processed_audio, face_vertices, face_templates, _ = batcher.get_validation_batch(config['batch_size'])
        processed_audio = np.expand_dims(np.tile(processed_audio, (num_training_subjects, 1, 1)), -1)
        face_vertices = np.expand_dims(np.tile(face_vertices, (num_training_subjects, 1, 1)), -1)
        face_templates = np.expand_dims(np.tile(face_templates, (num_training_subjects, 1, 1)), -1)

        # Convertimos los datos a tensores y los movemos al dispositivo
        processed_audio = torch.from_numpy(processed_audio).type(torch.float32).to(device)
        face_vertices = torch.from_numpy(face_vertices).type(torch.float32).to(device)
        face_templates = torch.from_numpy(face_templates).type(torch.float32).to(device)

        condition = np.reshape(np.repeat(np.arange(num_training_subjects)[:,np.newaxis],
                        repeats=config['num_consecutive_frames']*config['batch_size'], axis=-1), [-1,])
        condition = torch.from_numpy(condition)
        condition = nn.functional.one_hot(condition, batcher.get_num_training_subjects()).to(device)

        # Procesar los datos en la red para obtener los movimientos de vértices 3D
        predicted, exp_offset = model(processed_audio, condition, face_templates)

        # Calculamos la pérdida
        rec_loss = loss_fn_dict['rec'](predicted, face_vertices)
        vel_loss = loss_fn_dict['vel'](predicted, face_vertices)
        # Según el paper, usan las dos funciones de pérdida anteriores,
        # sin embargo también calculan dos más que tienen peso 0 (modificable en el archivo de configuración)
        acc_loss = loss_fn_dict['accel'](predicted, face_vertices)
        verts_reg_loss = loss_fn_dict['verts'](exp_offset)
        # Los pesos para cada función de pérdida se encuentran en el archivo de configuración
        # Para el reconstruction_loss, se mantiene el peso en 1 siempre
        loss = rec_loss + vel_loss + acc_loss + verts_reg_loss
        
        return loss

def train_model(config, batcher, model, optimizer, device, num_epochs, save=False, render=False, model_render=None):
    num_train_batches = batcher.get_num_batches(config['batch_size'], 'train')
    num_val_batches = batcher.get_num_batches(config['batch_size'], 'val')
    # Movemos el modelo al dispositivo
    model = model.to(device)

    train_loss_history = []
    val_loss_history = []

    rec_loss = nn.L1Loss()
    vel_loss = VelocityLoss(config, rec_loss)
    accel_loss = AccelerationLoss(config, rec_loss)
    verts_reg_loss = VertsRegularizerLoss(config)

    loss_fn_dict = {
        'rec': rec_loss,
        'vel': vel_loss,
        'accel': accel_loss,
        'verts': verts_reg_loss
    }

    # Iteramos sobre el número de épocas especificado
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        start = time.time()
        train_loss = 0.0
        val_loss = 0.0

        # Fase de Entrenamiento
        for batch_num in range(num_train_batches):
            train_batch_loss = train_step(config, batcher, model, optimizer, loss_fn_dict, device)
            train_loss += train_batch_loss.item() * config['batch_size']

            #if batch_num % 50 == 0:
                #print(f'Epoch: {epoch} | Iter: {batch_num} | Loss: {train_batch_loss}')
            if batch_num % 100 == 0:
                val_batch_loss = validation_step(config, batcher, model, loss_fn_dict, device)
                val_loss += val_batch_loss * config['batch_size']
                #print(f'Val Loss: {val_batch_loss}')

        if save and epoch % 10 == 0:
            save_model(epoch, model, optimizer, config)
        
        if render and (model_render is not None) and epoch % 20 == 0:
            model_render.render_sequences(model, device, out_folder=os.path.join(config['checkpoint_dir'], 'videos', 'training_epoch_%d_iter_%d' % (epoch, batch_num))
                                       , data_specifier='training')
            model_render.render_sequences(model, device, out_folder=os.path.join(config['checkpoint_dir'], 'videos', 'validation_epoch_%d_iter_%d' % (epoch, batch_num))
                                       , data_specifier='validation')

        epoch_train_loss = train_loss / batcher.get_training_size()
        train_loss_history.append(epoch_train_loss * 1000.0)
        epoch_val_loss = val_loss / batcher.get_validation_size()
        val_loss_history.append(epoch_val_loss * 1000.0)

        print('Train Loss: {:.4f} | Val Loss: {:.4f}'.format(epoch_train_loss * 1000.0, epoch_val_loss * 1000.0))

        end = time.time()
        timed = end - start
        print(f'epoch {epoch}/{num_epochs} took {timed/60.0} minutes')
    
    model_dict = {'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history}
    with open('model_losses.pickle', 'wb') as handle:
            pickle.dump(model_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return model_dict

def save_model(epoch, model, optimizer, config):
    save_path = os.path.join(config['checkpoint_dir'], 'checkpoints')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optimizer.state_dict()
    }, save_path + '/voca_test_checkpoint{}.pt'.format(epoch))

def plot_loss(model_dic, model_name=None, save=False, test=False):
    train_loss_history = model_dic['train_loss_history']
    val_loss_history = model_dic['val_loss_history']
    x_values = range(1, len(train_loss_history) + 1)
    plt.figure(figsize=(7, 5))
    if model_name is None:
        plt.title('Loss')
    else:
        plt.title(model_name + ' Loss')
    plt.plot(x_values, train_loss_history, '-o', label='train')
    plt.plot(x_values, val_loss_history, '-o', label='val')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.show()
    if save:
        if test:
            fname = './plots/{}_losses_test.png'.format(model_name)
        else:
            fname = './plots/{}_losses.png'.format(model_name)
        plt.savefig(fname)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))

    config, batcher = get_train_elements()
    
    model = VOCAModel(config, batcher)
    model_parameters = list(model.speech_encoder.parameters()) + list(model.expression_layer.parameters())
    optimizer = torch.optim.Adam(model_parameters, lr=config['learning_rate'], betas=(config['adam_beta1_value'], 0.999))
    model_render = ModelRender(config, batcher)
    epoch_num = 20 #config['epoch_num']
    model_dict = train_model(config, batcher, model, optimizer, device, epoch_num, save=True, render=True, model_render=model_render)
    plot_loss(model_dict, 'VOCA', save=True, test=True)

if __name__ == '__main__':
    main()