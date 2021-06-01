'''
Basado en el código de entrenamiento del modelo original en tensorflow de VOCA,
adaptado a funcionalidades en Pytorch, la mayoría de este código pertenece al autor del repositorio original.
Repositorio original: https://github.com/TimoBolkart/voca
'''

import os
import shutil
import configparser

import numpy as np
import torch
from torch import nn

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

def train_step(config, batcher, model, device):
    # Tamaño del conjunto de datos de entrenamiento
    n = batcher.get_num_batches(config['batch_size'], 'train')
    print(f'Processing training batches, total: {n}')

    # Pérdida de entrenamiento
    train_loss = 0.

    model.train()

    progress = 0
    # Iteramos sobre todos los lotes de entrenamiento usando el batcher
    for batch_num in range(n):

        current_progress = int((float(batch_num+1)/n)*100)
        if progress != current_progress and current_progress % 20 == 0:
            progress = current_progress
            print(f'Processing batch {batch_num+1}/{n}...')

        # Obtener datos para evaluarlos
        processed_audio, face_vertices, face_templates, subject_idx = batcher.get_training_batch(config['batch_size'])

        # Convertimos los datos a tensores y los movemos al dispositivo
        processed_audio = torch.from_numpy(processed_audio).type(torch.float32).to(device)
        face_vertices = torch.from_numpy(face_vertices).type(torch.float32).to(device)
        face_templates = torch.from_numpy(face_templates).type(torch.float32).to(device)
        subject_idx = torch.from_numpy(subject_idx)

        condition = nn.functional.one_hot(subject_idx, batcher.get_num_training_subjects()).to(device)

        # Procesar los datos en la red para obtener los movimientos de vértices 3D
        encoded = model.speech_encoder(processed_audio, condition)
        exp_offset = model.expression_layer(encoded)
        predicted = exp_offset + face_templates

        # Calculamos la pérdida
        rec_loss = model.reconstruction_loss(predicted, face_vertices)
        vel_loss = model.velocity_loss(predicted, face_vertices)
        # Según el paper, usan las dos funciones de pérdida anteriores,
        # sin embargo también calculan dos más que tienen peso 0 (modificable en el archivo de configuración)
        acc_loss = model.acceleration_loss(predicted, face_vertices)
        verts_reg_loss = model.verts_reg_loss(exp_offset)
        # Los pesos para cada función de pérdida se encuentran en el archivo de configuración
        # Para el reconstruction_loss, se mantiene el peso en 1 siempre
        loss = rec_loss + vel_loss + acc_loss + verts_reg_loss
        
        # Backpropagation y optimización
        loss.backward()
        model.encoder_optimizer.zero_grad()
        model.decoder_optimizer.zero_grad()
        model.encoder_optimizer.step()
        model.decoder_optimizer.step()

        # Adicionamos la pérdida de entrenamiento al total  
        train_loss += loss.item()

    # Promedio de función de pérdida y accuracy total
    train_loss /= n

    # Almacenamos todos los resultados en un diccionario
    train_dic = {'train_loss': train_loss}
    return train_dic

def validation_step(config, batcher, model, device):
    # Tamaño del conjunto de datos de validación
    n = batcher.get_num_batches(config['batch_size'], 'val')
    print(f'Processing validation batches, total: {n}')
    num_training_subjects = batcher.get_num_training_subjects()

    # Pérdida y accuracy total de entrenamiento
    val_loss = 0.
    
    model.eval()

    progress = 0
    with torch.no_grad():
        # Iteramos sobre todos los lotes de validación
        for batch_num in range(n):

            current_progress = int((float(batch_num+1)/n)*100)
            if progress != current_progress and current_progress % 20 == 0:
                progress = current_progress
                print(f'Processing batch {batch_num+1}/{n}...')
            
            # Obtener datos para evaluarlos
            processed_audio, face_vertices, face_templates, _ = batcher.get_validation_batch(config['batch_size'])

            # Convertimos los datos a tensores y los movemos al dispositivo
            processed_audio = torch.from_numpy(processed_audio).type(torch.float32).to(device)
            face_vertices = torch.from_numpy(face_vertices).type(torch.float32).to(device)
            face_templates = torch.from_numpy(face_templates).type(torch.float32).to(device)

            conditions = np.reshape(np.repeat(np.arange(num_training_subjects)[:,np.newaxis],
                            repeats=config['num_consecutive_frames']*config['batch_size'], axis=-1), [-1,])
            conditions = torch.from_numpy(conditions)
            conditions = nn.functional.one_hot(conditions, batcher.get_num_training_subjects()).to(device)

            # Procesar los datos en la red para obtener los movimientos de vértices 3D
            encoded = model.speech_encoder(processed_audio, conditions)
            exp_offset = model.expression_layer(encoded)
            predicted = exp_offset + face_templates

            # Calculamos la pérdida
            rec_loss = model.reconstruction_loss(predicted, face_vertices)
            vel_loss = model.velocity_loss(predicted, face_vertices)
            # Según el paper, usan las dos funciones de pérdida anteriores,
            # sin embargo también calculan dos más que tienen peso 0 (modificable en el archivo de configuración)
            acc_loss = model.acceleration_loss(predicted, face_vertices)
            verts_reg_loss = model.verts_reg_loss(exp_offset)
            # Los pesos para cada función de pérdida se encuentran en el archivo de configuración
            # Para el reconstruction_loss, se mantiene el peso en 1 siempre
            loss = rec_loss + vel_loss + acc_loss + verts_reg_loss

            # Adicionamos la pérdida de validación al total
            val_loss += loss.item()

    # Promedio de función de pérdida y accuracy total
    val_loss /= n

    # Almacenamos todos los resultados en un diccionario
    val_dic = {'val_loss': val_loss}
    return val_dic

def train_model(config, batcher, model, device, num_epochs):
    # Histórico de la pérdida por iteración en el entrenamiento
    train_loss_history = []
    val_loss_history = []

    # Movemos el modelo al dispositivo
    model = model.to(device)

    # Iteramos sobre el número de épocas especificado
    for epoch in range(num_epochs):

        # Fase de Entrenamiento
        train_dic = train_step(config, batcher, model, device)

        # Fase de Validación
        val_dic = validation_step(config, batcher, model, device)
    
        # Imprimimos el promedio de la pérdida y el accuracy de la fase 
        # de entrenamiento y validación
        train_loss, val_loss = train_dic['train_loss'], val_dic['val_loss']
        print(f'Época ({epoch+1}/{num_epochs}): ' \
            + f'train_loss = {train_loss:>7f}, val_loss= {val_loss:.8f}')

        # Adicionamos la pérdida la histórico por época
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)

    # Creamos diccionario con información del entrenamiento y validación
    model_dic = {'train_loss_history': train_loss_history,
                'val_loss_history': val_loss_history}

    return model_dic

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Usando {}'.format(device))

    config, data_handler, batcher = get_train_elements()
    
    model = VOCAModel(config=config)
    epoch_num = 1 #config['epoch_num']
    result = train_model(config, batcher, model, device, epoch_num)
       

if __name__ == '__main__':
    main()