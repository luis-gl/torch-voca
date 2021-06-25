import os
import torch

from run_training import get_train_elements
from utils.voca_model import VOCAModel
from utils.model_render import ModelRender

def load_model(epoch):
    config, batcher = get_train_elements()

    path = os.path.join(config['checkpoint_dir'], 'checkpoints')
    if not os.path.exists(path):
        print('Path not found in: ', path)
        return None, None, config, batcher
    checkpoint = torch.load(path + '/voca_test_checkpoint{}.pt'.format(epoch))

    model = VOCAModel(config, batcher)
    model_parameters = list(model.speech_encoder.parameters()) + list(model.expression_layer.parameters())
    optimizer = torch.optim.Adam(model_parameters, lr=config['learning_rate'], betas=(config['adam_beta1_value'], 0.999))

    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optim_state_dict'])

    return model, optimizer, config, batcher

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))
    
    model, optimizer, config, batcher = load_model(epoch=10)
    model.to(device)
    model_render = ModelRender(config, batcher)
    model_render.render_sequences(model, device, out_folder=os.path.join(config['checkpoint_dir'], 'videos', 'training_epoch')
                                , data_specifier='training')
    #model_render.render_sequences(model, device, out_folder=os.path.join(config['checkpoint_dir'], 'videos', 'validation_epoch')
    #                            , data_specifier='validation')

if __name__ == '__main__':
    main()