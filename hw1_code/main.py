"""
@author:rollingball
@time:2022/11/23

"""
import torch
import os
import csv
from torch.utils.data import Dataset, DataLoader

from net.net import HW1_net
from dataset import HW1Dataset, process_data
from utils import set_seed, train_valid_split, init_weights, predict
from trainer import trainer


def train(config):
    # dataset
    process_x, process_y, test_x = process_data()
    print(process_x.shape)
    print(test_x.shape)

    total_dataset = HW1Dataset(process_x, process_y)
    train_dataset, valid_dataset = train_valid_split(total_dataset, config['valid_ratio'], config['seed'])
    test_dataset = HW1Dataset(test_x)

    print("Train_dataset_len:{}".format(len(train_dataset)))
    print("valid_dataset_len:{}".format(len(valid_dataset)))
    print("test_dataset_len:{}".format(len(test_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, pin_memory=True)

    # net
    model = HW1_net(process_x.shape[1], 1, hidden_dim=config['hidden_dim'], dropout_number=config['dropout'])
    model.apply(init_weights)
    model.to(device)

    # train
    trainer(train_loader, valid_loader, model, config, device)

    # testing
    del model

    def save_pred(preds, file):
        ''' Save predictions to specified file '''
        with open(file, 'w') as fp:
            writer = csv.writer(fp)
            writer.writerow(['id', 'tested_positive'])
            for i, p in enumerate(preds):
                writer.writerow([i, p])

    new_model = HW1_net(input_dim=process_x.shape[1], output_dim=1, hidden_dim=config['hidden_dim'],
                        dropout_number=config['dropout']).to(device)
    new_model.load_state_dict(torch.load(config['model_save_path']))
    preds = predict(test_loader, new_model, device)
    save_pred(preds, os.path.join(config['log_save_path'], 'pred.csv'))


if __name__ == "__main__":
    # para
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dir_name = "hidden_dim"
    name = "64"
    seed = 697428
    config = {
        'seed': seed,  # Your seed number, you can pick your lucky number. :)
        'valid_ratio': 0.1,  # validation_size = train_size * valid_ratio
        'n_epochs': 5000,  # Number of epochs.
        'batch_size': 128,
        'learning_rate': 5e-5,
        'early_stop': 400,  # If model has not improved for this many consecutive epochs, stop training.
        'model_save_path': os.path.join('log', dir_name, name, 'model.ckpt'),  # Your model will be saved here.
        'log_save_path': os.path.join('log', dir_name, name),  # Your log will be saved here
        'current_episode': 0,
        'hidden_dim': 64,
        'dropout': 0.25,
    }
    set_seed(seed)
    print(config)
    train(config)
