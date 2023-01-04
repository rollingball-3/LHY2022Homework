"""
@author:rollingball
@time:2022/11/23

"""
import os
import torch
import csv
from torch.utils.data import Dataset, DataLoader

from net.net import HW2_net, HW2RNN_net
from dataset import HW2TestDataset, HW2TrainDataset, HW2RNNTestDataset, HW2RNNTrainDataset
from utils import set_seed, train_valid_split, init_weights
from trainer import trainer, Rnn_trainer
from tqdm import tqdm


def train(config, total_dataset, model, train_function, device):
    set_seed(config['seed'])

    train_dataset, valid_dataset = train_valid_split(total_dataset, config['valid_ratio'], config['seed'])

    print("Train_dataset_len:{}".format(len(train_dataset)))
    print("valid_dataset_len:{}".format(len(valid_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=config['batch_size'], shuffle=True, pin_memory=True, num_workers=4)

    model.apply(init_weights)
    model.to(device)

    train_function(train_loader, valid_loader, model, config, device)


def eval_model(config, test_dataset, new_model, device):
    new_model.load_state_dict(torch.load(config['model_save_path']))
    new_model.to(device)

    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False, pin_memory=True)

    i = 0
    with open(os.path.join(config['log_save_path'], 'pred.csv'), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Id', 'Class'])

        new_model.eval()  # Set your model to evaluation mode.
        for x in tqdm(test_loader):
            x = x.to(device)
            with torch.no_grad():
                pred = new_model(x)
                results = torch.argmax(pred, dim=1).cpu().numpy()
                for result in results:
                    writer.writerow([i, result])
                    i += 1


if __name__ == "__main__":
    # para
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 697444
    # dir_name = "RNN_normalize"
    dir_name = "DNN"
    name = str(seed)
    config = {
        'seed': seed,  # Your seed number, you can pick your lucky number. :)
        'valid_ratio': 0.1,  # validation_size = train_size * valid_ratio
        'n_epochs': 100,  # Number of epochs.
        'batch_size': 512,
        'learning_rate': 5e-4,
        'early_stop': 40,  # If model has not improved for this many consecutive epochs, stop training.
        'model_save_path': os.path.join('log', dir_name, name, 'model.ckpt'),  # Your model will be saved here.
        'log_save_path': os.path.join('log', dir_name, name),  # Your log will be saved here
        'current_episode': 0,
        'batch_normalize': True,
        'dropout': 0.25,
        'RNN_output_dim': 128,
        'RNN_layers_number': 2,
    }
    # dataset
    total_dataset = HW2TrainDataset()
    # total_dataset = HW2RNNTrainDataset()
    test_dataset = HW2TestDataset()
    # test_dataset = HW2RNNTestDataset()

    # net
    model = HW2_net(11 * 39, 41, use_BatchNorm=config['batch_normalize'], dropout=config['dropout'])
    # model = HW2RNN_net(39, output_dim=config['RNN_output_dim'], layers_number=config['RNN_layers_number'],
    #                    dropout=config['dropout'], batch_norm=config['batch_normalize'])

    # train
    # train_function = Rnn_trainer
    train_function = trainer
    train(config, total_dataset, model, train_function, device)

    # eval
    del model
    new_model = HW2_net(11 * 39, 41, use_BatchNorm=config['batch_normalize'], dropout=config['dropout'])
    # new_model = HW2RNN_net(39, output_dim=config['RNN_output_dim'], layers_number=config['RNN_layers_number'],
    #                        dropout=config['dropout'], batch_norm=config['batch_normalize'])
    eval_model(config, test_dataset, new_model, device)
