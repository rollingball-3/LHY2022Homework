"""
@author:rollingball
@time:2022/11/23

"""
import os
import torch
import csv
from torch.utils.data import DataLoader

from net.net import HW3_net, Res_Net18, Vgg16_net
from utils import set_seed, train_valid_split, init_weights
from dataset import FoodDataset, test_tfm128, train_tfm128, train_tfm224, test_tfm224, TestDataset
from trainer import trainer
from tqdm import tqdm


def train(config, train_loader, valid_loader, model, train_function, device):
    set_seed(config['seed'])

    model.apply(init_weights)
    model.to(device)

    train_function(train_loader, valid_loader, model, config, device)


def eval_model(config, test_loader, new_model, device):
    new_model.load_state_dict(torch.load(config['model_save_path']))
    new_model.to(device)

    def pad4(i):
        return "0" * (4 - len(str(i))) + str(i)

    i = 0
    with open(os.path.join(config['log_save_path'], 'pred.csv'), 'w') as fp:
        writer = csv.writer(fp)
        writer.writerow(['Id', 'Category'])

        new_model.eval()  # Set your model to evaluation mode.
        for x in tqdm(test_loader):
            x = x.to(device)
            with torch.no_grad():
                pred = new_model(x)
                pred = pred.mean(dim=0)
                results = torch.argmax(pred).cpu().numpy()

                i += 1
                writer.writerow([pad4(i), results])


if __name__ == "__main__":
    # para
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    seed = 664549
    dir_name = "ResNet"
    # dir_name = "dropout"
    name = str(seed)
    config = {
        'seed': seed,  # Your seed number, you can pick your lucky number. :)
        'n_epochs': 400,  # Number of epochs.
        'batch_size': 64,
        'learning_rate': 5e-4,
        'early_stop': 100,  # If model has not improved for this many consecutive epochs, stop training.
        'model_save_path': os.path.join('log', dir_name, name, 'model.ckpt'),  # Your model will be saved here.
        'log_save_path': os.path.join('log', dir_name, name),  # Your log will be saved here
        'current_episode': 0,
    }
    # dataset
    # total_dataset = HW2TrainDataset()
    path = "../../hw3/food11"
    train_set = FoodDataset(os.path.join(path, "training"), tfm=train_tfm128)
    total_dataset = FoodDataset(os.path.join(path, "validation"), tfm=train_tfm128, files=train_set.files)

    train_set, valid_set = train_valid_split(total_dataset, 0.1)
    train_loader = DataLoader(train_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True)
    valid_loader = DataLoader(valid_set, batch_size=config['batch_size'], shuffle=True, pin_memory=True)

    print("We have {} images, {} for train, {} for valid".format(len(total_dataset), len(train_set), len(valid_set)))

    test_set = TestDataset(os.path.join(path, "test"), tfm1=train_tfm128, tfm2=test_tfm128)
    # net

    # model = HW3_net()
    model = Res_Net18()
    # train
    train_function = trainer
    train(config, train_loader, valid_loader, model, train_function, device)

    # eval
    del model
    # model = HW3_net()
    new_model = Res_Net18()

    eval_model(config, test_set, new_model, device)
