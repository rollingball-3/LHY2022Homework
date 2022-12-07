"""
@author:rollingball
@time:2022/11/23

4.训练完整逻辑
"""
import torch
import torch.nn as nn
import math

from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def trainer(train_loader, valid_loader, model, config, device):
    # 1、定义损失函数和优化器
    criterion = nn.MSELoss(reduction='mean')  # Define your loss function, do not modify this.
    # optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9, weight_decay=0.001)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=0.001)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.5, last_epoch=-1)
    # 2、定义训练过程监控,以及log和模型的相应保存路径
    writer = SummaryWriter(config['log_save_path'])  # Writer of tensoboard.
    for key in config:
        writer.add_text(key, str(config[key]))

    # 3、初始化训练中的各个参数
    n_epochs, current_episode, batch_size, best_loss, step, early_stop_count = config['n_epochs'], config[
        'current_episode'], config['batch_size'], math.inf, 0, 0

    for epoch in range(current_episode, n_epochs):
        model.train()  # Set your model to train mode.
        loss_record = []
        # tqdm is a package to visualize your training progress.
        train_pbar = tqdm(train_loader, position=0, leave=True)

        # 4、train
        for x, y in train_loader:
            optimizer.zero_grad()  # Set gradient to zero.
            x, y = x.to(device), y.to(device)  # Move your data to device.
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()  # Compute gradient(backpropagation).
            optimizer.step()  # Update parameters.
            step += 1
            loss_record.append(loss.detach().item())

            # Display current epoch number and loss on tqdm progress bar.
            train_pbar.set_description(f'Epoch [{epoch + 1}/{n_epochs}]')
            train_pbar.set_postfix({'loss': loss.detach().item()})

        # optim update
        scheduler.step()

        # 5、add log
        mean_train_loss = sum(loss_record) / len(loss_record)
        writer.add_scalar('Loss/train', mean_train_loss, step)

        # 6、eval on valid_data
        model.eval()  # Set your model to evaluation mode.
        loss_record = []
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                pred = model(x)
                loss = criterion(pred, y)

            loss_record.append(loss.item())

        mean_valid_loss = sum(loss_record) / len(loss_record)
        print(f'Epoch [{epoch + 1}/{n_epochs}]: Train loss: {mean_train_loss:.4f}, Valid loss: {mean_valid_loss:.4f}')
        writer.add_scalar('Loss/valid', mean_valid_loss, step)

        if mean_valid_loss < best_loss:
            best_loss = mean_valid_loss
            torch.save(model.state_dict(), config['model_save_path'])  # Save your best model
            print('Saving model with loss {:.3f}...'.format(best_loss))
            early_stop_count = 0
        else:
            early_stop_count += 1

        if early_stop_count >= config['early_stop']:
            print('\nModel is not improving, so we halt the training session.')
            return
