import os
from torch.utils.data import TensorDataset
from methods.dice_cc import DICE_CC
from utils.losses import resample_loss
from utils.utils import set_seed
from torch.utils.data import DataLoader
from utils.metrics import evaluation, kld
import numpy as np
import torch
from utils.dg_dataset import load_dg_dataset
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='VSD', choices=['VSD', 'FBP', 'MRD'])
parser.add_argument('--loss_type', type=str, default='kl', choices=['kl', 'focal'])
parser.add_argument('--alpha', type=float, default=0.01)
parser.add_argument('--max_epoch', type=int, default=50)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--seed', type=int, default=0)


def get_model():
    model = DICE_CC(loss_func=resample_loss(loss_type=loss_type,
                                            class_freq=np.sum(y_train, axis=0),
                                            train_num=x_train.shape[0]),
                    num_feature=x_tests[0].shape[1],
                    num_classes=y_tests[0].shape[1],
                    max_epoch=max_epoch,
                    lr=lr,
                    alpha=alpha,
                    adjust_lr=adjust_lr,
                    device=device)
    return model


def _train():
    print('Start Training!')
    model = get_model()
    best_KLD = 10
    for epoch in range(max_epoch):
        KLD, avg_loss = model.train_loop(epoch, train_loaders, log=print, return_acc=True)
        print('Epoch %d training done | KLD: %f | Loss: %f | lr: %f' % (epoch, KLD, avg_loss, model.lr_now))
        y_pred, y_test = model.val_loop(val_loaders, return_loss=False)

        result = evaluation(y_test, y_pred)
        print(result)

        KLD = kld(y_test, y_pred)
        if KLD < best_KLD:
            best_KLD = KLD
            model.save(train_dir, epoch='best_KLD')
        if epoch == max_epoch - 1:
            model.save(train_dir, epoch=epoch)


def _test(data_loader, type='test'):
    model = get_model()
    model.load(train_dir, epoch='best_KLD')
    if type == 'val':
        y_pred, y_test = model.val_loop(data_loader)
    else:
        y_pred, y_test = model.test_loop(data_loader)
    result = evaluation(y_test, y_pred)
    for key in result.keys():
        print(f'{key}: {result[key]}')


if __name__ == '__main__':
    args = parser.parse_args()
    lr = args.lr
    batch_size = args.batch_size
    num_workers = args.num_workers
    max_epoch = args.max_epoch
    device = args.device
    dataset = args.dataset
    seed = args.seed
    loss_type = args.loss_type
    alpha = args.alpha

    algorithm = 'dice_cc'
    adjust_lr = True
    set_seed(seed)

    x_trains, x_vals, x_tests, y_trains, y_vals, y_tests = load_dg_dataset(dataset)
    for split in range(len(x_trains)):
        train_dir = os.path.join('save', f'{algorithm}_{loss_type}', dataset, str(split),
                                 f'{lr}_{adjust_lr}_{batch_size}_{max_epoch}_{alpha}_{seed}',
                                 'train')
        log_dir = os.path.join('save', f'{algorithm}_{loss_type}', dataset, str(split),
                               f'{lr}_{adjust_lr}_{batch_size}_{max_epoch}_{alpha}_{seed}',
                               'log')
        XTrain = x_trains[split]
        XTest = x_tests[split]
        XVal = x_vals[split]
        YTrain = y_trains[split]
        YVal = y_vals[split]
        YTest = y_tests[split]

        num_domains = len(XTrain)
        x_train = np.concatenate(XTrain, axis=0)
        y_train = np.concatenate(YTrain, axis=0)

        train_loaders, val_loaders, test_loader = [], [], []
        for i in range(len(XTrain)):
            train_dataset = TensorDataset(torch.from_numpy(XTrain[i]).float(), torch.from_numpy(YTrain[i]))
            train_loaders.append(DataLoader(train_dataset, batch_size=batch_size, shuffle=True))
            val_dataset = TensorDataset(torch.from_numpy(XVal[i]).float(), torch.from_numpy(YVal[i]))
            val_loaders.append(DataLoader(val_dataset, batch_size=batch_size, shuffle=False))
        test_dataset = TensorDataset(torch.from_numpy(XTest).float(), torch.from_numpy(YTest))
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        print(f'{dataset}, {split}, {algorithm}, {loss_type}, {max_epoch}, {seed}')

        if not os.path.exists(os.path.join(train_dir, f'{max_epoch - 1}.tar')):
            _train()
        print('Start Testing!')
        _test(test_loader, type='test')
