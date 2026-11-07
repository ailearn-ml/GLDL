import torch
import torch.nn as nn
import numpy as np
from abc import abstractmethod
import os
from collections import deque
from utils.metrics import kld


class LDLTemplate_DG(nn.Module):
    def __init__(self, num_feature, num_classes, adjust_lr=False, gradient_clip_value=5.0,
                 max_epoch=None, device='cuda:0'):
        super(LDLTemplate_DG, self).__init__()
        self.num_feature = num_feature
        self.num_classes = num_classes
        self.adjust_lr = adjust_lr
        self.epoch = 0
        self.gradient_clip_value, self.gradient_norm_queue = gradient_clip_value, deque([np.inf], maxlen=5)
        self.max_epoch = max_epoch
        self.device = device

    @abstractmethod
    def set_forward(self, x):
        # x -> predicted score
        pass

    @abstractmethod
    def set_forward_loss(self, data):
        # batch -> loss value
        pass

    def train_loop(self, epoch, train_loaders, log=None, return_acc=False):
        self.train()
        if not log:
            log = print
        self.epoch = epoch
        if self.adjust_lr:
            self.adjust_learning_rate()
        total_KLD = 0
        total_loss = 0
        num_samples = 0
        for batchs in zip(*train_loaders):
            data = []
            for batch in batchs:
                data.append(batch[0].to(self.device))
                data.append(batch[1].to(self.device))
            self.optimizer.zero_grad()
            loss = self.set_forward_loss(data)  # x_1, y_1, x_2, y_2
            loss.backward()
            self.clip_gradient()
            self.optimizer.step()
            total_loss += loss.item()
            with torch.no_grad():
                for i in range(len(data))[::2]:
                    if return_acc:
                        total_KLD += data[i].shape[0] * kld(data[i + 1].detach().cpu().numpy(),
                                                            torch.softmax(self.set_forward(data[i]),
                                                                          dim=1).detach().cpu().numpy())
                    num_samples += data[i].shape[0]


        if return_acc:
            KLD, avg_loss = total_KLD / num_samples, total_loss / num_samples
            return KLD, avg_loss
        else:
            avg_loss = total_loss / num_samples
            return avg_loss

    def val_loop(self, val_loaders, return_loss=False):
        self.eval()
        preds = []
        ys = []
        if return_loss:
            total_loss = 0
        with torch.no_grad():
            for val_loader in val_loaders:
                for batch in val_loader:
                    x = batch[0].to(self.device)
                    y = batch[1].numpy()
                    pred = self.set_forward(x)
                    if return_loss:
                        loss = self.set_forward_loss(pred, y.to(self.device))
                        total_loss += loss.item()
                    pred = torch.softmax(pred, dim=1).detach().cpu().numpy()
                    preds.extend(pred)
                    ys.extend(y)
        preds = np.array(preds)
        ys = np.array(ys)
        if return_loss:
            return preds, ys, total_loss / preds.shape[0]
        else:
            return preds, ys

    def test_loop(self, test_loader, return_loss=False):
        self.eval()
        preds = []
        ys = []
        if return_loss:
            total_loss = 0
        with torch.no_grad():
            for batch in test_loader:
                x = batch[0].to(self.device)
                y = batch[1].numpy()
                pred = self.set_forward(x)
                if return_loss:
                    loss = self.set_forward_loss(pred, y.to(self.device), )
                    total_loss += loss.item()
                pred = torch.softmax(pred, dim=1).detach().cpu().numpy()
                preds.extend(pred)
                ys.extend(y)
        preds = np.array(preds)
        ys = np.array(ys)
        if return_loss:
            return preds, ys, total_loss / preds.shape[0]
        else:
            return preds, ys

    def save(self, path, epoch=None, save_optimizer=False):
        os.makedirs(path, exist_ok=True)
        if type(epoch) is str:
            save_path = os.path.join(path, '%s.tar' % epoch)
        elif epoch is None:
            save_path = os.path.join(path, 'model.tar')
        else:
            save_path = os.path.join(path, '%d.tar' % epoch)
        while True:
            try:
                if not save_optimizer:
                    torch.save({'model': self.state_dict(), }, save_path)
                else:
                    torch.save({'model': self.state_dict(),
                                'optimizer': self.optimizer.state_dict(), }, save_path)
                return
            except:
                pass

    def load(self, path, epoch=None, load_optimizer=False):
        if type(epoch) is str:
            load_path = os.path.join(path, '%s.tar' % epoch)
        else:
            if epoch is None:
                files = os.listdir(path)
                files = np.array(list(map(lambda x: int(x.replace('.tar', '')), files)))
                epoch = np.max(files)
            load_path = os.path.join(path, '%d.tar' % epoch)
        tmp = torch.load(load_path, map_location=self.device)
        self.load_state_dict(tmp['model'])
        if load_optimizer:
            self.optimizer.load_state_dict(tmp['optimizer'])

    def clip_gradient(self):
        if self.gradient_clip_value is not None:
            max_norm = max(self.gradient_norm_queue)
            total_norm = torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm * self.gradient_clip_value)
            self.gradient_norm_queue.append(min(total_norm, max_norm * 2.0, 1.0))

    def adjust_learning_rate(self):
        scale = 10 ** (-((self.epoch + 1) // 10))
        self.lr_now = self.lr * scale
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_now
