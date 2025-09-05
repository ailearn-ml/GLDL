from torch import nn
from methods.template_DG import LDLTemplate_DG
import torch


class DICE_CC(LDLTemplate_DG):
    def __init__(self, loss_func, num_feature, num_classes, latent_dim=100,
                 max_epoch=None, lr=0.1, weight_decay=5e-5, alpha=0.1,
                 adjust_lr=False, device='cuda:0'):
        super(DICE_CC, self).__init__(num_feature=num_feature, num_classes=num_classes,
                                      adjust_lr=adjust_lr, max_epoch=max_epoch, device=device)
        self.lr = lr
        self.lr_now = lr
        self.latent_dim = latent_dim
        self.M_int = Module(hidden_dim=latent_dim)
        self.M_sub = Module(hidden_dim=latent_dim)
        self.f1 = nn.Linear(self.num_feature, latent_dim)
        self.f2 = nn.Linear(latent_dim, self.num_classes)
        self.loss_func = loss_func
        self.alpha = alpha

        params_decay = (p for name, p in self.named_parameters() if 'bias' not in name)
        params_no_decay = (p for name, p in self.named_parameters() if 'bias' in name)
        self.optimizer = torch.optim.AdamW(
            [{'params': params_decay, 'lr': lr, 'weight_decay': weight_decay},
             {'params': params_no_decay, 'lr': lr}], amsgrad=True)
        self.to(self.device)

    def set_forward(self, x):
        x = self.f2(self.f1(x))
        return x

    def set_forward_loss(self, data):
        xs = data[0::2]
        ys = data[1::2]
        loss_preds = []
        loss_ccs = []
        xf1s = []
        # pred
        for i in range(len(xs)):
            # pred
            x, y = xs[i], ys[i]
            xf1 = self.f1(x)
            xf1s.append(xf1)
            loss_preds.append(self.loss_func(self.f2(xf1), y))
        # cc
        for i in range(len(xs)):
            for j in range(i + 1, len(xs)):
                if xf1s[i].shape[0] == xf1s[j].shape[0]:
                    x_int = self.f2(self.M_int(xf1s[i], xf1s[j]))
                    y_int = self.label_normalize((ys[i] + 1e-7) * (ys[j] + 1e-7))
                    loss_int = self.loss_func(x_int, y_int)
                    loss_rec = nn.MSELoss()(self.M_int(xf1s[i], xf1s[j]), self.M_int(xf1s[j], xf1s[i]))
                    x_sub_1 = self.f2(self.M_sub(xf1s[i], xf1s[j]))
                    y_sub_1 = self.label_normalize(
                        torch.maximum(ys[i] - ys[j], torch.zeros_like(ys[i]).to(ys[i].device) + 1e-7))
                    x_sub_2 = self.f2(self.M_sub(xf1s[j], xf1s[i]))
                    y_sub_2 = self.label_normalize(
                        torch.maximum(ys[j] - ys[i], torch.zeros_like(ys[j]).to(ys[j].device) + 1e-7))
                    loss_sub = -self.loss_func(x_sub_1, y_sub_1) - self.loss_func(x_sub_2, y_sub_2)
                    loss_additional = loss_int + loss_rec + loss_sub
                else:
                    loss_additional = torch.tensor(0).to(self.device)
                loss_ccs.append(loss_additional)
        loss_pred = torch.sum(torch.stack(loss_preds))
        loss_cc = torch.sum(torch.stack(loss_ccs))
        return loss_pred + self.alpha * loss_cc

    def label_normalize(self, y):
        y = y / torch.sum(y, dim=1, keepdim=True)
        return y


class Net(nn.Module):
    def __init__(self, n_in, n_out):
        super(Net, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Linear(n_in, n_out),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
        )
        self.layer2 = nn.Linear(n_out, n_out)

    def forward(self, x):
        return self.layer1(x) + self.layer2(self.layer1(x))


class Module(nn.Module):
    def __init__(self, hidden_dim=100):
        super(Module, self).__init__()
        self.linear_block = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        return self.linear_block(x)
