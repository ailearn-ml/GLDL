from torch import nn
from methods.template_DG import LDLTemplate_DG
import torch


class DVC(LDLTemplate_DG):
    def __init__(self, loss_func, num_feature, num_classes, latent_dim=100,
                 max_epoch=None, lr=0.1, weight_decay=5e-5, alpha=0.1,
                 adjust_lr=False, device='cuda:0'):
        super(DVC, self).__init__(num_feature=num_feature, num_classes=num_classes,
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
        # self.loss_func = nn.CrossEntropyLoss()

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
        if len(data) == 4:
            x_1, y_1, x_2, y_2 = data
            x_1, x_2 = self.f1(x_1), self.f1(x_2)
            loss_pred = self.loss_func(self.f2(x_1), y_1) + self.loss_func(self.f2(x_2), y_2)
            if x_1.shape[0] == x_2.shape[0]:
                x_int = self.f2(self.M_int(x_1, x_2))
                y_int = self.label_normalize(y_1 * y_2)
                loss_int = self.loss_func(x_int, y_int)
                loss_rec = nn.MSELoss()(self.M_int(x_1, x_2), self.M_int(x_2, x_1))
                x_sub_1 = self.f2(self.M_sub(x_1, x_2))
                y_sub_1 = self.label_normalize(torch.maximum(y_1 - y_2, torch.zeros_like(y_1).to(y_1.device) + 1e-7))
                x_sub_2 = self.f2(self.M_sub(x_2, x_1))
                y_sub_2 = self.label_normalize(torch.maximum(y_2 - y_1, torch.zeros_like(y_2).to(y_2.device) + 1e-7))
                loss_sub = -self.loss_func(x_sub_1, y_sub_1) \
                           - self.loss_func(x_sub_2, y_sub_2)
                loss_additional = loss_int + loss_rec + loss_sub
            else:
                loss_additional = 0
            return loss_pred + self.alpha * loss_additional
        elif len(data) == 6:
            x_1, y_1, x_2, y_2, x_3, y_3 = data
            x_1, x_2, x_3 = self.f1(x_1), self.f1(x_2), self.f1(x_3)
            loss_pred = self.loss_func(self.f2(x_1), y_1) \
                        + self.loss_func(self.f2(x_2), y_2) \
                        + self.loss_func(self.f2(x_3), y_3)
            if x_1.shape[0] == x_2.shape[0] == x_3.shape[0]:
                x_int_a = self.f2(self.M_int(x_1, x_2))
                y_int_a = self.label_normalize(torch.minimum(y_1, y_2) + 1e-7)
                x_int_b = self.f2(self.M_int(x_1, x_3))
                y_int_b = self.label_normalize(torch.minimum(y_1, y_3) + 1e-7)
                x_int_c = self.f2(self.M_int(x_2, x_3))
                y_int_c = self.label_normalize(torch.minimum(y_2, y_3) + 1e-7)
                loss_int = self.loss_func(x_int_a, y_int_a) \
                           + self.loss_func(x_int_b, y_int_b) \
                           + self.loss_func(x_int_c, y_int_c)
                loss_rec = nn.MSELoss()(self.M_int(x_1, x_2), self.M_int(x_2, x_1)) \
                           + nn.MSELoss()(self.M_int(x_1, x_3), self.M_int(x_3, x_1)) \
                           + nn.MSELoss()(self.M_int(x_2, x_3), self.M_int(x_3, x_2))
                x_sub_a_1 = self.f2(self.M_sub(x_1, x_2))
                y_sub_a_1 = self.label_normalize(torch.maximum(y_1 - y_2, torch.zeros_like(y_1).to(y_1.device) + 1e-7))
                x_sub_a_2 = self.f2(self.M_sub(x_2, x_1))
                y_sub_a_2 = self.label_normalize(torch.maximum(y_2 - y_1, torch.zeros_like(y_2).to(y_2.device) + 1e-7))

                x_sub_b_1 = self.f2(self.M_sub(x_1, x_3))
                y_sub_b_1 = self.label_normalize(torch.maximum(y_1 - y_3, torch.zeros_like(y_1).to(y_1.device) + 1e-7))
                x_sub_b_2 = self.f2(self.M_sub(x_3, x_1))
                y_sub_b_2 = self.label_normalize(torch.maximum(y_3 - y_1, torch.zeros_like(y_3).to(y_3.device) + 1e-7))

                x_sub_c_1 = self.f2(self.M_sub(x_2, x_3))
                y_sub_c_1 = self.label_normalize(torch.maximum(y_2 - y_3, torch.zeros_like(y_2).to(y_2.device) + 1e-7))
                x_sub_c_2 = self.f2(self.M_sub(x_3, x_2))
                y_sub_c_2 = self.label_normalize(torch.maximum(y_3 - y_2, torch.zeros_like(y_3).to(y_3.device) + 1e-7))
                loss_sub = -self.loss_func(x_sub_a_1, y_sub_a_1) \
                           - self.loss_func(x_sub_a_2, y_sub_a_2) \
                           - self.loss_func(x_sub_b_1, y_sub_b_1) \
                           - self.loss_func(x_sub_b_2, y_sub_b_2) \
                           - self.loss_func(x_sub_c_1, y_sub_c_1) \
                           - self.loss_func(x_sub_c_2, y_sub_c_2)
                loss_additional = loss_int + loss_rec + loss_sub
            else:
                loss_additional = 0
            return loss_pred + self.alpha * loss_additional

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
