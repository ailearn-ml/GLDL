from torch import nn
from methods.template_DG import LDLTemplate_DG
import torch
import torch.nn.functional as F


class DICE_VI(LDLTemplate_DG):
    def __init__(self, loss_func, num_feature, num_classes, num_domains,
                 hidden_dim=128, latent_dim=100,
                 lambda1=0.1, lambda2=0.1,
                 max_epoch=None, lr=0.1, weight_decay=5e-5,
                 adjust_lr=False, device='cuda:0'):
        super(DICE_VI, self).__init__(num_feature=num_feature, num_classes=num_classes,
                                      adjust_lr=adjust_lr, max_epoch=max_epoch, device=device)
        self.lambda1 = lambda1
        self.lambda2 = lambda2

        self.lr = lr
        self.lr_now = lr
        self.VariationalModel = Variational(num_classes, self.num_feature, latent_dim=latent_dim,
                                            hidden_dim=hidden_dim)
        self.loss_func = loss_func
        self.pms = nn.Parameter(torch.zeros([num_domains, latent_dim]))
        self.pvs = nn.Parameter(torch.ones([num_domains, latent_dim]))
        params_decay = (p for name, p in self.named_parameters() if 'bias' not in name)
        params_no_decay = (p for name, p in self.named_parameters() if 'bias' in name)
        self.optimizer = torch.optim.AdamW(
            [{'params': params_decay, 'lr': lr, 'weight_decay': weight_decay},
             {'params': params_no_decay, 'lr': lr}], amsgrad=True)
        self.to(self.device)

    def set_forward(self, x):
        x = self.VariationalModel.predict(x)
        return x

    def set_forward_loss(self, data):
        xs = data[0::2]
        ys = data[1::2]
        loss_preds = []
        loss_sums = []
        loss_ps = []
        ps = []
        loss_ys = []
        for i in range(len(xs)):
            # pred
            x, y = xs[i], ys[i]
            pred = self.VariationalModel.predict(x)
            loss_preds.append(self.loss_func(pred, y))
            # variational
            x_encode, x_decode, y_decode, mu, var = self.VariationalModel(x, y)
            loss_dec = self.loss_func(y_decode, y)
            loss_tgt = nn.MSELoss()(x_encode, x_decode)
            loss_var = kl_normal(mu, var, self.pms[i], self.pvs[i]).mean()
            loss_sums.append(loss_tgt + loss_dec + loss_var)
            # correlation
            ps.append(torch.softmax(pred, dim=1))
        for i in range(len(xs)):
            # prior
            for j in range(i + 1, len(xs)):
                loss_ps.append(js_normal(self.pms[i], self.pvs[i], self.pms[j], self.pvs[j]))
                loss_ys.append(nn.MSELoss()(ps[i].t() @ ps[i], ps[j].t() @ ps[j]))
        loss_pred = torch.sum(torch.stack(loss_preds))
        loss_sum = torch.sum(torch.stack(loss_sums))
        loss_p = torch.sum(torch.stack(loss_ps))
        loss_y = torch.sum(torch.stack(loss_ys))
        return loss_pred + self.lambda1 * (loss_sum + loss_p) + self.lambda2 * loss_y


class Variational(nn.Module):
    def __init__(self, num_classes, num_features, latent_dim=100, hidden_dim=200):
        super(Variational, self).__init__()
        self.num_classes = num_classes
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.mlp = Net(n_in=2 * latent_dim, n_hidden=hidden_dim, n_out=hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, latent_dim)
        self.linear2 = nn.Linear(hidden_dim, latent_dim)

        self.encoder_x = nn.Linear(num_features, latent_dim)
        self.encoder_y = nn.Linear(num_classes, latent_dim)
        self.decoder_x = nn.Linear(latent_dim, latent_dim)
        self.decoder_y = nn.Linear(latent_dim, num_classes)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)

    def forward(self, x, y):
        x_encode = self.encoder_x(x)
        y_encode = self.encoder_y(y.float())

        d = self.mlp(torch.cat([x_encode, y_encode], dim=-1))
        mu = self.linear1(d)
        var = F.softplus(self.linear2(d)) + 1e-8

        rand = torch.normal(mean=0., std=1., size=mu.shape).to(x.device)
        z_down = mu + (var ** 0.5) * rand

        x_decode = self.decoder_x(z_down)
        y_decode = self.decoder_y(x_decode)
        return x_encode, x_decode, y_decode, mu, var

    def predict(self, x):
        x_encode = self.encoder_x(x)
        return self.decoder_y(x_encode)


class Net(nn.Module):
    def __init__(self, n_in, n_hidden, n_out):
        super(Net, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.LeakyReLU(),
            nn.Linear(n_hidden, n_out),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        x = self.net(x)
        return x


def kl_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise KL divergence between two normal distributions KL(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    element_wise = 0.5 * (torch.log(pv) - torch.log(qv) + qv / pv + (qm - pm).pow(2) / pv - 1)
    kl = element_wise.sum(-1)
    return kl


def js_normal(qm, qv, pm, pv):
    """
    Computes the elem-wise JS divergence between two normal distributions JS(q || p) and
    sum over the last dimension

    Args:
        qm: tensor: (batch, dim): q mean
        qv: tensor: (batch, dim): q variance
        pm: tensor: (batch, dim): p mean
        pv: tensor: (batch, dim): p variance

    Return:
        kl: tensor: (batch,): kl between each sample
    """
    m = 0.5 * (qm + pm)
    v = 0.5 * (qv + pv)
    js = 0.5 * (kl_normal(qm, qv, m, v) + kl_normal(pm, pv, m, v))
    return js
