from .resample_loss import ResampleLoss


def resample_loss(loss_type='kl', class_freq=None, train_num=None, reduction='mean'):
    loss_type = loss_type.lower()
    if loss_type == 'kl':
        loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
                                 focal=dict(focal=False, alpha=0.5, gamma=2),
                                 logit_reg=dict(),
                                 class_freq=class_freq, train_num=train_num,
                                 reduction=reduction)

    elif loss_type == 'focal':
        loss_func = ResampleLoss(reweight_func=None, loss_weight=1.0,
                                 focal=dict(focal=True, alpha=0.5, gamma=2),
                                 logit_reg=dict(),
                                 class_freq=class_freq, train_num=train_num,
                                 reduction=reduction)
    else:
        raise ValueError('Error loss type!')
    return loss_func
