import torch
from torch.autograd import Variable

def augment_criterion(augment, mode):
    if mode == 'train':
        if augment == 'original': return normal_criterion
        elif augment == 'MixUp':  return mixup_criterion
        elif augment == 'CutOut': return normal_criterion
        elif augment == 'CutMix': return cutmix_criterion
        elif augment == 'RICAP':  return ricap_criterion
        elif augment == 'APR':    return apr_criterion
    else :
        return normal_criterion

def normal_criterion(criterion, y_pred, augment_etc): # augment_etc = target
    return criterion(y_pred, augment_etc[0])

def mixup_criterion(criterion, y_pred, augment_etc): # augment_etc = (target_a, target_b), lam
    return augment_etc[1]*criterion(y_pred, augment_etc[0][0]) + (1 - augment_etc[1])*criterion(y_pred, augment_etc[0][1])

def cutmix_criterion(criterion, y_pred, augment_etc):
    if len(augment_etc) == 4: # (target_a, target_b), lam, prob, cutmix_prob
        return augment_etc[1]*criterion(y_pred, augment_etc[0][0]) + (1 - augment_etc[1])*criterion(y_pred, augment_etc[0][1])
    else: # target
        return criterion(y_pred, augment_etc[0])

def ricap_criterion(criterion, y_pred, augment_etc): # augment_etc = target
    (c_, W_) = augment_etc[0]
    return sum([W_[k] * criterion(y_pred, Variable(c_[k])) for k in range(4)])

def apr_criterion(criterion, y_pred, augment_etc):
    if len(augment_etc) == 3:
        y_true = torch.cat([augment_etc[0], augment_etc[0]], dim=0)
        return criterion(y_pred, y_true)
    return criterion(y_pred, augment_etc[0])