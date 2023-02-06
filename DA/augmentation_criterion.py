import torch


def augment_criterion(augment, mode):
    if mode == 'train':
        if augment=='original': return normal_criterion
        elif augment=='MixUp': return mixup_criterion
    else :
        return normal_criterion

def normal_criterion(criterion, y_pred, augment_etc): # augment_etc = target
    return criterion(y_pred, augment_etc[0])

def mixup_criterion(criterion, y_pred, augment_etc): # augment_etc = (target_a, target_b), lam
    return augment_etc[1]*criterion(y_pred, augment_etc[0][0]) + (1 - augment_etc[1])*criterion(y_pred, augment_etc[0][1])