import torch
from torch.autograd import Variable

import numpy as np

def augment_forward(augment, mode='train'):
    if mode == 'train':
        if augment=='original': return normal_forward # image, target
        elif augment=='MixUp': return mixup_forward # mixed_image, (target_a, target_b), lam
    else :
        return normal_forward

def normal_forward(image, target) :
    return image, target

def mixup_forward(image, target, alpha=1.0, ) :
    if alpha > 0 : lam = np.random.beta(alpha, alpha)
    else : lam = 1

    batch_size = image.size()[0]

    # if use_cuda : index = torch.randperm(batch_size).cuda()
    index = torch.randperm(batch_size)

    mixed_image = lam * image + (1 - lam) * image[index, :]

    target_a, target_b = target, target[index]
    mixed_image, targets_a, targets_b = map(Variable, (mixed_image, target_a, target_b))
    mixed_target = (targets_a, targets_b)

    return mixed_image, mixed_target, lam