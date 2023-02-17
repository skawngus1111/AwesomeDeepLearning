import torch
import torch.fft as fft
from torch.autograd import Variable

import numpy as np

from .common import rand_bbox

def augment_forward(augment, mode='train'):
    if mode == 'train':
        if augment == 'original': return normal_forward # image, target
        elif augment == 'MixUp':  return mixup_forward # mixed_image, (target_a, target_b), lam
        elif augment == 'CutOut': return cutout_forward
        elif augment == 'CutMix': return cutmix_forward # image, target, lam, prob, cutmix_prob or # image, target, prob, cutmix_prob
        elif augment == 'RICAP':  return ricap_forward # image, target
        elif augment == 'APR':    return apr_forward
    else :
        return normal_forward # image, target

def normal_forward(image, target) :
    return image, target

def mixup_forward(image, target, alpha=1.0) :
    if alpha > 0 : lam = np.random.beta(alpha, alpha)
    else : lam = 1

    batch_size = image.size()[0]
    index = torch.randperm(batch_size)

    mixed_image = lam * image + (1 - lam) * image[index, :]

    target_a, target_b = target, target[index]
    mixed_image, targets_a, targets_b = map(Variable, (mixed_image, target_a, target_b))
    mixed_target = (targets_a, targets_b)

    return mixed_image, mixed_target, lam

def cutout_forward(image, target, cutout_prob=0.5, beta=1.0):
    prob = np.random.rand(1)
    if prob <= cutout_prob:
        lam = np.random.beta(beta, beta)
        bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
        image[:, :, bbx1:bbx2, bby1:bby2] = 0

    return image, target

def cutmix_forward(image, target, cutmix_prob=0.5, beta=1.0):
    prob = np.random.rand(1)
    if prob <= cutmix_prob:
        lam = np.random.beta(beta, beta)
        batch_size = image.size()[0]
        index = torch.randperm(batch_size)

        bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
        image[:, :, bbx1:bbx2, bby1:bby2] = image[index, :, bbx1:bbx2, bby1:bby2]

        target_a, target_b = target, target[index]
        target = (target_a, target_b)

        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))

        return image, target, lam, prob, cutmix_prob

    return image, target

def ricap_forward(image, target, beta=1.0):
    # size of image
    I_x, I_y = image.size()[2:]

    # generate boundary position (w, h)
    w = int(np.round(I_x * np.random.beta(beta, beta)))
    h = int(np.round(I_y * np.random.beta(beta, beta)))
    w_ = [w, I_x - w, w, I_x - w]
    h_ = [h, h, I_y - h, I_y - h]

    # select four images
    cropped_images = {}
    c_ = {}
    W_ = {}
    for k in range(4):
        index = torch.randperm(image.size(0))
        x_k = np.random.randint(0, I_x - w_[k] + 1)
        y_k = np.random.randint(0, I_y - h_[k] + 1)
        cropped_images[k] = image[index][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
        c_[k] = target[index]#.to(image.get_device())
        W_[k] = (w_[k] * h_[k]) / (I_x * I_y)

    # patch cropped images
    image = torch.cat(
        (torch.cat((cropped_images[0], cropped_images[1]), 2),
         torch.cat((cropped_images[2], cropped_images[3]), 2)),
        3)

    target = (c_, W_)
    image = Variable(image)#.to(self.device)

    return image, target

def apr_forward(image, target, apr_prob=0.5):
    prob = np.random.rand(1)
    if prob <= apr_prob:
        batch_size = image.size()[0]
        index = torch.randperm(batch_size)

        fft_1 = fft.fftn(image, dim=(1, 2, 3))
        abs_1, angle_1 = torch.abs(fft_1), torch.angle(fft_1)

        fft_2 = fft.fftn(image[index, :], dim=(1, 2, 3))
        abs_2, angle_2 = torch.abs(fft_2), torch.angle(fft_2)

        fft_1 = abs_2 * torch.exp((1j) * angle_1)

        mixed_x = fft.ifftn(fft_1, dim=(1, 2, 3)).float()
        image = torch.cat([image, mixed_x], 0)

        return image, target, prob, apr_prob

    return image, target