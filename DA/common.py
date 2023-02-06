import torch.nn as nn

from .augmentation_forward import *
from .augmentation_criterion import *

class DataAugmentationForward(nn.Module):
    def __init__(self, args, device):
        super(DataAugmentationForward, self).__init__()

        self.args = args
        self.device = device
        self.augment_forward = self.get_augment_forward(args.augment)

    def forward(self, image, target):
        if self.args.distributed: image, target = image.cuda(), target.cuda()
        else: image, target = image.to(self.device), target.to(self.device)

        return self.augment_forward(image, target)

    def get_augment_forward(self, augment):
        if augment == 'original': return normal_forward
        elif augment == 'MixUp': return mixup_forward

class DataAugmentationCriterion(nn.Module):
    def __init__(self, augment='original'):
        super(DataAugmentationCriterion, self).__init__()

        self.augment_criterion = self.get_augment_criterion(augment)

    def forward(self, criterion, output, augment_etc):
        return self.augment_criterion(criterion, output, augment_etc)

    def get_augment_criterion(self, augment):
        if augment == 'original': return normal_criterion
        elif augment == 'MixUp': return mixup_criterion