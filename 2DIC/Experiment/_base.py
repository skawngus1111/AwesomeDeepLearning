import os
import sys
import random

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import numpy as np

from utils.get_functions import *

class BaseClassificationExperiment(object) :
    def __init__(self, args):
        super(BaseClassificationExperiment, self).__init__()

        self.args = args

        print("STEP1. Load {} Dataset Loader...".format(args.data_type))
        if args.data_type == 'ImageNet' :
            train_dir = os.path.join(args.dataset_dir, '2012', 'train')
            val_dir = os.path.join(args.dataset_dir, '2012', 'val')
            json_file = os.path.join(args.dataset_dir, 'imagenet_class_index.json')

            train_dataset = datasets.ImageFolder(train_dir, self.transform_generator('train'))
            test_dataset = datasets.ImageFolder(val_dir, self.transform_generator('test'))
        elif args.data_type == 'CIFAR10':
            train_dataset = datasets.CIFAR10(args.dataset_dir, train=True, transform=self.transform_generator('train'), download=True)
            test_dataset = datasets.CIFAR10(args.dataset_dir, train=False, transform=self.transform_generator('test'), download=True)
        elif args.data_type == 'CIFAR100':
            train_dataset = datasets.CIFAR100(args.dataset_dir, train=True, transform=self.transform_generator('train'), download=True)
            test_dataset = datasets.CIFAR100(args.dataset_dir, train=False, transform=self.transform_generator('test'), download=True)
        else:
            print("You choose wrong dataset.")
            sys.exit()

        self.train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, worker_init_fn=self.seed_worker)
        self.test_loader = DataLoader(test_dataset, batch_size=args.batch_size , shuffle=False, num_workers=args.num_workers, pin_memory=True, worker_init_fn=self.seed_worker)

        self.args = args
        self.device = get_deivce()
        self.fix_seed(self.device)

        print("STEP2. Load 2D Image Classification Model {}...".format(args.model_name))
        self.model = get_model(args.model_name, args.image_size, args.num_channels, args.num_classes)
        if torch.cuda.device_count() > 1:
            print('Multi GPU activate : ', torch.cuda.device_count())
            self.model = nn.DataParallel(self.model)
        self.model = self.model.to(self.device)

        print("STEP3. Load Optimizer {}...".format(args.optimizer_name))
        self.optimizer = get_optimizer(args.optimizer_name, self.model, args.lr, args.momentum, args.weight_decay)

        print("STEP4. Load Scheduler...")
        self.scheduler = get_scheduler(self.optimizer, args.epochs, len(self.train_loader), args.lr)

        print("STEP5. Load Criterion {}...".format(args.criterion))
        self.criterion = get_criterion(args.criterion).to(self.device)

    def print_params(self):
        print("\ndata type : {}".format(self.args.data_type))
        print("model : {}".format(self.args.model_name))
        print("main optimizer : {}".format(self.optimizer))
        print("epochs : {}".format(self.args.epochs))
        print("learning rate : {}".format(self.args.lr))
        print("loss function : {}".format(self.criterion))
        print("batch size : {}".format(self.args.batch_size))
        print("image size : ({}, {}, {})".format(self.args.image_size, self.args.image_size, self.args.num_channels))
        print("DA method : {}".format(self.args.augment))
        total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("number of trainable parameter : {}".format(total_params))

    def seed_worker(self, worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def fix_seed(self, device):
        random.seed(4321)
        np.random.seed(4321)
        torch.manual_seed(4321)
        torch.cuda.manual_seed(4321)
        torch.cuda.manual_seed_all(4321)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        print("your seed is fixed to '4321'")

    def current_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def _apply_transform(self, image, target):
        image, target = image.to(self.device), target.to(self.device)

        return image, target

    def _calculate_criterion(self, y_pred, y_true):
        loss = self.criterion(y_pred, y_true)

        return loss

    def forward(self, image, target):
        image, target = self._apply_transform(image, target)
        output = self.model(image)
        loss = self._calculate_criterion(output, target)

        return loss, output, target

    def backward(self, loss):
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

    def transform_generator(self, mode):
        if mode == 'train' :
            train_transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

            return transforms.Compose(train_transform_list)
        elif mode == 'test' :
            test_transform_list = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ]

            return transforms.Compose(test_transform_list)