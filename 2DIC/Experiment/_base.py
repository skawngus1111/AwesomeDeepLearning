import os
import sys
import random

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import yaml
import numpy as np
from ptflops import get_model_complexity_info

from utils.get_functions import *
from utils.load_functions import *
from DA.augmentation_forward import augment_forward
from DA.augmentation_criterion import augment_criterion

class BaseClassificationExperiment(object) :
    def __init__(self, args):
        super(BaseClassificationExperiment, self).__init__()

        self.args = args
        self.device = get_deivce()
        self.fix_seed(self.device)
        self.history_generator()
        self.scaler = torch.cuda.amp.GradScaler()
        self.start, self.end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        self.inference_time_list = []
        # self.configuration = yaml.safe_load(open(f'TwoDIC/configuration_files/configuration.yml'))

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
        elif args.data_type == 'STL10':
            train_dataset = datasets.STL10(args.dataset_dir, split='train', transform=self.transform_generator('train'), download=True)
            test_dataset = datasets.STL10(args.dataset_dir, split='test', transform=self.transform_generator('test'), download=True)
        else:
            print("You choose wrong dataset.")
            sys.exit()

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if args.distributed else None
        self.train_loader = DataLoader(train_dataset,
                                       batch_size=int(args.batch_size / args.ngpus_per_node) if args.distributed else args.batch_size,
                                       shuffle=(self.train_sampler is None),
                                       num_workers=int((args.num_workers+args.ngpus_per_node-1)/args.ngpus_per_node) if args.distributed else args.num_workers,
                                       pin_memory=True, sampler=self.train_sampler)
        self.test_loader = DataLoader(test_dataset,
                                      batch_size=int(args.batch_size / args.ngpus_per_node) if args.distributed else args.batch_size,
                                      shuffle=False,
                                      num_workers=int((args.num_workers+args.ngpus_per_node-1)/args.ngpus_per_node) if args.distributed else args.num_workers,
                                      pin_memory=True)

        print("STEP2. Load 2D Image Classification Model {}...".format(args.model_name))
        self.model = get_model(args.model_name, args.linear_node, args.image_size, args.num_channels, args.num_classes)


        # multiprocess 설정
        if args.distributed:
            print('Multi GPU activate : {} with DP'.format(torch.cuda.device_count()))
            if args.gpu is not None:
                torch.cuda.set_device(args.gpu)
                self.model.cuda(args.gpu)
                # when using a single GPU per process and per DDP, we need to divide tha batch size
                # ourselves based on the total number of GPUs we have
                self.model = nn.parallel.DistributedDataParallel(self.model, device_ids=[args.gpu])
            else:
                self.model.cuda()
                # DDP will divide and allocate batch_size to all available GPUs if device_ids are not set
                # 만약에 device_ids를 따로 설정해주지 않으면, 가능한 모든 gpu를 기준으로 ddp가 알아서 배치사이즈와 workers를 나눠준다는 뜻.
                self.model = torch.nn.parallel.DistributedDataParallel(self.model)
        elif not args.distributed and torch.cuda.device_count() > 1:
            print('Multi GPU activate : {}'.format(torch.cuda.device_count()))
            self.model = nn.DataParallel(self.model)
            self.model = self.model.to(self.device)
        else:
            self.model = self.model.to(self.device)

        print("STEP3. Load Optimizer {}...".format(args.optimizer_name))
        self.optimizer = get_optimizer(args.optimizer_name, self.model, args.lr, args.momentum, args.weight_decay)

        print("STEP4. Load LRS {}...".format(args.LRS_name))
        self.scheduler = get_scheduler(args.LRS_name, self.optimizer, args.final_epoch, len(self.train_loader), args.lr)

        print("STEP5. Load Criterion {}...".format(args.criterion))
        self.criterion = get_criterion(args.criterion)

        if args.distributed: self.criterion.cuda(args.gpu)

    def print_params(self):
        print("\ndata type : {}".format(self.args.data_type))
        print("model : {}".format(self.args.model_name))
        print("optimizer : {}".format(self.optimizer))
        print("learning rate : {}".format(self.args.lr))
        print("learning rate scheduler : {}".format(self.args.LRS_name))
        print("start epoch : {}".format(self.args.start_epoch))
        print("final epoch : {}".format(self.args.final_epoch))
        print("criterion : {}".format(self.criterion))
        print("batch size : {}".format(self.args.batch_size))
        print("image size : ({}, {}, {})".format(self.args.image_size, self.args.image_size, self.args.num_channels))
        print("DA : {}".format(self.args.augment))
        # print('{:<30}  {:<8}'.format('Computational complexity (MAC): ', self.complexity[0]))
        # print('{:<30}  {:<8}'.format('Number of parameters: ', self.complexity[1]))

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
        if self.args.reproducibility :
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        print("your seed is fixed to '4321' with reproducibility {}".format(self.args.reproducibility))

    def current_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def forward(self, image, target, mode):
        if self.args.distributed: image, target = image.cuda(), target.cuda()
        else: image, target = image.to(self.device), target.to(self.device)

        augment_ = augment_forward(self.args.augment, mode)(image, target) # image, target, etc ...
        with torch.cuda.amp.autocast():
            output = self.model(augment_[0])
            loss = augment_criterion(self.args.augment, mode)(self.criterion, output, augment_[1:]) # target, etc ...

        return loss, output, target

    def backward(self, loss):
        self.optimizer.zero_grad()
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.args.LRS_name == 'CALRS': self.scheduler.step()

    def transform_generator(self, mode):
        if mode == 'train' :
            train_transform_list = [
                transforms.RandomHorizontalFlip(),
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.RandomCrop(self.args.image_size, padding=self.args.crop_padding),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]

            return transforms.Compose(train_transform_list)
        elif mode == 'test' :
            test_transform_list = [
                transforms.Resize((self.args.image_size, self.args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ]

            return transforms.Compose(test_transform_list)

    def history_generator(self):
        self.history = dict()
        self.history['train_loss'] = list()
        self.history['train_top1_acc'] = list()
        self.history['train_top5_acc'] = list()

        self.history['test_loss'] = list()
        self.history['test_top1_acc'] = list()
        self.history['test_top5_acc'] = list()