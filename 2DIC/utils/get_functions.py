import os
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import numpy as np

def get_deivce() :
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    device = torch.cuda.current_device()
    print("You are using \"{}\" device.".format(device))

    return torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_dataset(data_dir, data_type) :
    data_dir = os.path.join(data_dir, data_type)
    if data_type == 'ImageNet' :
        train_dir = os.path.join(data_dir, '2012', 'train')
        val_dir = os.path.join(data_dir, '2012', 'val')
        json_file = os.path.join(data_dir, 'imagenet_class_index.json')

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        train_dataset = datasets.ImageFolder(train_dir, transform)
        val_dataset = datasets.ImageFolder(val_dir, transform)

    return train_dataset, val_dataset

def get_model(model_name, linear_node, image_size, num_channels, num_classes) :
    if model_name == 'VGG11' :
        from models.vgg import vgg11
        return vgg11(image_size, linear_node, num_channels, num_classes)
    elif model_name == 'VGG13' :
        from models.vgg import vgg13
        return vgg13(image_size, linear_node, num_channels, num_classes)
    elif model_name == 'VGG16' :
        from models.vgg import vgg16
        return vgg16(image_size, linear_node, num_channels, num_classes)
    elif model_name == 'VGG19' :
        from models.vgg import vgg19
        return vgg19(image_size, linear_node, num_channels, num_classes)
    elif model_name == 'WRN_40_2' :
        from models import wideresnet
        return wideresnet(num_classes, num_channels, depth=40, widen_factor=2)

def get_optimizer(optimizer, model, lr, momentum, weight_decay):
    params = list(filter(lambda p: p.requires_grad, model.parameters()))

    if optimizer == 'SGD' :
        optimizer = optim.SGD(params=params, lr=lr, momentum=momentum, nesterov=True, weight_decay=weight_decay)
    elif optimizer == 'Adam' :
        optimizer = optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
    elif optimizer == 'AdamW' :
        optimizer = optim.AdamW(params=params, lr=lr, weight_decay=weight_decay)
    else :
        print("Wrong optimizer")
        sys.exit()

    return optimizer

def get_lr(step, total_steps, lr_max, lr_min):
  """Compute learning rate according to cosine annealing schedule."""
  return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(step / total_steps * np.pi))

def get_scheduler(LRS_name, optimizer, epochs, train_loader_len, learning_rate) :
    if LRS_name == 'SLRS' : # step learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.1)
    elif LRS_name == 'MSLRS': # multi-step learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[60, 120, 160], gamma=0.2)
    elif LRS_name == 'CALRS': # cosine learning rate scheduler
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lr_lambda=lambda step: get_lr(  # pylint: disable=g-long-lambda
                step,
                epochs * train_loader_len,
                1,  # lr_lambda computes multiplicative factor
                1e-6 / learning_rate))
    else :
        scheduler = None

    return scheduler

def get_criterion(criterion) :
    if criterion == 'CCE' :
        criterion = nn.CrossEntropyLoss()
    elif criterion == 'BCE' :
        criterion = nn.BCEWithLogitsLoss()
    else :
        print("Wrong criterion")
        sys.exit()

    return criterion

def get_save_path(args):
    save_model_path = '{}_{}x{}_{}_{}_{}({}_{})_{}/{}'.format(args.data_type,
                                                              str(args.image_size), str(args.image_size),
                                                              str(args.batch_size),
                                                              args.model_name,
                                                              args.optimizer_name,
                                                              args.lr,
                                                              str(args.final_epoch).zfill(3),
                                                              str(args.LRS_name),
                                                              args.augment)

    model_dirs = os.path.join(args.save_path, save_model_path)
    if not os.path.exists(os.path.join(model_dirs, 'model_weights')): os.makedirs(os.path.join(model_dirs, 'model_weights'))
    if not os.path.exists(os.path.join(model_dirs, 'test_reports')): os.makedirs(os.path.join(model_dirs, 'test_reports'))
    if not os.path.exists(os.path.join(model_dirs, 'plot_results')): os.makedirs(os.path.join(model_dirs, 'plot_results'))

    return model_dirs