#-*- coding:utf-8 -*-
# Distributed Data Parallel 코드 https://dbwp031.tistory.com/32
import os
import sys
import argparse
import builtins

import torch
import torch.multiprocessing as mp

from Experiment.image_classification_experiment import ImageNetExperiment
# puzzle = __import__('8puzzle')

def IC2D_main(args) :
    print("Hello! We start experiment for 2D Image Classification!")
    print("Distributed Data Parallel {}".format(args.multiprocessing_distributed))

    dataset_rootdir = os.path.join('.', args.data_path)

    try:
        dataset_dir = os.path.join(dataset_rootdir, args.data_type)
    except TypeError:
        print("join() argument must be str, bytes, or os.PathLike object, not 'NoneType'")
        print("Please explicitely write the dataset type")
        sys.exit()

    args.dataset_dir = dataset_dir
    if args.data_type == 'ImageNet':
        args.image_size = 224
        args.num_channels = 3
        args.num_classes = 1000
        args.crop_padding = 32
    elif args.data_type == 'CIFAR10' or args.data_type == 'CIFAR100':
        args.image_size = 32
        args.num_channels = 3
        if args.data_type == 'CIFAR10': args.num_classes = 10
        elif args.data_type == 'CIFAR100': args.num_classes = 100
        args.crop_padding = 4
    elif args.data_type == 'STL10':
        args.image_size = 96
        args.num_channels = 3
        args.num_classes = 10
        args.crop_padding = 12

    args.distributed = False
    if args.multiprocessing_distributed and args.train:
        args.distributed = args.world_size > 1 or args.multiprocessing_distributed
        ngpus_per_node = torch.cuda.device_count()
        args.world_size = ngpus_per_node * args.world_size

        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else :
        experiment = ImageNetExperiment(args)
        experiment.fit()

def main_worker(gpu,ngpus_per_node, args):
    # 내용1 :gpu 설정
    args.gpu = gpu
    args.ngpus_per_node = ngpus_per_node

    if args.multiprocessing_distributed and args.gpu !=0:
        def print_pass(*args):
            pass
        builtins.print=print_pass

    if args.gpu is not None: print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url=='env://' and args.rank==-1:
            args.rank=int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            args.rank=args.rank*ngpus_per_node + gpu #gpu None아님?
        torch.distributed.init_process_group(backend=args.dist_backend,init_method=args.dist_url,
                                            world_size=args.world_size,rank=args.rank)

    experiment = ImageNetExperiment(args)
    if args.train: experiment.fit()

if __name__=='__main__' :
    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself!')
    parser.add_argument('--data_path', type=str, default='/media/kds/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/common material/Dataset Collection')
    parser.add_argument('--data_type', type=str, required=True, choices=['ImageNet', 'CIFAR10', 'CIFAR100', 'STL10'])
    parser.add_argument('--model_name', type=str, required=True, choices=['VGG11', 'VGG13', 'VGG16', 'VGG19',
                                                                          'ResNet_18', 'ResNet_34', 'ResNet_50', 'ResNet_101',
                                                                          'DenseNet_121',
                                                                          'WRN_28_10', 'WRN_40_2'])
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--save_path', type=str, default='TwoDIC/TwoDIC_model_weight')
    parser.add_argument('--save_cpt_interval', type=int, default=None)
    parser.add_argument('--train', default=False, action='store_true')
    parser.add_argument('--reproducibility', default=False, action='store_true')

    # Multi-Processing parameters
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    parser.add_argument('--world-size', default=-1, type=int, help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int, help='node rank for distributed training')
    parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str, help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str, help='distributed backend')

    # Train parameter
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--criterion', type=str, default='CCE', choices=['CCE', 'BCE'])
    parser.add_argument('--start_epoch', type=int, default=1)
    parser.add_argument('--final_epoch', type=int, default=200)
    parser.add_argument('--linear_node', type=int, default=4096)
    parser.add_argument('--augment', type=str, default='original', choices=['original', 'MixUp', 'CutOut', 'CutMix', 'RICAP'])

    # Optimizer Configuration
    parser.add_argument('--optimizer_name', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=1e-1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0005)

    # Learning Rate Scheduler (LRS) Configuration
    parser.add_argument('--LRS_name', type=str, default=None)

    # Print parameter
    parser.add_argument('--step', type=int, default=10)

    args = parser.parse_args()

    IC2D_main(args)