import os
import sys
import argparse

from Trainer import *

def TwoDIC_main(args) :
    print("Hello! We start experiment for 2D Image Classification!")

    dataset_rootdir = os.path.join('.', args.data_path)

    try:
        dataset_dir = os.path.join(dataset_rootdir, args.data_type)
    except TypeError:
        print("join() argument must be str, bytes, or os.PathLike object, not 'NoneType'")
        print("Please explicitely write the dataset type")
        sys.exit()

    args.dataset_dir = dataset_dir
    if args.data_type == 'ImageNet' : ImageNetTrainer(args)
    elif args.data_type == 'CIFAR10' or args.data_type == 'CIFAR100' : CIFARTrainer(args)

if __name__=='__main__' :
    # /media/lord-of-the-gay/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/common material/Dataset Collection
    parser = argparse.ArgumentParser(description='Following are the arguments that can be passed form the terminal itself!')
    parser.add_argument('--data_path', type=str, default='/media/lord-of-the-gay/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/common material/Dataset Collection')
    parser.add_argument('--data_type', type=str, required=True, choices=['ImageNet', 'CIFAR10', 'CIFAR100'])
    parser.add_argument('--model_name', type=str, required=True, choices=['VGG11', 'VGG13', 'VGG16', 'VGG19'])
    parser.add_argument('--num_workers', type=int, default=4, help='number of workers')
    parser.add_argument('--save_path', type=str, default='TwoDIC/TwoDIC_model_weight')
    parser.add_argument('--train', default=False, action='store_true')

    # Train parameter
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--criterion', type=str, default='CCE', choices=['CCE', 'BCE'])
    parser.add_argument('--epochs', type=int, default=1)

    # Optimizer Configuration
    parser.add_argument('--optimizer_name', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-2)

    # Print parameter
    parser.add_argument('--step', type=int, default=10)

    args = parser.parse_args()

    for model_name in ['VGG11', 'VGG13', 'VGG16', 'VGG19'] :
        args.model_name = model_name
        TwoDIC_main(args)