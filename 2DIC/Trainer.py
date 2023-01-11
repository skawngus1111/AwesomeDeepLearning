from utils import *
from Experiment.image_classification_experiment import ImageNetExperiment

def ImageNetTrainer(args):
    args.image_size = 224
    args.num_channels = 3
    args.num_classes = 1000

    experiment = ImageNetExperiment(args)
    if args.train:
        model, optimizer, history = experiment.fit()

def CIFARTrainer(args):
    args.image_size = 32
    args.num_channels = 3
    if args.data_type == 'CIFAR10' : args.num_classes = 10
    elif args.data_type == 'CIFAR100': args.num_classes = 100

    experiment = ImageNetExperiment(args)
    if args.train:
        model, optimizer, test_results = experiment.fit()

        save_result(args, model, optimizer, test_results)