from utils import *
from Experiment.image_classification_experiment import ImageNetExperiment

def ImageNetTrainer(args):
    args.image_size = 224
    args.num_channels = 3
    args.num_classes = 1000
    args.crop_padding = 32

    experiment = ImageNetExperiment(args)
    model, optimizer, history = experiment.fit()

def CIFARTrainer(args):
    args.image_size = 32
    args.num_channels = 3
    if args.data_type == 'CIFAR10' : args.num_classes = 10
    elif args.data_type == 'CIFAR100': args.num_classes = 100
    args.crop_padding = 4

    experiment = ImageNetExperiment(args)
    if args.train:
        model, optimizer, test_results = experiment.fit()
    else :
        test_results = experiment.fit()

        model_dirs = get_save_path(args)

        print("Save {} Model Test Results...".format(args.model_name))
        save_metrics(test_results, model_dirs, args.final_epoch)

        print("Load {} Model History...".format(args.model_name))
        history = load_history(model_dirs)

        print("Plot {} Model History...".format(args.model_name))
        plot_loss_acc(history, model_dirs)

def STL10Trainer(args):
    args.image_size = 96
    args.num_channels = 3
    args.num_classes = 10
    args.crop_padding = 12

    experiment = ImageNetExperiment(args)
    if args.train:
        model, optimizer, test_results = experiment.fit()
