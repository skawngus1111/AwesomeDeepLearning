import os

import torch

import pandas as pd

from common.utils.get_functions import get_save_path

def load_model(args, model) :
    model_dirs = get_save_path(args)

    load_path = os.path.join(model_dirs, 'model_weights/model_weight(EPOCH {}).pth.tar'.format(args.final_epoch))

    print("Your model is loaded from {}.".format(load_path))
    checkpoint = torch.load(load_path)
    print(".pth.tar keys() =  {}.".format(checkpoint.keys()))

    model.load_state_dict(checkpoint['model_state_dict'])
    args.start_epoch = checkpoint['current_epoch']

    return model

def load_history(model_dirs):
    load_path = os.path.join(model_dirs, 'loss.csv')
    history = pd.read_csv(load_path)

    return history