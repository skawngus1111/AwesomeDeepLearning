import os

import torch

import numpy as np

from .get_functions import get_save_path

def save_result(args, model, optimizer, test_results) :
    model_dirs = get_save_path(args)
    print("Your experiment is saved in {}.".format(model_dirs))

    print("STEP1. Save {} Model Weight...".format(args.model_name))
    save_model(model, optimizer, model_dirs, args.model_name)

    print("STEP2. Save {} Model Test Results...".format(args.model_name))
    save_metrics(test_results, model_dirs)

def save_model(model, optimizer, model_dirs, model_name):
    check_point = {
        'model': model.module if torch.cuda.device_count() > 1 else model,
        'model_name': model_name,
        'model_state_dict': model.module.state_dict() if torch.cuda.device_count() > 1 else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }

    torch.save(check_point, os.path.join(model_dirs, 'model_weight.pth'))

def save_metrics(test_results, model_dirs) :
    top1_error, top5_error = test_results

    print("###################### TEST REPORT ######################")
    print("test TOP1 Error : {}".format(np.round(top1_error, 4)))
    print("test TOP5 Error : {}".format(np.round(top5_error, 4)))
    print("###################### TEST REPORT ######################")

    f = open(os.path.join(model_dirs, 'test report.txt'), 'w')

    f.write("###################### TEST REPORT ######################\n")
    f.write("test TOP1 Error : {}\n".format(np.round(top1_error, 4)))
    f.write("test TOP5 Error : {}\n".format(np.round(top5_error, 4)))
    f.write("###################### TEST REPORT ######################")

    f.close()