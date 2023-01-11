from time import time

import torch

import numpy as np
from tqdm import tqdm

from ._base import BaseClassificationExperiment
from utils.calculate_metrics import calculate_top1_error, calculate_top5_error

class ImageNetExperiment(BaseClassificationExperiment) :
    def __init__(self, args):
        super(ImageNetExperiment, self).__init__(args)

    def fit(self):
        for epoch in tqdm(range(1, self.args.epochs + 1)):
            print('\n============ EPOCH {}/{} ============\n'.format(epoch, self.args.epochs))

            epoch_start_time = time()

            print("TRAINING")
            train_loss = self.train_epoch(epoch)

            print("EVALUATE")
            test_loss, test_top1_err, test_top5_err = self.val_epoch(epoch)

            total_epoch_time = time() - epoch_start_time
            m, s = divmod(total_epoch_time, 60)
            h, m = divmod(m, 60)

            print('\nEpoch {}/{} : train loss {} | test loss {} | test top1 err {} | test top5 err {} | current lr {} | took {} h {} m {} s'.format(
                epoch, self.args.epochs, train_loss, test_loss, test_top1_err, test_top5_err, self.current_lr(self.optimizer), int(h), int(m), int(s)))

        return self.model, self.optimizer, (test_top1_err, test_top5_err)

    def train_epoch(self, epoch):
        self.model.train()

        running_loss, total = 0., 0

        for batch_idx, (image, target) in enumerate(self.train_loader):
            loss, _, _ = self.forward(image, target)
            self.backward(loss)

            running_loss += loss.item()
            total += image.size(0)

            if (batch_idx + 1) % self.args.step == 0 or (batch_idx + 1) == len(self.train_loader):
                print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                    epoch, batch_idx + 1, len(self.train_loader), np.round((batch_idx + 1) / len(self.train_loader) * 100.0, 2),
                    running_loss / total
                ))

        return running_loss / total

    def val_epoch(self, epoch):
        self.model.eval()

        total_loss, total = .0, 0
        correct_top1, correct_top5 = 0, 0

        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.test_loader):
                if (batch_idx + 1) % self.args.step == 0:
                    print("EPOCH {} | {}/{}({}%) COMPLETE".format(epoch, batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                loss, output, target = self.forward(image, target)

                total_loss += loss.item()
                total += target.size(0)

                correct_top1 += calculate_top1_error(output, target)
                correct_top5 += calculate_top5_error(output, target)

        test_loss = total_loss / total
        test_top1_acc = correct_top1 / total
        test_top5_acc = correct_top5 / total

        return test_loss, 1. - test_top1_acc, 1. - test_top5_acc