from time import time

import torch

import numpy as np
from tqdm import tqdm

from ._base import BaseClassificationExperiment
from common.utils.save_functions import save_result
from common.utils.calculate_metrics import calculate_top1_error, calculate_top5_error

class ImageNetExperiment(BaseClassificationExperiment) :
    def __init__(self, args):
        super(ImageNetExperiment, self).__init__(args)

    def fit(self):
        self.print_params()
        if self.args.train:
            for epoch in tqdm(range(self.args.start_epoch, self.args.final_epoch + 1)):
                print('\n============ EPOCH {}/{} ============\n'.format(epoch, self.args.final_epoch))
                if self.args.distributed: self.train_sampler.set_epoch(epoch)
                if epoch % 10 == 0: self.print_params()
                if self.args.LRS_name == 'MSLRS' or self.args.LRS_name == 'SLRS': self.scheduler.step()
                epoch_start_time = time()

                print("TRAINING")
                train_results = self.train_epoch(epoch)

                print("EVALUATE")
                test_results = self.val_epoch(epoch)

                total_epoch_time = time() - epoch_start_time
                m, s = divmod(total_epoch_time, 60)
                h, m = divmod(m, 60)

                self.history['train_loss'].append(train_results[0])
                self.history['train_top1_acc'].append(1 - train_results[1])
                self.history['train_top5_acc'].append(1 - train_results[2])

                self.history['test_loss'].append(test_results[0])
                self.history['test_top1_acc'].append(1 - test_results[1])
                self.history['test_top5_acc'].append(1 - test_results[2])

                if self.best_top1_err > test_results[1]: self.best_top1_err = test_results[1]
                if self.best_top5_err > test_results[2]: self.best_top5_err = test_results[2]

                print('\nEpoch {}/{} : train loss {} | test loss {} | test top1 err {} % (best top1 err {} %) | test top5 err {} % (best top5 err {} %) | current lr {} | took {} h {} m {} s'.format(
                    epoch, self.args.final_epoch, np.round(train_results[0], 4), np.round(test_results[0], 4),
                    np.round(test_results[1] * 100, 2), np.round(self.best_top1_err * 100, 2), np.round(test_results[2] * 100, 2), np.round(self.best_top5_err * 100, 2),
                    self.current_lr(self.optimizer), int(h), int(m), int(s)))

                if self.args.save_cpt_interval is not None and epoch % self.args.save_cpt_interval == 0 or self.args.final_epoch == epoch:
                    save_result(self.args, self.model, self.optimizer, self.scheduler, self.history, test_results, (self.best_top1_err, self.best_top5_err), epoch)

            return self.model, self.optimizer, test_results
        else :
            print("INFERENCE")
            test_results = self.val_epoch(self.args.final_epoch)

            return test_results

    def train_epoch(self, epoch):
        self.model.train()

        total_loss, total = 0., 0
        correct_top1, correct_top5 = 0, 0

        for batch_idx, (image, target) in enumerate(self.train_loader):
            loss, output, target = self.forward(image, target, mode='train')
            self.backward(loss)

            total_loss += loss.item() * image.size(0)
            total += image.size(0)

            if self.args.augment == 'APR':
                output = output[:image.size(0)]

            correct_top1 += calculate_top1_error(output, target)
            correct_top5 += calculate_top5_error(output, target)

            if (batch_idx + 1) % self.args.step == 0 or (batch_idx + 1) == len(self.train_loader):
                print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                    epoch, batch_idx + 1, len(self.train_loader), np.round((batch_idx + 1) / len(self.train_loader) * 100.0, 2),
                    total_loss / total
                ))

        train_loss = total_loss / total
        train_top1_acc = correct_top1 / total
        train_top5_acc = correct_top5 / total

        return train_loss, 1 - train_top1_acc, 1 - train_top5_acc

    def val_epoch(self, epoch):
        self.model.eval()

        total_loss, total = .0, 0
        correct_top1, correct_top5 = 0, 0

        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.test_loader):
                if (batch_idx + 1) % self.args.step == 0:
                    print("EPOCH {} | {}/{}({}%) COMPLETE".format(epoch, batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                if self.args.final_epoch == epoch : self.start.record()
                loss, output, target = self.forward(image, target, mode='val')
                if self.args.final_epoch == epoch:
                    self.end.record()
                    torch.cuda.synchronize()
                    self.inference_time_list.append(self.start.elapsed_time(self.end))

                total_loss += loss.item() * image.size(0)
                total += target.size(0)

                correct_top1 += calculate_top1_error(output, target)
                correct_top5 += calculate_top5_error(output, target)

        test_loss = total_loss / total
        test_top1_acc = correct_top1 / total
        test_top5_acc = correct_top5 / total

        if self.args.final_epoch == epoch : print("Mean Inference Time (ms) : {} ({})".format(np.mean(self.inference_time_list), np.std(self.inference_time_list)))

        return test_loss, 1 - test_top1_acc, 1 - test_top5_acc