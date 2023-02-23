from time import time

import torch

import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from ._SC1Dbase import BaseSignalClassificationExperiment

class SignalClassificationExperiment(BaseSignalClassificationExperiment):
    def __init__(self, args):
        super(SignalClassificationExperiment, self).__init__(args)

    def fit(self):
        self.print_params()
        if self.args.train:
            for epoch in tqdm(range(self.args.start_epoch, self.args.final_epoch + 1)):
                print('\n============ EPOCH {}/{} ============\n'.format(epoch, self.args.final_epoch))
                if self.args.distributed: self.train_sampler.set_epoch(epoch)

                epoch_start_time = time()

                print("TRAINING")
                train_results = self.train_epoch(epoch)

                print("EVALUATE")
                val_results = self.val_epoch(epoch)

                self.history['train_loss'].append(train_results)
                self.history['val_loss'].append(val_results)

                total_epoch_time = time() - epoch_start_time
                m, s = divmod(total_epoch_time, 60)
                h, m = divmod(m, 60)

                print('\nEpoch {}/{} : train loss {} | val loss {} | current lr {} | took {} h {} m {} s'.format(
                    epoch, self.args.final_epoch, np.round(train_results, 4), np.round(val_results, 4),
                    self.current_lr(self.optimizer), int(h), int(m), int(s)))

        print("INFERENCE")
        test_results = self.inference(self.args.final_epoch)

        return self.model, self.optimizer, self.scheduler, self.history, test_results, self.metric_list

    def train_epoch(self, epoch):
        self.model.train()

        total_loss, total = 0., 0

        for batch_idx, (signal, target) in enumerate(self.train_loader):
            loss, output, target = self.forward(signal, target, mode='train')
            self.backward(loss)

            total_loss += loss.item() * signal.size(0)
            total += signal.size(0)

            if (batch_idx + 1) % self.args.step == 0 or (batch_idx + 1) == len(self.train_loader):
                print("Epoch {} | batch_idx : {}/{}({}%) COMPLETE | loss : {}".format(
                    epoch, batch_idx + 1, len(self.train_loader), np.round((batch_idx + 1) / len(self.train_loader) * 100.0, 2),
                    total_loss / total
                ))

        train_loss = total_loss / total

        return train_loss

    def val_epoch(self, epoch):
        self.model.eval()

        total_loss, total = .0, 0

        with torch.no_grad():
            for batch_idx, (signal, target) in enumerate(self.test_loader):
                if (batch_idx + 1) % self.args.step == 0:
                    print("EPOCH {} | {}/{}({}%) COMPLETE".format(epoch, batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                loss, output, target = self.forward(signal, target, mode='val')

                total_loss += loss.item() * signal.size(0)
                total += target.size(0)

        val_loss = total_loss / total

        return val_loss

    def inference(self, epoch):
        self.model.eval()

        total_loss, total = .0, 0
        y_true, y_pred = list(), list()
        accuracy_list, f1_score_list, precision_list, recall_list, auc_list = [], [], [], [], []

        with torch.no_grad():
            for batch_idx, (image, target) in enumerate(self.test_loader):
                if (batch_idx + 1) % self.args.step == 0:
                    print("EPOCH {} | {}/{}({}%) COMPLETE".format(epoch, batch_idx + 1, len(self.test_loader), np.round((batch_idx + 1) / len(self.test_loader) * 100), 4))

                self.start.record()
                loss, output, target = self.forward(image, target, mode='val')

                self.end.record()
                torch.cuda.synchronize()
                self.inference_time_list.append(self.start.elapsed_time(self.end))

                for y_true_, y_pred_ in zip(target, output) :
                    y_true_np = y_true_.cpu().detach().numpy()
                    y_pred_np = torch.sigmoid(y_pred_).cpu().detach().numpy() >= 0.5

                    y_true.append(y_true_np)
                    y_pred.append(y_pred_np)

                    accuracy_list.append(accuracy_score(y_true_np, y_pred_np))
                    f1_score_list.append(f1_score(y_true_np, y_pred_np, average='macro'))
                    precision_list.append(precision_score(y_true_np, y_pred_np, average='macro'))
                    recall_list.append(recall_score(y_true_np, y_pred_np, average='macro'))
                    auc_list.append(roc_auc_score(y_true_np, y_pred_np, average=None))

                total_loss += loss.item() * image.size(0)
                total += target.size(0)

        test_loss = total_loss / total
        accuracy = np.round(np.mean(accuracy_list), 4)
        f1_score = np.round(np.mean(f1_score_list), 4)
        precision = np.round(np.mean(precision_list), 4)
        recall = np.round(np.mean(recall_list), 4)
        auc = np.round(np.mean(auc_list), 4)

        if self.args.final_epoch == epoch : print("Mean Inference Time (ms) : {} ({})".format(np.mean(self.inference_time_list), np.std(self.inference_time_list)))

        return test_loss, accuracy, f1_score, precision, recall, auc