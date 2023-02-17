import os

import yaml
import numpy as np
import matplotlib.pyplot as plt


def plot_loss_acc(history, model_dirs):
    params = yaml.safe_load(open(f'configuration_files/plot_configurations/plot_loss_configuration.yml'))

    train_loss, test_loss = history['train_loss'], history['val_loss']
    # train_top1_acc, test_top1_acc = history['train_top1_acc'], history['test_top1_acc']
    # train_top5_acc, test_top5_acc = history['train_top5_acc'], history['test_top5_acc']

    plt.plot(np.arange(len(train_loss)), train_loss, label='train loss', color=params['train_color'])
    plt.plot(np.arange(len(test_loss)), test_loss, label='val loss', color=params['test_color'])

    plt.xlim([np.arange(len(train_loss))[0], np.arange(len(train_loss))[-1]])
    plt.ylim(params['loss_range'])
    plt.legend(loc='upper right')
    plt.grid(params['y_grid'], axis='y', linestyle='--')

    plt.savefig(os.path.join(model_dirs, 'plot_results/loss.png'), dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

    # plt.plot(np.arange(len(train_top1_acc)), train_top1_acc, label='train top1 accuracy', color=params['train_color'])
    # plt.plot(np.arange(len(test_top1_acc)), test_top1_acc, label='test top1 accuracy', color=params['test_color'])
    #
    # plt.xlim([np.arange(len(train_top1_acc))[0], np.arange(len(train_top1_acc))[-1]])
    # plt.ylim(params['acc_range'])
    # plt.legend(loc='lower right')
    # plt.grid(params['y_grid'], axis='y', linestyle='--')
    #
    # plt.savefig(os.path.join(model_dirs, 'plot_results/top1_accuracy.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
    # plt.close()
    #
    # plt.plot(np.arange(len(train_top5_acc)), train_top5_acc, label='train top5 accuracy', color=params['train_color'])
    # plt.plot(np.arange(len(test_top5_acc)), test_top5_acc, label='test top5 accuracy', color=params['test_color'])
    #
    # plt.xlim([np.arange(len(train_top5_acc))[0], np.arange(len(train_top5_acc))[-1]])
    # plt.ylim(params['acc_range'])
    # plt.legend(loc='lower right')
    # plt.grid(params['y_grid'], axis='y', linestyle='--')
    #
    # plt.savefig(os.path.join(model_dirs, 'plot_results/top5_accuracy.png'), dpi=300, bbox_inches = 'tight', pad_inches = 0)
    # plt.close()