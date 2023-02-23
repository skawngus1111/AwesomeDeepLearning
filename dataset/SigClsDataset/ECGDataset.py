import os
import re
import sys
import warnings
warnings.filterwarnings('ignore')

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

import pywt
import h5py
import signal
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import detrend, find_peaks
from tslearn.barycenters import dtw_barycenter_averaging

class ECGDataset(Dataset):
    def __init__(self, data_path, train):
        super(ECGDataset, self).__init__()

        self.data_path = data_path
        self.df = pd.read_csv(os.path.join(data_path, 'exams.csv'), index_col=False)
        passing_idx_list = []
        for part in range(18):
            f = open(os.path.join(data_path, 'tracings_2048', 'signal', 'exam_part{}'.format(part), 'exam_id_continue.txt'))
            while True:
                line = f.readline()
                if not line: break
                passing_exam_id = int(re.sub(r'[^0-9]', '', line))
                passing_idx = int(np.where(self.df.exam_id == passing_exam_id)[0])
                passing_idx_list.append(passing_idx)
            f.close()

        self.df = self.df.drop(passing_idx_list, axis=0)

        self.exam_part_list = []

        if train:
            train_trace_file_idx = list(np.where((self.df['trace_file'] != 'exams_part15.hdf5') &
                                                 (self.df['trace_file'] != 'exams_part16.hdf5') &
                                                 (self.df['trace_file'] != 'exams_part17.hdf5') &
                                                 (self.df['trace_file'] != 'exams_part18.hdf5'))[0])

            self.df = self.df.iloc[train_trace_file_idx]

            for part in range(15):
                self.exam_part_list.append("exam_part{}".format(part))
        else:
            test_trace_file_idx = list(np.where((self.df['trace_file'] == 'exams_part15.hdf5') &
                                                 (self.df['trace_file'] == 'exams_part16.hdf5') &
                                                 (self.df['trace_file'] == 'exams_part17.hdf5') &
                                                 (self.df['trace_file'] == 'exams_part18.hdf5'))[0])

            self.df = self.df.iloc[test_trace_file_idx]

            for part in range(15, 18 + 1):
                self.exam_part_list.append("exam_part{}".format(part))


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        exam_id, target, trace_file = self.df.iloc[idx]['exam_id'], self.df.iloc[idx][['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']], self.df.iloc[idx]['trace_file']
        trace_file_idx = int(re.sub(r'[^0-9]', '', trace_file.split('.')[0]))

        ecg = pd.read_csv(os.path.join(self.data_path, 'tracings_2048', 'signal', 'exam_part{}'.format(trace_file_idx), 'exam_id{}.csv'.format(exam_id))).to_numpy()
        target = np.array(target, dtype=np.int)

        return ecg, target

# class ECGDataset(Dataset):
#     def __init__(self, data_path, train):
#         super(ECGDataset, self).__init__()
#
#         self.data_path = data_path
#         self.df = pd.read_csv(os.path.join(data_path, 'exams.csv'), index_col=False)
#         # idx = np.where(self.df['trace_file'] == 'exams_part{}.hdf5'.format(part))
#         # self.df = self.df.loc[idx]
#         self.hdf5_file_list = []
#
#         # f = h5py.File(os.path.join(data_path, 'exams_part{}.hdf5'.format(part)), 'r')
#         # self.hdf5_file_list.append(f)
#
#         if train:
#             for i in range(18):
#                 f = h5py.File(os.path.join(data_path, 'exams_part{}.hdf5'.format(i)), 'r')
#                 self.hdf5_file_list.append(f)
#
#     def __len__(self):
#         return len(self.df)
#
#     def __getitem__(self, idx):
#         exam_id, target, trace_file = self.df.iloc[idx]['exam_id'], self.df.iloc[idx][['1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF']], self.df.iloc[idx]['trace_file']
#         trace_file_idx = int(re.sub(r'[^0-9]', '', trace_file.split('.')[0]))
#         hdf5_file = self.hdf5_file_list[trace_file_idx]
#         # hdf5_file = self.hdf5_file_list[0]
#         ecg = np.array(hdf5_file['tracings'][np.where(np.array(hdf5_file['exam_id']) == exam_id)[0][0]])
#         target = np.array(target, dtype=np.int)
#
#         return exam_id, ecg, target, trace_file_idx

def get_numpy_from_nonfixed_2d_array(aa, fixed_length, padding_value=0):
    aa = np.pad(aa, (0, fixed_length), 'constant', constant_values=padding_value)[:fixed_length]
    return aa

if __name__=='__main__':
    import matplotlib.pyplot as plt
    data_path = '/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/AwesomeDeepLearning/dataset/SC1D_dataset/12ECG'
    train_dataset = ECGDataset(data_path, train=True)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    import ecg_plot
    for batch_idx, (ecg, target, exam_id, trace_file_idx) in tqdm(enumerate(train_loader)):
        ecg_plot.plot_12(ecg.squeeze().cpu().numpy(), sample_rate=500, lead_index=None)
        if not os.path.exists(os.path.join(data_path, 'tracings_2048', 'image', 'exam_part{}'.format(int(trace_file_idx)))):
            os.makedirs(os.path.join(data_path, 'tracings_2048', 'image', 'exam_part{}'.format(int(trace_file_idx))))
        ecg_plot.save_as_png('exam_id{}.png'.format(int(exam_id)), os.path.join(data_path, 'tracings_2048', 'image', 'exam_part{}/'.format(int(trace_file_idx))))


    # data_path = '/media/jhnam0514/68334fe0-2b83-45d6-98e3-76904bf08127/home/namjuhyeon/Desktop/LAB/AwesomeDeepLearning/dataset/SC1D_dataset/12ECG'
    # # for part in range(18):
    # train_dataset = ECGDataset(data_path, train=True)
    # train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False)
    #
    # cnt = 1
    # number_of_continue = 0
    # exam_id_continue = []
    # for batch_idx, (exam_id, ecg, target, trace_file_idx) in tqdm(enumerate(train_loader)):
    #     # target_df = pd.DataFrame(columns=['exam_id', '1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF'])
    #     tracings_df = pd.DataFrame(columns=range(2048))
    #     # target_df.loc[batch_idx] = [int(exam_id), int(target[0][0]), int(target[0][1]), int(target[0][2]),
    #     #                             int(target[0][3]), int(target[0][4]), int(target[0][5])]
    #
    #     ecg = np.array(ecg.squeeze())
    #     target = np.array(target.squeeze())
    #     ecg_list = []
    #     ecg_detrend_list = []
    #
    #     import matplotlib.pyplot as plt
    #
    #     if cnt <= 1000:
    #         fig, ax = plt.subplots(5, 1, figsize=(10, 5))
    #     for lead, ecg_ in enumerate(np.transpose(ecg, (1, 0))):
    #         ecg_list.append(ecg_)
    #         ecg_detrend = detrend(ecg_)
    #         ecg_detrend_list.append(ecg_detrend)
    #         if cnt <= 1000:
    #             ax[0].plot(ecg_, 'k', alpha=0.5)
    #             ax[1].plot(ecg_detrend, 'k', alpha=0.5)
    #             ax[3].plot(ecg_detrend, 'k', alpha=0.5)
    #
    #     # plt.show()
    #     # plt.savefig('STEP1.png', bbox_inches='tight', pad_inches=0)
    #     # plt.show()
    #     # sys.exit()
    #
    #     # dba_ecg_detrend = dtw_barycenter_averaging(ecg_detrend_list, max_iter=1)
    #     dba_ecg_detrend = np.mean(ecg_detrend_list, axis=0)
    #     peaks, _ = find_peaks(dba_ecg_detrend.squeeze(), distance=150)
    #     t = np.arange(0, 4096)
    #     if cnt <= 1000:
    #         ax[1].plot(dba_ecg_detrend, 'b')
    #         ax[1].scatter(t[peaks], dba_ecg_detrend[peaks], c='red')
    #
    #         ax[2].plot(np.sort(dba_ecg_detrend[peaks].squeeze()), 'k', alpha=0.5)
    #         ax[2].plot(np.diff(np.sort(dba_ecg_detrend[peaks].squeeze())), 'r')
    #
    #     # dba_ecg_detrend[peaks] : ECG 신호 내의 R-Peak index 후보군
    #     # np.sort(dba_ecg_detrend[peaks]) : ECG 신호 내의 R-Peak index 후보군 정렬
    #     # np.diff(np.sort(dba_ecg_detrend[peaks])) : ECG 신호 내의 R-Peak index 후보군 정렬 후 1차 차분 계산
    #     # np.argmax(np.diff(np.sort(dba_ecg_detrend[peaks]))) : ECG 신호 내의 R-Peak index 후보군 정렬 후 1차 차분값 중 가장 큰 값
    #
    #     if len(dba_ecg_detrend[peaks]) <= 4:
    #         number_of_continue += 1
    #         exam_id_continue.append(exam_id)
    #         print("number_of_continue : ", number_of_continue)
    #         continue
    #
    #     max_diff_idx = np.argmax(np.diff(
    #         np.sort(dba_ecg_detrend[peaks].squeeze())))  # peak 값들의 모임 중 가장 변화율이 큰 지점으로 이전 영역은 R-Peak index 후보군에서 제외해야함
    #     sort_index = dba_ecg_detrend[peaks].squeeze().argsort()  # ECG 신호 내에서 R-Peak 값들을 sort했을 때 얻을 수 있는 index set을 얻음
    #     # R-Peak index 후보군들은 최대 0~ 4095까지의 값을 가지지만 max_diff_idx는 그보다 훨씬 적은 개수를 가지게 됨.
    #     # 이를 매칭해주기 위해서 sort_index를 기반으로 R-Peak index 집합을 정렬해준 뒤 max_idff_idx로 인덱싱을 하게 되면 ECG 신호 내의 R-Peak index 후보군 중 1차 차분값이 가장 큰 index를 찾을 수 있음
    #     sorted_peak = peaks[sort_index]
    #
    #     filtered_peak_idx = sorted_peak[max_diff_idx:]  # ECG 신호의 R-Peak index 후보군 중 1차 차분값이 가장 큰 index를 포함하여 이후의 모든 인덱스값
    #     if len(filtered_peak_idx) <= 4:
    #         number_of_continue += 1
    #         exam_id_continue.append(int(exam_id))
    #         print("number_of_continue : ", number_of_continue)
    #         continue
    #
    #     if cnt <= 1000:
    #         ax[3].plot(dba_ecg_detrend, 'b')
    #         ax[3].scatter(t[filtered_peak_idx], dba_ecg_detrend[filtered_peak_idx], c='red')
    #     # plt.savefig('STEP3.png', bbox_inches='tight', pad_inches=0)
    #     # plt.show()
    #
    #     mean_peak_idx = int(np.mean(filtered_peak_idx))
    #
    #     for lead, ecg_detrend in enumerate(ecg_detrend_list):
    #         new_ecg_detrend = ecg_detrend[mean_peak_idx-1024:mean_peak_idx+1024]
    #         if len(new_ecg_detrend) < 2048:
    #             new_ecg_detrend = get_numpy_from_nonfixed_2d_array(new_ecg_detrend, fixed_length=2048, padding_value=0)
    #
    #         if cnt <= 1000:
    #             ax[4].plot(new_ecg_detrend, 'k', alpha=0.5)
    #         tracings_df.loc[lead] = new_ecg_detrend
    #
    #     plt.savefig('STEP4.png', bbox_inches='tight', pad_inches=0)
    #     plt.show()
    #
    #     # if not os.path.exists(os.path.join(data_path, 'tracings_2048', 'exam_part{}'.format(part))):
    #     #     os.makedirs(os.path.join(data_path, 'tracings_2048', 'exam_part{}'.format(part)))
    #     #
    #     # tracings_df.to_csv(os.path.join(data_path, 'tracings_2048', 'exam_part{}'.format(part), 'exam_id{}.csv'.format(int(exam_id))), index=False)
    #     # # target_df.to_csv(os.path.join(data_path, 'exam_dba{}.csv'.format(part)), index=False)
    #
    #     # if cnt <= 1000:
    #     #     plt.tight_layout()
    #     #     plt.subplots_adjust(left=0, bottom=0, right=1, top=1, hspace=0, wspace=0)
    #     #     # plt.show()
    #     #     plt.savefig('example_{}.png'.format(cnt), bbox_inches='tight', pad_inches=0)
    #     #     plt.close()
    #     # cnt += 1
    #
    # # with open(os.path.join(data_path, 'tracings_2048', 'exam_part{}'.format(part), 'exam_id_continue.txt'), 'w', encoding='UTF-8') as f:
    # #     for continue_idx in exam_id_continue:
    # #         f.write(str(continue_idx) + '\n')