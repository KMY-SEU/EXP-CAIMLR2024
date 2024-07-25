import numpy as np
import pandas as pd
import torch

from utils.timefeatures import *


class Data:
    def __init__(self, args):
        self.args = args

        # read data
        if args.data_file == 'ETTh1.csv' or args.data_file == 'ETTh2.csv':
            data = pd.read_csv('../data/' + args.data_file)

            self.flow = data.iloc[:, 1:].values
            self.name_nodes = data.columns

            self.set_type = {'train': [0, 12 * 30 * 24],
                             'val': [12 * 30 * 24 - args.seq_len,
                                     12 * 30 * 24 + 4 * 30 * 24 - args.seq_len - args.pred_len],
                             'test': [12 * 30 * 24 + 4 * 30 * 24 - args.seq_len,
                                      12 * 30 * 24 + 8 * 30 * 24 - args.seq_len - args.pred_len]}

            df_stamp = data[['date']]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            self.data_stamp = time_features(df_stamp, timeenc=0, freq=args.freq)

        elif args.data_file == 'ETTm1.csv' or args.data_file == 'ETTm2.csv':
            data = pd.read_csv('../data/' + args.data_file)

            self.flow = data.iloc[:, 1:].values
            self.name_nodes = data.columns

            self.set_type = {'train': [0, 12 * 30 * 24 * 4],
                             'val': [12 * 30 * 24 * 4 - args.seq_len,
                                     12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - args.seq_len - args.pred_len],
                             'test': [12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - args.seq_len,
                                      12 * 30 * 24 * 4 + 8 * 30 * 24 * 4 - args.seq_len - args.pred_len]}

            df_stamp = data[['date']]
            df_stamp['date'] = pd.to_datetime(df_stamp.date)
            self.data_stamp = time_features(df_stamp, timeenc=0, freq=args.freq)

        if args.do_normalization:
            self.do_normalization()

    def do_normalization(self):

        self.max = np.max(self.flow, axis=0)
        self.min = np.min(self.flow, axis=0)
        self.flow = (self.flow - self.min) / (self.max - self.min)

    def do_inv_normalization(self, flow):
        return flow * (self.max - self.min) + self.min

    def get_data_set(self, set_type='train'):
        seq_x, seq_y, seq_x_mark, seq_y_mark, pred_y = [], [], [], [], []
        start, end = self.set_type[set_type][0], self.set_type[set_type][1]

        for i in range(start, end):
            seq_x += [self.flow[i: i + self.args.seq_len]]
            seq_x_mark += [self.data_stamp[i: i + self.args.seq_len]]
            seq_y += [np.pad(self.flow[i + self.args.seq_len - self.args.label_len: i + self.args.seq_len],
                             pad_width=((0, self.args.pred_len), (0, 0)), mode='constant', constant_values=0)]
            seq_y_mark += [self.data_stamp[
                           i + self.args.seq_len - self.args.label_len: i + self.args.seq_len + self.args.pred_len]]
            pred_y += [self.flow[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len]]

        return torch.tensor(seq_x, dtype=torch.float), \
               torch.tensor(seq_x_mark, dtype=torch.float), \
               torch.tensor(seq_y, dtype=torch.float), \
               torch.tensor(seq_y_mark, dtype=torch.float), \
               torch.tensor(pred_y, dtype=torch.float)
