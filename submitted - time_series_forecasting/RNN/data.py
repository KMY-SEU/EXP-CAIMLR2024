import numpy as np
import pandas as pd
import torch


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

        elif args.data_file == 'ETTm1.csv' or args.data_file == 'ETTm2.csv':
            data = pd.read_csv('../data/' + args.data_file)

            self.flow = data.iloc[:, 1:].values
            self.name_nodes = data.columns

            self.set_type = {'train': [0, 12 * 30 * 24 * 4],
                             'val': [12 * 30 * 24 * 4 - args.seq_len,
                                     12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - args.seq_len - args.pred_len],
                             'test': [12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - args.seq_len,
                                      12 * 30 * 24 * 4 + 8 * 30 * 24 * 4 - args.seq_len - args.pred_len]}

        if args.do_normalization:
            self.do_normalization()

    def do_normalization(self):

        self.max = np.max(self.flow, axis=0)
        self.min = np.min(self.flow, axis=0)
        self.flow = (self.flow - self.min) / (self.max - self.min)

    def do_inv_normalization(self, flow):
        return flow * (self.max - self.min) + self.min

    def get_train_set(self):
        x, y = [], []
        start, end = self.set_type['train'][0], self.set_type['train'][1]

        for i in range(start, end):
            x += [self.flow[i: i + self.args.seq_len]]
            y += [self.flow[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len]]

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

    def get_val_set(self):
        x, y = [], []
        start, end = self.set_type['val'][0], self.set_type['val'][1]

        for i in range(start, end):
            x += [self.flow[i: i + self.args.seq_len]]
            y += [self.flow[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len]]

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)

    def get_test_set(self):
        x, y = [], []
        start, end = self.set_type['test'][0], self.set_type['test'][1]

        for i in range(start, end):
            x += [self.flow[i: i + self.args.seq_len]]
            y += [self.flow[i + self.args.seq_len: i + self.args.seq_len + self.args.pred_len]]

        return torch.tensor(x, dtype=torch.float), torch.tensor(y, dtype=torch.float)
