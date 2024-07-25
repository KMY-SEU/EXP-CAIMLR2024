import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, configs):
        super(GRU, self).__init__()
        self.configs = configs

        self.gru = nn.GRU(input_size=configs.enc_in, hidden_size=configs.d_model,
                          num_layers=configs.n_layers, batch_first=True)

        self.fc_out = nn.Linear(configs.d_model, configs.enc_in)

    def forward(self, x):
        h, _ = self.gru(x)

        return self.fc_out(h)
