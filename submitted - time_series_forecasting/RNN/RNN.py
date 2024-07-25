import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, enc_in, d_model, n_layers, pred_len):
        super(RNN, self).__init__()
        self.enc_in = enc_in
        self.d_model = d_model
        self.n_layers = n_layers
        self.pred_len = pred_len

        self.rnn = nn.RNN(input_size=enc_in, hidden_size=d_model,
                          num_layers=n_layers, batch_first=True)
        self.linear = nn.Linear(in_features=d_model, out_features=pred_len * enc_in)

    def forward(self, x):
        out, _ = self.rnn(x)
        y = self.linear(out[:, -1, :])

        return y.view(x.size(0), -1, self.enc_in)
