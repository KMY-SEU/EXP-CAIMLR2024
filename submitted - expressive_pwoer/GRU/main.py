import os
import numpy as np
import torch.nn as nn
import pandas as pd

from utils import *
from configs import args
from GRU import GRU

if __name__ == '__main__':
    # trajectory
    trajs = generate_trajectory(args)
    print(compute_trajectory_length(trajs.numpy()))

    # modeling
    model = GRU(args)

    total_params = sum(p.numel() for p in model.parameters())
    print('Total number of parameters == {}'.format(total_params))

    # test
    trajs_len = []
    normal_sq_len = []
    angular_entropy = []

    for _ in range(10):
        for n_test in range(100):

            # initialization
            sigma = np.sqrt(1 / args.n_layers)
            for name, param in model.named_parameters():

                # print('name, param == {}{}', name, param)
                if 'weight' in name:
                    nn.init.normal_(param.data, mean=0., std=sigma)
                elif 'bias' in name:
                    nn.init.zeros_(param.data)

            # compute
            if n_test == 0:
                trajs_in = trajs.detach().clone().numpy()
            else:
                trajs_in = np.concatenate([trajs_in, trajs.detach().clone().numpy()], axis=0)

            out = model(trajs.detach().clone())

            if n_test == 0:
                trajs_out = out.detach().clone().numpy()
            else:
                trajs_out = np.concatenate([trajs_out, out.detach().clone().numpy()], axis=0)

            trajs_len += [compute_trajectory_length(out.detach().clone().numpy())]
            normal_sq_len += [compute_normalized_squared_length(out.detach().clone().numpy())]

        angular_entropy += [compute_angular_entropy(trajs_in, trajs_out)]

        print('trajs_len == {}, normal_sq_len == {}, angular_entropy == {}'.format(
            np.mean(trajs_len),
            np.mean(normal_sq_len),
            compute_angular_entropy(trajs_in, trajs_out)
        ))

    save_path = './results/seq_len_' + str(args.seq_len) + '/'
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    pd.DataFrame(trajs_len).to_csv(save_path + 'trajs_len.csv', header=False, index=False)
    pd.DataFrame(normal_sq_len).to_csv(save_path + 'normal_sq_len_.csv', header=False, index=False)
    pd.DataFrame(angular_entropy).to_csv(save_path + 'angular_entropy_.csv', header=False, index=False)
