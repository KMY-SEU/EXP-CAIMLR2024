import os
import numpy as np
import torch
import torch.nn as nn

from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader

from config import args
from data import Data

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    # read data
    data = Data(args)
    test_seq, test_pred = data.get_test_set()

    # load data
    test_dataset = TensorDataset(test_seq, test_pred)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size
    )

    # load model
    model = torch.load(args.save_path + 'predictor.pkl')

    if args.use_parallel:
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model = model.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))

    # test model
    MAEs = []
    RMSEs = []
    MAPEs = []

    for step, (bx, by) in enumerate(tqdm(test_loader)):
        # send batch data to cuda
        bx = bx.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))
        by = by.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))

        # prediction
        out = model(bx)

        # inverse normalization
        if args.do_normalization:
            out = data.do_inv_normalization(out.detach().clone().cpu().numpy())
            by = data.do_inv_normalization(by.detach().clone().cpu().numpy())

        # compute indices
        MAEs += [np.mean(np.abs(out - by))]
        RMSEs += [np.sqrt(np.mean(np.square(out - by)))]

        more_than_one = by >= 1.
        MAPEs += [np.mean(np.abs((out[more_than_one] - by[more_than_one]) / by[more_than_one]))]

    print('MAE == {}, RMSE == {}, MAPE == {}'.format(np.mean(MAEs), np.mean(RMSEs), np.mean(MAPEs)))
