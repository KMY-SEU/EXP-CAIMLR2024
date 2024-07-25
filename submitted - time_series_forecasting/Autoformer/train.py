import os

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import args
from data import Data
from Autoformer import Autoformer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    # read data
    data = Data(args)

    train_seq_x, train_seq_y, train_seq_x_mark, train_seq_y_mark, train_pred_y = data.get_data_set('train')
    train_dataset = TensorDataset(train_seq_x, train_seq_y, train_seq_x_mark, train_seq_y_mark, train_pred_y)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_seq_x, val_seq_y, val_seq_x_mark, val_seq_y_mark, val_pred_y = data.get_data_set('val')
    val_dataset = TensorDataset(val_seq_x, val_seq_y, val_seq_x_mark, val_seq_y_mark, val_pred_y)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # initialize modal
    model = Autoformer(args)

    if args.use_parallel:
        model = nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    model = model.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))

    # optimizer, lr
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # loss
    criterion = nn.MSELoss()

    # training
    lowest_loss = np.inf
    early_stop = 0

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    for epoch in range(args.training_epoch):

        # train model
        for step, (x_enc, x_mark_enc, x_dec, x_mark_dec, by) in enumerate(tqdm(train_loader)):
            # send batch data to cuda
            x_enc = x_enc.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))
            x_mark_enc = x_mark_enc.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))
            x_dec = x_dec.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))
            x_mark_dec = x_mark_dec.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))
            by = by.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))

            # prediction
            out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(out, by)

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validate loss
        val_loss = []
        for step, (x_enc, x_mark_enc, x_dec, x_mark_dec, by) in enumerate(val_loader):
            # send batch data to cuda
            x_enc = x_enc.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))
            x_mark_enc = x_mark_enc.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))
            x_dec = x_dec.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))
            x_mark_dec = x_mark_dec.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))
            by = by.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))

            # prediction
            out = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            loss = criterion(out, by)
            val_loss += [loss.detach().clone().cpu().numpy()]

        val_loss_mean = np.mean(val_loss)
        print('val_loss ==', val_loss_mean)

        if val_loss_mean < lowest_loss:

            torch.save(model, args.save_path + 'predictor.pkl')

            lowest_loss = val_loss_mean
            early_stop = 0
        else:
            early_stop += 1

            if early_stop >= args.early_stop:
                print('Early stop.')
                break
