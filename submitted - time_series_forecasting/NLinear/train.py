import os

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm

from config import args
from data import Data
from NLinear import NLinear

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    # read data
    data = Data(args)

    train_seq, train_pred = data.get_train_set()
    train_dataset = TensorDataset(train_seq, train_pred)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    val_seq, val_pred = data.get_val_set()
    val_dataset = TensorDataset(val_seq, val_pred)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        shuffle=True
    )

    # initialize modal
    model = NLinear(args)

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
        for step, (bx, by) in enumerate(tqdm(train_loader)):
            # send batch data to cuda
            bx = bx.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))
            by = by.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))

            # prediction
            out = model(bx)
            loss = criterion(out, by)

            # update parameters
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # validate loss
        val_loss = []
        for step, (bx, by) in enumerate(val_loader):
            # send batch data to cuda
            bx = bx.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))
            by = by.to(device=torch.device('cuda') if args.use_cuda else torch.device('cpu'))

            # prediction
            out = model(bx)
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
