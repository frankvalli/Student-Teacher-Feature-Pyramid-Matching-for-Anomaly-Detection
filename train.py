import os
import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from loss import DistillationLoss
from dataset import AnomalyDataset, get_transforms


def train(model: nn.Module, args: argparse.Namespace) -> None:
    '''Train the STFPM model

    Args:
        model (nn.Module): model to train
        args (argparse.Namespace): parsed arguments
    '''
    transform = get_transforms(size=args.size, mean=args.mean, std=args.std)
    root_dir = os.path.join(os.path.join(args.data_dir, args.category), 'train')
    train_dataset = AnomalyDataset(root_dir=root_dir, transform=transform)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    model.teacher.eval()
    model.student.train()
    optimizer = optim.SGD(model.student.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    criterion = DistillationLoss(args.alphas)

    elapsed_time = 0.

    print(f'Training model {model.__class__.__name__}...\n')
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        running_loss = 0.

        print('-'*20)
        print(f'Epoch [{epoch} / {args.epochs}]\n')

        for sample in loader:
            img = sample['image'].to(args.device)
            
            optimizer.zero_grad()
            t_maps, s_maps = model(img)
            loss = criterion(t_maps, s_maps)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        epoch_end = time.time()
        epoch_time = epoch_end - epoch_start
        elapsed_time += epoch_time
        
        print(f'Loss: {running_loss:.4f}')
        print(f'Epoch time: {epoch_time:.3f}')

    print('-'*20)
    print('\nTraining finished.')
    print(f'Elapsed time: {elapsed_time:.3f}\n')
    print('Saving weights...')

    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    torch.save(model.student.state_dict(), os.path.join(args.model_dir, f'student_{args.category}.pt'))
    print('Weights saved successfully.\n')