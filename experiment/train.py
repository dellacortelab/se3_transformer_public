import numpy as np
import pickle as pkl
import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
import dgl
from dgl.dataloading.pytorch import GraphDataLoader

from se3_transformer.utils import create_graph
from se3_transformer.se3_transformer import PointCloud
from se3_transformer.model import Model
from ani1x_dataset import ANI1xDataset



EPOCHS = 1000


def main(args):
    dev = f'cuda:{args.gpu}'
    model, optimizer, it = load_checkpoint(args.model_path, args.lr, args)
    dataset, dataloader = get_dataloader(args.batch_size, args.knn)
    train(args, model, optimizer, dataset, dataloader, it, dev, args.save_dir)

    
def load_checkpoint(model_path, lr, args):
    dev = f'cuda:{args.gpu}' 
    if args.mean == True:
        rdc = torch.mean
    else:
        rdc = torch.sum
    model = Model(
        heads=args.heads,
        c_in=4,
        c=8,
        c_hid=args.channels,
        c_out=1,
        depth=args.depth,
        d_in=0,
        d_hid=args.order,
        d_out=1,
        rdc=rdc,
    ).to(dev)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    it = 0
    if model_path:
        checkpoint = torch.load(model_path, map_location=dev)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for g in optimizer.param_groups:
            g['lr'] = lr
        it = checkpoint['it']
    return model, optimizer, it


def get_dataloader(batch_size, knn):
    dataset = ANI1xDataset(knn=knn)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader

 
def train(args, model, optimizer, dataset, dataloader, it, dev, save_dir):
    print('Beginning training')
    energy_errors = []
    force_errors = []
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    for epoch in range(EPOCHS):
        for batch in dataloader:
            try:
                loss, energy_error, force_error = train_step(args, model, optimizer, dataset, batch, dev)
                energy_errors.append(energy_error)
                force_errors.append(force_error)
                it += 1
                print(f'it:{it}, loss:{loss:.3e}, energy error:{energy_error:.3e}, force error:{force_error:.3e}')

                if it % 1000 == 0:
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'it': it},
                               os.path.join(save_dir, f'model_{it}.pkl'))
                    np.save(os.path.join(save_dir, f'energy_losses_{it}.npy'),
                            np.array(energy_errors))
                    np.save(os.path.join(save_dir, f'force_losses_{it}.npy'),
                            np.array(force_errors))
                    energy_errors = []
                    force_errors = []

            except: 
                print('ERROR: CUDA memory full or NaN encountered')


def train_step(args, model, optimizer, dataset, batch, dev):
    batch = batch.to(dev)
    pc = PointCloud(graph=batch, J_max=2*args.order).to(dev)
    f = {0: batch.ndata['species'].unsqueeze(-1)}
    model.zero_grad(set_to_none=True)

    y_hat = model(pc, f)
    energy_loss, force_loss = get_losses(y_hat, batch)
    loss = energy_loss + force_loss
    if torch.isnan(loss):
        print('NaN encountered')
        raise Exception('NaN' )
    loss.backward()
    total_norm = torch.nn.utils.clip_grad_value_(model.parameters(), 10.0) 
    optimizer.step()

    energy_error = 627.5 * dataset.energy_std * np.sqrt(energy_loss.item())
    force_error = 627.5 * dataset.denormalize_force(np.sqrt(force_loss.item()))

    return loss.item(), energy_error, force_error


def get_losses(y_hat, batch):
    energy_loss = 0
    force_loss = 0
    idx = 0
    for i, graph in enumerate(dgl.unbatch(batch)):
        l = graph.num_nodes()
        energy_out = torch.sum(y_hat[0][idx:idx+l, 0, 0])
        energy_target = graph.ndata['energy'][0]
        energy_loss += F.mse_loss(energy_out, energy_target)

        force_out = y_hat[1][idx:idx+l, 0, :]
        force_target = graph.ndata['force']
        force_loss += F.mse_loss(force_out, force_target)

        idx += l

    energy_loss /= i
    force_loss /= i
    return energy_loss, force_loss


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', help='The model path to load from', default=None)
    parser.add_argument('--gpu', help='Which GPU to train on', default=0, type=int)
    parser.add_argument('--lr', help='Learning rate', default=1e-3, type=float)
    parser.add_argument('--save_dir', help='Directory to save new models and losses', default='./models/basic/')
    parser.add_argument('--knn', help='Neighborhood size', default=None, type=int)
    parser.add_argument('--order', help='Order of spherical harmonics', default=2, type=int)
    parser.add_argument('--channels', help='How wide model gets', default=2, type=int)
    parser.add_argument('--mean', help='Whether to do mean pooling', default=False, type=bool)
    parser.add_argument('--depth', help='Number of convolutional layers', default=3, type=int)
    parser.add_argument('--heads', help='Number of heads', default=4, type=int)
    parser.add_argument('--batch_size', help='Graphs per batch', default=100, type=int)
    args = parser.parse_args()
    main(args)
