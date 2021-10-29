import numpy as np
import pickle as pkl
import os
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import dgl
from dgl.dataloading.pytorch import GraphDataLoader

from se3_transformer.utils import BatchSampler, create_graph
from se3_transformer.se3_transformer import PointCloud
from se3_transformer.model import Model
from ani1x_dataset import ANI1xDataset



def main(args):
    dev = f'cuda:{args.gpu}'
    model = get_model(args, dev)
    dataset, dataloader = get_dataloader(args.max_edges, args.knn)
    val_dict = load_val_dict(args.model_dir)
    run_validation(args, model, args.model_dir, dataset, dataloader, val_dict, dev)

def get_model(args, dev):
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
    model.eval()
    return model

def get_dataloader(max_edges, knn):
    dataset = ANI1xDataset(knn=knn, mode='val')
    sizes = [len(pos) for pos in dataset.pos_list]
    batch_sampler = BatchSampler(sizes, args.max_edges)
    dataloader = GraphDataLoader(dataset, batch_sampler=batch_sampler, num_workers=4) 
    return dataset, dataloader

def load_val_dict(directory):
    file_path = os.path.join(directory, 'validation.pkl')
    if os.path.isfile(file_path):
        with open(file_path, 'rb') as f:
            val_dict = pkl.load(f)
    else:
        val_dict = {
            'energy': dict(),
            'force': dict()
        }
    return val_dict


def save_val_dict(directory, val_dict):
    file_path = os.path.join(directory, 'validation.pkl')
    with open(file_path, 'wb') as f:
        pkl.dump(val_dict, f)


def run_validation(args, model, model_dir, dataset, dataloader, val_dict, dev, freq=10000):
    if len(val_dict['energy'].keys()) > 0:
        n = max(val_dict['energy'].keys()) + freq
    else:
        n = freq
    file_name = f'model_{n}.pkl'
    model_path = os.path.join(model_dir, file_name)
    while os.path.isfile(model_path) and n<1000000:
        print(f'Running validation on model num {n}')
        checkpoint = torch.load(model_path, map_location=dev)
        model.load_state_dict(checkpoint['model_state_dict'])
        energy_error, force_error = validate(args, model, dataset, dataloader, dev)
        val_dict['energy'][n] = energy_error
        val_dict['force'][n] = force_error
        n += freq
        file_name = f'model_{n}.pkl'
        model_path = os.path.join(model_dir, file_name)
        save_val_dict(model_dir, val_dict)

def validate(args, model, dataset, dataloader, dev):
    energy_list = []
    force_list = []

    for i, batch in enumerate(dataloader):
        energy_losses, force_losses = validation_step(args, model, dataset, batch, dev)
        energy_list += list(energy_losses)
        force_list += list(force_losses)
        print(f'Iteration {i}, Batch size {len(energy_losses)}, Dataset position {len(energy_list)}')
    energy_error = 627.5 * dataset.energy_std * np.sqrt(np.mean(energy_list))
    force_error = 627.5 * dataset.force_std * np.sqrt(np.mean(force_list))
    print(f'Energy error: {energy_error}, Force error: {force_error}')
    return energy_error, force_error


def validation_step(args, model, dataset, batch, dev):
    batch = batch.to(dev)
    pc = PointCloud(graph=batch, J_max=2*args.order).to(dev)
    f = {0: batch.ndata['species'].unsqueeze(-1)}
    with torch.no_grad():
        y_hat = model(pc, f)
    energy_losses, force_losses = get_losses(y_hat, batch)
    return energy_losses, force_losses


def get_losses(y_hat, batch):
    energy_losses = []
    force_losses = []
    idx = 0
    for i, graph in enumerate(dgl.unbatch(batch)):
        l = graph.num_nodes()
        energy_out = torch.sum(y_hat[0][idx:idx+l, 0, 0])
        energy_target = graph.ndata['energy'][0]
        energy_losses.append(F.mse_loss(energy_out, energy_target).item())

        force_out = y_hat[1][idx:idx+l, 0, :]
        force_target = graph.ndata['force']
        force_losses.append(F.mse_loss(force_out, force_target).item())

        idx += l
    return energy_losses, force_losses



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', help='Directory for the model files', default='./models/')
    parser.add_argument('--gpu', help='While GPU to run validation on', default=0, type=int)
    parser.add_argument('--knn', help='Neighborhood size', default=None, type=int)
    parser.add_argument('--order', help='Order of spherical harmonics', default=2, type=int)
    parser.add_argument('--channels', help='How wide model gets', default=2, type=int)
    parser.add_argument('--mean', help='Whether to do mean pooling', default=False, type=bool)
    parser.add_argument('--depth', help='Number of convolutional layers', default=3, type=int)
    parser.add_argument('--heads', help='Number of heads', default=4, type=int)
    parser.add_argument('--max_edges', help='Max number of edges per batch', default=1000000, type=int)
    args = parser.parse_args()
    main(args)
