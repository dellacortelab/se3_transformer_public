import numpy as np
import pickle as pkl
import os
import argparse
import re

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
    test_dict = run_test(args, model, dataset, dataloader, dev)
    save_test_dict(test_dict, args.model_file)

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
    checkpoint = torch.load(args.model_file, map_location=dev)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_dataloader(max_edges, knn):
    dataset = ANI1xDataset(knn=knn, mode='test')
    sizes = [len(pos) for pos in dataset.pos_list]
    batch_sampler = BatchSampler(sizes, args.max_edges)
    dataloader = GraphDataLoader(dataset, batch_sampler=batch_sampler, num_workers=4) 
    return dataset, dataloader

def save_test_dict(test_dict, model_file):
    model_dir = os.path.dirname(model_file)
    model_idx = re.split('_|\.', model_file)[-2] # Expected format: model_dir/model_1000.pkl
    test_filename = f'test_{model_idx}.pkl'
    test_file_path = os.path.join(model_dir, test_filename)
    with open(test_file_path, 'wb') as f:
        pkl.dump(test_dict, f)

def run_test(args, model, dataset, dataloader, dev):
    test_dict = dict()
    energy_preds, force_preds = test(args, model, dataset, dataloader, dev)
    test_dict['energy_pred'] = energy_preds
    test_dict['force_pred'] = force_preds
    return test_dict

def test(args, model, dataset, dataloader, dev):
    energy_preds = []
    force_preds = []

    print('Starting Test')
    for i, batch in enumerate(dataloader):
        print(f'Iteration {i}, idx {len(energy_preds)}')
        energy_pred, force_pred = test_step(args, model, dataset, batch, dev)
        energy_preds += energy_pred
        force_preds += force_pred
    return energy_preds, force_preds

def test_step(args, model, dataset, batch, dev):
    batch = batch.to(dev)
    pc = PointCloud(graph=batch, J_max=2*args.order)
    f = {0: batch.ndata['species'].unsqueeze(-1)}
    with torch.no_grad():
        y_hat = model(pc, f)
    energy_preds, force_preds = get_preds(y_hat, batch, dataset)
    return energy_preds, force_preds

def get_preds(y_hat, batch, dataset):
    energy_preds = []
    force_preds = []
    idx = 0 
    cum_sum = 0  
    for i, graph in enumerate(dgl.unbatch(batch)):
        l = graph.num_nodes()
        species = graph.ndata['species'].cpu()
        energy_pred = torch.sum(y_hat[0][idx:idx+l,0,0]).detach().cpu()
        energy_pred = dataset.denormalize_energy(energy_pred, species)
        energy_preds.append(energy_pred.item())
        force_pred = y_hat[1][idx:idx+l,0,:].detach().cpu().numpy()
        force_pred = dataset.denormalize_force(force_pred) 
        force_preds.append(force_pred)

        idx += l

    return energy_preds, force_preds






if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_file', help='Model file')
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
