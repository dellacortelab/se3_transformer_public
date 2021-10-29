import h5py
import numpy as np
import os
import random

import torch
from torch.utils.data import Dataset
from dgl.data import DGLDataset
import dgl

from se3_transformer.utils import create_graph


class ANI1xDataset(DGLDataset):
    def __init__(self, h5file='./ani1xrelease.h5', mode='train', knn=None):
        self.h5file = h5file
        self.mode = mode
        self.knn = knn
 
        eye = torch.eye(4)
        self.species_dict = {1: eye[0], 6: eye[1], 7: eye[2], 8: eye[3]}
        self.si_energies = torch.tensor([
            -0.600952980000,
            -38.08316124000,
            -54.70775770000,
            -75.19446356000])
        self.energy_mean = 0.0184
        self.energy_std = 0.1062
        self.force_std = 0.0709

        super(ANI1xDataset, self).__init__(name='ani1x_dataset')

    def process(self):
        print('Loading dataset')

        species_list = []
        pos_list = []
        force_list = []
        energy_list = []

        it = iter_data_buckets(self.h5file, keys=['wb97x_dz.forces', 'wb97x_dz.energy'])

        for i, molecule in enumerate(it):
            if (self.mode=='train' and i%10!=0) or (self.mode=='val' and i%20==0) or (self.mode=='test' and i%20==10):
                species = molecule['atomic_numbers']
                for pos, force, energy in zip(molecule['coordinates'], molecule['wb97x_dz.forces'], molecule['wb97x_dz.energy']):
                    species_list.append(species)
                    pos_list.append(pos)
                    force_list.append(force)
                    energy_list.append(energy)

        self.species_list = species_list
        self.pos_list = pos_list
        self.force_list = force_list
        self.energy_list = energy_list

 
    def normalize_energy(self, energy, species):
        adjustment = torch.sum(self.si_energies[None,:] * species)
        new_energy = energy - adjustment
        new_energy = (new_energy-self.energy_mean) / self.energy_std # Normalize
        return new_energy
    
    def denormalize_energy(self, energy, species):
        adjustment = torch.sum(self.si_energies[None,:] * species)
        new_energy = energy*self.energy_std + self.energy_mean
        new_energy += adjustment
        return new_energy

    def normalize_force(self, force):
        return force / self.force_std

    def denormalize_force(self, force):
        return force * self.force_std

    def __len__(self):
        return len(self.species_list)

    def __getitem__(self, i):
        pos = self.pos_list[i]
        species = self.species_list[i]
        force = self.force_list[i]
        energy = self.energy_list[i]

        pos = torch.tensor(pos)
        species = [self.species_dict[atom] for atom in species]
        species = torch.stack(species)
        force = torch.tensor(force)

        graph = create_graph(pos, knn=self.knn)
        graph.ndata['pos'] = pos
        graph.ndata['species'] = species
        graph.ndata['force'] = self.normalize_force(force)
        graph.ndata['energy'] = torch.ones(len(species)) * self.normalize_energy(energy, species)
        return graph

    
def iter_data_buckets(h5filename, keys=['wb97x_dz.energy']):
    """ Iterate over buckets of data in ANI HDF5 file. 
    Yields dicts with atomic numbers (shape [Na,]) coordinated (shape [Nc, Na, 3])
    and other available properties specified by `keys` list, w/o NaN values.
    """
    keys = set(keys)
    keys.discard('atomic_numbers')
    keys.discard('coordinates')
    with h5py.File(h5filename, 'r') as f:
        for grp in f.values():
            Nc = grp['coordinates'].shape[0]
            mask = np.ones(Nc, dtype=np.bool)
            data = dict((k, grp[k][()]) for k in keys)
            for k in keys:
                v = data[k].reshape(Nc, -1)
                mask = mask & ~np.isnan(v).any(axis=1)
            if not np.sum(mask):
                continue
            d = dict((k, data[k][mask]) for k in keys)
            d['atomic_numbers'] = grp['atomic_numbers'][()]
            d['coordinates'] = grp['coordinates'][()][mask]
            yield d
