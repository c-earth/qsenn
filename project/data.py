import torch
import glob
import os
import pandas as pd
import numpy as np
import pickle as pkl
from ase.neighborlist import neighbor_list
from torch_geometric.data import Data
from ase import Atom
import itertools
from copy import copy

default_dtype = torch.float64
torch.set_default_dtype(default_dtype)

def get_node_attr(atomic_numbers):
    z = []
    for atomic_number in atomic_numbers:
        node_attr = [0.0] * 118
        node_attr[atomic_number - 1] = 1
        z.append(node_attr)
    return torch.from_numpy(np.array(z, dtype = np.float64))

def get_node_feature(atomic_numbers):
    x = []
    for atomic_number in atomic_numbers:
        node_feature = [0.0] * 118
        node_feature[atomic_number - 1] = Atom(atomic_number).mass
        x.append(node_feature)
    return torch.from_numpy(np.array(x, dtype = np.float64))


def get_node_deg(edge_dst, n):
    node_deg = np.zeros((n, 1), dtype = np.float64)
    for dst in edge_dst:
        node_deg[dst] += 1
    node_deg += node_deg == 0
    return torch.from_numpy(node_deg)


# def build_data(id, system, spins, rot, r_max):
#     symbols = list(system.symbols).copy()
#     positions = torch.from_numpy(system.get_positions().copy())
#     numb = len(positions)
#     # lattice = torch.from_numpy(structure.cell.array.copy()).unsqueeze(0)
#     edge_src, edge_dst, edge_shift, edge_vec, edge_len = neighbor_list("ijSDd", a = system, cutoff = r_max, self_interaction = True)
#     z = get_node_attr(system.arrays['numbers'])
#     x =  get_node_feature(system.arrays['numbers'])
#     node_deg = get_node_deg(edge_dst, len(x))
#     # print(spins)
#     # print('positions', positions)
#     y = spins
#     data = Data(id = id,
#                 pos = positions,
#                 symbol = symbols,
#                 x = x,
#                 z = z,
#                 y = y,
#                 rot = rot,
#                 node_deg = node_deg,
#                 edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim = 0),
#                 edge_shift = torch.tensor(edge_shift, dtype = torch.float64),
#                 edge_vec = torch.tensor(edge_vec, dtype = torch.float64),
#                 edge_len = torch.tensor(edge_len, dtype = torch.float64),
#                 r_max = r_max,
#                 # ucs = None,
#                 numb = numb) 
#     return data

def build_data(id, system, spins, rot, r_max):
    symbols = list(system.symbols).copy()
    positions = torch.from_numpy(system.get_positions().copy())
    numb = len(positions)
    # lattice = torch.from_numpy(structure.cell.array.copy()).unsqueeze(0)
    edge_src, edge_dst, edge_shift, edge_vec, edge_len = neighbor_list("ijSDd", a = system, cutoff = r_max, self_interaction = True)
    z = torch.ones(spins.shape)    #get_node_attr(system.arrays['numbers'])
    x =  torch.tensor(spins)  #get_node_feature(system.arrays['numbers'])
    rot = torch.tensor(rot)
    node_deg = get_node_deg(edge_dst, len(x))
    # print(spins)
    # print('positions', positions)
    # y = spins
    # print('check!!!!')
    data = Data(id = id,
                pos = positions,
                symbol = symbols,
                x = x,
                z = z,
                rot = rot,
                node_deg = node_deg,
                edge_index = torch.stack([torch.LongTensor(edge_src), torch.LongTensor(edge_dst)], dim = 0),
                edge_shift = torch.tensor(edge_shift, dtype = torch.float64),
                edge_vec = torch.tensor(edge_vec, dtype = torch.float64),
                edge_len = torch.tensor(edge_len, dtype = torch.float64),
                r_max = r_max,
                numb = numb) 
    return data


def generate_data_dict(data, r_max):
    data_dict = dict()
    ids = data['id']
    systems = data['system']
    # print(systems)
    spins = data['spins']
    rots = data['rot']
    # print(spins)
    # print('spins----')
    for id, system, spin, rot in zip(ids, systems, spins, rots):
        data_dict[id] = build_data(id, system, spin, rot, r_max)
    # pkl.dump(data_dict, open(data_dict_path, 'wb'))
    return data_dict
