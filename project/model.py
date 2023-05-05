import logging
import math

import torch
import numpy as np
from torch_cluster import radius_graph
from torch_geometric.data import Data, DataLoader
from torch_scatter import scatter
from typing import Union,TypeVar
from typing import Dict, Union

from su2nn_e3nn_core import su2
from su2nn_e3nn_core.su2 import Irrep, Irreps
from su2nn_e3nn_core.math import soft_one_hot_linspace
from su2nn_e3nn_core.nn import Gate, FullyConnectedNet

from su2nn_e3nn_core.su2 import FullyConnectedTensorProduct, TensorProduct
from su2nn_e3nn_core.util.test import assert_equivariant

from e3nn import o3


def irreps_o3_to_su2(irreps):
    list_irrep = str(irreps).split('+')
    irreps_su2 = ''
    for ir in list_irrep:
        irreps_su2 += ir + 'e+'
    return Irreps(irreps_su2[:-1])

def irreps_su2_to_o3(irreps):
    list_irrep = str(irreps).split('+')
    irreps_o3 = ''
    for ir in list_irrep:
        irreps_o3 += ir[:-1] + '+'
    return o3.Irreps(irreps_o3[:-1])


def tp_path_exists(irreps_in1, irreps_in2, ir_out):
    irreps_in1 = Irreps(irreps_in1).simplify()
    irreps_in2 = Irreps(irreps_in2).simplify()
    ir_out = Irrep(ir_out)

    for _, ir1 in irreps_in1:
        for _, ir2 in irreps_in2:
            if ir_out in ir1 * ir2:
                return True
    return False

class CustomCompose(torch.nn.Module):
    def __init__(self, first, second):
        super().__init__()
        self.first = first
        self.second = second
        self.irreps_in = self.first.irreps_in
        self.irreps_out = self.second.irreps_out

    def forward(self, *input):
        x = self.first(*input)
        self.first_out = x.clone()
        x = self.second(x)
        self.second_out = x.clone()
        return x

class GraphConvolution(torch.nn.Module):
    def __init__(self,
                 irreps_in,
                 irreps_node_attr,
                 irreps_edge_attr,
                 irreps_out,
                 number_of_basis,
                 radial_layers,
                 radial_neurons):
        super().__init__()
        self.irreps_in = Irreps(irreps_in)
        self.irreps_node_attr = Irreps(irreps_node_attr)
        self.irreps_edge_attr = Irreps(irreps_edge_attr)
        self.irreps_out = Irreps(irreps_out)

        self.linear_input = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_in)
        self.linear_mask = FullyConnectedTensorProduct(self.irreps_in, self.irreps_node_attr, self.irreps_out)
        
        irreps_mid = []
        instructions = []
        for i, (mul, irrep_in) in enumerate(self.irreps_in):
            for j, (_, irrep_edge_attr) in enumerate(self.irreps_edge_attr):
                for irrep_mid in irrep_in * irrep_edge_attr:
                    if irrep_mid in self.irreps_out:
                        k = len(irreps_mid)
                        irreps_mid.append((mul, irrep_mid))
                        instructions.append((i, j, k, 'uvu', True))
        irreps_mid = Irreps(irreps_mid)
        irreps_mid, p, _ = irreps_mid.sort()

        instructions = [(i_1, i_2, p[i_out], mode, train) for (i_1, i_2, i_out, mode, train) in instructions]

        self.tensor_edge = TensorProduct(self.irreps_in,
                                         self.irreps_edge_attr,
                                         irreps_mid,
                                         instructions,
                                         internal_weights=False,
                                         shared_weights=False)
        
        self.edge2weight = FullyConnectedNet([number_of_basis] + radial_layers * [radial_neurons] + [self.tensor_edge.weight_numel], torch.nn.functional.silu)
        self.linear_output = FullyConnectedTensorProduct(irreps_mid, self.irreps_node_attr, self.irreps_out)

    def forward(self,
                node_input,
                node_attr,
                node_deg,
                edge_src,
                edge_dst,
                edge_attr,
                edge_length_embedded):

        node_input_features = self.linear_input(node_input, node_attr)
        node_features = torch.div(node_input_features, torch.pow(node_deg, 0.5))

        node_mask = self.linear_mask(node_input, node_attr)

        edge_weight = self.edge2weight(edge_length_embedded)
        edge_features = self.tensor_edge(node_features[edge_src], edge_attr, edge_weight)

        node_features = scatter(edge_features, edge_dst, dim = 0, dim_size = node_features.shape[0])
        node_features = torch.div(node_features, torch.pow(node_deg, 0.5))

        node_output_features = self.linear_output(node_features, node_attr)

        node_output = node_output_features

        c_s, c_x = math.sin(math.pi / 8), math.cos(math.pi / 8)
        mask = self.linear_mask.output_mask
        c_x = (1 - mask) + c_x * mask
        return c_s * node_mask + c_x * node_output




class GraphNetwork(torch.nn.Module):
    def __init__(self,
                 mul,
                 irreps_out,
                 lmax,
                 nlayers,
                 number_of_basis,
                 radial_layers,
                 radial_neurons,
                 node_dim,
                 node_embed_dim,
                 input_dim,
                 input_embed_dim):
        super().__init__()
        
        self.mul = mul
        self.irreps_in = Irreps(str(input_embed_dim)+'x0ee')
        self.irreps_node_attr = Irreps(str(node_embed_dim)+'x0ee')
        self.irreps_edge_attr_o3 = o3.Irreps.spherical_harmonics(lmax)
        self.irreps_edge_attr = irreps_o3_to_su2(self.irreps_edge_attr_o3)
        self.irreps_hidden = Irreps([(self.mul, (l, p, t)) for l in range(lmax + 1) for p in [-1, 1] for t in [-1, 1]])
        self.irreps_out = Irreps(irreps_out)
        self.number_of_basis = number_of_basis

        act = {1: torch.nn.functional.silu,
               -1: torch.tanh}
        act_gates = {1: torch.sigmoid,
                     -1: torch.tanh}

        self.layers = torch.nn.ModuleList()
        irreps_in = self.irreps_in

        for _ in range(nlayers):
            irreps_scalars = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l == 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            irreps_gated = Irreps([(mul, ir) for mul, ir in self.irreps_hidden if ir.l > 0 and tp_path_exists(irreps_in, self.irreps_edge_attr, ir)])
            for z_ir in ['0oo','0oe', '0eo', '0ee']:
                if tp_path_exists(irreps_in, self.irreps_edge_attr, z_ir):
                    ir = z_ir
            irreps_gates = Irreps([(mul, ir) for mul, _ in irreps_gated])

            gate = Gate(irreps_scalars, [act[ir.p] for _, ir in irreps_scalars],
                        irreps_gates, [act_gates[ir.p] for _, ir in irreps_gates],
                        irreps_gated)
            conv = GraphConvolution(irreps_in,
                                    self.irreps_node_attr,
                                    self.irreps_edge_attr,
                                    gate.irreps_in,
                                    number_of_basis,
                                    radial_layers,
                                    radial_neurons)
            irreps_in = gate.irreps_out
            self.layers.append(CustomCompose(conv, gate))

        self.layers.append(
            GraphConvolution(
                irreps_in,
                self.irreps_node_attr,
                self.irreps_edge_attr,
                self.irreps_out,
                number_of_basis,
                radial_layers,
                radial_neurons
            )
        )
        self.emx = torch.nn.Linear(input_dim, input_embed_dim, dtype = torch.float64)
        self.emz = torch.nn.Linear(node_dim, node_embed_dim, dtype = torch.float64)

    def forward(self, data):
        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]
        edge_vec = data['edge_vec']
        edge_len = data['edge_len']
        edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'].item(), self.number_of_basis, basis = 'gaussian', cutoff = False)
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr_o3, edge_vec, True, normalization = 'component')
        edge_attr = edge_sh
        numb = data['numb']
        x = torch.relu(self.emx(torch.relu(data['x'])))
        z = torch.relu(self.emz(torch.relu(data['z'])))
        node_deg = data['node_deg']
        # count = 0
        for layer in self.layers:
            x = layer(x, z, node_deg, edge_src, edge_dst, edge_attr, edge_length_embedded)
            # print(f'x [{count}]: ', x.norm())
            # x /= x.norm()+1e-7
            # count += 1
        return x




class Convolution(torch.nn.Module):
    def __init__(self, irreps_in, irreps_sh, irreps_out, num_neighbors) -> None:
        super().__init__()

        self.num_neighbors = num_neighbors

        tp = FullyConnectedTensorProduct(
            irreps_in1=irreps_in,
            irreps_in2=irreps_sh,
            irreps_out=irreps_out,
            internal_weights=False,
            shared_weights=False,
        )
        self.fc = FullyConnectedNet([3, 256, tp.weight_numel], torch.relu)
        self.tp = tp
        self.irreps_out = self.tp.irreps_out

    def forward(self, node_features, edge_src, edge_dst, edge_attr, edge_scalars) -> torch.Tensor:
        weight = self.fc(edge_scalars)
        edge_features = self.tp(node_features[edge_src], edge_attr, weight)
        node_features = scatter(edge_features, edge_dst, dim=0).div(self.num_neighbors**0.5)
        return node_features

class Network(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.num_neighbors = 3.8  # typical number of neighbors
        self.irreps_sh_o3 = o3.Irreps.spherical_harmonics(3)
        self.irreps_sh = irreps_o3_to_su2(self.irreps_sh_o3)

        irreps = self.irreps_sh

        # First layer with gate
        gate = Gate(
            "4x0ee + 4x0eo + 4x0oe + 4x0oo + 4x0ee + 4x0eo + 4x0oe + 4x0oo",
            [torch.relu, torch.abs],  # scalar
            "4x0ee + 4x0eo + 4x0oe + 4x0oo + 4x0ee + 4x0eo + 4x0oe + 4x0oo",
            [torch.relu, torch.tanh, torch.relu, torch.tanh],  # gates (scalars)
            "4x0ee + 4x0eo + 4x0oe + 4x0oo + 4x0ee + 4x0eo + 4x0oe + 4x0oo",  # gated tensors, num_irreps has to match with gates
        )
        self.conv = Convolution(irreps, self.irreps_sh, gate.irreps_in, self.num_neighbors)
        self.gate = gate
        irreps = self.gate.irreps_out

        # Final layer
        self.final = Convolution(irreps, self.irreps_sh, "1x0eo", self.num_neighbors)
        self.irreps_out = self.final.irreps_out

    def forward(self, data) -> torch.Tensor:
        num_nodes = 4  # typical number of nodes

        edge_src = data['edge_index'][0]
        edge_dst = data['edge_index'][1]
        edge_vec = data['edge_vec']
        edge_len = data['edge_len']
        edge_length_embedded = soft_one_hot_linspace(edge_len, 0.0, data['r_max'].item(), self.number_of_basis, basis = 'gaussian', cutoff = False)
        edge_sh = o3.spherical_harmonics(self.irreps_edge_attr, edge_vec, True, normalization = 'component')
        edge_attr = edge_sh
        numb = data['numb']

        x = scatter(edge_attr, edge_dst, dim=0).div(self.num_neighbors**0.5)

        x = self.conv(x, edge_src, edge_dst, edge_attr, edge_length_embedded)
        x = self.gate(x)
        x = self.final(x, edge_src, edge_dst, edge_attr, edge_length_embedded)

        return x    #scatter(x, data.batch, dim=0).div(num_nodes**0.5)
        



