from typing import Dict, List, Tuple, NamedTuple, Any

import random
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

import math
# model related
from modules.graph_networks import SubGraph, GlobalGraphRes
from modules.shallow_networks import DecoderResCat, MLP_vectornet

# data related functions
from utils.utils import get_from_mapping, batch_init, merge_tensors


# tensor operations
from utils.utils import merge_tensors_not_add_dim


class VectorNet(nn.Module):

    def __init__(self, hidden_size, device, data_augmentation):
        super(VectorNet, self).__init__()

        self.sub_graph = SubGraph(hidden_size)
        self.global_graph = GlobalGraphRes(hidden_size)
        self.device = device
        self.hidden_size = hidden_size
        self.data_augmentation = data_augmentation

    def augment_tensor(self, tensor, polyline_idx, map_start_polyline_idx, random_scale):

        # === random scale and random polyline perturbation ===
        if len(tensor) >= 3:
            perturbation = torch.randn(2).to(self.device)*0.2
            if polyline_idx >= map_start_polyline_idx:  # lane vector
                # scale points
                tensor[:, -4:] *= random_scale
                # shift y, x values
                tensor[:, -4:-2] += perturbation
                tensor[:, -2:] += perturbation

            else:                                       # agent vector
                # scale points
                tensor[:, :4] *= random_scale
                # shift x, y values
                tensor[:, 2:4] += perturbation
                tensor[:, :2] += perturbation

        # === =================== ===

        return tensor

    def forward_encode_sub_graph(self, mapping, matrix, polyline_spans, batch_size, validate):

        input_list_list = []

        labels = []
        if (not self.data_augmentation) or validate:
            labels = get_from_mapping(mapping, 'labels')

        original_labels = get_from_mapping(mapping, 'labels')

        for i in range(batch_size):
            random_scale = 0.75 + torch.rand(1).item()/2.0

            if self.data_augmentation and not validate:
                new_label = original_labels[i]
                new_label *= random_scale
                labels.append(new_label)

            input_list = []
            map_start_polyline_idx = mapping[i]['map_start_polyline_idx']

            for j, polyline_span in enumerate(polyline_spans[i]):
                tensor = torch.tensor(
                    matrix[i][polyline_span], device=self.device)

                if self.data_augmentation and not validate:
                    tensor = self.augment_tensor(
                        tensor, j, map_start_polyline_idx, random_scale)

                input_list.append(tensor)

            input_list_list.append(input_list)

        polyline_vectors = input_list_list
        # run subgraph on vector elements
        element_states_batch = merge_tensors_not_add_dim(polyline_vectors,
                                                         module=self.sub_graph,
                                                         sub_batch_size=64,
                                                         device=self.device,
                                                         hidden_size=self.hidden_size)

        return element_states_batch, labels

    def forward(self, mapping, validate):

        matrix = get_from_mapping(mapping, 'matrix')
        polyline_spans = get_from_mapping(mapping, 'polyline_spans')

        batch_size = len(matrix)

        batch_init(mapping)

        element_states_batch, labels_list = self.forward_encode_sub_graph(
            mapping, matrix, polyline_spans, batch_size, validate)

        inputs, inputs_lengths = merge_tensors(
            element_states_batch, device=self.device, hidden_size=self.hidden_size)

        max_poly_num = max(inputs_lengths)
        attention_mask = torch.zeros(
            [batch_size, max_poly_num, max_poly_num], device=self.device)

        for i, length in enumerate(inputs_lengths):
            attention_mask[i][:length][:length].fill_(1)

        # L2 normalize polyline nodes
        inputs = F.normalize(inputs, p=2, dim=2)

        hidden_states = self.global_graph(
            inputs, attention_mask, mapping)

        return hidden_states, inputs_lengths, labels_list
