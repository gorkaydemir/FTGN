import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, Tensor

from modules.shallow_networks import MLP_vectornet, LayerNorm, DecoderResCat
from utils.utils import get_dis_point_2_points


class GlobalGraph(nn.Module):
    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1):
        super(GlobalGraph, self).__init__()
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads if attention_head_size is None else attention_head_size
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.num_qkv = 1

        self.query = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.key = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)
        self.value = nn.Linear(hidden_size, self.all_head_size * self.num_qkv)

    def get_extended_attention_mask(self, attention_mask):
        extended_attention_mask = attention_mask.unsqueeze(1)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        return extended_attention_mask

    def transpose_for_scores(self, x):
        sz = x.size()[:-1] + (self.num_attention_heads,
                              self.attention_head_size)
        # (batch, max_vector_num, head, head_size)
        x = x.view(*sz)
        # (batch, head, max_vector_num, head_size)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None, mapping=None):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = nn.functional.linear(hidden_states, self.key.weight)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        # print(attention_scores.shape, attention_mask.shape)
        if attention_mask is not None:
            attention_scores = attention_scores + \
                self.get_extended_attention_mask(attention_mask)

        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer


class CrossAttention(GlobalGraph):
    def __init__(self, hidden_size, attention_head_size=None, num_attention_heads=1, key_hidden_size=None,
                 query_hidden_size=None):
        super(CrossAttention, self).__init__(
            hidden_size, attention_head_size, num_attention_heads)
        if query_hidden_size is not None:
            self.query = nn.Linear(
                query_hidden_size, self.all_head_size * self.num_qkv)
        if key_hidden_size is not None:
            self.key = nn.Linear(
                key_hidden_size, self.all_head_size * self.num_qkv)
            self.value = nn.Linear(
                key_hidden_size, self.all_head_size * self.num_qkv)

    def forward(self, hidden_states_query, hidden_states_key=None, attention_mask=None, mapping=None,
                return_scores=False):
        mixed_query_layer = self.query(hidden_states_query)
        mixed_key_layer = self.key(hidden_states_key)
        mixed_value_layer = self.value(hidden_states_key)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(
            query_layer / math.sqrt(self.attention_head_size), key_layer.transpose(-1, -2))
        if attention_mask is not None:
            assert hidden_states_query.shape[1] == attention_mask.shape[1] \
                   and hidden_states_key.shape[1] == attention_mask.shape[2]
            attention_scores = attention_scores + \
                self.get_extended_attention_mask(attention_mask)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[
                                  :-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        if return_scores:
            return context_layer, torch.squeeze(attention_probs, dim=1)
        return context_layer


class GlobalGraphRes(nn.Module):
    def __init__(self, hidden_size):
        super(GlobalGraphRes, self).__init__()
        self.global_graph = GlobalGraph(hidden_size, hidden_size // 2)
        self.global_graph2 = GlobalGraph(hidden_size, hidden_size // 2)

    def forward(self, hidden_states, attention_mask=None, mapping=None):
        hidden_states = torch.cat([self.global_graph(hidden_states, attention_mask, mapping),
                                   self.global_graph2(hidden_states, attention_mask, mapping)], dim=-1)
        return hidden_states


class SubGraph(nn.Module):
    def __init__(self, hidden_size, depth=3):
        super(SubGraph, self).__init__()

        self.layers = nn.ModuleList(
            [MLP_vectornet(hidden_size, hidden_size // 2) for _ in range(depth)])

    def forward(self, hidden_states, li_vector_num=None):
        sub_graph_batch_size = hidden_states.shape[0]
        max_vector_num = hidden_states.shape[1]
        if li_vector_num is None:
            li_vector_num = [max_vector_num] * sub_graph_batch_size
        hidden_size = hidden_states.shape[2]
        device = hidden_states.device

        attention_mask = torch.zeros([sub_graph_batch_size, max_vector_num, hidden_size // 2],
                                     device=device)
        zeros = torch.zeros([hidden_size // 2], device=device)
        for i in range(sub_graph_batch_size):
            assert li_vector_num[i] > 0
            attention_mask[i][li_vector_num[i]:max_vector_num].fill_(-10000.0)
        for layer_index, layer in enumerate(self.layers):
            new_hidden_states = torch.zeros([sub_graph_batch_size, max_vector_num, hidden_size],
                                            device=device)

            encoded_hidden_states = layer(hidden_states)
            for j in range(max_vector_num):
                attention_mask[:, j] += -10000.0
                max_hidden, _ = torch.max(
                    encoded_hidden_states + attention_mask, dim=1)
                max_hidden = torch.max(max_hidden, zeros)
                attention_mask[:, j] += 10000.0
                new_hidden_states[:, j] = torch.cat(
                    (encoded_hidden_states[:, j], max_hidden), dim=-1)
            hidden_states = new_hidden_states
        return torch.max(hidden_states, dim=1)[0]


class MemoryEncoder(torch.nn.Module):
    def __init__(self, hidden_size, depth=3):
        super(MemoryEncoder, self).__init__()

        self.layer_0 = MLP_vectornet(hidden_size)
        self.layers = nn.ModuleList(
            [GlobalGraph(hidden_size, num_attention_heads=2) for _ in range(depth)])
        self.layers_2 = nn.ModuleList(
            [LayerNorm(hidden_size) for _ in range(depth)])

    def forward(self, hidden_states, polyline_nums):
        device = hidden_states.device
        max_vector_num = hidden_states.shape[1]
        batch_size = hidden_states.shape[0]

        attention_mask = torch.zeros(
            [batch_size, max_vector_num, max_vector_num], device=device)

        hidden_states = self.layer_0(hidden_states.clone())
        for i in range(batch_size):
            assert polyline_nums[i] > 0
            attention_mask[i, :polyline_nums[i], :polyline_nums[i]].fill_(1)

        for layer_index, layer in enumerate(self.layers):
            temp = hidden_states
            hidden_states = layer(hidden_states, attention_mask)
            hidden_states = F.relu(hidden_states)
            hidden_states = hidden_states + temp
            hidden_states = self.layers_2[layer_index](hidden_states)

        return torch.max(hidden_states, dim=1)[0]


class PointSubGraph(nn.Module):
    def __init__(self, hidden_size, agent_hidden_size=None):
        super(PointSubGraph, self).__init__()
        self.hidden_size = hidden_size
        if agent_hidden_size is None:
            agent_hidden_size = self.hidden_size

        self.agent_decoder = DecoderResCat(
            agent_hidden_size, agent_hidden_size, self.hidden_size)
        self.layers = torch.nn.ModuleList([MLP_vectornet(2, hidden_size // 2),
                                           MLP_vectornet(
                                         hidden_size, hidden_size // 2),
                                     MLP_vectornet(hidden_size, hidden_size)])

    def forward(self, hidden_states, agent):
        predict_agent_num, point_num = hidden_states.shape[0], hidden_states.shape[1]
        hidden_size = self.hidden_size
        agent = self.agent_decoder(agent)
        assert (agent.shape[0], agent.shape[1]) == (
            predict_agent_num, hidden_size)

        agent = agent[:, :hidden_size // 2].unsqueeze(1).expand(
            [predict_agent_num, point_num, hidden_size // 2])
        for layer_index, layer in enumerate(self.layers):
            if layer_index == 0:
                hidden_states = layer(hidden_states)
            else:
                hidden_states = layer(
                    torch.cat([hidden_states, agent], dim=-1))

        return hidden_states


class Agent_Feature_Enhancer(nn.Module):
    def __init__(self, args):
        super(Agent_Feature_Enhancer, self).__init__()
        self.device = args.device
        self.hidden_size = args.feature_dim

        # agent decoding related
        self.agent_map_cross_attention = CrossAttention(self.hidden_size)
        self.agent_scene_cross_attention = CrossAttention(self.hidden_size)

    def forward(self, node_features, agent_feature, input_lengths, lane_polylines):

        batch_size = len(agent_feature)
        improved_agent_features = torch.zeros(
            batch_size, self.hidden_size*3, device=self.device)

        for i in range(batch_size):
            # agent_feature_i.shape = (1, 1, hidden_size)
            agent_feature_i = agent_feature[i].unsqueeze(
                dim=0).unsqueeze(dim=0)
            # scene_nodes.shape = (1, #nodes, hidden_size)
            scene_nodes = node_features[i, :input_lengths[i]].unsqueeze(dim=0)
            # map_nodes.shape = (1, #lane_nodes, hidden_size)
            lane_nodes = node_features[i, lane_polylines[i]: input_lengths[i]].unsqueeze(dim=0)

            # agent_lane_attention.shape = agent_scene_attention.shape = (1, hidden_size)
            agent_lane_attention = self.agent_map_cross_attention(
                agent_feature_i, lane_nodes).squeeze(dim=0)
            agent_scene_attention = self.agent_scene_cross_attention(
                agent_feature_i, scene_nodes).squeeze(dim=0)

            # agent_feature_cat.shape = (1, hidden_size*3)
            agent_feature_cat = torch.cat((agent_feature_i.squeeze(
                dim=0), agent_scene_attention, agent_lane_attention), dim=-1)

            improved_agent_features[i] = agent_feature_cat

        return improved_agent_features


class Endpoint_Refinement(nn.Module):
    def __init__(self, args):
        super(Endpoint_Refinement, self).__init__()
        self.device = args.device
        self.hidden_size = args.feature_dim
        self.n_endpoints = 6

        self.endpoint_source = args.endpoint_source
        self.temperature = args.temperature
        self.temperature_validation = args.temperature_validation

        self.all_endpoints = None
        # endpoint proposal related
        self.gt_endpoint_distribution = None

        if self.endpoint_source == "agent":
            self.endpoint_proposer_agent = DecoderResCat(
                self.hidden_size, self.hidden_size*3, self.n_endpoints*2)

        elif self.endpoint_source == "context":
            self.endpoint_proposer_map = DecoderResCat(
                self.hidden_size, self.hidden_size, self.n_endpoints*2)

        elif self.endpoint_source == "both":
            self.endpoint_proposer_agent = DecoderResCat(
                self.hidden_size, self.hidden_size*3, self.n_endpoints)
            self.endpoint_proposer_map = DecoderResCat(
                self.hidden_size, self.hidden_size, self.n_endpoints)

        # endpoint decoding
        self.endpoint_proposal_decoder = PointSubGraph(
            self.hidden_size, agent_hidden_size=self.hidden_size*3)
        self.endpoint_scene_cross_attention = CrossAttention(self.hidden_size)
        self.endpoint_decoder = DecoderResCat(
            self.hidden_size*3, self.hidden_size*5, self.hidden_size)

        self.scorer = DecoderResCat(self.hidden_size, self.hidden_size + 2, 1)
        self.refiner = DecoderResCat(
            self.hidden_size, self.hidden_size + 2, 2)
        self.traj_regressor = DecoderResCat(
            self.hidden_size, 4*self.hidden_size + 2, 29*2)

        self.traj_completion_criterion = torch.nn.SmoothL1Loss()

    def get_point_feature(self, i, endpoints_i, node_features, agent_feature, input_lengths, lane_polylines):
        endpoint_features = self.endpoint_proposal_decoder(
            endpoints_i.unsqueeze(dim=0), agent_feature[i].unsqueeze(dim=0)).squeeze(dim=0)
        endpoint_features_attention = \
            self.endpoint_scene_cross_attention(endpoint_features.unsqueeze(
                dim=0), node_features[i, :input_lengths[i]].unsqueeze(dim=0)).squeeze(dim=0)
        endpoint_features = torch.cat([agent_feature[i].unsqueeze(0).repeat(
            len(endpoints_i), 1), endpoint_features, endpoint_features_attention], dim=-1)
        # endpoint_features.shape = (n_endpoints, hidden_size)
        endpoint_features = self.endpoint_decoder(endpoint_features)
        # concated_endpoint_features.shape = (n_endpoints, hidden_size + 2)
        concated_endpoint_features = torch.cat(
            (endpoint_features, endpoints_i), dim=-1)

        return concated_endpoint_features

    def forward(self, node_features, agent_feature, map_features, input_lengths, lane_polylines, gt_endpoints, test=False, validate=False):
        batch_size = len(node_features)
        scores = torch.zeros(
            batch_size, self.n_endpoints, device=self.device)

        all_endpoints = torch.zeros(
            batch_size, self.n_endpoints, 2, device=self.device)
        endpoints = torch.zeros(
            batch_size, 6, 2, device=self.device)

        metric_probs = torch.zeros(
            batch_size, 6, device=self.device)

        # if not self.dense_points:
        # endpoint_proposals.shape = (batch_size, n_endpoints, 2)
        if self.endpoint_source == "agent":
            endpoint_proposals = self.endpoint_proposer_agent(
                agent_feature).view(batch_size, self.n_endpoints, 2)

        elif self.endpoint_source == "context":
            endpoint_proposals = self.endpoint_proposer_map(
                map_features).view(batch_size, self.n_endpoints, 2)

        elif self.endpoint_source == "both":
            endpoint_proposals_map = self.endpoint_proposer_map(
                map_features).view(batch_size, self.n_endpoints//2, 2)
            endpoint_proposals_agent = self.endpoint_proposer_agent(
                agent_feature).view(batch_size, self.n_endpoints//2, 2)
            endpoint_proposals = torch.cat(
                [endpoint_proposals_map, endpoint_proposals_agent], dim=1)

        if test:
            probs = torch.zeros(
                batch_size, 6, device=self.device)
            for i in range(batch_size):
                # === getting endpoint specific feature ===
                concated_endpoint_features = self.get_point_feature(
                    i, endpoint_proposals[i].detach().clone(), node_features, agent_feature, input_lengths, lane_polylines)

                offsets_i = self.refiner(concated_endpoint_features)
                endpoints_i = endpoint_proposals[i] + offsets_i
                final_endpoint_features = self.get_point_feature(
                    i, endpoints_i.detach().clone(), node_features, agent_feature, input_lengths, lane_polylines)

                # === getting probs from endpoint features ===
                scores_i = self.scorer(
                    final_endpoint_features).squeeze(dim=-1)/self.temperature

                probs[i] = F.softmax(scores_i, dim=-1)
                endpoints[i] = endpoints_i

            return endpoints, probs

        for i in range(batch_size):
            # === getting endpoint specific feature ===
            # endpoint specific features
            concated_endpoint_features = self.get_point_feature(
                i, endpoint_proposals[i].detach().clone(), node_features, agent_feature, input_lengths, lane_polylines)

            # offsets from these features
            offsets_i = self.refiner(concated_endpoint_features)

            # final endpoints and it's features for batch
            endpoints_i = endpoint_proposals[i] + offsets_i
            final_endpoint_features = self.get_point_feature(
                i, endpoints_i.detach().clone(), node_features, agent_feature, input_lengths, lane_polylines)

            # scores_i corresponds to scores of each endpoint
            scores_i = self.scorer(final_endpoint_features).squeeze(
                dim=-1)

            if validate or not self.temperature_validation:
                scores_i = scores_i/self.temperature

            scores[i] = scores_i
            metric_probs[i] = F.softmax(scores_i, dim=-1)

            endpoints[i] = endpoints_i
            all_endpoints[i] = endpoints_i
        self.all_endpoints = all_endpoints
        self.get_true_regions(all_endpoints, gt_endpoints)

        endpoint_loss = self.endpoint_loss(endpoints, gt_endpoints, scores)

        # if not self.dense_points:
        endpoint_CE = F.cross_entropy(scores, self.gt_endpoint_distribution)

        # return endpoints, endpoint_CE, metric_probs, diversity_loss
        return endpoints, endpoint_CE, metric_probs, endpoint_loss

    def get_true_regions(self, endpoint_proposals, gt_endpoints):
        # === getting probs for endpoints, similar to TNT trajectory scoring ===
        # batch_size = len(gt_endpoints)
        gt_endpoints = torch.vstack(gt_endpoints).unsqueeze(dim=1)
        distances = torch.norm(endpoint_proposals.detach()
                               - gt_endpoints, dim=-1, p=2)

        min_indices = torch.argmin(distances, dim=-1)
        self.gt_endpoint_distribution = min_indices

    def get_goal_conditioned_traj(self, node_features, agent_feature, input_lengths, lane_polylines, endpoints):
        batch_size = len(node_features)
        outputs = torch.zeros(batch_size, 6, 30, 2, device=self.device)

        for i in range(batch_size):
            # === getting endpoint specific feature ===
            concated_endpoint_features = self.get_point_feature(
                i, endpoints[i].detach().clone(), node_features, agent_feature, input_lengths, lane_polylines)

            concated_endpoint_features = torch.cat(
                (concated_endpoint_features, agent_feature[i].unsqueeze(0).repeat(6, 1)), dim=1)
            # === regress trajectory ===
            traj = self.traj_regressor(
                concated_endpoint_features).view(6, 29, 2)
            traj = torch.cat((traj, endpoints[i].unsqueeze(dim=1)), dim=1)
            outputs[i] = traj

        return outputs

    def endpoint_loss(self, endpoints, gt_endpoints, scores):
        batch_size = len(gt_endpoints)

        closest_endpoint_diff = torch.zeros(batch_size, device=self.device)
        for i in range(batch_size):
            # ==== loss on endpoint proposals ====
            proposal_argmin = np.argmin(get_dis_point_2_points(
                                gt_endpoints[i].cpu().numpy(),
                                np.array(endpoints[i].tolist())))
            closest_endpoint_diff[i] = self.traj_completion_criterion(
                                endpoints[i, proposal_argmin],
                                    gt_endpoints[i])

        return closest_endpoint_diff.mean()
