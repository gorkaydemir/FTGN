import numpy as np
import torch

# encoder modules
from model.vectornet import VectorNet
from modules.graph_networks import GlobalGraphRes, MemoryEncoder
from modules.graph_networks import Agent_Feature_Enhancer, Endpoint_Refinement
from modules.shallow_networks import DecoderResCat, TimeFuser

# data related functions
from utils.utils import get_from_mapping, get_labels

# coordinate operations
from utils.utils import get_dis_point_2_points, to_origin_coordinate


class Forecasting_TGN(torch.nn.Module):
    def __init__(self, args):
        super(Forecasting_TGN, self).__init__()

        self.feature_dim = args.feature_dim
        self.device = args.device
        self.node_features = None
        self.agent_node_features = None
        self.validate = None
        self.epoch_num = args.epoch_num
        self.test = args.test

        # Loss coefficients
        self.lambda1 = args.lambda1     # Trajectory L1 loss
        self.lambda2 = args.lambda2     # Endpoint L1 loss
        self.lambda3 = args.lambda3     # Endpoint CE loss

        # Temporal Graph related variables
        self.time_split_size = args.time_split_size
        self.fuse_time = args.fuse_time
        self.exclude_temporal_encoding = args.exclude_temporal_encoding
        self.sequential_encoding = args.sequential_encoding
        self.scene_memory = args.scene_memory

        # Feature Extraction Modules
        self.vectornet = VectorNet(
            args.feature_dim, args.device, args.data_augmentation)
        self.temporal_graph = GlobalGraphRes(self.feature_dim)
        self.time_fuser = TimeFuser(self.feature_dim)
        self.sequential_encoder = torch.nn.GRUCell(
            self.feature_dim, self.feature_dim)
        self.agent_feature_enhancer = Agent_Feature_Enhancer(args)

        fuser_dim_scale = 2
        if self.scene_memory:
            fuser_dim_scale += 1
        if self.sequential_encoding:
            fuser_dim_scale += 1

        self.feature_fuser = DecoderResCat(
            2*self.feature_dim, fuser_dim_scale*self.feature_dim, out_features=self.feature_dim)

        self.map_decoder = DecoderResCat(
            self.feature_dim, 2*self.feature_dim, self.feature_dim)
        # Memories
        self.graph_memory_encoder = MemoryEncoder(self.feature_dim)
        self.sequential_memory_encoder = torch.nn.GRUCell(
            self.feature_dim, self.feature_dim)

        # Endpoint realted
        self.prediction_heads = Endpoint_Refinement(args)

        # criterions
        self.traj_completion_criterion = torch.nn.SmoothL1Loss()

    def forward(self, mapping, validate=False):
        # timestamps = [scene[2] for scene in batch]
        # edge_pairs = [scene[1] for scene in batch]
        # edge_pairs = [[i.to(self.device) for i in scene]
        #               for scene in edge_pairs]
        # mapping = [scene[0] for scene in batch]

        # ================= Static Graph Encoding =================
        edge_pairs = get_from_mapping(mapping, 'scenes')
        edge_pairs = [[i.to(self.device) for i in scene]
                      for scene in edge_pairs]
        timestamps = get_from_mapping(mapping, 'scene_timestamps')
        # labels = get_from_mapping(mapping, 'labels')
        labels_is_valid = get_from_mapping(
                mapping, 'labels_is_valid')
        lane_polylines = get_from_mapping(mapping, "map_start_polyline_idx")

        self.validate = validate
        batch_size = len(mapping)
        # node_features.shape = (batch_size, max(#polylines), feature_dim)
        node_features, polyline_nums, labels = self.vectornet(
            mapping, validate)

        self.label_dict = get_labels(labels, self.device)

        global_features = node_features.clone()
        self.node_features = node_features
        hidden_agent_states = torch.zeros(
            node_features[:, 0, :].shape, device=self.device)
        scene_memories = self.graph_memory_encoder(
            self.node_features, polyline_nums)

        # ================= Temporal Graph Encoding =================
        if not self.exclude_temporal_encoding:
            for timestamp_id in range(0, 20, self.time_split_size):
                ts_edge_pairs = [torch.unique(torch.cat(scene_edges[timestamp_id: timestamp_id
                                                                    + self.time_split_size]), dim=0) for scene_edges in edge_pairs]
                ts = torch.tensor([scene_timestamps[timestamp_id]
                                  for scene_timestamps in timestamps], device=self.device).float()
                self.encode_temp_edges(mapping, ts_edge_pairs, ts)

                if self.sequential_encoding:
                    temporal_agent_states = self.node_features[:, 0, :].clone()
                    hidden_agent_states = self.sequential_encoder(
                        temporal_agent_states, hidden_agent_states)

                if self.scene_memory:
                    temporal_graph_feature = self.graph_memory_encoder(
                        self.node_features.clone(), polyline_nums)
                    scene_memories = self.sequential_memory_encoder(
                        temporal_graph_feature, scene_memories)

            # residual connection between global and temporal features
            self.agent_node_features = self.node_features[:, 0, :]
            cat_feature = torch.cat((self.agent_node_features,
                                     global_features[:, 0, :].clone()), dim=-1)
            if self.scene_memory:
                cat_feature = torch.cat((cat_feature,
                                         scene_memories), dim=-1)
            if self.sequential_encoding:
                cat_feature = torch.cat((cat_feature,
                                         hidden_agent_states), dim=-1)

            self.agent_node_features = self.feature_fuser(cat_feature)
            self.agent_node_features = self.agent_feature_enhancer(
                self.node_features, self.agent_node_features, polyline_nums, lane_polylines)
            # now, self.agent_node_features.shape = (batch_size, hidden_size*3)

            map_features = self.get_map_feature(
                scene_memories, polyline_nums, lane_polylines)

        # ================= No Temporal Graph Encoding =================
        else:
            self.agent_node_features = self.node_features[:, 0, :]

        # ================= Endpoint Oriented Trajectory Regression =================

        if self.test:
            endpoints, pred_probs = self.endpoint_refiner(
                self.node_features, self.agent_node_features, map_features, polyline_nums, lane_polylines, self.label_dict["gt_endpoints_torch"], True)
            outputs = self.endpoint_refiner.get_goal_conditioned_traj(
                self.node_features, self.agent_node_features, polyline_nums, lane_polylines, endpoints)

            outputs = np.array(outputs.tolist())
            pred_probs = np.array(pred_probs.tolist(
            ), dtype=np.float32) if pred_probs is not None else pred_probs
            for i in range(batch_size):
                for each in outputs[i]:
                    to_origin_coordinate(each, i)

            return outputs, pred_probs

        # ========= Get Endpoints =========
        endpoints, endpoint_CE, metric_probs, endpoint_loss = self.prediction_heads(
            self.node_features, self.agent_node_features, map_features, polyline_nums,
            lane_polylines, self.label_dict["gt_endpoints_torch"], validate=self.validate)

        # ========= Prediction of Trajectories =========
        outputs = self.prediction_heads.get_goal_conditioned_traj(
            self.node_features, self.agent_node_features, polyline_nums, lane_polylines, endpoints)

        traj_loss = self.get_traj_loss(batch_size, outputs, labels_is_valid)

        total_traj_loss = traj_loss*self.lambda1
        total_endpoint_loss = endpoint_loss*self.lambda2
        total_endpoint_CE = endpoint_CE*self.lambda3

        if not self.validate:
            return total_traj_loss + total_endpoint_loss + total_endpoint_CE

        outputs = np.array(outputs.tolist())
        metric_probs = np.array(metric_probs.tolist(
        ), dtype=np.float32)

        for i in range(batch_size):
            for each in outputs[i]:
                to_origin_coordinate(each, i)

        return outputs, metric_probs

    def get_traj_loss(self, batch_size, outputs, labels_is_valid):
        traj_l1_losses = torch.zeros(batch_size, device=self.device)

        for i in range(batch_size):
            argmin = np.argmin(get_dis_point_2_points(
                    self.label_dict["gt_endpoints_np"][i],
                    np.array(outputs[i, :, -1, :].tolist())))
            loss_ = self.traj_completion_criterion(outputs[i, argmin],
                                                   self.label_dict["gt_points_torch"][i],)
            loss_ = loss_ * \
                torch.tensor(labels_is_valid[i], device=self.device, dtype=torch.float).view(
                        30, 1)

            if labels_is_valid[i].sum() > 1e-5:
                traj_l1_losses[i] += loss_.sum() / labels_is_valid[i].sum()

        traj_loss = traj_l1_losses.mean()

        return traj_loss

    def encode_temp_edges(self, mapping, ts_edge_pairs, ts):
        lane_polylines = get_from_mapping(mapping, "map_start_polyline_idx")

        batch_size = self.node_features.shape[0]
        max_poly_num = self.node_features.shape[1]
        attention_mask = torch.zeros(
            [batch_size, max_poly_num, max_poly_num], device=self.device)
        for i in range(batch_size):
            pairs = ts_edge_pairs[i]
            attention_mask[i][pairs[:, 0], pairs[:, 1]] = 1

        if self.fuse_time:
            # time encoded node features
            self.node_features = self.time_fuser(
                self.node_features, ts, lane_polylines)

        self.node_features = self.temporal_graph(
            self.node_features, attention_mask, mapping)

    def get_map_feature(self, scene_memories, polyline_nums, lane_polylines):
        batch_size = len(scene_memories)
        map_features = torch.zeros(
            batch_size, self.feature_dim, device=self.device)

        for i in range(batch_size):
            scene_memory_i = scene_memories[i]
            lane_nodes = self.node_features[i,
                                            lane_polylines[i]: polyline_nums[i]]
            lane_nodes = torch.max(lane_nodes, dim=0)[0]

            cat_map_feature = torch.cat(
                [scene_memory_i, lane_nodes], dim=0).unsqueeze(dim=0)
            map_features[i] = self.map_decoder(cat_map_feature)

        return map_features
