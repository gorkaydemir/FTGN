import numpy as np
import torch


def get_edge_pairs(mappings, device, agent_of_interest_edges, edge_distance_treshold, distance_based_edges):
    scenes = []
    scene_timestamps = []
    for mappins_idx, mapping in enumerate(mappings):
        scene = []
        edges, unique_ts = get_edge_data(
            mapping, agent_of_interest_edges, edge_distance_treshold, distance_based_edges)
        scene_timestamps.append(unique_ts)
        for ts_value in unique_ts:
            ts_edges = edges[edges[:, 2] == ts_value][:, :2]
            ts_edges = torch.from_numpy(ts_edges).to(device).long()
            scene.append(ts_edges)

        scenes.append(scene)

    return scenes, scene_timestamps


class Map_Distance_Instance:
    def __init__(self, mapping):
        matrix = mapping["matrix"]
        map_start_polyline_idx = mapping["map_start_polyline_idx"]
        map_polyline_spans = mapping["polyline_spans"][map_start_polyline_idx:]

        lane_coordinates = []
        for i, polyline_span in enumerate(map_polyline_spans):
            map_polyline = matrix[polyline_span]

            # vector[-1], vector[-2]: x_0, y_0
            # vector[-3], vector[-4]: x_1, y_1
            samples = [i[-4:] for i in map_polyline]
            for j, coordinates in enumerate(samples):
                y_1, x_1, y_0, x_0 = coordinates
                start = (i + map_start_polyline_idx, x_0, y_0)
                lane_coordinates.append(start)

                if j == len(samples) - 1:
                    end = (i + map_start_polyline_idx, x_1, y_1)
                    lane_coordinates.append(end)

        # lane_coordinates[i] = (lane_segment_id, x, y)
        # lane_coordinates.shape = (sum(len(lane_segments)), 3)
        lane_coordinates = np.array(lane_coordinates)
        self.lane_indexes = lane_coordinates[:, 0]
        self.lane_locations = lane_coordinates[:, -2:]

    def smallest_lane(self, timestamp):
        location = timestamp[1:3]
        norms = np.linalg.norm(self.lane_locations - location, axis=1)
        smallest_index = np.argsort(norms)[0]
        smallest_lane = self.lane_indexes[smallest_index]
        return smallest_lane

    def close_lanes(self, related_timestamps, treshold):
        targets = []
        timestamps = []
        for ts_idx in range(len(related_timestamps)):
            ts = related_timestamps[ts_idx][-1]
            timestamp = related_timestamps[related_timestamps[:, -1] == ts]
            location = timestamp[0, 1:3]
            norms = np.linalg.norm(self.lane_locations - location, axis=1)
            indices = np.where(norms < treshold)
            close_lanes = self.lane_indexes[indices]
            close_lanes = np.unique(close_lanes).tolist()

            targets.extend(close_lanes)
            timestamps.extend([ts]*len(close_lanes))

        return targets, timestamps

    def get_closest_lanes_to_agent(self, related_timestamps):
        lane_list = np.zeros((20, 2))
        for ts_idx in range(len(related_timestamps)):
            ts = related_timestamps[ts_idx][-1]
            target_id = self.smallest_lane(related_timestamps[ts_idx])
            lane_list[ts_idx, 0] = target_id
            lane_list[ts_idx, 1] = ts

        return lane_list


def get_edge_data(mapping, agent_of_interest_edges, edge_distance_treshold, distance_based_edges):

    map_distance = Map_Distance_Instance(mapping)
    agents = mapping["agents"]
    timestamps = []

    # create agent node coordinates based on timestamps
    for i, agent in enumerate(agents):
        N = agent.shape[0]
        for j in range(N):
            x, y, timestamp = agent[j]
            timestamps.append((i, x, y, timestamp))

    # timestamps[i] = (agent_id, x, y, timestamp)
    # timestamps.shape = (sum(len(trajectories)), 4)
    timestamps = np.array(timestamps)
    # timestamps = timestamps[np.lexsort((timestamps[:, 0], timestamps[:, 3]))]

    edges = []
    max_agent = np.max(timestamps[:, 0]).astype(int)
    for src in range(max_agent + 1):
        related_timestamps = timestamps[timestamps[:, 0] == src]
        target_ids = []
        timestamp_acc = []

        # agent to lane segment edges
        if distance_based_edges:
            target_ids, timestamp_acc = map_distance.close_lanes(
                related_timestamps, edge_distance_treshold)

        else:
            lane_list = map_distance.get_closest_lanes_to_agent(
                related_timestamps)
            for ts_idx in range(len(related_timestamps)):
                target_id_t = lane_list[ts_idx, 0]
                ts = lane_list[ts_idx, 1]
                target_ids.append(target_id_t)
                timestamp_acc.append(ts)

                if ts_idx + 1 < len(related_timestamps):
                    target_id_t1 = lane_list[ts_idx + 1, 0]
                    if target_id_t1 != target_id_t:
                        target_ids.append(target_id_t1)
                        timestamp_acc.append(ts)

                    if ts_idx + 2 < len(related_timestamps):
                        target_id_t2 = lane_list[ts_idx + 2, 0]
                        if target_id_t2 != target_id_t and target_id_t2 != target_id_t1:
                            target_ids.append(target_id_t2)
                            timestamp_acc.append(ts)

        # agent to agent edges
        if (not agent_of_interest_edges) or (src == 0):
            for ts_idx in range(len(related_timestamps)):
                ts = related_timestamps[ts_idx, -1]
                available_agents = np.unique(
                    timestamps[timestamps[:, -1] == ts][:, 0])
                target_ids.extend(available_agents.tolist())
                timestamp_acc.extend([ts]*len(available_agents))

        for trg, ts in zip(target_ids, timestamp_acc):
            edges.append([src, trg, ts])
            edges.append([trg, src, ts])
            # if trg > max_agent:
            #     edges.append([trg, src, ts])

    edges = np.array(edges)
    # edges = edges[np.lexsort((edges[:, 1], edges[:, 0], edges[:, 2]))]
    return edges, np.unique(timestamps[:, 3])
