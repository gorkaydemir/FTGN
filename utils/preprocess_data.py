import sys
import argparse
import math
import numpy as np
import os
from tqdm import tqdm
import zlib
import pickle
from argoverse.map_representation.map_api import ArgoverseMap


import multiprocessing
from multiprocessing import Process
from utils import rotate
from create_edges import get_edge_pairs

parser = argparse.ArgumentParser("Forecasting TGN Argoverse Preprocess")

# === Data Related Parameters ===
parser.add_argument('--data_dir', type=str, help="Path of dataset file")
parser.add_argument('--output_dir', type=str, help="Path of output ex-file")
parser.add_argument('--core_num', type=int, default=8)
parser.add_argument('--feature_size', type=int, default=128)
parser.add_argument('--max_distance', type=float, default=50.0)
parser.add_argument('--validation', action="store_true")
parser.add_argument('--test', action="store_true")

parser.add_argument('--agent_of_interest_edges', action="store_true")
parser.add_argument('--distance_based_edges', action="store_true")
parser.add_argument('--edge_distance_treshold', type=float, default=2.0,
                    help="Distance treshold to create edge between lane nodes")

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


class Arguments:
    def __init__(self):
        self.data_dir = [args.data_dir]
        self.output_dir = args.output_dir
        self.core_num = args.core_num
        self.validation = args.validation
        self.test = args.test
        self.feature_size = args.feature_size
        self.max_distance = args.max_distance

        self.agent_of_interest_edges = args.agent_of_interest_edges
        self.edge_distance_treshold = args.edge_distance_treshold
        self.distance_based_edges = args.distance_based_edges


def get_angle(x, y):
    return math.atan2(y, x)


def larger(a, b):
    return a > b + 1e-5


def get_pad_vector(li, feature_size):
    """
    Pad vector to length of args.hidden_size
    """
    assert len(li) <= feature_size
    li.extend([0] * (feature_size - len(li)))
    return li


def get_dis(points: np.ndarray, point_label):
    return np.sqrt(np.square((points[:, 0] - point_label[0])) + np.square((points[:, 1] - point_label[1])))


TIMESTAMP = 0
TRACK_ID = 1
OBJECT_TYPE = 2
X = 3
Y = 4
CITY_NAME = 5

type2index = {}
type2index["OTHERS"] = 0
type2index["AGENT"] = 1
type2index["AV"] = 2

max_vector_num = 0

VECTOR_PRE_X = 0
VECTOR_PRE_Y = 1
VECTOR_X = 2
VECTOR_Y = 3


def get_sub_map(args, x, y, city_name, vectors=[], polyline_spans=[], mapping=None):
    """
    Calculate lanes which are close to (x, y) on map.
    Only take lanes which are no more than args.max_distance away from (x, y).
    """

    assert isinstance(am, ArgoverseMap)

    lane_ids = []
    for agent_idx, agent in enumerate(mapping["agents"]):
        sample_num = agent.shape[0]
        for i in range(sample_num):

            bias_x, bias_y = rotate(
                agent[i][0], agent[i][1], -1*mapping["angle"])
            temp_x, temp_y = (bias_x + x), (bias_y + y)
            lane_ids_temp = am.get_lane_ids_in_xy_bbox(
                temp_x, temp_y, city_name, query_search_range_manhattan=args.max_distance)
            lane_ids.extend(lane_ids_temp)
            # if agent_idx == 0:
            #     bias_x, bias_y = rotate(
            #         agent[i][0], agent[i][1] + 50, -1*mapping["angle"])
            #     temp_x, temp_y = (bias_x + x), (bias_y + y)
            #     lane_ids_temp = am.get_lane_ids_in_xy_bbox(
            #         temp_x, temp_y, city_name, query_search_range_manhattan=args.max_distance)
            #     lane_ids.extend(lane_ids_temp)

    lane_ids = list(set(lane_ids))
    local_lane_centerlines = [am.get_lane_segment_centerline(
        lane_id, city_name) for lane_id in lane_ids]
    polygons = local_lane_centerlines

    polygons = [polygon[:, :2].copy() for polygon in polygons]
    angle = mapping['angle']
    for index_polygon, polygon in enumerate(polygons):
        for i, point in enumerate(polygon):
            point[0], point[1] = rotate(point[0] - x, point[1] - y, angle)

    local_lane_centerlines = [polygon for polygon in polygons]

    def dis_2(point):
        return point[0] * point[0] + point[1] * point[1]

    def get_dis(point_a, point_b):
        return np.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2)

    def get_dis_for_points(point, polygon):
        dis = np.min(
            np.square(polygon[:, 0] - point[0]) + np.square(polygon[:, 1] - point[1]))
        return np.sqrt(dis)

    def ok_dis_between_points(points, points_, limit):
        dis = np.inf
        for point in points:
            dis = np.fmin(dis, get_dis_for_points(point, points_))
            if dis < limit:
                return True
        return False

    def get_hash(point):
        return round((point[0] + 500) * 100) * 1000000 + round((point[1] + 500) * 100)

    lane_idx_2_polygon_idx = {}
    for polygon_idx, lane_idx in enumerate(lane_ids):
        lane_idx_2_polygon_idx[lane_idx] = polygon_idx

    for index_polygon, polygon in enumerate(polygons):
        assert 2 <= len(polygon) <= 10

        start = len(vectors)

        assert len(lane_ids) == len(polygons)
        lane_id = lane_ids[index_polygon]
        lane_segment = am.city_lane_centerlines_dict[city_name][lane_id]
        assert len(polygon) >= 2
        for i, point in enumerate(polygon):
            if i > 0:
                vector = [0] * args.feature_size
                vector[-1 - VECTOR_PRE_X], vector[-1
                                                  - VECTOR_PRE_Y] = point_pre[0], point_pre[1]
                vector[-1 - VECTOR_X], vector[-1
                                              - VECTOR_Y] = point[0], point[1]
                vector[-5] = 1
                vector[-6] = i
                vector[-7] = len(polyline_spans)
                vector[-8] = 1 if lane_segment.has_traffic_control else -1
                vector[-9] = 1 if lane_segment.turn_direction == 'RIGHT' else \
                    -1 if lane_segment.turn_direction == 'LEFT' else 0
                vector[-10] = 1 if lane_segment.is_intersection else -1
                point_pre_pre = (
                    2 * point_pre[0] - point[0], 2 * point_pre[1] - point[1])
                if i >= 2:
                    point_pre_pre = polygon[i - 2]
                vector[-17] = point_pre_pre[0]
                vector[-18] = point_pre_pre[1]

                vectors.append(vector)
            point_pre = point

        end = len(vectors)
        if start < end:
            polyline_spans.append([start, end])

    return (vectors, polyline_spans)


def preprocess(args, id2info, mapping):
    """
    This function calculates matrix based on information from get_instance.
    """
    polyline_spans = []
    keys = list(id2info.keys())
    assert 'AV' in keys
    assert 'AGENT' in keys
    keys.remove('AV')
    keys.remove('AGENT')
    keys = ['AGENT', 'AV'] + keys
    vectors = []
    two_seconds = mapping['two_seconds']
    mapping['trajs'] = []
    mapping['agents'] = []
    for id in keys:
        polyline = {}

        info = id2info[id]
        start = len(vectors)

        agent = []
        for i, line in enumerate(info):
            if larger(line[TIMESTAMP], two_seconds):
                break
            agent.append((line[X], line[Y], line[TIMESTAMP]))

        for i, line in enumerate(info):
            if larger(line[TIMESTAMP], two_seconds):
                break
            x, y = line[X], line[Y]
            if i > 0:
                # print(x-line_pre[X], y-line_pre[Y])
                vector = [line_pre[X], line_pre[Y], x, y, line[TIMESTAMP], line[OBJECT_TYPE] == 'AV',
                          line[OBJECT_TYPE] == 'AGENT', line[OBJECT_TYPE] == 'OTHERS', len(polyline_spans), i]
                vectors.append(get_pad_vector(vector, args.feature_size))
            line_pre = line

        end = len(vectors)
        if end - start == 0:
            assert id != 'AV' and id != 'AGENT'
        else:
            mapping['agents'].append(np.array(agent))

            polyline_spans.append([start, end])

    assert len(mapping['agents']) == len(polyline_spans)

    assert len(vectors) <= max_vector_num

    t = len(vectors)
    mapping['map_start_polyline_idx'] = len(polyline_spans)

    vectors, polyline_spans = get_sub_map(args, mapping['cent_x'], mapping['cent_y'], mapping['city_name'],
                                          vectors=vectors,
                                          polyline_spans=polyline_spans, mapping=mapping)

    matrix = np.array(vectors)
    labels = []
    info = id2info['AGENT']
    info = info[mapping['agent_pred_index']:]
    if not args.test:
        assert len(info) == 30
    for line in info:
        labels.append(line[X])
        labels.append(line[Y])

    if args.test:
        labels = [0.0 for _ in range(60)]

    mapping.update(dict(
        matrix=matrix,
        labels=np.array(labels).reshape([30, 2]),
        polyline_spans=[slice(each[0], each[1]) for each in polyline_spans],
        labels_is_valid=np.ones(30, dtype=np.int64),
        eval_time=30,
    ))

    scenes, scene_timestamps = get_edge_pairs(
        [mapping], "cpu", args.agent_of_interest_edges, args.edge_distance_treshold,
        args.distance_based_edges)

    mapping.update(
        {"scenes": scenes[0], "scene_timestamps": scene_timestamps[0]})

    return mapping


def argoverse_get_instance(lines, file_name, args):
    """
    Extract polylines from one example file content.
    """

    global max_vector_num
    vector_num = 0
    id2info = {}
    mapping = {}
    mapping['file_name'] = file_name

    for i, line in enumerate(lines):

        line = line.strip().split(',')
        if i == 0:
            mapping['start_time'] = float(line[TIMESTAMP])
            mapping['city_name'] = line[CITY_NAME]

        line[TIMESTAMP] = float(line[TIMESTAMP]) - mapping['start_time']
        line[X] = float(line[X])
        line[Y] = float(line[Y])
        id = line[TRACK_ID]

        if line[OBJECT_TYPE] == 'AV' or line[OBJECT_TYPE] == 'AGENT':
            line[TRACK_ID] = line[OBJECT_TYPE]

        if line[TRACK_ID] in id2info:
            id2info[line[TRACK_ID]].append(line)
            vector_num += 1
        else:
            id2info[line[TRACK_ID]] = [line]

        if line[OBJECT_TYPE] == 'AGENT' and len(id2info['AGENT']) == 20:
            assert 'AV' in id2info
            assert 'cent_x' not in mapping
            agent_lines = id2info['AGENT']
            mapping['cent_x'] = agent_lines[-1][X]
            mapping['cent_y'] = agent_lines[-1][Y]
            mapping['agent_pred_index'] = len(agent_lines)
            mapping['two_seconds'] = line[TIMESTAMP]
            span = agent_lines[-6:]
            intervals = [2]
            angles = []
            for interval in intervals:
                for j in range(len(span)):
                    if j + interval < len(span):
                        der_x, der_y = span[j + interval][X] - \
                            span[j][X], span[j + interval][Y] - span[j][Y]
                        angles.append([der_x, der_y])

            der_x, der_y = agent_lines[-1][X] - \
                agent_lines[-2][X], agent_lines[-1][Y] - agent_lines[-2][Y]

    if not args.test:
        assert len(id2info['AGENT']) == 50

    if vector_num > max_vector_num:
        max_vector_num = vector_num

    if 'cent_x' not in mapping:
        return None

    if args.validation:
        origin_labels = np.zeros([30, 2])
        for i, line in enumerate(id2info['AGENT'][20:]):
            origin_labels[i][0], origin_labels[i][1] = line[X], line[Y]
        mapping['origin_labels'] = origin_labels

    angle = -get_angle(der_x, der_y) + math.radians(90)

    angles = np.array(angles)
    der_x, der_y = np.mean(angles, axis=0)
    angle = -get_angle(der_x, der_y) + math.radians(90)

    mapping['angle'] = angle
    for id in id2info:
        info = id2info[id]
        for line in info:
            line[X], line[Y] = rotate(
                line[X] - mapping['cent_x'], line[Y] - mapping['cent_y'], angle)
        if 'scale' in mapping:
            scale = mapping['scale']
            line[X] *= scale
            line[Y] *= scale
    return preprocess(args, id2info, mapping)


def create_dataset(args):
    global am
    am = ArgoverseMap()
    files = []
    for each_dir in args.data_dir:
        root, dirs, cur_files = os.walk(each_dir).__next__()
        files.extend([os.path.join(each_dir, file) for file in cur_files if
                      file.endswith("csv") and not file.startswith('.')])
    print(files[:5], files[-5:])

    pbar = tqdm(total=len(files))
    queue = multiprocessing.Queue(args.core_num)
    queue_res = multiprocessing.Queue()

    def calc_ex_list(queue, queue_res, args):
        res = []
        while True:
            file = queue.get()
            if file is None:
                break
            if file.endswith("csv"):
                with open(file, "r", encoding='utf-8') as fin:
                    lines = fin.readlines()[1:]
                instance = argoverse_get_instance(
                    lines, file, args)
                if instance is not None:
                    data_compress = zlib.compress(
                        pickle.dumps(instance))
                    res.append(data_compress)
                    queue_res.put(data_compress)
                else:
                    queue_res.put(None)

    processes = [Process(target=calc_ex_list, args=(
        queue, queue_res, args,)) for _ in range(args.core_num)]
    for each in processes:
        each.start()

    for file in files:
        assert file is not None
        queue.put(file)
        pbar.update(1)

    while not queue.empty():
        pass

    pbar.close()
    ex_list = []

    pbar = tqdm(total=len(files))
    for i in range(len(files)):
        t = queue_res.get()
        if t is not None:
            ex_list.append(t)
        pbar.update(1)
    pbar.close()

    for i in range(args.core_num):
        queue.put(None)
    for each in processes:
        each.join()

    if args.test:
        ex_file_name = "test.ex_list"
    elif args.validation:
        ex_file_name = "eval.ex_list"
    else:
        ex_file_name = "ex_list"
    pickle_file = open(os.path.join(
        args.output_dir, ex_file_name), 'wb')
    pickle.dump(ex_list, pickle_file)
    pickle_file.close()
    assert len(ex_list) > 0
    print("valid data size is", len(ex_list))


if __name__ == '__main__':
    args = Arguments()
    create_dataset(args)
