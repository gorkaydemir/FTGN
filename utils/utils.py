import os
import math
import torch
import torch.distributed as dist
import numpy as np
import random
from argoverse.evaluation import eval_forecasting
from datetime import datetime


def setup(rank, world_size):
    now = datetime.now()
    s = int(now.second)
    m = int(now.minute)

    os.environ['MASTER_ADDR'] = "localhost"
    os.environ['MASTER_PORT'] = f"{12300 + s*2 + m*5}"
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def rotate(x, y, angle):
    res_x = x * math.cos(angle) - y * math.sin(angle)
    res_y = x * math.sin(angle) + y * math.cos(angle)
    return res_x, res_y


origin_point = None
origin_angle = None


def get_labels(labels, device):
    batch_size = len(labels)
    gt_points = []
    gt_points_torch = []
    gt_endpoints_torch = []
    gt_endpoints_np = []

    for i in range(batch_size):
        gt_points_i = np.array(labels[i]).reshape([30, 2])
        gt_points_torch_i = torch.tensor(
            gt_points_i, device=device, dtype=torch.float)
        gt_endpoints_np_i = gt_points_i[-1]
        gt_endpoints_torch_i = gt_points_torch_i[-1]

        gt_points.append(gt_points_i)
        gt_points_torch.append(gt_points_torch_i)
        gt_endpoints_np.append(gt_endpoints_np_i)
        gt_endpoints_torch.append(gt_endpoints_torch_i)

    label_dict = {"gt_points": gt_points,
                  "gt_points_torch": gt_points_torch,
                  "gt_endpoints_np": gt_endpoints_np,
                  "gt_endpoints_torch": gt_endpoints_torch}
    return label_dict


def batch_init(mapping):
    global origin_point, origin_angle
    batch_size = len(mapping)

    origin_point = np.zeros([batch_size, 2])
    origin_angle = np.zeros([batch_size])
    for i in range(batch_size):
        origin_point[i][0], origin_point[i][1] = rotate(0 - mapping[i]['cent_x'], 0 - mapping[i]['cent_y'],
                                                        mapping[i]['angle'])
        origin_angle[i] = -mapping[i]['angle']


def to_origin_coordinate(points, idx_in_batch, scale=None):
    for point in points:
        point[0], point[1] = rotate(point[0] - origin_point[idx_in_batch][0],
                                    point[1] - origin_point[idx_in_batch][1], origin_angle[idx_in_batch])
        if scale is not None:
            point[0] *= scale
            point[1] *= scale


def get_from_mapping(mapping, key):
    return [each[key] for each in mapping]


def get_dis_point_2_points(point, points):
    assert points.ndim == 2
    return np.sqrt(np.square(points[:, 0] - point[0]) + np.square(points[:, 1] - point[1]))


def merge_tensors(tensors, device, hidden_size):
    lengths = []
    for tensor in tensors:
        lengths.append(tensor.shape[0] if tensor is not None else 0)
    res = torch.zeros([len(tensors), max(lengths), hidden_size], device=device)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            res[i][:tensor.shape[0]] = tensor
    return res, lengths


def merge_tensors_not_add_dim(tensor_list_list, module, sub_batch_size, device, hidden_size):
    batch_size = len(tensor_list_list)
    output_tensor_list = []
    for start in range(0, batch_size, sub_batch_size):
        end = min(batch_size, start + sub_batch_size)
        sub_tensor_list_list = tensor_list_list[start:end]
        sub_tensor_list = []
        for each in sub_tensor_list_list:
            sub_tensor_list.extend(each)
        inputs, lengths = merge_tensors(
            sub_tensor_list, device=device, hidden_size=hidden_size)
        outputs = module(inputs, lengths)
        sub_output_tensor_list = []
        sum = 0
        for each in sub_tensor_list_list:
            sub_output_tensor_list.append(outputs[sum:sum + len(each)])
            sum += len(each)
        output_tensor_list.extend(sub_output_tensor_list)
    return output_tensor_list


method2FDEs = []


def batch_list_to_batch_tensors(batch):
    return [each for each in batch]


def __iter__(self):  # iterator to load data
    for __ in range(math.ceil(len(self.ex_list) / float(self.batch_size))):
        batch = []
        for __ in range(self.batch_size):
            idx = random.randint(0, len(self.ex_list) - 1)
            batch.append(self.__getitem__(idx))
        # To Tensor
        yield batch_list_to_batch_tensors(batch)


def eval_instance_argoverse(batch_size, pred, pred_probs, mapping, file2pred, file2labels, file2probs, DEs, iter_bar, first_time):
    global method2FDEs
    if first_time:
        method2FDEs = []

    for i in range(batch_size):
        a_pred = pred[i]
        a_prob = pred_probs[i]
        # a_endpoints = all_endpoints[i]
        assert a_pred.shape == (6, 30, 2)
        assert a_prob.shape == (6, )

        file_name_int = int(os.path.split(mapping[i]['file_name'])[1][:-4])
        file2pred[file_name_int] = a_pred
        file2labels[file_name_int] = mapping[i]['origin_labels']
        file2probs[file_name_int] = a_prob

    DE = np.zeros([batch_size, 30])
    for i in range(batch_size):
        origin_labels = mapping[i]['origin_labels']
        FDE = np.min(get_dis_point_2_points(
                origin_labels[-1], pred[i, :, -1, :]))
        method2FDEs.append(FDE)
        for j in range(30):
            DE[i][j] = np.sqrt((origin_labels[j][0] - pred[i, 0, j, 0]) ** 2 + (
                    origin_labels[j][1] - pred[i, 0, j, 1]) ** 2)
    DEs.append(DE)
    miss_rate = 0.0
    miss_rate = np.sum(np.array(method2FDEs) > 2.0) / len(method2FDEs)

    iter_bar.set_description('Iter (MR=%5.3f)' % (miss_rate))


def post_eval(file2pred, file2labels, file2probs, DEs, logger):

    metric_results = eval_forecasting.get_displacement_errors_and_miss_rate(
        file2pred, file2labels, 6, 30, 2.0, file2probs)

    for key in metric_results.keys():
        logger.info(f"{key}: {metric_results[key]:.5f}")

    DE = np.concatenate(DEs, axis=0)
    length = DE.shape[1]
    DE_score = [0, 0, 0, 0]
    for i in range(DE.shape[0]):
        DE_score[0] += DE[i].mean()
        for j in range(1, 4):
            index = round(float(length) * j / 3) - 1
            assert index >= 0
            DE_score[j] += DE[i][index]
    for j in range(4):
        score = DE_score[j] / DE.shape[0]
        logger.info(
            f" {'ADE' if j == 0 else 'DE@1' if j == 1 else 'DE@2' if j == 2 else 'DE@3'}: {score:.5f}")
