import logging
import time
import datetime
import sys
import argparse
import torch
import numpy as np
from pathlib import Path
import os
from tqdm import tqdm
import utils.utils
import random
import pickle

from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# DDP related functions
from utils.utils import setup

# Dataset related functions
from utils.dataloader import Argoverse_Edge_Dataset
from utils.utils import batch_list_to_batch_tensors

# model related
from utils.get_model import get_model_v2, save_model_v2

# evaluation related
from utils.utils import eval_instance_argoverse, post_eval

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
random.seed(0)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser("Forecasting TGN")

# === Data Related Parameters ===
parser.add_argument('--ex_file_path', type=str,
                    help="Path of preprocessed train scene list")
parser.add_argument('--val_ex_file_path', type=str,
                    help="Path of preprocessed validation scene list")
parser.add_argument('--test_ex_file_path', type=str,
                    help="Path of preprocessed test scene list", default=None)
parser.add_argument('--data_augmentation', action="store_true",
                    help="If data augmentation is applied to scenes")

# === Test Evaluation Related Parameters ===
parser.add_argument('--test', action="store_true")
parser.add_argument('--validate', action="store_true")
parser.add_argument('--output_path', type=str, default=None)

# === Common Hyperparameters ===
parser.add_argument('--feature_dim', type=int,
                    default=128, help="Dimenson of features")
parser.add_argument('--batch_size', type=int, default=64,
                    help="Batch size as scene")
parser.add_argument('--epoch', type=int, default=36,
                    help="Number of epochs")
parser.add_argument('--learning_rate', type=float,
                    default=0.0001, help="Learning rate")

# === Vectornet Related Parameters ===
parser.add_argument('--pretrained_vectornet', action="store_true",
                    help="If backbone is initialized from pretrained weights")
parser.add_argument('--pretrain_vectornet_path', type=str, default=None,
                    help="Path of pretrained vectornet module")

# === Temporal Graph Related Parameters ===
parser.add_argument('--exclude_temporal_encoding', action="store_true",
                    help="If Temporal Encoding is discarded in training")
parser.add_argument('--time_split_size', type=int, default=2,
                    help="Number of timestamps for an edge update")
parser.add_argument('--fuse_time', action="store_true",
                    help="If sinusoidal time encoding applied in TG")
parser.add_argument('--sequential_encoding', action="store_true",
                    help="If sequential agent memory is used")
parser.add_argument('--scene_memory', action="store_true",
                    help="If scene memory is used")

# === Model Saving/Loading Parameters ===
parser.add_argument('--model_save_path', type=str, default=None,
                    help="Path to save per epoch model")
parser.add_argument('--pretrain_path', type=str, default=None,
                    help="Path of pretrained/checkpoint model")
parser.add_argument('--load_epoch', type=int, default=None,
                    help="Epoch of training to be loaded")
parser.add_argument('--training_name', type=str, default=None,
                    help="Name of the training to be saved as")

# === Misc Training Parameters ===
parser.add_argument('--world_size', type=int, default=4,
                    help="Number of available GPUs")
parser.add_argument('--validation_epoch', type=int,
                    default=2, help="Validation per n epoch")

# === Endpoint Prediction Parameters ===
parser.add_argument('--endpoint_source', type=str,
                    default="both", choices=["agent", "context", "both"], help="Which features endpoints are generated from")
parser.add_argument('--temperature', type=float,
                    default=0.3, help="Temperature of softmax")
parser.add_argument('--temperature_validation', action="store_true",
                    help="If aplly temperature on validation")


# === Loss Coefficients ===
parser.add_argument('--lambda1', type=float, default=1.0,
                    help="Trajectory L1 loss coefficient")
parser.add_argument('--lambda2', type=float, default=1.0,
                    help="Endpoint L1 loss coefficient")
parser.add_argument('--lambda3', type=float, default=1.0,
                    help="Endpoint CE loss coefficient")

try:
    args = parser.parse_args()
except:
    parser.print_help()
    sys.exit(0)


class Arguments:
    def __init__(self):
        self.ex_file_path = args.ex_file_path
        self.val_ex_file_path = args.val_ex_file_path
        self.test_ex_file_path = args.test_ex_file_path

        self.data_augmentation = True if args.data_augmentation else False

        self.test = True if args.test else False
        self.validate = True if args.validate else False
        self.output_path = args.output_path

        self.feature_dim = args.feature_dim
        self.batch_size = args.batch_size
        self.epoch_num = args.epoch
        self.learning_rate = args.learning_rate

        self.pretrained_vectornet = True if args.pretrained_vectornet else False
        self.pretrain_vectornet_path = args.pretrain_vectornet_path

        self.exclude_temporal_encoding = True if args.exclude_temporal_encoding else False
        self.time_split_size = args.time_split_size
        self.fuse_time = True if args.fuse_time else False
        self.sequential_encoding = True if args.sequential_encoding else False
        self.scene_memory = True if args.scene_memory else False

        self.save_path = args.model_save_path
        self.pretrain_path = args.pretrain_path
        self.load_epoch = args.load_epoch
        self.training_name = args.training_name

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.world_size = args.world_size
        self.validation_epoch = args.validation_epoch

        self.endpoint_source = args.endpoint_source
        self.temperature = args.temperature
        self.temperature_validation = True if args.temperature_validation else False

        self.lambda1 = args.lambda1
        self.lambda2 = args.lambda2
        self.lambda3 = args.lambda3

        if self.load_epoch is not None:
            assert self.pretrain_path is not None

        if self.pretrained_vectornet:
            assert self.pretrain_vectornet_path is not None


def create_logger():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    Path("log/").mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler('log/{}.log'.format(str(time.time())))
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARN)
    formatter = logging.Formatter(
      '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(args)
    return logger


def test_evaluate(args, model, logger):
    from argoverse.evaluation.competition_util import generate_forecasting_h5

    test_dataset = Argoverse_Edge_Dataset(args)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size//args.world_size,
                                 pin_memory=False, drop_last=False, shuffle=False, collate_fn=batch_list_to_batch_tensors)
    file2pred = {}
    file2prob = {}
    logger.info("Test Prediction starting")
    iter_bar = tqdm(test_dataloader)
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(iter_bar):
            pred_trajectory, pred_probs = model(batch)

            batch_size = pred_trajectory.shape[0]

            for i in range(batch_size):
                a_pred = pred_trajectory[i]
                a_prob = pred_probs[i]
                assert a_pred.shape == (6, 30, 2)
                assert a_prob.shape == (6, )
                file_name_int = int(os.path.split(
                    batch[i]['file_name'])[1][:-4])
                file2pred[file_name_int] = a_pred
                file2prob[file_name_int] = a_prob

    generate_forecasting_h5(
        data=file2pred, probabilities=file2prob, output_path=args.output_path)


def train(model, iter_bar, optimizer, rank, dataset, main_device, args, logger, writer, scheduler, i_epoch):
    total_loss = 0.0
    for step, batch in enumerate(iter_bar):

        traj_loss = model(batch)
        loss = traj_loss
        total_loss += loss.item()

        loss.backward()
        if main_device:
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            loss_desc = f"lr = {lr:.6f} loss = {total_loss/((step+1)*args.batch_size):.5f}"
            iter_bar.set_description(loss_desc)

            writer.add_scalar(
                "traj_loss/batch", traj_loss.item(), step + i_epoch*len(iter_bar))

        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()


def validate(model, dataloader, dataset, logger):
    file2pred = {}
    file2probs = {}
    file2labels = {}
    DEs = []
    iter_bar = tqdm(dataloader, desc='Iter (loss=X.XXX)')

    with torch.no_grad():
        for step, batch in enumerate(iter_bar):

            pred_trajectory, pred_probs = model(batch, True)
            batch_size = pred_trajectory.shape[0]
            for i in range(batch_size):
                assert pred_trajectory[i].shape == (6, 30, 2)
                assert pred_probs[i].shape == (6, )

            # batch = [scene[0] for scene in batch]
            eval_instance_argoverse(
                batch_size, pred_trajectory, pred_probs, batch, file2pred, file2labels, file2probs, DEs, iter_bar, step == 0)

    post_eval(file2pred, file2labels, file2probs, DEs, logger)
    predictions = {"file2pred": file2pred,
                   "file2probs": file2probs,
                   "file2labels": file2labels}
    return predictions


def main(rank, args):
    main_device = True if rank == 0 else False
    args.device = rank

    logger = create_logger() if main_device else None
    writer = SummaryWriter() if main_device else None

    if main_device and args.save_path is not None:
        assert os.path.exists(args.save_path)
        now = datetime.datetime.now()
        if args.training_name is not None:
            args.save_path = f"{args.save_path}/{args.training_name}_{now.hour}:{now.minute}_{now.day}-{now.month}-{now.year}"
        else:
            args.save_path = f"{args.save_path}/{now.hour}:{now.minute}_{now.day}-{now.month}-{now.year}"
        os.mkdir(args.save_path)

    world_size = args.world_size
    setup(rank, world_size)

    model, optimizer, start_epoch, scheduler = get_model_v2(args)

    if args.test:
        if main_device:
            test_evaluate(args, model, logger)

        dist.barrier()
        dist.destroy_process_group()
        return

    if not args.validate:
        # set train dataset and dataloader
        train_dataset = Argoverse_Edge_Dataset(args)
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=args.world_size, rank=rank, shuffle=True, drop_last=False)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size//world_size,
                                      pin_memory=False, drop_last=False, shuffle=False, sampler=train_sampler,
                                      collate_fn=batch_list_to_batch_tensors)

    # if main device, load validation dataset
    if main_device:
        val_dataset = Argoverse_Edge_Dataset(args, validation=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size//world_size,
                                    pin_memory=False, drop_last=False, shuffle=False, collate_fn=batch_list_to_batch_tensors)

    if args.validate:
        if main_device:
            model.eval()
            file2pred = validate(model, val_dataloader, val_dataset, logger)

            pred_save_path = os.path.join(
                args.save_path, "predictions")
            pickle_file = open(pred_save_path, "wb")
            pickle.dump(file2pred, pickle_file,
                        protocol=pickle.HIGHEST_PROTOCOL)
            pickle_file.close()

        dist.barrier()
        dist.destroy_process_group()
        return

    for i_epoch in range(start_epoch, args.epoch_num):
        if main_device:
            logger.info(
                f"====== [Epoch {i_epoch + 1}/{args.epoch_num}] ======")
        train_sampler.set_epoch(i_epoch)

        if main_device:
            iter_bar = tqdm(train_dataloader, desc='Iter (loss=X.XXX)')
        else:
            iter_bar = train_dataloader

        model.train()
        train(model, iter_bar, optimizer, rank,
              train_dataset, main_device, args, logger, writer, scheduler, i_epoch)

        if main_device:
            if args.save_path is not None:
                save_model_v2(args, i_epoch, model, optimizer, scheduler)

            if (i_epoch % args.validation_epoch == args.validation_epoch - 1):
                model.eval()
                predictions = validate(model, val_dataloader, val_dataset,
                                       logger)
                if i_epoch == args.epoch_num - 1:
                    try:
                        pred_save_path = os.path.join(
                            args.save_path, "predictions")
                        pickle_file = open(pred_save_path, "wb")
                        pickle.dump(predictions, pickle_file,
                                    protocol=pickle.HIGHEST_PROTOCOL)
                        pickle_file.close()
                    except:
                        logger.info("Prediction could not be saved")

            logger.info("====== ====== ======\n")

        dist.barrier()

    dist.destroy_process_group()


if __name__ == '__main__':
    args = Arguments()
    world_size = args.world_size
    torch.set_num_threads(args.world_size*6)
    mp.spawn(main, args=[args], nprocs=world_size)
