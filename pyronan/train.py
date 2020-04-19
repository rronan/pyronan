import json
import logging
import math
import time
from argparse import ArgumentParser
from collections import defaultdict
from copy import deepcopy

import numpy as np
from pyronan.utils.misc import args2dict, checkpoint
from torch.utils.tensorboard import SummaryWriter

from tqdm import tqdm

parser_train = ArgumentParser(add_help=False)
parser_train.add_argument("--train_epochs", type=int, default=200)
parser_train.add_argument("--batch_size", type=int, help="batch size")
parser_train.add_argument("--num_workers", type=int, default=20)
parser_train.add_argument("--pin_memory", action="store_true")
parser_train.add_argument("--save_all", action="store_true")
parser_train.add_argument("--chkpt_interval", type=int, default=None)
parser_train.add_argument("--image_interval_val", type=int, default=None)
parser_train.add_argument("--image_interval_train", type=int, default=None)
parser_train.add_argument("--tensorboard", action="store_true")
parser_train.add_argument("--tensorboard_interval", type=int, default=1000)


def loss2str(set_, i, n, loss):
    loss = {
        k.replace("train_", "").replace("val_", ""): v
        for k, v in loss.items()
        if k != "epoch"
    }
    x = f"{set_} {i}/{n-1}," + " | ".join(f"{k}: {v:.3e}" for k, v in loss.items())
    return x


class Trainer:
    def __init__(self, model=None, args=None, from_chkpt=None):
        self.model = model
        self.args = args
        self.chkpt_interval = getattr(args, "chkpt_inverval", None)
        self.image_interval = {
            set_: getattr(args, f"image_interval_{set_}", None)
            for set_ in ["train", "val"]
        }
        self.cutoff = getattr(args, "cutoff", 0)
        self.tensorboard = None
        if from_chkpt is not None:
            with open(from_chkpt / "log.json") as f:
                self.log = json.load(f)
        else:
            self.log = {"step": 1, "loss": []}
        if getattr(args, "tensorboard", False):
            self.tensorboard_interval = getattr(args, "tensorboard_interval", 100)
            self.tensorboard = SummaryWriter(log_dir=getattr(args, "checkpoint", "."))
            self.tensorboard.add_text(
                "args",
                json.dumps(args2dict(args), sort_keys=True, indent=4),
                global_step=0,
            )

    def start_epoch(self):
        self.t0 = time.time()
        epoch = len(self.log["loss"])
        self.log["loss"].append({"epoch": epoch})
        if self.tensorboard is not None:
            self.loss_buffer = defaultdict(list)
        return epoch

    def end_epoch(self, i):
        self.log["loss"][i]["lr"] = self.model.get_lr()
        self.log["loss"][i]["time"] = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - self.t0)
        )
        logging.info(f"saving checkpoint to {self.args.checkpoint}")
        checkpoint(f"{i:04d}", self.log, model=self.model, args=self.args)
        return self.log["loss"][i]

    def log_batch(self, loss, set_, i, j, last=False):
        train = set_ == "train"
        step = self.log["step"]
        log = self.log["loss"][i]
        for key, value in loss.items():
            x = log.get(f"{set_}_{key}", value)
            log[f"{set_}_{key}"] = (x * (j - 1) + value) / j
        if self.tensorboard is not None:
            for key, value in loss.items():
                self.loss_buffer[key].append(value)
            if (train and j % self.tensorboard_interval == 0) or last:
                for key, value in self.loss_buffer.items():
                    self.tensorboard.add_scalar(f"{key}/{set_}", np.mean(value), step)
                    self.loss_buffer = defaultdict(list)
        if (
            self.chkpt_interval is not None
            and step % self.chkpt_interval == 0
            and train
        ):
            checkpoint(f"{i:04d}", self.log, f"{set_}_step_{step}")
        if (
            self.image_interval[set_] is not None
            and j % self.image_interval[set_] == 0
            and self.tensorboard is not None
            and hasattr(self.model, "get_image")
        ):
            images = self.model.get_image(self.cutoff)
            self.tensorboard.add_images(f"{set_}/{j}", images)
            self.tensorboard.flush()
        self.log["step"] += 1
        return log

    def process_epoch(self, set_, loader, i, n):
        j, L, loss = 1, len(loader), {}
        pbar = tqdm(total=L, dynamic_ncols=True, leave=False)
        iterator = iter(loader)
        while True:
            pbar.update(n=1)
            try:
                batch = next(iterator)
            except (RuntimeError, TimeoutError) as e:
                # this is a fix to hanging pytorch dataloader in some multithreaded cases
                print("*" * 80)
                logging.warning(f"Exception caught in process_epoch(): \n {e}")
                j, L, loss = 1, len(loader), {}
                pbar = tqdm(total=L, dynamic_ncols=True, leave=False)
                iterator = iter(deepcopy(loader))
                continue
            except StopIteration:
                break
            loss = self.model.step(batch, set_)
            loss_avg = self.log_batch(loss, set_, i, j, last=(j >= L))
            desc_short = f'{set_} {i}/{n - 1} | {loss_avg[set_ + "_loss"]:.3e}'
            try:
                desc = loss2str(set_, i, n, loss_avg)
                pbar.set_description(desc=desc, desc_short=desc_short)
            except Exception as e:
                pbar.set_description(desc=desc_short)
            j += 1
        return loss

    def train(self, loader_dict, train_epochs):
        i = 0
        while i < train_epochs:
            i = self.start_epoch()
            for set_, loader in loader_dict.items():
                self.process_epoch(set_, loader, i, train_epochs)
            log = self.end_epoch(i)
            print(log)
            self.model.lr_scheduler.step(log["val_loss"])
            if self.model.get_lr() < 5e-8 or math.isnan(log["train_loss"]):
                break
