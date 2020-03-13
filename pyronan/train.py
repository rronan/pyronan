import gc
import json
import logging
import math
import time
from argparse import ArgumentParser
from pathlib import Path

from pympler.tracker import SummaryTracker
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pyronan.utils.misc import args2dict, checkpoint

parser_train = ArgumentParser(add_help=False)
parser_train.add_argument("--train_epochs", type=int, default=200)
parser_train.add_argument("--batch_size", type=int, help="batch size")
parser_train.add_argument("--num_workers", type=int, default=20)
parser_train.add_argument("--pin_memory", action="store_true")
parser_train.add_argument("--save_all", action="store_true")
parser_train.add_argument("--save_last", action="store_true")
parser_train.add_argument("--chkpt_interval", type=int, default=None)
parser_train.add_argument("--image_interval_val", type=int, default=None)
parser_train.add_argument("--image_interval_train", type=int, default=None)
parser_train.add_argument("--gc_collect_interval", type=int, default=None)
parser_train.add_argument("--tensorboard", action="store_true")
parser_train.add_argument("--step", type=int, default=0)


class Callback:
    def __init__(self, model=None, args=None, from_chkpt=None):
        self.model = model
        self.args = args
        self.chkpt_interval = getattr(args, "chkpt_inverval", None)
        self.image_interval = {
            set_: getattr(args, f"image_interval_{set_}", None)
            for set_ in ["train", "val"]
        }
        self.gc_collect_interval = getattr(args, "gc_collect_interval", None)
        self.cutoff = getattr(args, "cutoff", 0)
        self.tensorboard = None
        self.is_graph_written = False
        if from_chkpt is not None:
            with open(from_chkpt / "log.json") as f:
                self.log = json.load(f)
        else:
            self.log = {"step": 0, "loss": []}
        if getattr(args, "tensorboard", False):
            self.tensorboard = SummaryWriter(log_dir=getattr(args, "checkpoint", "."))
            self.tensorboard.add_text(
                "args", json.dumps(args2dict(args), sort_keys=True, indent=4)
            )

    def start_epoch(self):
        self.t0 = time.time()
        epoch = len(self.log["loss"])
        self.log["loss"].append({"epoch": epoch})
        return epoch

    def end_epoch(self, i):
        self.log["loss"][i]["lr"] = self.model.get_lr()
        self.log["loss"][i]["time"] = time.strftime(
            "%H:%M:%S", time.gmtime(time.time() - self.t0)
        )
        logging.info(f"saving checkpoint to {self.args.checkpoint}")
        checkpoint(f"{i:04d}", self.log, model=self.model, args=self.args)
        self.add_graph()
        return self.log["loss"][i]

    def batch(self, loss, set_, i, j):
        step = self.log["step"]
        log = self.log["loss"][i]
        for key, value in loss.items():
            if j == 1:
                log[f"{set_}_{key}"] = value
            else:
                log[f"{set_}_{key}"] *= j - 1
                log[f"{set_}_{key}"] += value
                log[f"{set_}_{key}"] /= j
        if self.tensorboard is not None:
            for key, value in loss.items():
                self.tensorboard.add_scalar(f"{key}/{set_}", value, step)
        if (
            self.chkpt_interval is not None
            and step % self.chkpt_interval == 0
            and set_ == "train"
        ):
            checkpoint(f"{i:04d}", self.log, f"{set_}_step_{step}")
        if (
            self.image_interval[set_] is not None
            and j % self.image_interval[set_] == 0
            and self.tensorboard is not None
            and hasattr(self.model, "get_image")
        ):
            self.tensorboard.add_images(
                f"{set_}/{j}", self.model.get_image(self.cutoff), i
            )
            self.tensorboard.flush()
        if self.gc_collect_interval is not None and j % self.gc_collect_interval == 0:
            if self.tensorboard is not None:
                self.tensorboard.flush()
            gc.collect()
        self.log["step"] += 1
        return log

    def add_graph(self, overwrite=False):
        if self.is_graph_written and not overwrite:
            return
        try:
            tensorboard = getattr(self, "tensorboard", None)
            if tensorboard is not None and hasattr(self.model, "get_input_to_model"):
                tensorboard.add_graph(self.model, self.model.get_input_to_model())
                self.is_graph_written = True
        except Exception as e:
            print("Exception in Callback.add_graph():", e)
            self.is_graph_written = True


def loss2str(set_, i, n, loss, verbose):
    if verbose:
        loss = {
            k.replace("train_", "").replace("val_", ""): v
            for k, v in loss.items()
            if k != "epoch"
        }
        x = f"{set_} {i}/{n-1}," + " | ".join(f"{k}: {v:.3e}" for k, v in loss.items())
    else:
        x = f"{set_} {i}/{n - 1} | {loss[set_]:.3e}"
    return x


def process_epoch(model, set_, loader, i, n, verbose, callback):
    loss = {}
    pbar = tqdm(loader, dynamic_ncols=True, leave=False)
    # tracker = SummaryTracker()
    for j, batch in enumerate(pbar, 1):
        loss = model.step(batch, set_)
        loss_avg = callback.batch(loss, set_, i, j)
        pbar.set_description(loss2str(set_, i, n, loss_avg, verbose))
    # tracker.print_diff()
    return loss


def trainer(model, loader_dict, train_epochs, callback=Callback(), verbose=True):
    i = 0
    while i < train_epochs:
        i = callback.start_epoch()
        for set_, loader in loader_dict.items():
            process_epoch(model, set_, loader, i, train_epochs, verbose, callback)
        log = callback.end_epoch(i)
        print(log)
        model.lr_scheduler.step(log["val_loss"])
        if model.get_lr() < 5e-8 or math.isnan(log["train_loss"]):
            break
