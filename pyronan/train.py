import math
import time
from argparse import ArgumentParser

from tqdm import tqdm

from pyronan.utils.misc import Nop

parser_train = ArgumentParser(add_help=False)
parser_train.add_argument("--train_epochs", type=int, default=200)
parser_train.add_argument("--bsz", type=int, help="batch size")
parser_train.add_argument("--num_workers", type=int, default=20)
parser_train.add_argument("--pin_memory", action="store_true")
parser_train.add_argument("--save_all", action="store_true")
parser_train.add_argument("--save_last", action="store_true")
parser_train.add_argument("--subcheck", type=int, default=None)
parser_train.add_argument("--tensorboard", action="store_true")


def process_batch(model, batch, loss_avg, set_, j):
    loss = model.step(batch, set_)
    for key, value in loss.items():
        try:
            loss_avg[key] = (loss_avg[key] * (j - 1) + value) / j
        except KeyError:
            loss_avg[key] = value
    return loss, loss_avg


def _loss2str(set_, i, n, loss, verbose):
    if verbose:
        x = f"{set_} {i}/{n-1}, " + " | ".join(f"{k}: {v:.3e}" for k, v in loss.items())
    else:
        loss_value = loss["loss"]
        x = f"{set_} {i}/{n - 1} | {loss_value:.3e}"
    return x


def process_epoch(model, set_, loader, i, n, verbose, callback):
    loss = {}
    pbar = tqdm(loader, dynamic_ncols=True, leave=False)
    for j, batch in enumerate(pbar, 1):
        loss = process_batch(model, batch, loss, set_, j)
        pbar.set_description(_loss2str(set_, i, n, loss, verbose))
        callback.step(loss, set_, i, j)
    return loss


class DummyCallback(Nop):
    step = 0
    interval = None


def trainer(model, loader_dict, train_epochs, verbose=True, callback=DummyCallback):
    for i in range(train_epochs):
        callback.start_epoch(i)
        for set_, loader in loader_dict.items():
            process_epoch(model, set_, loader, i, train_epochs, verbose, callback)
        log = callback.epoch(i)
        print(log)
        model.lr_scheduler.step(log["val_loss"])
        if model.get_lr() < 5e-8 or math.isnan(log["train_loss"]):
            break
