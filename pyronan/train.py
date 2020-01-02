import math
import time
from argparse import ArgumentParser
from pydoc import locate

from path import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from pyronan.utils.misc import append_timestamp

parser_train = ArgumentParser(add_help=False)
parser_train.add_argument("--bsz", type=int, help="batch size")
parser_train.add_argument("--checkpoint", type=Path)
parser_train.add_argument("--seed", type=int, default=0)
parser_train.add_argument("--name", type=append_timestamp, default="")
parser_train.add_argument("--data_parallel", action="store_true")
parser_train.add_argument("--gpu", action="store_true", help="Use NVIDIA GPU")
parser_train.add_argument("--num_workers", type=int, default=20)
parser_train.add_argument("--n_epochs", type=int, default=200)
parser_train.add_argument("--pin_memory", action="store_true")
parser_train.add_argument("--save_all", action="store_true")
parser_train.add_argument("--save_last", action="store_true")
parser_train.add_argument("--subcheck", type=float, default=None)


def make_model(model, args, load=None, gpu=False, data_parallel=False):
    model = locate(model)(args)
    if load is not None:
        model.load(load)
    if gpu:
        model.gpu()
    if data_parallel:
        model.data_parallel()
    print(f"n parameters: {model.get_num_parameters()}")
    return model


def make_loader(dataset, args, set_, bsz, num_workers, pin_memory, shuffle=True):
    dataset = locate(dataset)(args, set_)
    loader = DataLoader(
        dataset,
        batch_size=bsz,
        num_workers=num_workers,
        shuffle=shuffle,
        drop_last=True,
        pin_memory=pin_memory,
    )
    return loader


def process_batch(model, batch, loss, set_, j):
    res = model.step(batch, set_)
    for key, value in res.items():
        try:
            loss[key] = (loss[key] * j + value) / (j + 1)
        except KeyError:
            loss[key] = value
    return loss


def process_epoch(model, set_, loader, log, i, n, callback=None, verbose=True):
    # model.requires_grad_(set_ == "train")
    loss = {}
    pbar = tqdm(loader, dynamic_ncols=True, leave=False)
    for j, batch in enumerate(pbar):
        loss = process_batch(model, batch, loss, set_, j)
        if verbose:
            pbar.set_description(
                f"{set_} {i}/{n - 1}, "
                + " | ".join(f"{k}: {v:.3e}" for k, v in loss.items())
            )
        else:
            pbar.set_description(f"{set_} {i}/{n - 1}")
        for key, value in loss.items():
            log[i][f"{set_}_{key}"] = value
        if callback is not None and (j + 1) % callback[1] == 0:
            callback[0](f"{i:03d}_{j:06d}_", log)
    return log


def trainer(model, loader_dict, n_epochs, checkpoint_func, subcheck=None, verbose=True):
    callback = (checkpoint_func, subcheck) if subcheck is not None else None
    log = []
    for i in range(n_epochs):
        t0 = time.time()
        log.append({"epoch": i})
        for set_, loader in loader_dict.items():
            process_epoch(model, set_, loader, log, i, n_epochs, callback, verbose)
        log[i]["lr"] = model.get_lr()
        log[i]["time"] = time.strftime("%H:%M:%S", time.gmtime(time.time() - t0))
        checkpoint_func(f"{i:03d}", log)
        print(log[i])
        model.scheduler.step(log[-1]["val_loss"])
        if model.get_lr() < 5e-8 or math.isnan(log[-1]["train_loss"]):
            break
