import argparse
import inspect
import json
import os
import pathlib
import pdb
import pickle
import re
import time
from functools import wraps


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def debug(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if os.environ.get("PYRONAN_DEBUG"):
            return func(*args, **kwargs)
        try:
            func(*args, **kwargs)
        except Exception as e:
            print(e, "\n", "*" * 80)
            print("function:", inspect.getsource(func))
            print("args:", args)
            print("kwargs:", kwargs)
            pdb.set_trace()

    return wrapper


def init_shared_dict():
    if os.environ.get("DISABLE_MP_CACHE"):
        print("mp_cache() is disabled")
        return None
    from multiprocessing import Manager

    return Manager().dict()


def mp_cache(mp_dict):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if mp_dict is None:
                return func(*args, **kwargs)
            k = func.__name__
            k += "_".join(map(str, args))
            k += "_".join(map(lambda k, v: f"{k}_{v}", kwargs.items()))
            if k in mp_dict:
                return mp_dict[k]
            res = func(*args, **kwargs)
            mp_dict[k] = res
            return res

        return wrapper

    return decorate


def parse_slice(s):
    a_list = []
    for a in s.split(":"):
        try:
            a_list.append(int(a))
        except ValueError:
            a_list.append(None)
    while len(a_list) < 3:
        a_list.append(None)
    return slice(*a_list)


def append_timestamp(name, end=False):
    if re.search("[\d]{6}_[\d]{6}", name):
        return name
    else:
        if end:
            return name + "_" + time.strftime("%y%m%d_%H%M%S")
        else:
            return time.strftime("%y%m%d_%H%M%S") + "_" + name


def checkpoint(epoch, log, model=None, args=None, path=None):
    if path is None:
        path = pathlib.Path(args.checkpoint)
    path.mkdir(exist_ok=True)
    if args is not None:
        with open(path / "args.json", "w") as f:
            dict_ = {k: str(v) for k, v in vars(args).items()}
            json.dump(dict_, f, indent=4, sort_keys=True)
        with open(path / "args.pickle", "wb") as f:
            pickle.dump(args, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(path / "log.json", "w") as f:
        json.dump(log, f, indent=4)
    if getattr(args, "save_all", False):
        model.save(path, epoch)
    model.save(path, "last")
    last_val_loss = log["loss"][-1].get("val_loss")
    if last_val_loss is not None:
        for loss in log["loss"][:-1]:
            if "val_loss" in loss and loss["val_loss"] < last_val_loss:
                return
        model.save(path, "best")
