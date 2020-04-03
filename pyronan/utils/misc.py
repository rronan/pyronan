import argparse
import inspect
import json
import os
import pathlib
import pdb
import re
import time
from copy import copy
from functools import wraps

from tqdm import tqdm


class Nop(object):
    def __init__(self):
        pass

    def nop(self, *foo, **bar):
        pass

    def __getattr__(self, _):
        return self.nop


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i : i + n]


def tqdm_(x, *args, **kwargs):
    if type(x) is int:
        x = range(x)
    if os.environ.get("DISABLE_TQDM"):
        return x
    return tqdm(x, *args, **kwargs, dynamic_ncols=True)


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


def write_slice(s):
    return f"{s.start if s.start is not None else ''}:{s.stop if s.stop is not None else ''}"


def write_namespace(s):
    out = []
    for k, v in vars(s).items():
        if type(v) is not bool or v:
            out.append("--" + k)
            if type(v) is bool:
                out.append('""')
            elif type(v) is slice:
                out.append(write_slice(v))
            else:
                out.append(str(v))
    return " ".join(out)


class to_namespace:
    def __init__(self, d):
        vars(self).update(dict([(key, value) for key, value in d.items()]))

    def __str__(self):
        return str(vars(self))


def append_timestamp(name, end=False):
    if re.search("[\d]{6}_[\d]{6}", name):
        return name
    else:
        if end:
            return name + "_" + time.strftime("%y%m%d_%H%M%S")
        else:
            return time.strftime("%y%m%d_%H%M%S") + "_" + name


def tostring(v):
    if isinstance(v, pathlib.PosixPath):
        return str(v)
    if type(v) is slice:
        return write_slice(v)
    elif type(v) is argparse.Namespace:
        return write_namespace(v)
    return v


def args2dict(args):
    args_copy = copy(args)
    for k, v in vars(args_copy).items():
        if type(v) is list:
            vars(args_copy)[k] = [tostring(w) for w in v]
        else:
            vars(args_copy)[k] = tostring(v)
    return vars(args_copy)


def save_args(path, args):
    args_dict = args2dict(args)
    with open(path, "w") as f:
        json.dump(args_dict, f, indent=4, sort_keys=True)


def load_args(path, type_dict):
    with open(str(path), "r") as f:
        dict_ = json.load(f)
    for k, v in dict_.items():
        if k != "lr" and type_dict[k] is not None:
            if type(v) is list:
                v = [type_dict[k](e) for e in v]
            elif v is not None:
                v = type_dict[k](v)
        dict_[k] = v
    args = to_namespace(dict_)
    return args


def checkpoint(epoch, log, model=None, args=None, path=None):
    if path is None:
        path = pathlib.Path(args.checkpoint)
    path.mkdir(exist_ok=True)
    if args is not None:
        save_args(path / "args.json", args)
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
